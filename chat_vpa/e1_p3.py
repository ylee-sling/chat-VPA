# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_p3.py -- P3 of E1's Step 1 (docs/e1.md, Section 11.7): what LiteRT-LM's model file holds, and whether the builder
that ships with LiteRT-LM unpacks it and packs it again unchanged. It runs in LiteRT-LM's environment, without
torch: the builder (``litert-lm-builder``) is a dependency of ``litert-lm`` and carries the TFLite schema.

    python -m chat_vpa.e1_p3 inspect --model ASSETS/litert/gemma-4-E2B-it.litertlm \\
        --assistant ASSETS/models/gemma-4-E2B-it-assistant --out runs/e1-p3/inspect
    python -m chat_vpa.e1_p3 repack --model ASSETS/litert/gemma-4-E2B-it.litertlm --out runs/e1-p3/repack

``inspect`` reads the file and writes its report (``inspect.json``, ``inspect.md``) and the drafter's section:

- the header: the file format's version and the system metadata;
- every section: its data type, model type, other items, offsets, size and SHA-256;
- every TFLite section: its signatures (each input's and output's name, shape and type), its subgraphs, tensors and
  operators (counted by kind), and its weights' bytes by type and quantisation;
- the drafter's section (model type ``tf_lite_mtp_drafter``): every tensor (name, shape, type, quantisation, where
  its bytes are), and the section itself as ``drafter.tflite``;
- with ``--assistant`` (A1's directory with its safetensors): every weight tensor of the drafter's section that
  holds values in the section, dequantised, against every tensor of A1 with as many elements, as it is and, for two
  matrices, transposed: the closest, its relative error ||x - a|| / ||a||, and the next closest's. This is what
  route (a) of 11.7 needs to know: which of A1's weights each tensor holds, and what its quantisation costs.

``repack`` is the null test of route (a)'s tooling at the level of bytes: the builder's ``unpack`` writes the
sections and a ``model.toml``; the builder packs them again with every weight kept inside its model's graph and no
deduplication of shared weights (its defaults move weights out of the graphs); every section of the result is then
compared byte for byte with the original's. Identical sections can only behave identically wherever sections are
read as they are; the header can differ (the builder writes its own format version), and the engine test of 11.7
checks that this does not matter. It needs about twice the file's size free under ``--out``; the unpacked sections
are deleted after the comparison unless ``--keep`` is given, and the repacked file stays for the engine test.

Nothing here is a measurement of speed or of text; every number is read from the files (*reported* by them) or
computed from what they hold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import mmap
import shutil
import struct
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["dequantise", "inspect", "read_safetensors", "repack", "section_table", "tflite_summary"]

DRAFTER = "tf_lite_mtp_drafter"
_CHUNK = 1 << 24


def _schema() -> tuple[Any, Any, Any]:
    from litert_lm_builder import litertlm_core, litertlm_peek
    from litert_lm_builder import tflite_schema_py_generated as tfl

    return litertlm_core, litertlm_peek, tfl


def _names(enum: Any) -> dict[int, str]:
    return {v: k for k, v in vars(enum).items() if not k.startswith("_") and isinstance(v, int)}


# ---------------------------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------------------------

def _sha256(handle: Any, begin: int, end: int) -> str:
    digest = hashlib.sha256()
    handle.seek(begin)
    remaining = end - begin
    while remaining > 0:
        block = handle.read(min(_CHUNK, remaining))
        if not block:
            break
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def section_table(path: Path) -> dict[str, Any]:
    """The header's version and system metadata, and every section: data type, model type, items, offsets, size and
    SHA-256 of its bytes."""
    core, peek, _ = _schema()
    meta = peek.read_litertlm_header(str(path), io.StringIO())
    with open(path, "rb") as handle:
        handle.seek(8)
        version = list(struct.unpack("<III", handle.read(12)))
        system = meta.SystemMetadata()
        entries = [peek.kvp_to_dict(system.Entries(i)) for i in range(system.EntriesLength())] if system else []
        objects = meta.SectionMetadata()
        sections = []
        for i in range(objects.ObjectsLength() if objects else 0):
            o = objects.Objects(i)
            begin, end = int(o.BeginOffset()), int(o.EndOffset())
            sections.append({"index": i, "data_type": core.any_section_data_type_to_string(o.DataType()),
                             "model_type": peek.get_model_type(o),
                             "items": [peek.kvp_to_dict(o.Items(j)) for j in range(o.ItemsLength())],
                             "begin": begin, "end": end, "size": end - begin, "sha256": _sha256(handle, begin, end)})
    return {"path": str(path), "bytes": Path(path).stat().st_size, "version": version, "system_metadata": entries,
            "sections": sections}


# ---------------------------------------------------------------------------------------------
# TFLite graphs
# ---------------------------------------------------------------------------------------------

def _shape(tensor: Any) -> list[int]:
    return [] if tensor.ShapeIsNone() else [int(x) for x in tensor.ShapeAsNumpy()]


def _quantisation(tensor: Any, tfl: Any) -> dict[str, Any] | None:
    q = tensor.Quantization()
    if q is None or (q.ScaleLength() == 0 and q.DetailsType() == 0):
        return None
    out: dict[str, Any] = {"scales": int(q.ScaleLength()), "zero_points": int(q.ZeroPointLength()),
                           "dimension": int(q.QuantizedDimension()),
                           "details": _names(tfl.QuantizationDetails).get(int(q.DetailsType()), str(q.DetailsType()))}
    if q.ZeroPointLength():
        zeros = q.ZeroPointAsNumpy()
        out["zero_point_range"] = [int(zeros.min()), int(zeros.max())]
    if q.DetailsType() == tfl.QuantizationDetails.BlockwiseQuantization:
        details = tfl.BlockwiseQuantization()
        table = q.Details()
        details.Init(table.Bytes, table.Pos)
        out["block_size"] = int(details.BlockSize())
        out["block_scales_tensor"] = int(details.Scales())
    return out


def _storage(model: Any, tensor: Any) -> dict[str, Any]:
    if hasattr(tensor, "ExternalBuffer") and tensor.ExternalBuffer():
        return {"external_buffer": int(tensor.ExternalBuffer())}
    index = int(tensor.Buffer())
    if index <= 0 or index >= model.BuffersLength():
        return {"buffer": index, "bytes": 0}
    b = model.Buffers(index)
    if b.DataLength() > 0:
        return {"buffer": index, "inline": int(b.DataLength())}
    if b.Offset() > 1:
        return {"buffer": index, "offset": int(b.Offset()), "size": int(b.Size())}
    return {"buffer": index, "bytes": 0}


def _tensor_info(model: Any, tensor: Any, tfl: Any) -> dict[str, Any]:
    return {"name": tensor.Name().decode("utf-8", "replace"), "shape": _shape(tensor),
            "type": _names(tfl.TensorType).get(int(tensor.Type()), str(tensor.Type())),
            "quantisation": _quantisation(tensor, tfl), "storage": _storage(model, tensor)}


def _weight_bytes(storage: dict[str, Any]) -> int:
    return int(storage.get("inline", storage.get("size", 0)))


def tflite_summary(buffer: Any, every_tensor: bool = False) -> dict[str, Any]:
    """A TFLite graph's signatures, subgraphs, operators by kind and weight bytes by type and quantisation; with
    ``every_tensor``, every tensor of every subgraph."""
    _, _, tfl = _schema()
    model = tfl.Model.GetRootAs(buffer, 0)
    operators = _names(tfl.BuiltinOperator)
    codes = []
    for i in range(model.OperatorCodesLength()):
        code = model.OperatorCodes(i)
        builtin = max(int(code.BuiltinCode()), int(code.DeprecatedBuiltinCode()))
        custom = code.CustomCode()
        codes.append(custom.decode() if custom else operators.get(builtin, str(builtin)))
    signatures = []
    for i in range(model.SignatureDefsLength()):
        sd = model.SignatureDefs(i)
        sub = model.Subgraphs(sd.SubgraphIndex())

        def io_of(count: int, get: Any, sub: Any = sub) -> list[dict[str, Any]]:
            rows = []
            for j in range(count):
                m = get(j)
                t = sub.Tensors(m.TensorIndex())
                rows.append({"name": m.Name().decode("utf-8", "replace"), "shape": _shape(t),
                             "type": _names(tfl.TensorType).get(int(t.Type()), str(t.Type()))})
            return rows

        signatures.append({"key": sd.SignatureKey().decode("utf-8", "replace"), "subgraph": int(sd.SubgraphIndex()),
                           "inputs": io_of(sd.InputsLength(), sd.Inputs),
                           "outputs": io_of(sd.OutputsLength(), sd.Outputs)})
    subgraphs, weights, tensors, external = [], Counter(), [], Counter()
    seen: set[int] = set()
    for s in range(model.SubgraphsLength()):
        sub = model.Subgraphs(s)
        kinds = Counter(codes[sub.Operators(i).OpcodeIndex()] for i in range(sub.OperatorsLength()))
        subgraphs.append({"index": s, "name": (sub.Name() or b"").decode("utf-8", "replace"),
                          "tensors": int(sub.TensorsLength()), "operators": dict(sorted(kinds.items()))})
        for j in range(sub.TensorsLength()):
            info = _tensor_info(model, sub.Tensors(j), tfl)
            buffer_index = info["storage"].get("buffer")
            if "external_buffer" in info["storage"]:
                external[info["type"]] += 1
            if _weight_bytes(info["storage"]) and buffer_index not in seen:
                seen.add(buffer_index)
                q = info["quantisation"]
                kind = "none" if q is None else (f"{q['details']} block {q.get('block_size')}"
                                                 if q["details"] != "NONE" else
                                                 "per-tensor" if q["scales"] == 1 else f"per-channel (dim {q['dimension']})")
                weights[f"{info['type']}, {kind}"] += _weight_bytes(info["storage"])
            if every_tensor:
                tensors.append({"subgraph": s, "index": j, **info})
    out = {"subgraphs": subgraphs, "buffers": int(model.BuffersLength()), "signatures": signatures,
           "weight_bytes": dict(sorted(weights.items())), "external_tensors": dict(sorted(external.items())),
           "operator_codes": codes}
    if hasattr(model, "ExternalBuffersLength"):
        out["external_buffers"] = int(model.ExternalBuffersLength())
    if every_tensor:
        out["tensors"] = tensors
    return out


def dequantise(raw: np.ndarray, kind: str, shape: Sequence[int], quantisation: dict[str, Any] | None,
               scales: np.ndarray | None = None, zero_points: np.ndarray | None = None) -> np.ndarray | None:
    """A tensor's values as float32 from its bytes (``raw``, uint8) and type: float32, float16 and bfloat16 as they
    are; int8, uint8, int16 and int4 (two per byte, the first in the low four bits, sign-extended) through their
    per-tensor or per-channel scales and zero points along the quantised dimension. None for what it cannot read
    (other types, blockwise or multi-axis quantisation)."""
    count = int(np.prod(shape)) if len(shape) else 1
    raw = np.asarray(raw, dtype=np.uint8)
    if kind == "FLOAT32":
        return raw.view(np.float32)[:count].reshape(shape)
    if kind == "FLOAT16":
        return raw.view(np.float16)[:count].astype(np.float32).reshape(shape)
    if kind == "BFLOAT16":
        return (raw.view(np.uint16)[:count].astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if kind == "INT8":
        values = raw.view(np.int8)[:count].astype(np.float32)
    elif kind == "UINT8":
        values = raw[:count].astype(np.float32)
    elif kind == "INT16":
        values = raw.view(np.int16)[:count].astype(np.float32)
    elif kind == "INT4":
        low = (raw & 0x0F).astype(np.int16)
        high = (raw >> 4).astype(np.int16)
        values = np.empty(2 * raw.size, dtype=np.int16)
        values[0::2], values[1::2] = low, high
        values = np.where(values > 7, values - 16, values)[:count].astype(np.float32)
    else:
        return None
    if quantisation is None or quantisation["details"] != "NONE" or scales is None or scales.size == 0:
        return None
    values = values.reshape(shape)
    zeros = np.zeros_like(scales) if zero_points is None or zero_points.size == 0 else zero_points.astype(np.float32)
    if scales.size == 1:
        return (values - zeros.reshape(-1)[0]) * scales.reshape(-1)[0]
    axis = quantisation["dimension"]
    view = [1] * len(shape)
    view[axis] = scales.size
    return (values - zeros.reshape(view)) * scales.astype(np.float32).reshape(view)


def _values(model: Any, buffer: Any, tensor: Any, tfl: Any) -> np.ndarray | None:
    info = _tensor_info(model, tensor, tfl)
    storage = info["storage"]
    if "inline" in storage:
        raw = model.Buffers(storage["buffer"]).DataAsNumpy()
    elif "offset" in storage:
        raw = np.frombuffer(buffer, dtype=np.uint8, count=storage["size"], offset=storage["offset"])
    else:
        return None
    q = tensor.Quantization()
    scales = q.ScaleAsNumpy() if q is not None and q.ScaleLength() else None
    zeros = q.ZeroPointAsNumpy() if q is not None and q.ZeroPointLength() else None
    return dequantise(raw, info["type"], info["shape"], info["quantisation"],
                      None if scales is None else np.asarray(scales, dtype=np.float32), zeros)


# ---------------------------------------------------------------------------------------------
# A1's weights, and which of them each drafter tensor holds
# ---------------------------------------------------------------------------------------------

_SAFETENSORS = {"F32": np.float32, "F16": np.float16, "F64": np.float64}


def read_safetensors(directory: Path) -> dict[str, np.ndarray]:
    """Every floating-point tensor of the ``*.safetensors`` files of a directory, as float32 (read without the
    safetensors package: an 8-byte little-endian header length, a JSON header, the data)."""
    out: dict[str, np.ndarray] = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        data = path.read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8 + length])
        base = 8 + length
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            raw = np.frombuffer(data, dtype=np.uint8, count=end - begin, offset=base + begin)
            if meta["dtype"] == "BF16":
                values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif meta["dtype"] in _SAFETENSORS:
                values = raw.view(_SAFETENSORS[meta["dtype"]]).astype(np.float32)
            else:
                continue
            out[name] = values.reshape(meta["shape"])
    return out


def _closest(values: np.ndarray, reference: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    candidates = []
    flat = values.reshape(-1).astype(np.float32)
    for name, a in reference.items():
        if a.size != values.size:
            continue
        layouts = [("as is", flat)]
        if values.ndim == 2 and a.ndim == 2 and values.shape[::-1] == a.shape and a.shape[0] != a.shape[1]:
            layouts.append(("transposed", np.ascontiguousarray(values.T).reshape(-1).astype(np.float32)))
        elif values.ndim == 2 and a.ndim == 2 and values.shape == a.shape and a.shape[0] == a.shape[1]:
            layouts.append(("transposed", np.ascontiguousarray(values.T).reshape(-1).astype(np.float32)))
        target = a.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(target))
        for layout, x in layouts:
            error = float(np.linalg.norm(x - target)) / norm if norm > 0 else math.inf
            candidates.append({"name": name, "shape": list(a.shape), "layout": layout, "relative_error": error})
    return sorted(candidates, key=lambda c: c["relative_error"])[:2]


def match_weights(buffer: Any, reference: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """For every tensor of a TFLite graph that holds values in it, dequantised: the closest tensors of
    ``reference``."""
    _, _, tfl = _schema()
    model = tfl.Model.GetRootAs(buffer, 0)
    out, seen = [], set()
    for s in range(model.SubgraphsLength()):
        sub = model.Subgraphs(s)
        for j in range(sub.TensorsLength()):
            tensor = sub.Tensors(j)
            info = _tensor_info(model, tensor, tfl)
            if not _weight_bytes(info["storage"]) or info["storage"]["buffer"] in seen:
                continue
            seen.add(info["storage"]["buffer"])
            values = _values(model, buffer, tensor, tfl)
            row = {"subgraph": s, "name": info["name"], "shape": info["shape"], "type": info["type"],
                   "quantisation": info["quantisation"], "bytes": _weight_bytes(info["storage"])}
            if values is None:
                row["read"] = False
            elif values.size < 2:
                row["read"] = True
                row["closest"] = []
            else:
                row["read"] = True
                row["closest"] = _closest(values, reference)
            out.append(row)
    return out


# ---------------------------------------------------------------------------------------------
# The two commands
# ---------------------------------------------------------------------------------------------

def _release(view: memoryview) -> None:
    gc.collect()                                       # the arrays that numpy made over the view, if any are left
    try:
        view.release()
    except BufferError:                                # pragma: no cover - freed with the process instead
        pass


def inspect(model: Path, out: Path, assistant: Path | None = None) -> dict[str, Any]:
    """The report of the module docstring. The file is mapped into memory, not read into it."""
    table = section_table(model)
    result: dict[str, Any] = {"what": "the contents of a LiteRT-LM model file (docs/e1.md, Section 11.7, P3); "
                                      "read from the file", **table, "graphs": [], "drafter": None}
    out.mkdir(parents=True, exist_ok=True)
    handle = open(model, "rb")
    mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        for section in table["sections"]:
            if section["data_type"] != "TFLiteModel":
                continue
            view = memoryview(mm)[section["begin"]:section["end"]]
            try:
                drafter = section["model_type"] == DRAFTER
                summary = tflite_summary(view, every_tensor=drafter)
                result["graphs"].append({"section": section["index"], "model_type": section["model_type"],
                                         **{k: v for k, v in summary.items() if k != "tensors"}})
                if drafter:
                    (out / "drafter.tflite").write_bytes(bytes(view))
                    result["drafter"] = {"section": section["index"], "tensors": summary["tensors"],
                                         "extracted": str(out / "drafter.tflite")}
                    if assistant is not None:
                        result["drafter"]["matches"] = match_weights(view, read_safetensors(assistant))
                        result["drafter"]["assistant"] = str(assistant)
            finally:
                _release(view)
    finally:
        try:
            mm.close()
        except BufferError:                            # pragma: no cover
            pass
        handle.close()
    return result


def _short(name: str) -> str:
    """A tensor's name for a table: the first of fused names, its last three path parts without module prefixes
    (the full names are in the JSON)."""
    first = name.split(";")[0]
    parts = [part.rsplit(".", 1)[-1] for part in first.split("/") if part]
    return "/".join(parts[-3:]) or name


def render_inspect(result: dict[str, Any]) -> str:
    lines = ["# What the LiteRT-LM file holds (docs/e1.md, Section 11.7, P3)", "",
             f"`{result['path']}`: {result['bytes']:,} bytes, format version "
             f"{'.'.join(map(str, result['version']))}. Read from the file (*reported* by it).", "",
             "| section | data type | model type | bytes | SHA-256 (first 16) |", "|---|---|---|---|---|"]
    for s in result["sections"]:
        lines.append(f"| {s['index']} | {s['data_type']} | {s['model_type'] or '--'} | {s['size']:,} | "
                     f"{s['sha256'][:16]} |")
    for g in result["graphs"]:
        lines += ["", f"## Section {g['section']}: {g['model_type']}", "",
                  f"{len(g['subgraphs'])} subgraphs, {g['buffers']} buffers"
                  + (f", {g['external_buffers']} external buffers" if g.get("external_buffers") else "")
                  + "; weights by type and quantisation: "
                  + ", ".join(f"{k}: {v:,} bytes" for k, v in g["weight_bytes"].items())
                  + ("; tensors in external buffers: " + ", ".join(f"{k}: {v}" for k, v in g["external_tensors"].items())
                     if g["external_tensors"] else "")]
        for sig in g["signatures"]:
            ins = ", ".join(f"{x['name']} {x['shape']} {x['type']}" for x in sig["inputs"])
            outs = ", ".join(f"{x['name']} {x['shape']} {x['type']}" for x in sig["outputs"])
            lines.append(f"- signature `{sig['key']}` (subgraph {sig['subgraph']}): inputs {ins}; outputs {outs}")
    drafter = result.get("drafter")
    if drafter is None:
        lines += ["", f"No section of model type {DRAFTER}."]
        return "\n".join(lines)
    lines += ["", f"The drafter's section {drafter['section']} is extracted to `{drafter['extracted']}`; its "
              f"{len(drafter['tensors'])} tensors are listed in `drafter_tensors.md`."]
    if "matches" in drafter:
        lines += ["", f"## Against A1's weights (`{drafter['assistant']}`): the closest tensor, dequantised", "",
                  "| name | shape | type | closest in A1 | layout | relative error | next closest |",
                  "|---|---|---|---|---|---|---|"]
        for m in drafter["matches"]:
            if not m["read"]:
                lines.append(f"| {_short(m['name'])} | {m['shape']} | {m['type']} | (not read: its quantisation) | | | |")
                continue
            if not m["closest"]:
                lines.append(f"| {_short(m['name'])} | {m['shape']} | {m['type']} | (no tensor of A1 with as many elements) "
                             "| | | |")
                continue
            best = m["closest"][0]
            second = (f"{m['closest'][1]['name']} {m['closest'][1]['relative_error']:.3g}" if len(m["closest"]) > 1
                      else "--")
            lines.append(f"| {_short(m['name'])} | {m['shape']} | {m['type']} | {best['name']} {best['shape']} | "
                         f"{best['layout']} | {best['relative_error']:.3g} | {second} |")
    return "\n".join(lines)


def render_tensors(result: dict[str, Any]) -> str:
    """Every tensor of the drafter's section, as a table."""
    drafter = result["drafter"]
    lines = [f"# The drafter's tensors (section {drafter['section']} of `{result['path']}`)", "",
             "| subgraph | tensor | name | shape | type | quantisation | bytes |", "|---|---|---|---|---|---|---|"]
    for t in drafter["tensors"]:
        q = t["quantisation"]
        quant = "--" if q is None else (f"{q['details']}" + (f" block {q['block_size']}" if "block_size" in q else "")
                                        if q["details"] != "NONE" else f"{q['scales']} scales, dim {q['dimension']}")
        lines.append(f"| {t['subgraph']} | {t['index']} | {_short(t['name'])} | {t['shape']} | {t['type']} | {quant} | "
                     f"{_weight_bytes(t['storage']):,} |")
    return "\n".join(lines)


def _builder(*args: str) -> None:
    command = [sys.executable, "-I", "-m", "litert_lm_builder.litertlm_builder_cli", *args]
    subprocess.run(command, check=True)


def _differences(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> list[str]:
    """The keys of system metadata that one side lacks or that differ in value."""
    left, right = {x["key"]: x["value"] for x in a}, {x["key"]: x["value"] for x in b}
    return sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))


def repack(model: Path, out: Path, keep: bool = False) -> dict[str, Any]:
    """The null test of the module docstring."""
    out.mkdir(parents=True, exist_ok=True)
    size = Path(model).stat().st_size
    free = shutil.disk_usage(out).free
    if free < 2.2 * size:
        raise SystemExit(f"{out} has {free / 2**30:.1f} GiB free; the null test needs about "
                         f"{2.2 * size / 2**30:.1f} GiB (the unpacked sections and the repacked file)")
    unpacked, repacked = out / "unpacked", out / "repacked.litertlm"
    shutil.rmtree(unpacked, ignore_errors=True)
    _builder("unpack", "--path", str(model), "--output_dir", str(unpacked))
    _builder("toml", "--path", str(unpacked / "model.toml"), "output", "--path", str(repacked),
             "--weights-in-model-graph", "true", "--dedupe-shared-weights", "false")
    original, again = section_table(model), section_table(repacked)
    rows = []
    for i in range(max(len(original["sections"]), len(again["sections"]))):
        a = original["sections"][i] if i < len(original["sections"]) else None
        b = again["sections"][i] if i < len(again["sections"]) else None
        same = a is not None and b is not None and a["sha256"] == b["sha256"] and a["data_type"] == b["data_type"]
        rows.append({"index": i, "original": a, "repacked": b, "identical_bytes": same,
                     "same_items": a is not None and b is not None and a["items"] == b["items"]})
    result = {"what": "the builder's unpack and pack, section by section (docs/e1.md, Section 11.7, P3's null test)",
              "model": str(model), "repacked": str(repacked), "original_version": original["version"],
              "repacked_version": again["version"], "original_bytes": original["bytes"],
              "repacked_bytes": again["bytes"],
              "same_system_metadata": original["system_metadata"] == again["system_metadata"],
              "system_metadata_differences": _differences(original["system_metadata"], again["system_metadata"]),
              "same_offsets": [(x["begin"], x["end"]) for x in original["sections"]]
              == [(x["begin"], x["end"]) for x in again["sections"]], "sections": rows,
              "identical_sections": all(r["identical_bytes"] for r in rows),
              "toml": (unpacked / "model.toml").read_text(encoding="utf-8")}
    if not keep:
        shutil.rmtree(unpacked, ignore_errors=True)
    return result


def render_repack(result: dict[str, Any]) -> str:
    lines = ["# P3's null test: the builder's unpack and pack (docs/e1.md, Section 11.7)", "",
             f"`{result['model']}` ({result['original_bytes']:,} bytes, version "
             f"{'.'.join(map(str, result['original_version']))}) against `{result['repacked']}` "
             f"({result['repacked_bytes']:,} bytes, version {'.'.join(map(str, result['repacked_version']))}); "
             + ("the same system metadata" if result["same_system_metadata"] else
                "system metadata differing in " + ", ".join(result["system_metadata_differences"]))
             + f"; section offsets {'the same' if result['same_offsets'] else 'moved'}.", "",
             "| section | data type | model type | bytes | identical bytes | same items |", "|---|---|---|---|---|---|"]
    for r in result["sections"]:
        a = r["original"] or r["repacked"]
        lines.append(f"| {r['index']} | {a['data_type']} | {a['model_type'] or '--'} | {a['size']:,} | "
                     f"{'yes' if r['identical_bytes'] else 'NO'} | {'yes' if r['same_items'] else 'no'} |")
    lines += ["", "Every section identical: " + ("yes" if result["identical_sections"] else "NO")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 1, P3: LiteRT-LM's model file", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("--model", type=Path, required=True)
    i.add_argument("--assistant", type=Path, default=None, help="A1's directory (its safetensors), to match against")
    i.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("repack")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--keep", action="store_true", help="keep the unpacked sections")
    args = p.parse_args(argv)
    from chat_vpa.e1_common import write_json

    if args.command == "inspect":
        result = inspect(args.model, args.out, args.assistant)
        write_json(args.out / "inspect.json", result)
        text = render_inspect(result)
        (args.out / "inspect.md").write_text(text + "\n", encoding="utf-8")
        if result["drafter"] is not None:
            (args.out / "drafter_tensors.md").write_text(render_tensors(result) + "\n", encoding="utf-8")
    else:
        result = repack(args.model, args.out, args.keep)
        write_json(args.out / "repack.json", result)
        text = render_repack(result)
        (args.out / "repack.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
