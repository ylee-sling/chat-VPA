# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_p3_route.py -- route (a) of P3 (docs/e1.md, Sections 11.7 and 11.13): a drafter's weights written into the
drafter section of Google's LiteRT-LM file, in place, in a copy of the file. It runs in LiteRT-LM's environment,
without torch (``litert-lm-builder``, a dependency of ``litert-lm``, carries the schemas).

    python -m chat_vpa.e1_p3_route relate --model FILE --assistant A1_DIR --out runs/e1-p3/route/relate
    python -m chat_vpa.e1_p3_route codec --model FILE --out runs/e1-p3/route/codec
    python -m chat_vpa.e1_p3_route patch --model FILE --null --out runs/e1-p3/route/null/FILE_NAME
    python -m chat_vpa.e1_p3_route patch --model FILE --weights DRAFTER_DIR --mapping .../relate/mapping.json \\
        --out runs/e1-p3/a2/FILE_NAME
    python -m chat_vpa.e1_p3_route compare --a FILE --b FILE --out runs/e1-p3/route/compare

**The drafter's weights** are the buffers of the drafter section that hold bytes in it: quantised matrices (4- and
8-bit with per-channel scales in Google's file) and float vectors and scalars. A buffer can be read by several
tensors (one per signature's subgraph, or several in one subgraph where the converter merged equal constants); each
tensor has its own scales, and all of them are written.

**What each weight is** (``relate``) is read from the graph, not from the values, because the values of Google's
drafter are not A1's (11.7). The matrices are named by their role in Google's file (``ROLES``): the input and
output projections, each layer's query, output, two gating and down projections, the output embedding; an operator
that only dequantises, reshapes or transposes a matrix hands its role to the operator that reads its output. A
layer is the rank of its operator among those of the same role in execution order; of a layer's two gating
matrices, the gate is the one whose branch to the product of the two passes through the activation (GELU, TANH,
LOGISTIC) or is the longer. Every other float constant that a normalisation multiplies (an ``rms_norm`` composite,
or a product with a value computed through RSQRT, SQRT or POW at most four operators upstream), and every argument
of the converted function (``jax2tf_arg_*``) that is not a matrix, is placed by the nearest matrix downstream of
the operator that reads it (ties to the earliest in execution order; upstream, the residual stream would make the
nearest matrix the wrong one), in execution order within its group:

    downstream                     what it is, in execution order
    the first layer's query,       the layer's input norm
    key or value
    a later layer's query, key     the previous layer's post-feed-forward norm, then this layer's input norm;
    or value                       a scalar between them: the previous layer's scalar
    an output projection           the layer's query (key, value) norm, by the matrix nearest upstream
    a gating matrix                the post-attention norm, then the pre-feed-forward norm
    the output side, or none       the last layer's post-feed-forward norm, then the final norm; a scalar between
                                   them: the last layer's scalar (with no matrix downstream but a query, key or
                                   value upstream, a norm is that matrix's: a prefill graph's last layer writes
                                   only its keys and values)

A group with more or fewer constants than its row lists is not placed (*unresolved*). How a vector is stored is
read from the operator that uses it (past operators that only reshape, cast or dequantise it): a product (or the product inside a composite's decomposition) stores
Transformers' weight as it is (``as is``); a sum with 1 before the product stores the weight minus one (``minus
one``: Gemma's JAX normalisation multiplies by 1 + scale, Transformers' Gemma 4 by its weight). How a matrix is laid
out is read from the shapes: Transformers' (``as is``) or transposed; a square matrix is placed only where a fully
connected operator reads it (TFLite's [out, in] is PyTorch's). Beside every placement the values are compared with
A1's tensor of that role: the relative error ||x - a|| / ||a||, the cosine, the error left after the best scale per
row, per column and both, what quantising A1's own tensor the same way costs, the singular values against A1's
(equal under any permutation of rows or columns), and for vectors the errors under each of the three conventions.
They are evidence, never the rule. ``mapping.json`` lists, per buffer, every source and how it is written (a buffer
with several sources is *shared*), and the buffers a patch must write: every quantised matrix, every argument and
every normalisation's constant, placed or not.

``codec`` is the null test of the integer codec: every quantised weight is decoded to its integers and encoded
again, and its bytes must come back unchanged (two 4-bit values per byte, the first in the low four bits). It also
reports whether every tensor that reads a weight has the same scales, and the share of channels whose largest
|integer| is the largest one (qmax: 7 or 127), which is what max-abs scales give.

``patch`` copies the file and writes, at the same offsets: each mapped weight from ``--weights``' safetensors
(transposed and shifted as the mapping says), quantised per channel along the file's quantised dimension with
symmetric max-abs scales at the file's bit width (or written as float32); its scales into every tensor that reads
it; and, if the drafter section's ``sha256`` item is the hash of its bytes, the section's new hash. A shared buffer
whose sources differ is refused (``--shared refuse``) or written with their mean (``--shared mean``), with the
largest relative difference of a source from the mean reported. It refuses a mapping made for another file or one
that leaves a buffer it must write out. It then checks that every byte outside the written ranges is the
original's and that every tensor of a written weight decodes alike. ``--null`` writes the file's own integers and
scales back: the copy must be the original byte for byte.

``compare`` sets two files side by side: each section's bytes and items and, for the LLM metadata, its fields.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mmap
import shutil
import struct
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from chat_vpa.e1_p3 import (
    DRAFTER,
    _names,
    _schema,
    _short,
    _storage,
    _tensor_info,
    _weight_bytes,
    read_safetensors,
    section_table,
)

__all__ = ["ROLES", "assign_roles", "buffer_roles", "codec", "compare", "decode_integers", "encode_integers", "mapping_for",
           "patch", "quantise", "read_graph", "relate", "role_of", "weights_of"]

QMAX = {"INT4": 7, "INT8": 127, "INT16": 32_767}
FLOATS = ("FLOAT32", "FLOAT16", "BFLOAT16")
CONVENTIONS = {"as is": 0.0, "minus one": -1.0, "plus one": 1.0}   # what the file holds: a + shift
# Google's names of the drafter's matrices (11.7): path segments that appear in a row in a tensor's name, the role.
ROLES = (
    (("mtp_pre_proj",), "pre_projection"),
    (("mtp_post_proj",), "post_projection"),
    (("decode_softmax",), "embedding"),
    (("q_einsum",), "q"),
    (("attn_vec_einsum",), "o"),
    (("gating_einsum1",), "gating1"),
    (("gating_einsum2",), "gating2"),
    (("linear",), "down"),
)
PER_LAYER = ("q", "k", "v", "o", "gate", "up", "gating1", "gating2", "down")
INPUT_SIDE = ("start", "pre_projection", "embedding_lookup")
ATTENTION_IN = ("q", "k", "v")
MLP_IN = ("gate", "up", "gating1", "gating2")
OUTPUT_SIDE = ("end", "post_projection", "embedding", "lm_head")
WEIGHT_TRANSFORMS = ("DEQUANTIZE", "RESHAPE", "TRANSPOSE", "CAST", "SQUEEZE", "EXPAND_DIMS")
SOURCES = {
    "pre_projection": "pre_projection.weight",
    "post_projection": "post_projection.weight",
    "embedding": "model.embed_tokens.weight",
    "embedding_lookup": "model.embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "q": "model.layers.{L}.self_attn.q_proj.weight",
    "k": "model.layers.{L}.self_attn.k_proj.weight",
    "v": "model.layers.{L}.self_attn.v_proj.weight",
    "o": "model.layers.{L}.self_attn.o_proj.weight",
    "gate": "model.layers.{L}.mlp.gate_proj.weight",
    "up": "model.layers.{L}.mlp.up_proj.weight",
    "down": "model.layers.{L}.mlp.down_proj.weight",
    "input_layernorm": "model.layers.{L}.input_layernorm.weight",
    "q_norm": "model.layers.{L}.self_attn.q_norm.weight",
    "k_norm": "model.layers.{L}.self_attn.k_norm.weight",
    "v_norm": "model.layers.{L}.self_attn.v_norm.weight",
    "post_attention_layernorm": "model.layers.{L}.post_attention_layernorm.weight",
    "pre_feedforward_layernorm": "model.layers.{L}.pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm": "model.layers.{L}.post_feedforward_layernorm.weight",
    "layer_scalar": "model.layers.{L}.layer_scalar",
    "norm": "model.norm.weight",
}
NORM_MARKERS = ("RSQRT", "SQRT", "POW")
SMALL = 4_096                      # float32 constants up to this size carry their values (a broadcast 1, a scale)
ACTIVATIONS = ("GELU", "TANH", "LOGISTIC")
ARGUMENT = "jax2tf_arg_"


def _source(role: str, layer: int | None) -> str:
    return SOURCES[role].format(L=layer)


# ---------------------------------------------------------------------------------------------
# The drafter section, its graphs and its weights
# ---------------------------------------------------------------------------------------------

def _address(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


@contextmanager
def drafter_section(path: Path) -> Iterator[tuple[dict[str, Any], memoryview, Any]]:
    """The drafter's section of a LiteRT-LM file (its table row), a read-only view of its bytes and its TFLite
    model, the file mapped into memory."""
    table = section_table(path)
    rows = [s for s in table["sections"] if s["model_type"] == DRAFTER]
    if len(rows) != 1:
        raise ValueError(f"{path} has {len(rows)} sections of model type {DRAFTER}, not one")
    section = rows[0]
    _, _, tfl = _schema()
    handle = open(path, "rb")
    mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    view = memoryview(mm)[section["begin"]:section["end"]]
    try:
        yield section, view, tfl.Model.GetRootAs(view, 0)
    finally:
        import gc

        gc.collect()
        try:
            view.release()
            mm.close()
        except BufferError:                                      # pragma: no cover - freed with the process
            pass
        handle.close()


def _origins(name: str) -> list[list[str]]:
    """A tensor's name as the converter's list of origins (fused names are joined by ';'), each as its path's
    segments without module prefixes."""
    return [[part.rsplit(".", 1)[-1] for part in origin.split("/") if part] for origin in name.split(";") if origin]


def role_of(name: str, patterns: Sequence[tuple[tuple[str, ...], str]] = ROLES) -> str | None:
    """The role of the first origin of ``name`` that a pattern matches (its segments appear in a row), by the first
    such pattern."""
    for segments in _origins(name):
        for pattern, role in patterns:
            n = len(pattern)
            if any(tuple(segments[i:i + n]) == tuple(pattern) for i in range(len(segments) - n + 1)):
                return role
    return None


def _is_argument(name: str) -> bool:
    return any(segment.startswith(ARGUMENT) for origin in _origins(name) for segment in origin)


def _small_values(view: memoryview, model: Any, tensor: Any, kind: str) -> list[float] | None:
    storage = _storage(model, tensor)
    count = 1 if tensor.ShapeIsNone() else int(np.prod([int(x) for x in tensor.ShapeAsNumpy()]))
    if kind != "FLOAT32" or count > SMALL or not _weight_bytes(storage):
        return None
    if "inline" in storage:
        data = model.Buffers(storage["buffer"]).DataAsNumpy()
        return np.frombuffer(data.tobytes(), dtype=np.float32)[:count].astype(float).tolist()
    return np.frombuffer(view, dtype=np.float32, count=count, offset=int(storage["offset"])).astype(float).tolist()


def read_graph(view: memoryview, model: Any, s: int) -> dict[str, Any]:
    """Subgraph ``s`` as plain data: its operators in execution order (kind, inputs, outputs, and for a composite
    its name and decomposition), its inputs, and its constants (name, shape, type, buffer; values up to SMALL)."""
    _, _, tfl = _schema()
    operators = _names(tfl.BuiltinOperator)
    sub = model.Subgraphs(s)
    ops = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        builtin = max(int(code.BuiltinCode()), int(code.DeprecatedBuiltinCode()))
        kind = code.CustomCode().decode() if code.CustomCode() else operators.get(builtin, str(builtin))
        row = {"kind": kind, "inputs": [int(t) for t in (op.InputsAsNumpy() if op.InputsLength() else [])],
               "outputs": [int(t) for t in (op.OutputsAsNumpy() if op.OutputsLength() else [])],
               "composite": None, "body": None}
        if kind == "STABLEHLO_COMPOSITE":
            table = op.BuiltinOptions2()
            if table is not None:
                options = tfl.StableHLOCompositeOptions()
                options.Init(table.Bytes, table.Pos)
                row["composite"] = (options.Name() or b"").decode("utf-8", "replace")
                row["body"] = int(options.DecompositionSubgraphIndex())
        ops.append(row)
    constants = {}
    for j in range(sub.TensorsLength()):
        tensor = sub.Tensors(j)
        info = _tensor_info(model, tensor, tfl)
        if not _weight_bytes(info["storage"]):
            continue
        constants[j] = {"name": info["name"], "shape": info["shape"], "type": info["type"],
                        "buffer": info["storage"]["buffer"],
                        "quantised": bool(info["quantisation"] and info["quantisation"]["scales"]),
                        "values": _small_values(view, model, tensor, info["type"])}
    return {"index": s, "ops": ops, "constants": constants,
            "inputs": [int(t) for t in (sub.InputsAsNumpy() if sub.InputsLength() else [])]}


def signature_subgraphs(model: Any) -> list[int]:
    return sorted({int(model.SignatureDefs(i).SubgraphIndex()) for i in range(model.SignatureDefsLength())})


def weights_of(view: memoryview, model: Any) -> list[dict[str, Any]]:
    """The drafter's weights: one row per buffer that holds bytes in the section and is a float or quantised
    tensor's, with where its bytes are, every tensor that reads it (subgraph, index, name, shape, type,
    quantisation, where its scales are) and how many operators read those tensors."""
    _, _, tfl = _schema()
    base = _address(np.frombuffer(view, dtype=np.uint8, count=1))
    rows: dict[int, dict[str, Any]] = {}
    for s in range(model.SubgraphsLength()):
        sub = model.Subgraphs(s)
        readers: dict[int, int] = defaultdict(int)
        for i in range(sub.OperatorsLength()):
            op = sub.Operators(i)
            for t in (op.InputsAsNumpy() if op.InputsLength() else []):
                if t >= 0:
                    readers[int(t)] += 1
        for j in range(sub.TensorsLength()):
            tensor = sub.Tensors(j)
            info = _tensor_info(model, tensor, tfl)
            storage = info["storage"]
            if not _weight_bytes(storage):
                continue
            quantised = info["quantisation"] is not None and info["quantisation"]["scales"] > 0
            if not quantised and info["type"] not in FLOATS:
                continue                                          # integer constants: not weights
            entry = {"subgraph": s, "index": j, "name": info["name"], "shape": info["shape"], "type": info["type"],
                     "quantisation": info["quantisation"], "readers": readers.get(j, 0)}
            if quantised:
                q = tensor.Quantization()
                entry["scale_position"] = _address(q.ScaleAsNumpy()) - base
                entry["scales"] = int(q.ScaleLength())
                zeros = q.ZeroPointAsNumpy() if q.ZeroPointLength() else np.zeros(0, dtype=np.int64)
                entry["zero_points_zero"] = bool(np.all(np.asarray(zeros) == 0))
            b = storage["buffer"]
            if b not in rows:
                if "inline" in storage:
                    position = _address(model.Buffers(b).DataAsNumpy()) - base
                    size = int(storage["inline"])
                else:
                    position, size = int(storage["offset"]), int(storage["size"])
                rows[b] = {"buffer": b, "name": info["name"], "short": _short(info["name"]), "shape": info["shape"],
                           "type": info["type"], "quantisation": info["quantisation"], "quantised": quantised,
                           "position": position, "bytes": size, "tensors": [], "readers": 0}
            rows[b]["tensors"].append(entry)
            rows[b]["readers"] += entry["readers"]
    out = sorted(rows.values(), key=lambda r: r["position"])
    for row in out:
        row["argument"] = any(_is_argument(t["name"]) for t in row["tensors"])
    return out


# ---------------------------------------------------------------------------------------------
# The integer codec and quantisation
# ---------------------------------------------------------------------------------------------

def decode_integers(raw: np.ndarray, kind: str, count: int) -> np.ndarray:
    """The integers of a quantised tensor's bytes: int8, uint8, int16 as they are; int4 two per byte, the first in
    the low four bits, sign-extended."""
    raw = np.asarray(raw, dtype=np.uint8)
    if kind == "INT8":
        return raw.view(np.int8)[:count].astype(np.int16)
    if kind == "UINT8":
        return raw[:count].astype(np.int16)
    if kind == "INT16":
        return raw.view(np.int16)[:count].astype(np.int32)
    if kind == "INT4":
        values = np.empty(2 * raw.size, dtype=np.int8)
        values[0::2] = (raw & 0x0F).astype(np.int8)
        values[1::2] = (raw >> 4).astype(np.int8)
        return np.where(values > 7, values - 16, values).astype(np.int8)[:count]
    raise ValueError(f"no integer codec for {kind}")


def encode_integers(values: np.ndarray, kind: str, size: int) -> bytes:
    """The inverse of ``decode_integers`` into ``size`` bytes (an odd count of int4 values leaves the last high
    four bits zero)."""
    values = np.asarray(values)
    if kind == "INT8":
        out = values.astype(np.int8).view(np.uint8)
    elif kind == "UINT8":
        out = values.astype(np.uint8)
    elif kind == "INT16":
        out = values.astype(np.int16).view(np.uint8)
    elif kind == "INT4":
        nibbles = (values.astype(np.int16) & 0x0F).astype(np.uint8)
        if nibbles.size % 2:
            nibbles = np.append(nibbles, np.uint8(0))
        out = nibbles[0::2] | (nibbles[1::2] << 4)
    else:
        raise ValueError(f"no integer codec for {kind}")
    data = out.tobytes()
    if len(data) != size:
        raise ValueError(f"{kind}: {len(data)} bytes encoded where the file has {size}")
    return data


def _per_channel(values: np.ndarray, dimension: int) -> np.ndarray:
    """``values`` with the quantised dimension first and every other one flattened behind it."""
    return np.moveaxis(values, dimension, 0).reshape(values.shape[dimension], -1)


def quantise(values: np.ndarray, kind: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric max-abs quantisation per channel along ``dimension``: (integers in the tensor's order, float32
    scales); a channel of zeros gets the scale 1."""
    qmax = QMAX[kind]
    values = np.asarray(values, dtype=np.float64)
    channels = _per_channel(values, dimension)
    scales = (np.abs(channels).max(axis=1) / qmax).astype(np.float32)
    scales = np.where(scales > 0, scales, np.float32(1.0)).astype(np.float32)
    integers = np.clip(np.rint(channels / scales[:, None].astype(np.float64)), -qmax, qmax)
    moved = integers.reshape(np.moveaxis(values, dimension, 0).shape)
    return np.moveaxis(moved, 0, dimension).reshape(-1).astype(np.int32), scales


def _scale_shape(shape: Sequence[int], dimension: int, count: int) -> list[int]:
    return [count if i == dimension else 1 for i in range(len(shape))]


def _dequantised(view: memoryview, w: dict[str, Any], tensor: dict[str, Any] | None = None) -> np.ndarray:
    """A weight's values (float64) through one of its tensors' scales (the first's by default)."""
    tensor = tensor or w["tensors"][0]
    shape = tensor["shape"]
    count = int(np.prod(shape)) if shape else 1
    raw = np.frombuffer(view, dtype=np.uint8, count=w["bytes"], offset=w["position"])
    if not w["quantised"]:
        if w["type"] == "FLOAT32":
            return raw.view(np.float32)[:count].reshape(shape).astype(np.float64)
        if w["type"] == "FLOAT16":
            return raw.view(np.float16)[:count].reshape(shape).astype(np.float64)
        if w["type"] == "BFLOAT16":
            return (raw.view(np.uint16)[:count].astype(np.uint32) << 16).view(np.float32).reshape(shape).astype(
                np.float64)
        raise ValueError(f"{w['short']}: no reader for {w['type']}")
    q = tensor["quantisation"]
    if q["details"] != "NONE" or not tensor.get("zero_points_zero", True):
        raise ValueError(f"{w['short']}: only symmetric per-tensor or per-channel quantisation is read")
    scales = np.frombuffer(view, dtype=np.float32, count=tensor["scales"], offset=tensor["scale_position"])
    integers = decode_integers(raw, w["type"], count).astype(np.float64).reshape(shape)
    if scales.size == 1:
        return integers * float(scales[0])
    return integers * scales.astype(np.float64).reshape(_scale_shape(shape, q["dimension"], scales.size))


# ---------------------------------------------------------------------------------------------
# Roles from the graph
# ---------------------------------------------------------------------------------------------

def _links(graph: dict[str, Any]) -> tuple[dict[int, int], dict[int, list[int]]]:
    producer, consumers = {}, defaultdict(list)
    for i, op in enumerate(graph["ops"]):
        for t in op["outputs"]:
            producer[t] = i
        for t in op["inputs"]:
            if t >= 0:
                consumers[t].append(i)
    return producer, consumers


def _variable_inputs(graph: dict[str, Any], op: dict[str, Any]) -> list[int]:
    return [t for t in op["inputs"] if t >= 0 and t not in graph["constants"]]


def _names_matrix(c: dict[str, Any]) -> bool:
    return (c["quantised"] or c["type"] in FLOATS) and len([d for d in c["shape"] if d > 1]) >= 2


def buffer_roles(graphs: Sequence[dict[str, Any]],
                 patterns: Sequence[tuple[tuple[str, ...], str]] = ROLES) -> tuple[dict[int, str], list[str]]:
    """The role of every buffer that holds a named matrix: the role most of its tensors' names give (the first
    origin that a pattern matches; ties to the first seen), and the buffers whose names disagree."""
    votes: dict[int, list[str]] = defaultdict(list)
    names: dict[int, str] = {}
    for graph in graphs:
        for _, c in sorted(graph["constants"].items()):
            if not _names_matrix(c):
                continue                                          # shapes, indices and vectors name no matrix
            role = role_of(c["name"], patterns)
            if role is not None:
                votes[c["buffer"]].append(role)
                names.setdefault(c["buffer"], c["name"])
    out, disagreeing = {}, []
    for buffer, roles in votes.items():
        out[buffer] = max(dict.fromkeys(roles), key=roles.count)
        if len(set(roles)) > 1:
            disagreeing.append(f"{_short(names[buffer])}: {', '.join(sorted(set(roles)))} (read as {out[buffer]})")
    return out, disagreeing


def _matrix_operators(graph: dict[str, Any], roles: dict[int, str],
                      bodies: dict[int, dict[str, Any]] | None = None) -> dict[int, tuple[str, int, int]]:
    """The operators that apply a named matrix: {operator: (role, the constant in this graph, its buffer)}. An
    operator that only dequantises, reshapes or transposes a named matrix hands its role on to the operators that
    read its output; a composite whose decomposition holds a named matrix as a constant of its own applies it (the
    constant is then -1 here)."""
    carried: dict[int, tuple[str, int]] = {}
    for t, c in graph["constants"].items():
        if c["buffer"] in roles and _names_matrix(c):
            carried[t] = (roles[c["buffer"]], t)
    out: dict[int, tuple[str, int, int]] = {}
    for i, op in enumerate(graph["ops"]):
        held = [carried[t] for t in op["inputs"] if t in carried]
        if not held:
            continue
        rest = [t for t in _variable_inputs(graph, op) if t not in carried]
        if op["kind"] in WEIGHT_TRANSFORMS and not rest:
            for t in op["outputs"]:
                carried[t] = held[0]
            continue
        role, constant = held[0]
        if role == "embedding" and op["kind"] in ("EMBEDDING_LOOKUP", "GATHER"):
            role = "embedding_lookup"
        out[i] = (role, constant, graph["constants"][constant]["buffer"])
    for i, op in enumerate(graph["ops"]):
        body = (bodies or {}).get(op["body"]) if op["body"] is not None else None
        if i in out or body is None:
            continue
        for _, c in sorted(body["constants"].items()):
            if c["buffer"] in roles and _names_matrix(c):
                out[i] = (roles[c["buffer"]], -1, c["buffer"])
                break
    return out


def _nearest(graph: dict[str, Any], start: int, matrices: dict[int, Any], forward: bool,
             links: tuple[dict[int, int], dict[int, list[int]]]) -> int | None:
    """The matrix operator nearest to operator ``start`` downstream (``forward``) or upstream, by the number of
    operators between them, never walking through another matrix; ties go upstream to the latest and downstream to
    the earliest in execution order."""
    producer, consumers = links
    seen, frontier = {start}, [start]
    while frontier:
        found, following = [], []
        for i in frontier:
            op = graph["ops"][i]
            if forward:
                nxt = [c for t in op["outputs"] for c in consumers.get(t, [])]
            else:
                nxt = [producer[t] for t in _variable_inputs(graph, op) if t in producer]
            for j in nxt:
                if j in seen:
                    continue
                seen.add(j)
                (found if j in matrices else following).append(j)
        if found:
            return min(found) if forward else max(found)
        frontier = following
    return None


def _through_transforms(graph: dict[str, Any], op_index: int, tensor: int, links: Any) -> tuple[int, int]:
    """The operator that uses a constant, past operators that only reshape, cast or dequantise it (each read by
    one operator), and the tensor it reads: (operator, tensor)."""
    _, consumers = links
    while True:
        op = graph["ops"][op_index]
        if op["kind"] not in WEIGHT_TRANSFORMS or _variable_inputs(graph, op) or len(op["outputs"]) != 1:
            return op_index, tensor
        readers = consumers.get(op["outputs"][0], [])
        if len(readers) != 1:
            return op_index, tensor
        op_index, tensor = readers[0], op["outputs"][0]


def _is_ones(constant: dict[str, Any] | None) -> bool:
    return bool(constant and constant.get("values") and all(v == 1.0 for v in constant["values"]))


def _reader_convention(graph: dict[str, Any], op: dict[str, Any], tensor: int,
                       bodies: dict[int, dict[str, Any]], links: Any) -> tuple[str, int | None]:
    """How a vector read by ``op`` is stored (``as is``, ``minus one`` or ``unknown``), and the product that applies
    it where that is a later operator of this graph."""
    _, consumers = links
    if op["composite"] is not None:
        body = bodies.get(op["body"])
        if body is None or tensor not in op["inputs"]:
            return "unknown", None
        position = op["inputs"].index(tensor)
        if position >= len(body["inputs"]):
            return "unknown", None
        inner = body["inputs"][position]
        _, inner_consumers = _links(body)
        readers = [body["ops"][c] for c in inner_consumers.get(inner, [])]
        if len(readers) == 1 and readers[0]["kind"] == "MUL":
            return "as is", None
        if len(readers) == 1 and readers[0]["kind"] == "ADD" and any(
                _is_ones(body["constants"].get(t)) for t in readers[0]["inputs"] if t != inner):
            return "minus one", None
        return "unknown", None
    if op["kind"] == "MUL":
        return "as is", None
    if op["kind"] == "ADD" and any(_is_ones(graph["constants"].get(t)) for t in op["inputs"] if t != tensor):
        nxt = [c for t in op["outputs"] for c in consumers.get(t, [])]
        if len(nxt) == 1 and graph["ops"][nxt[0]]["kind"] == "MUL":
            return "minus one", nxt[0]
    return "unknown", None


def _normalises(graph: dict[str, Any], op_index: int, tensor: int, bodies: dict[int, dict[str, Any]],
                links: Any) -> bool:
    """Whether the constant ``tensor`` read by operator ``op_index`` is a normalisation's scale."""
    op = graph["ops"][op_index]
    if op["composite"] is not None:
        return "rms_norm" in op["composite"] or "layer_norm" in op["composite"]
    convention, product = _reader_convention(graph, op, tensor, bodies, links)
    if convention == "unknown":
        return False
    mul = graph["ops"][product] if product is not None else op
    producer, _ = links
    frontier = [producer[t] for t in _variable_inputs(graph, mul) if t != tensor and t in producer]
    seen: set[int] = set()
    for _ in range(4):
        following = []
        for i in frontier:
            if i in seen:
                continue
            seen.add(i)
            o = graph["ops"][i]
            if o["kind"] in NORM_MARKERS or "rms" in (o["composite"] or ""):
                return True
            following += [producer[t] for t in _variable_inputs(graph, o) if t in producer]
        frontier = following
    return False


def _gate(graph: dict[str, Any], a: int, b: int, links: Any) -> int | None:
    """Of two gating matrices' operators, the one whose branch to the product of the two passes through the
    activation (or, failing that, is the longer); None if the graph does not say."""
    _, consumers = links

    def distances(start: int) -> dict[int, int]:
        out, frontier, d = {start: 0}, [start], 0
        while frontier:
            d += 1
            following = []
            for i in frontier:
                for t in graph["ops"][i]["outputs"]:
                    for c in consumers.get(t, []):
                        if c not in out:
                            out[c] = d
                            following.append(c)
            frontier = following
        return out

    da, db = distances(a), distances(b)
    joins = [i for i in da if i in db and i not in (a, b)]
    if not joins:
        return None
    join = min(joins, key=lambda i: (max(da[i], db[i]), i))

    def activated(own: dict[int, int], other: dict[int, int]) -> bool:
        return any(0 < own[i] < own[join] and i not in other and (
            graph["ops"][i]["kind"] in ACTIVATIONS or "gelu" in (graph["ops"][i]["composite"] or ""))
            for i in own)

    on_a, on_b = activated(da, db), activated(db, da)
    if on_a != on_b:
        return a if on_a else b
    if da[join] != db[join]:
        return a if da[join] > db[join] else b
    return None


def assign_roles(graph: dict[str, Any], bodies: dict[int, dict[str, Any]],
                 roles: dict[int, str] | None = None) -> dict[str, Any]:
    """Every weight's role in one graph (the module docstring), the matrices' roles by buffer (``buffer_roles``; by
    default from this graph's names): ``placed``, one row per (operator, constant) with its role, layer, source,
    kind and convention and why; ``unresolved``, the constants that could not be placed, with why;
    ``normalising``, every normalisation's constant and argument considered; ``layers``, the count of each
    per-layer role."""
    links = _links(graph)
    if roles is None:
        roles, _ = buffer_roles([graph, *bodies.values()])
    applied = _matrix_operators(graph, roles, bodies)
    by_role: dict[str, list[int]] = defaultdict(list)
    for i, (role, _, _) in sorted(applied.items()):
        by_role[role].append(i)
    matrices: dict[int, tuple[str, int | None]] = {}
    for role, indices in by_role.items():
        for rank, i in enumerate(indices):
            matrices[i] = (role, rank if role in PER_LAYER else None)
    placed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    resolved: dict[int, str] = {}
    blocked: set[int] = set()
    pairs = [*zip(by_role.get("gating1", []), by_role.get("gating2", []), strict=False),
             *zip(by_role.get("gate", []), by_role.get("up", []), strict=False)]
    for a, b in pairs:
        gate = _gate(graph, a, b, links)
        layer = matrices[a][1]
        named = {matrices[a][0], matrices[b][0]}
        why = None
        if gate is None:
            why = f"the graph does not say which gating matrix of layer {layer} is the gate"
        elif named == {"gate", "up"} and matrices[gate][0] != "gate":
            why = f"layer {layer}: the activated branch is the matrix named up"
        if why:
            blocked |= {a, b}
            unresolved += [{"tensor": applied[i][1], "buffer": applied[i][2], "op": i, "why": why} for i in (a, b)]
            continue
        resolved[gate] = "gate"
        resolved[b if gate == a else a] = "up"
    for i, (role, layer) in sorted(matrices.items()):
        if i in blocked:
            continue
        role = resolved.get(i, role)
        if role in ("gating1", "gating2"):                       # a gating matrix without its pair
            unresolved.append({"tensor": applied[i][1], "buffer": applied[i][2], "op": i,
                               "why": f"{role} of layer {layer} has no pair"})
            continue
        placed.append({"tensor": applied[i][1], "buffer": applied[i][2], "op": i, "role": role, "layer": layer,
                       "source": _source(role, layer), "kind": "matrix", "convention": "as is", "why": "name"})
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    normalising: list[dict[str, Any]] = []
    named = {applied[i][1] for i in applied}
    for i, op in enumerate(graph["ops"]):
        if i in matrices:
            continue
        for t in op["inputs"]:
            c = graph["constants"].get(t)
            if c is None or t in named or c["quantised"] or c["type"] not in FLOATS:
                continue
            size = int(np.prod(c["shape"])) if c["shape"] else 1
            argument = _is_argument(c["name"])
            user, read = _through_transforms(graph, i, t, links)
            if not argument and (size == 1 or not _normalises(graph, user, read, bodies, links)):
                continue
            down = _nearest(graph, i, matrices, True, links)
            role, layer = matrices[down] if down is not None else ("end", None)
            side = (("attention", layer) if role in ATTENTION_IN else ("mlp", layer) if role in MLP_IN
                    else ("output", None) if role in OUTPUT_SIDE else (role, layer))
            if role == "end":                                     # a key norm whose attention was pruned away
                up = _nearest(graph, i, matrices, False, links)
                if up is not None and matrices[up][0] in ATTENTION_IN:
                    side = ("o", matrices[up][1])
            convention, _ = _reader_convention(graph, graph["ops"][user], read, bodies, links)
            member = {"tensor": t, "buffer": c["buffer"], "op": i, "size": size, "convention": convention}
            groups[side].append(member)
            normalising.append(member)
    last = len(by_role.get("down", [])) - 1
    for side, members in sorted(groups.items(), key=lambda kv: min(m["op"] for m in kv[1])):
        members.sort(key=lambda m: m["op"])
        vectors = [m for m in members if m["size"] > 1]
        scalars = [m for m in members if m["size"] == 1]
        kind, layer = side
        expected: list[tuple[str, int | None]] = []
        scalar_roles: list[tuple[str, int | None]] = []
        if kind == "attention" and layer == 0:
            expected = [("input_layernorm", 0)]
        elif kind == "attention":
            expected = [("post_feedforward_layernorm", layer - 1), ("input_layernorm", layer)]
            scalar_roles = [("layer_scalar", layer - 1)]
        elif kind == "mlp":
            expected = [("post_attention_layernorm", layer), ("pre_feedforward_layernorm", layer)]
        elif kind == "output" and last >= 0:
            expected = [("post_feedforward_layernorm", last), ("norm", None)]
            scalar_roles = [("layer_scalar", last)]
        elif kind == "o":                                          # the query (key, value) norms, by their matrix
            for m in vectors:
                up = _nearest(graph, m["op"], matrices, False, links)
                upstream = matrices[up] if up is not None else ("start", None)
                if upstream[0] in ATTENTION_IN and upstream[1] == layer:
                    expected.append((f"{upstream[0]}_norm", layer))
            if len(set(expected)) != len(expected):
                expected = []
        why = f"before {kind}{'' if layer is None else layer}"
        if not expected or len(vectors) != len(expected) or len(scalars) > len(scalar_roles):
            unresolved += [{"tensor": m["tensor"], "buffer": m["buffer"], "op": m["op"], "why": f"{why}: "
                            f"{len(vectors)} vectors and {len(scalars)} scalars where {len(expected)} and at most "
                            f"{len(scalar_roles)} are expected"} for m in members]
            continue
        for m, (role, at) in [*zip(vectors, expected, strict=True), *zip(scalars, scalar_roles, strict=False)]:
            convention = m["convention"]
            if m["size"] == 1 and convention == "unknown":
                convention = "as is"
            placed.append({"tensor": m["tensor"], "buffer": m["buffer"], "op": m["op"], "role": role, "layer": at,
                           "source": _source(role, at), "kind": "vector" if m["size"] > 1 else "scalar",
                           "convention": convention, "why": why})
    layers = {role: len(indices) for role, indices in by_role.items() if role in PER_LAYER}
    return {"subgraph": graph["index"], "placed": placed, "unresolved": unresolved, "normalising": normalising,
            "layers": layers}


# ---------------------------------------------------------------------------------------------
# The mapping and its evidence
# ---------------------------------------------------------------------------------------------

def _relative(x: np.ndarray, a: np.ndarray) -> float:
    norm = float(np.linalg.norm(a))
    if norm > 0:
        return float(np.linalg.norm(x - a)) / norm
    return 0.0 if not np.any(x) else math.inf


def _fits(x: np.ndarray, a: np.ndarray) -> dict[str, float]:
    """The error left after the best scale per row, per column, and both (alternating)."""
    if x.ndim != 2:
        return {}
    rows = (x * a).sum(1) / np.maximum((a * a).sum(1), 1e-300)
    cols = (x * a).sum(0) / np.maximum((a * a).sum(0), 1e-300)
    out = {"after_row_scales": _relative(x, rows[:, None] * a), "after_column_scales": _relative(x, a * cols[None, :])}
    r, c = np.ones(a.shape[0]), np.ones(a.shape[1])
    for _ in range(10):
        b = a * c[None, :]
        r = (x * b).sum(1) / np.maximum((b * b).sum(1), 1e-300)
        b = r[:, None] * a
        c = (x * b).sum(0) / np.maximum((b * b).sum(0), 1e-300)
    out["after_both"] = _relative(x, r[:, None] * a * c[None, :])
    return out


def _spectrum(x: np.ndarray, k: int = 8) -> list[float]:
    if x.ndim != 2:
        return []
    small = x if x.shape[0] <= x.shape[1] else x.T
    eig = np.linalg.eigvalsh(small @ small.T)[::-1][:k]
    return [float(e) for e in np.sqrt(np.clip(eig, 0, None))]


def _layout(file_shape: Sequence[int], source_shape: Sequence[int], reader: str | None) -> str | None:
    file_shape, source_shape = list(file_shape), list(source_shape)
    if int(np.prod(file_shape or [1])) != int(np.prod(source_shape or [1])):
        return None
    if len(source_shape) == 2 and len(file_shape) == 2:
        if source_shape[0] == source_shape[1]:
            return "as is" if file_shape == source_shape and reader == "FULLY_CONNECTED" else None
        if file_shape == source_shape:
            return "as is"
        if file_shape == source_shape[::-1]:
            return "transposed"
        return None
    if len([d for d in file_shape if d != 1]) <= 1 and len([d for d in source_shape if d != 1]) <= 1:
        return "as is"                                            # vectors and scalars: only their length matters
    return "as is" if file_shape == source_shape else None


def transformed(values: np.ndarray, layout: str, convention: str, shape: Sequence[int]) -> np.ndarray:
    """A source tensor as the file holds it (a vector or scalar takes the file's shape; a matrix must have it)."""
    out = np.asarray(values, dtype=np.float64)
    if layout == "transposed":
        out = out.T
    if list(out.shape) != list(shape):
        if len([d for d in out.shape if d != 1]) > 1 or len([d for d in shape if d != 1]) > 1 or (
                out.size != int(np.prod(shape or [1]))):
            raise ValueError(f"a source of shape {list(out.shape)} ({layout}) where the file has {list(shape)}")
        out = out.reshape(list(shape))
    return out + CONVENTIONS[convention]


def mapping_for(weights: Sequence[dict[str, Any]], assignments: Sequence[dict[str, Any]],
                graphs: dict[int, dict[str, Any]], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    """The mapping per buffer from the roles found in every analysed graph (the union of each buffer's sources), what
    a patch must write, and what could not be placed and why."""
    by_buffer = {w["buffer"]: w for w in weights}
    found: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    problems: list[dict[str, Any]] = []
    required = {w["buffer"] for w in weights if w["quantised"] or w["argument"]}
    per_graph: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for result in assignments:
        graph = graphs[result["subgraph"]]
        for m in result["normalising"]:
            required.add(m["buffer"])
        for p in result["placed"]:
            buffer = p["buffer"]
            if buffer not in by_buffer:
                continue
            row = found[buffer].setdefault(p["source"], {**p, "reader": graph["ops"][p["op"]]["kind"],
                                                         "subgraphs": set()})
            row["subgraphs"].add(result["subgraph"])
            per_graph[buffer][result["subgraph"]].add(p["source"])
        for u in result["unresolved"]:
            w = by_buffer.get(u["buffer"])
            problems.append({"buffer": u["buffer"], "short": w["short"] if w else str(u["buffer"]), "why": u["why"],
                             "subgraph": result["subgraph"]})
    mapping = []
    written_to: dict[str, list[int]] = defaultdict(list)
    for buffer, sources in sorted(found.items(), key=lambda kv: by_buffer[kv[0]]["position"]):
        w = by_buffer[buffer]
        sets = list(per_graph[buffer].values())
        if any(not (a <= b or b <= a) for a in sets for b in sets):   # a subgraph that reads less is fine
            problems.append({"buffer": buffer, "short": w["short"], "why": "the subgraphs give it different roles: "
                             + "; ".join(f"{g}: {', '.join(sorted(x))}" for g, x in per_graph[buffer].items())})
            continue
        rows = []
        for source, p in sources.items():
            why = None
            if source not in reference:
                why = f"{source} is not among the drafter's weights"
            else:
                layout = _layout(w["shape"], reference[source].shape, p["reader"])
                if layout is None:
                    why = f"{source} has shape {list(reference[source].shape)}, the file {w['shape']}"
                elif p["convention"] == "unknown":
                    why = f"{source}: the graph does not say how it is stored"
            if why:
                problems.append({"buffer": buffer, "short": w["short"], "why": why})
                continue
            rows.append({"source": source, "role": p["role"], "layer": p["layer"], "kind": p["kind"],
                         "convention": p["convention"], "layout": layout, "why": p["why"],
                         "subgraphs": sorted(p["subgraphs"])})
            written_to[source].append(buffer)
        if rows and len(rows) == len(sources):
            mapping.append({"buffer": buffer, "name": w["name"], "short": w["short"], "position": w["position"],
                            "shape": w["shape"], "type": w["type"], "shared": len(rows) > 1, "sources": rows})
    several = {s: b for s, b in written_to.items() if len(b) > 1}
    return {"mapping": mapping, "required": sorted(required), "problems": problems,
            "sources_in_several_buffers": several}


def evidence_for(view: memoryview, weights: Sequence[dict[str, Any]], mapping: Sequence[dict[str, Any]],
                 reference: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """The values of every placed weight against A1's tensor of its role (module docstring)."""
    by_buffer = {w["buffer"]: w for w in weights}
    out = []
    for m in mapping:
        w = by_buffer[m["buffer"]]
        try:
            x = _dequantised(view, w)
        except ValueError as error:
            out.append({"short": m["short"], "problem": str(error)})
            continue
        for s in m["sources"]:
            a = transformed(reference[s["source"]], s["layout"], s["convention"], w["shape"])
            row: dict[str, Any] = {"short": m["short"], "shape": w["shape"], "type": w["type"],
                                   "source": s["source"], "kind": s["kind"], "convention": s["convention"],
                                   "layout": s["layout"], "relative_error": _relative(x, a),
                                   "cosine": float((x * a).sum() / max(float(np.linalg.norm(x) * np.linalg.norm(a)),
                                                                       1e-300))}
            if s["kind"] == "matrix":
                row.update(_fits(x.reshape(a.shape), a))
                if w["quantised"] and w["type"] in QMAX:
                    dimension = w["tensors"][0]["quantisation"]["dimension"]
                    integers, scales = quantise(a, w["type"], dimension)
                    back = integers.reshape(a.shape) * scales.astype(np.float64).reshape(
                        _scale_shape(a.shape, dimension, scales.size))
                    row["quantising_the_reference"] = _relative(back, a)
                row["spectrum_ratio"] = [float(p / r) if r > 0 else math.inf
                                         for p, r in zip(_spectrum(x), _spectrum(a), strict=False)]
            else:
                plain = transformed(reference[s["source"]], s["layout"], "as is", w["shape"])
                row["by_convention"] = {c: _relative(x, plain + shift) for c, shift in CONVENTIONS.items()}
            out.append(row)
    return out


def _reference(directory: Path) -> dict[str, np.ndarray]:
    reference = {k: v.astype(np.float64) for k, v in read_safetensors(directory).items()}
    if "model.embed_tokens.weight" not in reference and "lm_head.weight" in reference:
        reference["model.embed_tokens.weight"] = reference["lm_head.weight"]   # tied: saved once
    return reference


def relate(model: Path, assistant: Path, patterns: Sequence[tuple[tuple[str, ...], str]] = ROLES) -> dict[str, Any]:
    reference = _reference(assistant)
    with drafter_section(model) as (section, view, tfl_model):
        weights = weights_of(view, tfl_model)
        graphs = {s: read_graph(view, tfl_model, s) for s in range(tfl_model.SubgraphsLength())}
        analysed = signature_subgraphs(tfl_model) or [0]
        roles, disagreeing = buffer_roles([graphs[s] for s in analysed]
                                          + [g for s, g in graphs.items() if s not in analysed], patterns)
        assignments = [assign_roles(graphs[s], graphs, roles) for s in analysed]
        result = mapping_for(weights, assignments, graphs, reference)
        evidence = evidence_for(view, weights, result["mapping"], reference)
    placed = {m["buffer"] for m in result["mapping"]}
    by_buffer = {w["buffer"]: w for w in weights}
    return {"what": "the LiteRT drafter's weights and their roles (docs/e1.md, Section 11.13)", "model": str(model),
            "assistant": str(assistant), "section": section["index"], "section_sha256": section["sha256"],
            "subgraphs": analysed, "layers": [a["layers"] for a in assignments], "weights": len(weights),
            "required": result["required"], "mapped": len(result["mapping"]),
            "unplaced": [{"short": by_buffer[b]["short"], "type": by_buffer[b]["type"], "shape": by_buffer[b]["shape"]}
                         for b in result["required"] if b not in placed],
            "constants": [w["short"] for w in weights if w["buffer"] not in result["required"]
                          and w["buffer"] not in placed],
            "shared": [{"short": m["short"], "sources": [s["source"] for s in m["sources"]]}
                       for m in result["mapping"] if m["shared"]],
            "sources_in_several_buffers": result["sources_in_several_buffers"], "names_disagreeing": disagreeing,
            "problems": result["problems"],
            "evidence": evidence, "mapping": result["mapping"]}


def mapping_file(result: dict[str, Any]) -> dict[str, Any]:
    """What ``patch`` reads: the file it is for, the buffers it must write, the mapping."""
    return {"what": "relate's mapping (docs/e1.md, Section 11.13)", "model": result["model"],
            "section_sha256": result["section_sha256"], "required": result["required"], "mapping": result["mapping"]}


def render_relate(result: dict[str, Any]) -> str:
    lines = ["# The LiteRT drafter's weights and their roles (docs/e1.md, Section 11.13)", "",
             f"`{result['model']}` (section {result['section']}, subgraphs {result['subgraphs']}) against "
             f"`{result['assistant']}`: {result['weights']} weights (buffers with bytes in the section), "
             f"{len(result['required'])} that a patch must write, {result['mapped']} placed. Roles are read from "
             "the graph; the values beside them are evidence (*computed* from the files).", "",
             f"- Per-layer matrices found in each analysed subgraph, by role: {result['layers']}.",
             "- To be written but not placed: " + (", ".join(f"{u['short']} ({u['type']} {u['shape']})"
                                                             for u in result["unplaced"]) or "none") + ".",
             "- Constants of the graph, left as they are: " + (", ".join(result["constants"]) or "none") + ".",
             "- Shared buffers (several sources in one buffer): " + ("; ".join(
                 f"{s['short']}: {', '.join(s['sources'])}" for s in result["shared"]) or "none") + ".",
             "- Matrices whose tensors' names give different roles: " + ("; ".join(result["names_disagreeing"])
                                                                             or "none") + ".",
             "- Sources written into several buffers: " + ("; ".join(
                 f"{s} ({len(b)})" for s, b in result["sources_in_several_buffers"].items()) or "none") + "."]
    if result["problems"]:
        lines += ["", "| not placed | why |", "|---|---|"]
        lines += [f"| {p['short']} | {p['why']} |" for p in result["problems"]]
    lines += ["", "| weight | shape | type | source | layout | relative error | cosine | after row scales | after "
              "column scales | after both | quantising A1 | singular values / A1's |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in result["evidence"]:
        if e.get("kind") != "matrix":
            continue
        spectrum = ", ".join(f"{x:.2f}" for x in e.get("spectrum_ratio", [])[:4])
        lines.append(f"| {e['short']} | {e['shape']} | {e['type']} | {e['source']} | {e['layout']} | "
                     f"{e['relative_error']:.3f} | {e['cosine']:.3f} | {e.get('after_row_scales', math.nan):.3f} | "
                     f"{e.get('after_column_scales', math.nan):.3f} | {e.get('after_both', math.nan):.3f} | "
                     f"{e.get('quantising_the_reference', math.nan):.3f} | {spectrum} |")
    lines += ["", "| vector or scalar | shape | source | stored (graph) | relative error as is / minus one / plus one |",
              "|---|---|---|---|---|"]
    for e in result["evidence"]:
        if e.get("kind") not in ("vector", "scalar"):
            continue
        c = e["by_convention"]
        lines.append(f"| {e['short']} | {e['shape']} | {e['source']} | {e['convention']} | "
                     f"{c['as is']:.3g} / {c['minus one']:.3g} / {c['plus one']:.3g} |")
    lines += [f"\n{e['short']}: {e['problem']}" for e in result["evidence"] if "problem" in e]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------------------------

def codec(model: Path) -> dict[str, Any]:
    rows = []
    with drafter_section(model) as (section, view, tfl_model):
        for w in weights_of(view, tfl_model):
            if not w["quantised"] or w["type"] not in QMAX:
                continue
            raw = np.frombuffer(view, dtype=np.uint8, count=w["bytes"], offset=w["position"])
            first = w["tensors"][0]
            count = int(np.prod(first["shape"])) if first["shape"] else 1
            integers = decode_integers(raw, w["type"], count)
            again = encode_integers(integers, w["type"], w["bytes"])
            scales = [np.frombuffer(view, dtype=np.float32, count=t["scales"], offset=t["scale_position"])
                      for t in w["tensors"]]
            dimension = first["quantisation"]["dimension"] if first["scales"] > 1 else 0
            per_channel = _per_channel(integers.reshape(first["shape"]).astype(np.int32), dimension)
            if first["scales"] == 1:
                per_channel = per_channel.reshape(1, -1)
            reach = np.abs(per_channel).max(axis=1)
            rows.append({"short": w["short"], "shape": w["shape"], "type": w["type"], "tensors": len(w["tensors"]),
                         "bytes_back": again == raw.tobytes(),
                         "same_scales": all(np.array_equal(scales[0], s) for s in scales[1:]),
                         "same_shapes": all(t["shape"] == first["shape"] for t in w["tensors"]),
                         "zero_points_zero": all(t.get("zero_points_zero", True) for t in w["tensors"]),
                         "channels": int(per_channel.shape[0]), "scales": int(scales[0].size),
                         "channels_reaching_qmax": float((reach == QMAX[w["type"]]).mean()),
                         "channels_at_minus_qmax_minus_one": float(
                             (per_channel.min(axis=1) == -QMAX[w["type"]] - 1).mean()),
                         "scale_range": [float(scales[0].min()), float(scales[0].max())]})
    return {"what": "the codec's null test on the drafter's quantised weights (docs/e1.md, Section 11.13)",
            "model": str(model), "section": section["index"], "tensors": rows,
            "all_bytes_back": all(r["bytes_back"] for r in rows)}


def render_codec(result: dict[str, Any]) -> str:
    lines = ["# The codec's null test (docs/e1.md, Section 11.13)", "",
             f"`{result['model']}`, section {result['section']}: every quantised weight decoded and encoded again. "
             f"All bytes back: **{'yes' if result['all_bytes_back'] else 'NO'}**.", "",
             "| weight | shape | type | tensors | bytes back | same scales | zero points 0 | channels at qmax | "
             "channels at -qmax - 1 | scales |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["tensors"]:
        lines.append(f"| {r['short']} | {r['shape']} | {r['type']} | {r['tensors']} | "
                     f"{'yes' if r['bytes_back'] else 'NO'} | {'yes' if r['same_scales'] else 'NO'} | "
                     f"{'yes' if r['zero_points_zero'] else 'no'} | {r['channels_reaching_qmax']:.3f} | "
                     f"{r['channels_at_minus_qmax_minus_one']:.3f} | {r['scale_range'][0]:.3g} .. "
                     f"{r['scale_range'][1]:.3g} |")
    lines += ["", "A channel whose largest |integer| is qmax is what max-abs scales give; a share below 1 means the "
              "file's scales leave headroom or clip, which `patch` does not reproduce (it uses max-abs)."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------------------------

def _header_sha256(path: Path, index: int) -> tuple[int, int, str] | None:
    """Where the ``sha256`` item of section ``index`` keeps its value in the file: (start, length, value)."""
    core, peek, _ = _schema()
    from litert_lm_builder import litertlm_header_schema_py_generated as schema

    with open(path, "rb") as handle:
        handle.seek(core.HEADER_END_LOCATION_BYTE_OFFSET)
        end = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(core.HEADER_BEGIN_BYTE_OFFSET)
        data = handle.read(end - core.HEADER_BEGIN_BYTE_OFFSET)
    meta = peek.read_litertlm_header(str(path), io.StringIO())
    section = meta.SectionMetadata().Objects(index)
    for j in range(section.ItemsLength()):
        item = section.Items(j)
        if item.Key() is None or item.Key().decode() != "sha256" or item.ValueType() != schema.VData.StringValue:
            continue
        table = item.Value()
        value = schema.StringValue()
        value.Init(table.Bytes, table.Pos)
        o = value._tab.Offset(4)
        start = value._tab.Vector(o)
        length = value._tab.VectorLen(o)
        text = bytes(table.Bytes[start:start + length]).decode()
        if bytes(data[start:start + length]).decode() != text:                    # pragma: no cover
            raise ValueError("the header's sha256 item is not where it was read")
        return core.HEADER_BEGIN_BYTE_OFFSET + start, length, text
    return None


def _sha256_of(path: Path, begin: int, end: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(begin)
        remaining = end - begin
        while remaining > 0:
            block = handle.read(min(1 << 24, remaining))
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _encoded(w: dict[str, Any], values: np.ndarray) -> tuple[bytes, bytes | None, np.ndarray]:
    """A weight's new bytes and scales (None for a float weight) and what they decode to."""
    if not w["quantised"]:
        if w["type"] != "FLOAT32":
            raise ValueError(f"{w['short']}: only float32 constants are written as floats")
        data = np.asarray(values, dtype=np.float32).reshape(-1).tobytes()
        return data, None, np.frombuffer(data, dtype=np.float32).astype(np.float64).reshape(values.shape)
    if w["type"] not in QMAX:
        raise ValueError(f"{w['short']}: no encoder for {w['type']}")
    first = w["tensors"][0]
    dimension = first["quantisation"]["dimension"]
    for t in w["tensors"]:
        if (t["shape"] != first["shape"] or t["quantisation"]["dimension"] != dimension
                or t["scales"] != first["scales"] or not t.get("zero_points_zero", True)
                or t["quantisation"]["details"] != "NONE"):
            raise ValueError(f"{w['short']}: its tensors are not quantised alike (shape, dimension, scales, zero "
                             "points): not written")
    if first["scales"] == 1:
        qmax = QMAX[w["type"]]
        scale = np.float32(float(np.abs(values).max()) / qmax or 1.0)
        integers = np.clip(np.rint(values / np.float64(scale)), -qmax, qmax).reshape(-1).astype(np.int32)
        scales = np.array([scale], dtype=np.float32)
        decoded = integers.reshape(values.shape).astype(np.float64) * np.float64(scale)
    else:
        integers, scales = quantise(values, w["type"], dimension)
        if scales.size != first["scales"]:
            raise ValueError(f"{w['short']}: {scales.size} scales where the file has {first['scales']}")
        decoded = integers.reshape(values.shape).astype(np.float64) * scales.astype(np.float64).reshape(
            _scale_shape(values.shape, dimension, scales.size))
    return encode_integers(integers, w["type"], w["bytes"]), scales.tobytes(), decoded


def _plan(weights: Sequence[dict[str, Any]], mapping: dict[str, Any], reference: dict[str, np.ndarray],
          shared: str) -> tuple[list[tuple[int, bytes]], list[dict[str, Any]]]:
    by_buffer = {w["buffer"]: w for w in weights}
    plan = {m["buffer"]: m for m in mapping["mapping"]}
    required = set(mapping["required"]) | {w["buffer"] for w in weights if w["quantised"] or w["argument"]}
    missing = [by_buffer[b]["short"] for b in sorted(required) if b in by_buffer and b not in plan]
    if missing:
        raise ValueError(f"the mapping leaves {len(missing)} of the {len(required)} weights a patch must write out: "
                         + ", ".join(missing[:8]))
    writes, rows = [], []
    for buffer, m in plan.items():
        w = by_buffer.get(buffer)
        if w is None or w["position"] != m["position"]:
            raise ValueError(f"{m['short']}: the mapping is not this file's (buffer {buffer})")
        candidates = []
        for s in m["sources"]:
            if s["source"] not in reference:
                raise ValueError(f"{s['source']} is not among the drafter's weights")
            candidates.append(transformed(reference[s["source"]], s["layout"], s["convention"], w["shape"]))
        values = np.mean(candidates, axis=0) if len(candidates) > 1 else candidates[0]
        spread = max(_relative(c, values) for c in candidates)
        if spread > 0 and shared == "refuse":
            raise ValueError(f"{m['short']} is one buffer for {len(candidates)} sources that differ (largest "
                             f"relative difference from their mean {spread:.3g}): --shared mean writes their mean")
        data, scale_bytes, decoded = _encoded(w, values)
        writes.append((w["position"], data))
        if scale_bytes is not None:
            writes += [(t["scale_position"], scale_bytes) for t in w["tensors"]]
        rows.append({"short": m["short"], "sources": [s["source"] for s in m["sources"]], "type": w["type"],
                     "convention": m["sources"][0]["convention"], "layout": m["sources"][0]["layout"],
                     "tensors": len(w["tensors"]), "quantisation_error": _relative(decoded, values),
                     "shared_spread": spread if len(candidates) > 1 else None})
    return writes, rows


def _null_plan(view: memoryview, weights: Sequence[dict[str, Any]]) -> list[tuple[int, bytes]]:
    writes = []
    for w in weights:
        raw = np.frombuffer(view, dtype=np.uint8, count=w["bytes"], offset=w["position"])
        if w["quantised"] and w["type"] in QMAX:
            count = int(np.prod(w["tensors"][0]["shape"])) if w["tensors"][0]["shape"] else 1
            writes.append((w["position"], encode_integers(decode_integers(raw, w["type"], count), w["type"],
                                                          w["bytes"])))
            writes += [(t["scale_position"], bytes(view[t["scale_position"]:t["scale_position"] + 4 * t["scales"]]))
                       for t in w["tensors"]]
        else:
            writes.append((w["position"], raw.tobytes()))
    return writes


def patch(model: Path, out: Path, weights: Path | None, mapping: dict[str, Any] | None,
          null: bool = False, shared: str = "refuse") -> dict[str, Any]:
    """The module docstring's ``patch``."""
    if not null and (weights is None or mapping is None):
        raise ValueError("a patch needs --weights and --mapping, or --null")
    size = Path(model).stat().st_size
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 1.1 * size:
        raise ValueError(f"{out.parent} has not the {1.1 * size / 2**30:.1f} GiB free that a copy of {model} needs")
    reference = None if null else _reference(weights)
    with drafter_section(model) as (section, view, tfl_model):
        if not null and mapping.get("section_sha256") != section["sha256"]:
            raise ValueError("the mapping was made for another file (its drafter section's SHA-256 differs)")
        found = weights_of(view, tfl_model)
        writes, rows = (_null_plan(view, found), []) if null else _plan(found, mapping, reference, shared)
    tmp = out.with_name(out.name + ".tmp")
    shutil.copyfile(model, tmp)
    with open(tmp, "r+b") as handle:
        for position, data in writes:
            handle.seek(section["begin"] + position)
            handle.write(data)
    stamp = _header_sha256(model, section["index"])
    hashed = None
    if stamp is not None and stamp[2] == section["sha256"]:
        hashed = _sha256_of(tmp, section["begin"], section["end"])
        with open(tmp, "r+b") as handle:
            handle.seek(stamp[0])
            handle.write(hashed.encode())
    tmp.replace(out)
    result = _verify(model, out, section, writes, stamp, hashed, rows, null)
    if not null:
        with drafter_section(out) as (_, view, tfl_model):          # every tensor of a written weight, decoded
            again = {w["buffer"]: w for w in weights_of(view, tfl_model)}
            for m in mapping["mapping"]:
                w = again[m["buffer"]]
                first = _dequantised(view, w)
                for t in w["tensors"][1:]:
                    if not np.array_equal(_dequantised(view, w, t), first):          # pragma: no cover
                        raise ValueError(f"{m['short']}: its tensors decode differently after the patch")
    return result


def _differing(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("the copy is not as long as the original")
    return int(np.count_nonzero(np.frombuffer(left, dtype=np.uint8) != np.frombuffer(right, dtype=np.uint8)))


def _verify(model: Path, out: Path, section: dict[str, Any], writes: Sequence[tuple[int, bytes]],
            stamp: tuple[int, int, str] | None, hashed: str | None, rows: list[dict[str, Any]],
            null: bool) -> dict[str, Any]:
    """Every byte of the copy outside the written ranges is the original's; the written ranges hold what was
    written."""
    ranges = sorted((section["begin"] + p, section["begin"] + p + len(d)) for p, d in writes)
    if stamp is not None and hashed is not None:
        ranges = sorted([*ranges, (stamp[0], stamp[0] + stamp[1])])
    changed, outside = 0, 0
    total = Path(model).stat().st_size
    if Path(out).stat().st_size != total:
        raise ValueError("the copy is not as long as the original")
    with open(model, "rb") as a, open(out, "rb") as b:
        position = 0
        for start, end in [*ranges, (total, total)]:
            start = max(start, position)
            while position < start:
                n = min(1 << 24, start - position)
                left, right = a.read(n), b.read(n)
                if left != right:
                    outside += _differing(left, right)
                position += n
            if end > position:
                left, right = a.read(end - position), b.read(end - position)
                changed += _differing(left, right)
                position = end
    with open(out, "rb") as handle:
        for p, data in writes:
            handle.seek(section["begin"] + p)
            if handle.read(len(data)) != data:                    # pragma: no cover
                raise ValueError("a written range does not hold what was written")
    if outside:
        raise ValueError(f"{outside} bytes outside the written ranges changed: the copy is not the original's")
    if null and changed:
        raise ValueError(f"the null patch changed {changed} bytes: the codec does not give the file back")
    return {"what": "a drafter's weights written into a copy of the file in place (docs/e1.md, Section 11.13)",
            "model": str(model), "out": str(out), "null": null, "ranges": len(ranges),
            "bytes_changed_inside": changed, "bytes_changed_outside": outside,
            "identical": changed == 0 and outside == 0,
            "sha256_item": None if stamp is None else {"was": stamp[2], "now": hashed or stamp[2],
                                                        "was_the_sections": hashed is not None},
            "weights": rows}


def render_patch(result: dict[str, Any]) -> str:
    lines = ["# A drafter written into a copy of the LiteRT-LM file (docs/e1.md, Section 11.13)", "",
             f"`{result['out']}` from `{result['model']}`"
             + (" (the null patch: the file's own integers and scales back)" if result["null"] else "") + ": "
             f"{result['ranges']} ranges written, {result['bytes_changed_inside']:,} bytes changed in them, "
             f"{result['bytes_changed_outside']} outside them; identical to the original: "
             f"{'yes' if result['identical'] else 'no'}."]
    s = result["sha256_item"]
    if s is None:
        lines.append("- The drafter section has no `sha256` item.")
    else:
        lines.append(f"- The drafter section's `sha256` item: {s['was'][:16]}... -> {s['now'][:16]}..."
                     + ("" if s["was_the_sections"] else " (it was not the section's hash: left as it was)"))
    spread = [r for r in result["weights"] if r["shared_spread"]]
    if spread:
        lines.append("- Shared buffers written with the mean of their sources (largest relative difference of a "
                     "source from it): " + "; ".join(f"{r['short']} {r['shared_spread']:.3g}" for r in spread) + ".")
    if result["weights"]:
        lines += ["", "| weight | from | stored | layout | type | tensors | quantisation error |",
                  "|---|---|---|---|---|---|---|"]
        for r in result["weights"]:
            lines.append(f"| {r['short']} | {', '.join(r['sources'])} | {r['convention']} | {r['layout']} | "
                         f"{r['type']} | {r['tensors']} | {r['quantisation_error']:.4f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------

def _section_bytes(path: Path, section: dict[str, Any]) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(section["begin"])
        return handle.read(section["size"])


def _metadata(path: Path, section: dict[str, Any]) -> Any:
    from litert_lm_builder.runtime.proto import llm_metadata_pb2

    message = llm_metadata_pb2.LlmMetadata()
    message.ParseFromString(_section_bytes(path, section))
    return message


def compare(a: Path, b: Path) -> dict[str, Any]:
    left, right = section_table(a), section_table(b)
    rows = []
    for i in range(max(len(left["sections"]), len(right["sections"]))):
        x = left["sections"][i] if i < len(left["sections"]) else None
        y = right["sections"][i] if i < len(right["sections"]) else None
        row: dict[str, Any] = {"index": i, "data_type": (x or y)["data_type"], "model_type": (x or y)["model_type"],
                               "same_bytes": bool(x and y and x["sha256"] == y["sha256"])}
        if x and y:
            items_x = {it["key"]: it["value"] for it in x["items"]}
            items_y = {it["key"]: it["value"] for it in y["items"]}
            row["items_only_in_a"] = sorted(set(items_x) - set(items_y))
            row["items_only_in_b"] = sorted(set(items_y) - set(items_x))
            row["items_differing"] = sorted(k for k in set(items_x) & set(items_y) if items_x[k] != items_y[k])
            if x["data_type"] == "LlmMetadataProto" and not row["same_bytes"]:
                try:
                    mx, my = _metadata(a, x), _metadata(b, y)
                    row["metadata_equal_as_messages"] = mx == my
                    names = {f.name for f, _ in mx.ListFields()} | {f.name for f, _ in my.ListFields()}
                    row["metadata_fields_differing"] = sorted(n for n in names if getattr(mx, n) != getattr(my, n))
                    row["metadata_serialised_again_is_a"] = mx.SerializeToString() == _section_bytes(a, x)
                except Exception as error:                        # noqa: BLE001 - reported, not raised
                    row["metadata_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return {"what": "two LiteRT-LM files side by side (docs/e1.md, Section 11.13)", "a": str(a), "b": str(b),
            "versions": [left["version"], right["version"]], "bytes": [left["bytes"], right["bytes"]],
            "sections": rows}


def render_compare(result: dict[str, Any]) -> str:
    lines = ["# Two LiteRT-LM files side by side (docs/e1.md, Section 11.13)", "",
             f"a: `{result['a']}` ({result['bytes'][0]:,} bytes, version {'.'.join(map(str, result['versions'][0]))});"
             f" b: `{result['b']}` ({result['bytes'][1]:,} bytes, version {'.'.join(map(str, result['versions'][1]))})"
             ".", "", "| section | data type | model type | same bytes | items only in a | items only in b | items "
             "differing | metadata |", "|---|---|---|---|---|---|---|---|"]
    for r in result["sections"]:
        meta = ""
        if "metadata_equal_as_messages" in r:
            meta = ("equal as messages" if r["metadata_equal_as_messages"] else
                    "differs in " + ", ".join(r["metadata_fields_differing"]))
            meta += ("; a's serialised again is a's bytes" if r["metadata_serialised_again_is_a"]
                     else "; a's serialised again is not a's bytes")
        elif "metadata_error" in r:
            meta = r["metadata_error"]
        lines.append(f"| {r['index']} | {r['data_type']} | {r['model_type'] or '--'} | "
                     f"{'yes' if r['same_bytes'] else 'NO'} | {', '.join(r.get('items_only_in_a', [])) or '--'} | "
                     f"{', '.join(r.get('items_only_in_b', [])) or '--'} | "
                     f"{', '.join(r.get('items_differing', [])) or '--'} | {meta or '--'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 1, P3's route (a): a drafter in Google's LiteRT-LM file",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("relate")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--assistant", type=Path, required=True, help="A1's directory (its safetensors)")
    r.add_argument("--roles", type=Path, default=None,
                   help="a JSON list of [[segments...], role] that replaces ROLES (names other than Google's)")
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("codec")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("patch")
    w.add_argument("--model", type=Path, required=True)
    w.add_argument("--weights", type=Path, default=None, help="the drafter's directory (its safetensors)")
    w.add_argument("--mapping", type=Path, default=None, help="relate's mapping.json")
    w.add_argument("--shared", choices=("refuse", "mean"), default="refuse",
                   help="a shared buffer whose sources differ: refuse, or write their mean")
    w.add_argument("--null", action="store_true", help="write the file's own integers and scales back")
    w.add_argument("--out", type=Path, required=True, help="the new .litertlm file (its report beside it)")
    k = sub.add_parser("compare")
    k.add_argument("--a", type=Path, required=True)
    k.add_argument("--b", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    from chat_vpa.e1_common import write_json

    try:
        if args.command == "relate":
            patterns = ROLES
            if args.roles is not None:
                patterns = tuple((tuple(seg), role) for seg, role in json.loads(args.roles.read_text("utf-8")))
                unknown = sorted({role for _, role in patterns} - set(SOURCES) - {"gating1", "gating2"})
                if unknown:
                    raise ValueError(f"--roles names roles that do not exist: {', '.join(unknown)}")
            result, name = relate(args.model, args.assistant, patterns), "relate"
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "mapping.json", mapping_file(result))
            text = render_relate(result)
        elif args.command == "codec":
            result, name = codec(args.model), "codec"
            text = render_codec(result)
        elif args.command == "patch":
            mapping = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None
            result = patch(args.model, args.out, args.weights, mapping, args.null, args.shared)
            name, text = "patch", render_patch(result)
            args.out = args.out.parent
        else:
            result, name = compare(args.a, args.b), "compare"
            text = render_compare(result)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"{name}.json", result)
    (args.out / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
