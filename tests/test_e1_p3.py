# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""P3's route (a) (``chat_vpa.e1_p3_route``, docs/e1.md Sections 11.7 and 11.13), the parts that need no LiteRT-LM
package: the integer codec against bytes written by hand (two 4-bit values per byte, the first in the low four
bits), symmetric max-abs quantisation per channel against an explicit loop, and the proposed mapping on a section
built by hand, whose weights are a reference's quantised, or its norm scales minus one. The parts that read a real
``.litertlm`` file (``drafter_section``, ``codec``, ``patch``, ``compare``) run in LiteRT-LM's environment; their dry
run on a stand-in file is in docs/e1.md, Section 11.13."""

from __future__ import annotations

import numpy as np
import pytest

from chat_vpa import e1_p3
from chat_vpa import e1_p3_route as rt


def test_the_integer_codec_by_hand():
    # int4: 1, -2 -> low 0x1, high 0xE -> 0xE1; 7, -8 -> 0x07 | 0x80 = 0x87; an odd count leaves the high bits 0.
    assert rt.encode_integers(np.array([1, -2, 7, -8]), "INT4", 2) == bytes([0xE1, 0x87])
    assert rt.encode_integers(np.array([3, -1, 5]), "INT4", 2) == bytes([0xF3, 0x05])
    assert rt.decode_integers(np.frombuffer(bytes([0xE1, 0x87]), np.uint8), "INT4", 4).tolist() == [1, -2, 7, -8]
    assert rt.decode_integers(np.frombuffer(bytes([0xF3, 0x05]), np.uint8), "INT4", 3).tolist() == [3, -1, 5]
    assert rt.encode_integers(np.array([-128, 127, -1]), "INT8", 3) == bytes([0x80, 0x7F, 0xFF])
    rng = np.random.default_rng(0)
    for kind, low, high in (("INT4", -8, 8), ("INT8", -128, 128), ("INT16", -32768, 32768)):
        values = rng.integers(low, high, 37)
        size = {"INT4": 19, "INT8": 37, "INT16": 74}[kind]
        data = rt.encode_integers(values, kind, size)
        assert rt.decode_integers(np.frombuffer(data, np.uint8), kind, 37).tolist() == values.tolist()
    with pytest.raises(ValueError, match="bytes encoded"):
        rt.encode_integers(np.array([1, 2]), "INT8", 3)
    # The same unpacking as inspect's dequantisation (e1_p3.dequantise) with unit scales.
    raw = np.frombuffer(bytes([0xE1, 0x87]), np.uint8)
    q = {"details": "NONE", "dimension": 0, "scales": 1}
    assert e1_p3.dequantise(raw, "INT4", [4], q, np.ones(1, np.float32)).tolist() == [1, -2, 7, -8]


def test_quantisation_per_channel_is_an_explicit_loop():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(4, 6))
    for dimension in (0, 1):
        integers, scales = rt.quantise(values, "INT4", dimension)
        integers = integers.reshape(values.shape)
        for c in range(values.shape[dimension]):
            channel = np.take(values, c, axis=dimension)
            scale = np.float32(np.abs(channel).max() / 7)
            assert scales[c] == scale
            expected = np.clip(np.rint(channel / np.float64(scale)), -7, 7)
            assert np.take(integers, c, axis=dimension).tolist() == expected.astype(int).tolist()
        # Dequantised, every value is within half a step of its channel's scale.
        shape = [scales.size if i == dimension else 1 for i in range(2)]
        assert np.all(np.abs(integers * scales.reshape(shape) - values) <= scales.reshape(shape) / 2 + 1e-6)
    zero = rt.quantise(np.zeros((2, 3)), "INT8", 0)
    assert zero[1].tolist() == [1.0, 1.0] and not zero[0].any()                   # a channel of zeros: scale 1


def _section(reference):
    """A section of two weights written by hand: q_proj of layer 0 as 8-bit per channel, and a norm stored as the
    reference's minus one; their weight rows, and a mapping of both."""
    q_integers, q_scales = rt.quantise(reference["model.layers.0.self_attn.q_proj.weight"], "INT8", 0)
    data = rt.encode_integers(q_integers, "INT8", q_integers.size)
    norm = (reference["model.layers.0.input_layernorm.weight"] - 1.0).astype(np.float32).tobytes()
    view = memoryview(data + q_scales.astype(np.float32).tobytes() + norm)
    q = {"buffer": 1, "short": "q", "shape": [4, 6], "type": "INT8", "quantised": True, "position": 0,
         "bytes": len(data), "quantisation": {"details": "NONE", "dimension": 0, "scales": 4},
         "tensors": [{"shape": [4, 6], "quantisation": {"details": "NONE", "dimension": 0, "scales": 4},
                      "scales": 4, "scale_position": len(data), "zero_points_zero": True}]}
    n = {"buffer": 2, "short": "n", "shape": [6], "type": "FLOAT32", "quantised": False, "quantisation": None,
         "position": len(data) + 16, "bytes": 24, "tensors": [{"shape": [6], "quantisation": None}]}
    mapping = [{"buffer": 1, "short": "q", "sources": [{"source": "model.layers.0.self_attn.q_proj.weight",
                                                         "kind": "matrix", "layout": "as is", "convention": "as is"}]},
               {"buffer": 2, "short": "n", "sources": [{"source": "model.layers.0.input_layernorm.weight",
                                                         "kind": "vector", "layout": "as is",
                                                         "convention": "minus one"}]}]
    return view, [q, n], mapping


def test_the_evidence_beside_a_placement():
    rng = np.random.default_rng(2)
    reference = {"model.layers.0.self_attn.q_proj.weight": rng.normal(size=(4, 6)),
                 "model.layers.0.input_layernorm.weight": 1.0 + rng.uniform(1, 3, 6)}
    view, weights, mapping = _section(reference)
    rows = {e["short"]: e for e in rt.evidence_for(view, weights, mapping, reference)}
    q = rows["q"]
    # The section holds the reference quantised at 8 bits: its error is what quantising the reference costs.
    assert q["relative_error"] == pytest.approx(q["quantising_the_reference"], rel=1e-6)
    assert q["relative_error"] < 0.02 and q["cosine"] > 0.999
    assert q["after_row_scales"] <= q["relative_error"] + 1e-12
    assert all(abs(r - 1) < 0.02 for r in q["spectrum_ratio"])
    n = rows["n"]["by_convention"]
    assert n["minus one"] < 1e-6 < n["as is"] and n["plus one"] > n["as is"]
    assert rows["n"]["relative_error"] == pytest.approx(n["minus one"])


class _Graph:
    """A TFLite subgraph written by hand, in ``read_graph``'s plain form."""

    def __init__(self, index: int = 0):
        self.index, self.ops, self.constants, self.count = index, [], {}, 0

    def tensor(self) -> int:
        self.count += 1
        return self.count - 1

    def const(self, name, shape, kind="FLOAT32", values=None, buffer=None) -> int:
        t = self.tensor()
        self.constants[t] = {"name": name, "shape": list(shape), "type": kind, "buffer": 1000 + t if buffer is None
                             else buffer, "quantised": kind in ("INT4", "INT8"), "values": values}
        return t

    def op(self, kind, *inputs, composite=None, body=None) -> int:
        out = self.tensor()
        self.ops.append({"kind": kind, "inputs": list(inputs), "outputs": [out], "composite": composite,
                         "body": body})
        return out

    def plain(self, inputs=()) -> dict:
        return {"index": self.index, "ops": self.ops, "constants": self.constants, "inputs": list(inputs)}


def _rms(g, x, w, minus_one=False):
    """Gemma's RMS normalisation, decomposed; ``minus_one``: the weight is stored minus one and 1 added back."""
    mean = g.op("MEAN", g.op("MUL", x, x), g.const("axis", [1], "INT32"))
    normed = g.op("MUL", x, g.op("RSQRT", g.op("ADD", mean, g.const("eps", [], values=[1e-6]))))
    if minus_one:
        w = g.op("ADD", w, g.const("one", [], values=[1.0]))
    return g.op("MUL", normed, w)


def _drafter_graph(activated_second=(), scalar_layers=(0, 1, 2, 3)):
    """Google's drafter as the roles of route (a) see it: the input projection, four KV-shared layers (three
    sliding ones whose query norms share one buffer, one full), the final norm, the output projection and the
    output embedding as a composite. Every vector is a ``jax2tf_arg_*``; the input norms are stored minus one, the
    others as they are. In the layers of ``activated_second`` the second gating matrix is the activated one."""
    g = _Graph()
    n = iter(range(100))
    arg = lambda shape: g.const(f"jax2tf_arg_{next(n)}/ReadVariableOp", shape)  # noqa: E731
    mat = lambda name, shape: g.const(name, shape, "INT4")  # noqa: E731
    shared_q = arg([4])
    h = g.op("FULLY_CONNECTED", g.tensor(), mat("mtp/mtp_pre_proj/btm,md->btd/dot_general", [8, 24]))
    for layer in range(4):
        x = _rms(g, h, arg([8]), minus_one=True)
        q = g.op("RESHAPE", g.op("FULLY_CONNECTED", x, mat(f"layer_{layer}/attn/_pre_attention_query_fn/q_einsum/"
                                                                "reshape", [8, 8])), g.const("shape", [3], "INT32"))
        q = _rms(g, q, shared_q if layer < 3 else arg([8]))
        scores = g.op("SOFTMAX", g.op("BATCH_MATMUL", g.op("MUL", q, g.tensor()), g.tensor()))
        attention = g.op("RESHAPE", g.op("BATCH_MATMUL", scores, g.tensor()), g.const("shape", [2], "INT32"))
        o = g.op("FULLY_CONNECTED", attention, mat(f"layer_{layer}/attn/attn_vec_einsum/btH,Hd->btd/dot_general",
                                                   [8, 8]))
        h = g.op("ADD", h, _rms(g, o, arg([8])))
        x = _rms(g, h, arg([8]))
        first = g.op("FULLY_CONNECTED", x, mat(f"layer_{layer}/mlp/gating_einsum1/btd,df->btf/dot_general", [16, 8]))
        second = g.op("FULLY_CONNECTED", x, mat(f"layer_{layer}/mlp/gating_einsum2/btd,df->btf/dot_general", [16, 8]))
        if layer in activated_second:
            first, second = second, first
        down = g.op("FULLY_CONNECTED", g.op("MUL", g.op("GELU", first), second),
                    mat(f"layer_{layer}/mlp/linear/btf,fd->btd/dot_general", [8, 16]))
        h = g.op("ADD", h, _rms(g, down, arg([8])))
        if layer in scalar_layers:
            h = g.op("MUL", h, arg([1]))
    x = _rms(g, h, arg([8]))
    g.op("FULLY_CONNECTED", x, mat("mtp/mtp_post_proj/btd,dm->btm/dot_general", [12, 8]))
    g.op("STABLEHLO_COMPOSITE", x, mat("decode_softmax/decode/composite", [40, 8]), composite="odml.decode_softmax")
    return g


def _roles(result):
    return {(p["role"], p["layer"]): p for p in result["placed"]}


def test_role_of_reads_fused_names_by_their_segments():
    assert rt.role_of("a/b/layer_0/mlp/linear/btf,fd->btd/dot_general") == "down"
    assert rt.role_of("a/nonlinear/x") is None                               # a segment, not a substring
    assert rt.role_of("x.y.Model_model/q_einsum/reshape") == "q"            # module prefixes are dropped
    patterns = ((("Linear_up_proj",), "up"), (("Linear_gate_proj",), "gate"))
    fused = "m/torch.nn.Linear_gate_proj;m/torch.nn.Linear_up_proj"
    assert rt.role_of(fused, patterns) == "gate"                             # the first origin that matches
    assert rt.role_of("m/Attention_self_attn;m/torch.nn.Linear_up_proj", patterns) == "up"
    roles, disagreeing = rt.buffer_roles([{"constants": {
        0: {"name": fused, "shape": [4, 4], "type": "INT4", "buffer": 7, "quantised": True},
        1: {"name": "m/torch.nn.Linear_up_proj", "shape": [4, 4], "type": "INT4", "buffer": 7, "quantised": True},
        2: {"name": "m/torch.nn.Linear_up_proj", "shape": [4, 4], "type": "INT4", "buffer": 7, "quantised": True},
        3: {"name": "m/torch.nn.Linear_up_proj", "shape": [2], "type": "INT32", "buffer": 8, "quantised": False}}}],
        patterns)
    assert roles == {7: "up"}                                            # most names win; shapes name nothing
    assert disagreeing == ["m/Linear_gate_proj: gate, up (read as up)"]


def test_every_weight_of_a_drafter_built_by_hand_gets_its_role():
    g = _drafter_graph()
    result = rt.assign_roles(g.plain(), {})
    assert result["unresolved"] == []
    assert result["layers"] == {"q": 4, "o": 4, "gating1": 4, "gating2": 4, "down": 4}
    roles = _roles(result)
    for layer in range(4):
        for role in ("q", "o", "gate", "up", "down", "input_layernorm", "q_norm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm", "layer_scalar"):
            assert (role, layer) in roles, (role, layer)
        assert roles[("input_layernorm", layer)]["convention"] == "minus one"
        assert roles[("post_feedforward_layernorm", layer)]["convention"] == "as is"
        assert roles[("layer_scalar", layer)]["kind"] == "scalar"
    for role in ("pre_projection", "post_projection", "embedding", "norm"):
        assert (role, None) in roles, role
    # The gating matrices: the first is the gate (it is activated); the names say which layer by rank.
    gate = g.constants[roles[("gate", 2)]["tensor"]]["name"]
    assert "layer_2/mlp/gating_einsum1" in gate
    # The query norms of the three sliding layers are one tensor read three times; the full layer's is its own.
    tensors = {layer: roles[("q_norm", layer)]["tensor"] for layer in range(4)}
    assert tensors[0] == tensors[1] == tensors[2] != tensors[3]
    # Each argument is placed once per reader: 4 layers x 6 + the final norm, minus the two re-reads of q_norm.
    assert len(result["normalising"]) == 4 * 6 + 1
    # In a layer whose second gating matrix is the activated one, that one is the gate.
    other = rt.assign_roles(_drafter_graph(activated_second=(1,)).plain(), {})
    swapped = _roles(other)
    assert "gating_einsum2" in _drafter_graph(activated_second=(1,)).constants[swapped[("gate", 1)]["tensor"]]["name"]
    assert "gating_einsum1" in g.constants[_roles(result)[("gate", 1)]["tensor"]]["name"]
    # Without the layers' scalars (folded away by a converter), every norm is still placed.
    bare = _roles(rt.assign_roles(_drafter_graph(scalar_layers=()).plain(), {}))
    assert ("layer_scalar", 0) not in bare and ("input_layernorm", 3) in bare and ("norm", None) in bare


def test_the_mapping_unites_sources_by_buffer_and_lists_what_must_be_written():
    g = _drafter_graph()
    plain = g.plain()
    result = rt.assign_roles(plain, {})
    weights, reference = [], {}
    for t, c in sorted(g.constants.items()):
        if c["type"] == "INT32" or (c["type"] == "FLOAT32" and not c["name"].startswith("jax2tf_arg_")):
            continue
        weights.append({"buffer": c["buffer"], "name": c["name"], "short": c["name"], "shape": c["shape"],
                        "type": c["type"], "quantised": c["quantised"], "argument": c["name"].startswith("jax2tf"),
                        "position": t})
    for p in result["placed"]:
        shape = g.constants[p["tensor"]]["shape"]
        reference[p["source"]] = np.ones((1,) if p["kind"] == "scalar" else shape)
    out = rt.mapping_for(weights, [result], {0: plain}, reference)
    assert out["problems"] == []
    assert {w["buffer"] for w in weights} == set(out["required"])
    shared = [m for m in out["mapping"] if m["shared"]]
    assert len(shared) == 1 and [s["source"] for s in shared[0]["sources"]] == [
        f"model.layers.{k}.self_attn.q_norm.weight" for k in range(3)]
    assert len(out["mapping"]) == len(weights)
    # A source that the drafter does not have, or of another shape, is a problem, and its buffer is left out.
    del reference["model.norm.weight"]
    reference["model.layers.0.mlp.down_proj.weight"] = np.ones((3, 3))
    worse = rt.mapping_for(weights, [result], {0: plain}, reference)
    whys = " ".join(p["why"] for p in worse["problems"])
    assert "model.norm.weight is not among" in whys and "has shape [3, 3]" in whys
    assert len(worse["mapping"]) == len(weights) - 2


def test_what_does_not_fit_is_left_unresolved():
    g = _drafter_graph()
    # An argument read between a query and its attention, beside the query norm: neither is placed.
    q = next(i for i, op in enumerate(g.ops) if op["kind"] == "FULLY_CONNECTED" and "q_einsum" in str(
        [g.constants.get(t, {}).get("name") for t in op["inputs"]]))
    out, scaled = g.ops[q]["outputs"][0], g.tensor()
    g.ops[q + 1]["inputs"][0] = scaled                                   # the query's reshape reads the product
    g.ops.insert(q + 1, {"kind": "MUL", "inputs": [out, g.const("jax2tf_arg_99", [8])], "outputs": [scaled],
                         "composite": None, "body": None})
    result = rt.assign_roles(g.plain(), {})
    assert len(result["unresolved"]) == 2 and all("before o0: 2 vectors" in u["why"] for u in result["unresolved"])
    assert ("q_norm", 0) not in _roles(result) and ("q_norm", 1) in _roles(result)
    # A constant that nothing downstream reads joins the output side's group, which then does not fit.
    g = _drafter_graph()
    g.op("MUL", g.tensor(), g.const("jax2tf_arg_98", [8]))
    result = rt.assign_roles(g.plain(), {})
    assert {u["why"] for u in result["unresolved"]} == {"before output: 3 vectors and 1 scalars where 2 and at most 1 "
                                                        "are expected"}


def test_a_one_broadcast_to_the_vectors_length_still_means_minus_one():
    g = _Graph()
    h = g.op("FULLY_CONNECTED", g.tensor(), g.const("mtp_pre_proj/w", [8, 24], "INT4"))
    w = g.op("ADD", g.const("ones", [8], values=[1.0] * 8), g.const("model/scale", [8]))
    mean = g.op("MEAN", g.op("MUL", h, h), g.const("axis", [1], "INT32"))
    x = g.op("MUL", g.op("MUL", h, g.op("RSQRT", g.op("ADD", mean, g.const("eps", [], values=[1e-6])))), w)
    g.op("FULLY_CONNECTED", x, g.const("q_einsum/w", [8, 8], "INT4"))
    placed = _roles(rt.assign_roles(g.plain(), {}))
    assert placed[("input_layernorm", 0)]["convention"] == "minus one"


def test_a_weight_reshaped_before_its_product_is_still_read():
    g = _Graph()
    h = g.op("FULLY_CONNECTED", g.tensor(), g.const("mtp_pre_proj/w", [8, 24], "INT4"))
    w = g.op("RESHAPE", g.const("model/scale", [8]), g.const("shape", [3], "INT32"))
    x = _rms(g, h, w, minus_one=True)
    g.op("FULLY_CONNECTED", x, g.const("q_einsum/w", [8, 8], "INT4"))
    placed = _roles(rt.assign_roles(g.plain(), {}))
    assert placed[("input_layernorm", 0)]["convention"] == "minus one"
    assert g.constants[placed[("input_layernorm", 0)]["tensor"]]["name"] == "model/scale"


def test_a_matrix_inside_a_composites_decomposition_is_applied_by_the_composite():
    g = _drafter_graph()
    plain = g.plain()
    composite = next(op for op in plain["ops"] if op["composite"] == "odml.decode_softmax")
    embedding = composite["inputs"].pop(1)                          # the table lives in the decomposition instead
    constant = plain["constants"].pop(embedding)
    body = _Graph(index=1)
    body.const(constant["name"], constant["shape"], "INT4", buffer=constant["buffer"])
    composite["body"] = 1
    result = rt.assign_roles(plain, {1: body.plain()})
    roles = _roles(result)
    assert roles[("embedding", None)]["buffer"] == constant["buffer"] and roles[("embedding", None)]["tensor"] == -1
    assert ("norm", None) in roles and result["unresolved"] == []


def test_how_a_composite_norm_stores_its_weight():
    for minus_one, expected in ((False, "as is"), (True, "minus one")):
        body = _Graph(index=1)
        x, w = body.tensor(), body.tensor()
        _rms(body, x, w, minus_one)
        g = _Graph()
        h = g.op("FULLY_CONNECTED", g.tensor(), g.const("mtp_pre_proj/w", [8, 24], "INT4"))
        weight = g.const("model/RMSNorm_input_layernorm", [8])
        normed = g.op("STABLEHLO_COMPOSITE", h, weight, composite="odml.rms_norm", body=1)
        g.op("FULLY_CONNECTED", normed, g.const("q_einsum/w", [8, 8], "INT4"))
        result = rt.assign_roles(g.plain(), {1: body.plain(inputs=[x, w])})
        assert _roles(result)[("input_layernorm", 0)]["convention"] == expected


def test_layouts():
    assert rt._layout([8, 24], [8, 24], "BATCH_MATMUL") == "as is"
    assert rt._layout([24, 8], [8, 24], None) == "transposed"
    assert rt._layout([8, 8], [8, 8], "FULLY_CONNECTED") == "as is"
    assert rt._layout([8, 8], [8, 8], "BATCH_MATMUL") is None            # a square matrix's layout is not seen
    assert rt._layout([1, 1, 8], [8], None) == "as is" and rt._layout([], [1], "MUL") == "as is"
    assert rt._layout([4, 6], [6, 5], None) is None


def _weight(buffer, shape, kind, positions, dimension=0):
    tensors = [{"shape": list(shape), "quantisation": {"details": "NONE", "dimension": dimension,
                                                       "scales": shape[dimension]},
                "scales": shape[dimension], "scale_position": p, "zero_points_zero": True} for p in positions]
    size = int(np.prod(shape))
    return {"buffer": buffer, "short": f"w{buffer}", "shape": list(shape), "type": kind, "quantised": True,
            "argument": False, "position": 0, "bytes": size // 2 if kind == "INT4" else size, "tensors": tensors}


def test_the_plan_writes_every_tensors_scales_and_refuses_what_it_must():
    rng = np.random.default_rng(3)
    w = _weight(5, (4, 6), "INT4", positions=(100, 200))
    norm = {"buffer": 6, "short": "w6", "shape": [6], "type": "FLOAT32", "quantised": False, "argument": True,
            "position": 300, "bytes": 24, "tensors": [{"shape": [6]}]}
    a = rng.normal(size=(4, 6))
    reference = {"m.a": a, "m.n1": rng.normal(size=6), "m.n2": rng.normal(size=6)}
    mapping = {"required": [5, 6], "mapping": [
        {"buffer": 5, "short": "w5", "position": 0, "sources": [{"source": "m.a", "layout": "transposed",
                                                                 "convention": "as is"}]},
        {"buffer": 6, "short": "w6", "position": 300, "sources": [{"source": "m.n1", "layout": "as is",
                                                                   "convention": "minus one"}]}]}
    with pytest.raises(ValueError, match=r"shape \[6, 4\] \(transposed\) where the file has \[4, 6\]"):
        rt._plan([w, norm], mapping, reference, "refuse")
    reference["m.a"] = a.T.copy()
    writes, rows = rt._plan([w, norm], mapping, reference, "refuse")
    by_position = dict(writes)
    assert set(by_position) == {0, 100, 200, 300}                       # the bytes, both tensors' scales, the norm
    assert by_position[100] == by_position[200]
    integers, scales = rt.quantise(a, "INT4", 0)
    assert by_position[0] == rt.encode_integers(integers, "INT4", 12) and by_position[100] == scales.tobytes()
    assert np.frombuffer(by_position[300], np.float32).tolist() == (reference["m.n1"] - 1).astype(
        np.float32).tolist()
    assert rows[0]["quantisation_error"] == pytest.approx(rt._relative(
        integers.reshape(4, 6) * scales[:, None].astype(np.float64), a))
    # A buffer the mapping must write but leaves out; a shared buffer whose sources differ.
    with pytest.raises(ValueError, match="leaves 1 of the 2"):
        rt._plan([w, norm], {**mapping, "mapping": mapping["mapping"][:1]}, reference, "refuse")
    mapping["mapping"][1]["sources"].append({"source": "m.n2", "layout": "as is", "convention": "minus one"})
    with pytest.raises(ValueError, match="2 sources that differ"):
        rt._plan([w, norm], mapping, reference, "refuse")
    writes, rows = rt._plan([w, norm], mapping, reference, "mean")
    mean = ((reference["m.n1"] + reference["m.n2"]) / 2 - 1).astype(np.float32)
    assert np.frombuffer(dict(writes)[300], np.float32).tolist() == mean.tolist()
    assert rows[1]["shared_spread"] == pytest.approx(rt._relative(reference["m.n1"] - 1, mean.astype(np.float64)),
                                                     rel=1e-6)
