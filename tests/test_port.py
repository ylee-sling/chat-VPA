# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Guards of the import from text-VPA (docs/provenance.md): a checkpoint of the predecessor must keep loading.

Not ported tests: they check the two things the import itself could break, the names of the realizer's trained
parameters and the checkpoint format that ``runtime.load_checkpoint`` accepts.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from test_stage0 import randomize_adapters, tiny_realizer

from chat_vpa.runtime import CHECKPOINT_FORMAT, load_checkpoint, save_checkpoint

# sha256 of the sorted (name, shape) pairs of the tiny realizer's trainable parameters, computed with text-VPA's
# own ``Stage0Realizer`` at commit 6f97346 (``tiny_realizer()`` of its tests/test_stage0.py; 98 tensors).
TEXT_VPA_TRAINABLE_DIGEST = "2cbbd63551e499ad97695a8898e117361e4f528f642a85300af649e2eba64840"


def test_trainable_parameter_names_are_those_of_text_vpa():
    """``~/text-VPA/runs/stage0-a/last.pt`` loads only if no parameter or submodule of the realizer was renamed."""
    realizer, _ = tiny_realizer()
    names = sorted((n, tuple(p.shape)) for n, p in realizer.named_parameters() if p.requires_grad)
    assert len(names) == 98
    assert hashlib.sha256(json.dumps(names).encode()).hexdigest() == TEXT_VPA_TRAINABLE_DIGEST
    families = {n.split(".")[0] for n, _ in names}
    assert families == {"text", "mask_embedding", "t_mlp", "h_mlp", "mod_proj", "mod_tables"}


def test_checkpoints_of_the_predecessor_are_readable(tmp_path):
    realizer, _ = tiny_realizer()
    randomize_adapters(realizer)
    state = realizer.trainable_state()
    new = save_checkpoint(tmp_path / "new.pt", {"stage": "stage0", "trainable": state})
    assert torch.load(new, weights_only=True)["format"] == CHECKPOINT_FORMAT == "chat-vpa/1"
    old = tmp_path / "old.pt"                                          # as text-VPA's save_checkpoint wrote it
    torch.save({"format": "text-vpa/1", "stage": "stage0", "trainable": state}, old)
    other, _ = tiny_realizer()
    for path in (new, old):
        payload = load_checkpoint(path)
        other.load_trainable_state(payload["trainable"])
        for name, value in other.trainable_state().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0, msg=name)
    foreign = tmp_path / "foreign.pt"
    torch.save({"format": "something-else/1", "trainable": state}, foreign)
    with pytest.raises(ValueError, match="not a chat-VPA or text-VPA checkpoint"):
        load_checkpoint(foreign)
