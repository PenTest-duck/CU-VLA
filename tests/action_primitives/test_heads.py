"""Unit tests for ActionHeads."""

import torch

from experiments.action_primitives.config import HEAD_LOGITS, MODEL
from experiments.action_primitives.heads import ActionHeads


def test_action_heads_output_shapes():
    heads = ActionHeads()
    q = torch.randn(3, MODEL.n_queries, MODEL.d_model)
    out = heads(q)
    for name, n in HEAD_LOGITS.items():
        assert out[name].shape == (3, n), f"{name} got {out[name].shape}"


def test_action_heads_produce_all_six_outputs():
    heads = ActionHeads()
    q = torch.randn(1, MODEL.n_queries, MODEL.d_model)
    out = heads(q)
    assert set(out.keys()) == {"dx", "dy", "click", "scroll", "keys", "done"}
