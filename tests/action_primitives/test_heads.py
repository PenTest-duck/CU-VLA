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


def test_action_heads_outputs_two_click_heads():
    heads = ActionHeads()
    queries = torch.randn(2, MODEL.n_queries, MODEL.d_model)
    out = heads(queries)
    # New B0 architecture: dx, dy, click_left, click_right, scroll, keys, done
    assert "click_left" in out
    assert "click_right" in out
    assert out["click_left"].shape == (2, 3)
    assert out["click_right"].shape == (2, 3)
    assert "click" not in out, "5-way click head should be removed"


def test_head_logits_total_matches_new_arch():
    # 21+21+3+3+21+231+1 = 301
    expected = 21 + 21 + 3 + 3 + 21 + 231 + 1
    actual = sum(HEAD_LOGITS.values())
    assert actual == expected
