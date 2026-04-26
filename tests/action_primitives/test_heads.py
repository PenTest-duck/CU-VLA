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


def test_aux_target_head_output_shape():
    """AuxTargetHead: flattened (B, K, d) → (B, n_cells) logits."""
    import torch
    from experiments.action_primitives.heads import AuxTargetHead

    head = AuxTargetHead(n_queries=16, d_model=768, n_cells=9, hidden_dim=256)
    queries = torch.randn(4, 16, 768)
    logits = head(queries)
    assert logits.shape == (4, 9)


def test_aux_target_head_param_count_modest():
    """Aux head should be small (order of 3-4M for 16×768→256→9)."""
    from experiments.action_primitives.heads import AuxTargetHead

    head = AuxTargetHead(n_queries=16, d_model=768, n_cells=9, hidden_dim=256)
    n = sum(p.numel() for p in head.parameters())
    # ~12288*256 + 256*9 = ~3.15M
    assert 1_000_000 < n < 10_000_000, f"AuxTargetHead has {n} params; expected 1M-10M"
