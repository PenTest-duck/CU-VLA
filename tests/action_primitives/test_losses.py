"""Unit tests for loss functions."""

import torch

from experiments.action_primitives.config import HEAD_LOGITS, NUM_BINS_MOUSE, NUM_CLICK_EVENTS, NUM_KEYS
from experiments.action_primitives.losses import (
    focal_ce_loss,
    keys_focal_loss,
    done_loss,
    total_loss,
)


def test_focal_ce_loss_scalar_and_finite():
    logits = torch.randn(8, NUM_BINS_MOUSE)
    target = torch.randint(0, NUM_BINS_MOUSE, (8,))
    loss = focal_ce_loss(logits, target, gamma=2.0, label_smoothing=0.05)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_focal_ce_loss_decreases_when_confidence_on_target_increases():
    target = torch.tensor([0])
    logits_wrong = torch.zeros(1, NUM_BINS_MOUSE)
    logits_right = torch.zeros(1, NUM_BINS_MOUSE); logits_right[0, 0] = 10.0
    l_wrong = focal_ce_loss(logits_wrong, target)
    l_right = focal_ce_loss(logits_right, target)
    assert l_right < l_wrong


def test_keys_focal_loss_shape_handling():
    B = 4
    logits = torch.randn(B, NUM_KEYS * 3)
    target = torch.full((B, NUM_KEYS), 2, dtype=torch.long)  # all idle
    loss = keys_focal_loss(logits, target, gamma=2.0, idle_smoothing=0.05)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_done_loss_shape_handling():
    logits = torch.randn(8, 1)
    target = torch.randint(0, 2, (8,))
    loss = done_loss(logits, target)
    assert torch.isfinite(loss)


def test_total_loss_all_heads_contribute():
    B = 4
    head_logits = {
        "dx":     torch.randn(B, NUM_BINS_MOUSE),
        "dy":     torch.randn(B, NUM_BINS_MOUSE),
        "click":  torch.randn(B, NUM_CLICK_EVENTS),
        "scroll": torch.randn(B, NUM_BINS_MOUSE),
        "keys":   torch.randn(B, NUM_KEYS * 3),
        "done":   torch.randn(B, 1),
    }
    targets = {
        "dx":     torch.randint(0, NUM_BINS_MOUSE, (B,)),
        "dy":     torch.randint(0, NUM_BINS_MOUSE, (B,)),
        "click":  torch.randint(0, NUM_CLICK_EVENTS, (B,)),
        "scroll": torch.randint(0, NUM_BINS_MOUSE, (B,)),
        "keys":   torch.full((B, NUM_KEYS), 2, dtype=torch.long),
        "done":   torch.randint(0, 2, (B,)),
    }
    weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss(head_logits, targets, weights)
    assert torch.isfinite(total)
    assert set(per_head.keys()) == set(head_logits.keys())
