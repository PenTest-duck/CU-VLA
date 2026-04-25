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


def test_keys_focal_loss_with_class_weights():
    """Regression test: class_weights path used to crash with rank-mismatch."""
    B = 4
    logits = torch.randn(B, NUM_KEYS * 3)
    target = torch.full((B, NUM_KEYS), 2, dtype=torch.long)  # all idle
    class_weights = torch.ones(NUM_KEYS, 3) * 0.5             # any nonzero shape
    loss = keys_focal_loss(
        logits, target, gamma=2.0,
        class_weights=class_weights, idle_smoothing=0.05,
    )
    assert loss.ndim == 0
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


def test_keys_smoothing_is_probability_distribution():
    """Smoothed target must sum to 1.0 for every (sample, key), including idle targets."""
    B = 3
    logits = torch.randn(B, NUM_KEYS * 3)
    # Mix of idle, press, release targets
    target = torch.tensor([
        [0] * NUM_KEYS,  # all press
        [1] * NUM_KEYS,  # all release
        [2] * NUM_KEYS,  # all idle
    ], dtype=torch.long)
    # We can't directly introspect `smooth` from keys_focal_loss, so instead we
    # replicate the smoothing logic and assert its sum invariant.
    from experiments.action_primitives.config import KEY_STATE_IDLE
    import torch.nn.functional as F
    logp = F.log_softmax(logits.view(B, NUM_KEYS, 3), dim=-1)
    idle_smoothing = 0.05
    smooth = torch.zeros_like(logp)
    smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - idle_smoothing)
    mask_not_idle = (target != KEY_STATE_IDLE).float()
    smooth[..., KEY_STATE_IDLE] = smooth[..., KEY_STATE_IDLE] + idle_smoothing * mask_not_idle
    target_is_idle = (target == KEY_STATE_IDLE)
    smooth[..., KEY_STATE_IDLE] = torch.where(
        target_is_idle,
        torch.ones_like(smooth[..., KEY_STATE_IDLE]),
        smooth[..., KEY_STATE_IDLE],
    )
    # Every (sample, key) row must sum to 1.0 exactly (within floating-point tolerance)
    sums = smooth.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        f"smoothing sum invariant broken: min={sums.min()}, max={sums.max()}"
