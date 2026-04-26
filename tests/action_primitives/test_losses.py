"""Unit tests for loss functions."""

import numpy as np
import torch

from experiments.action_primitives.config import HEAD_LOGITS, MOUSE_BIN_CENTERS, NUM_BINS_MOUSE, NUM_CLICK_EVENTS, NUM_KEYS
from experiments.action_primitives.losses import (
    focal_ce_loss,
    keys_focal_loss,
    done_loss,
    soft_label_ce,
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


def test_soft_label_ce_concentrates_at_target_bin():
    """When expert is exactly on a bin center, soft label = hard label."""
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    bin_idx = 10  # center bin (zero)
    expert_continuous = torch.tensor([centers[bin_idx].item()])
    # Use logits that heavily favor target bin
    logits = torch.zeros(1, 21)
    logits[0, bin_idx] = 5.0
    loss = soft_label_ce(logits, expert_continuous, centers)
    # Loss should be very small (logits favor the correct bin)
    assert loss.item() < 0.5


def test_soft_label_ce_interpolates_between_bins():
    """When expert is between two bins, matched logits give lower loss than uniform."""
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    # Pick a value between bin 10 (0.0) and bin 11 (~0.026)
    expert_continuous = torch.tensor([centers[10].item() * 0.5 + centers[11].item() * 0.5])
    # Logits: zero everywhere → uniform softmax → high loss
    logits1 = torch.zeros(1, 21)
    loss_uniform = soft_label_ce(logits1, expert_continuous, centers)
    # Logits matching the soft target (bins 10 + 11)
    logits2 = torch.zeros(1, 21)
    logits2[0, 10] = 3.0
    logits2[0, 11] = 3.0
    loss_matched = soft_label_ce(logits2, expert_continuous, centers)
    assert loss_matched < loss_uniform  # matching distribution gets lower loss


def test_total_loss_b0_architecture():
    """total_loss_b0 should compute losses for click_left + click_right (no 'click')."""
    from experiments.action_primitives.losses import total_loss_b0
    B = 4
    head_logits = {
        "dx":          torch.randn(B, 21),
        "dy":          torch.randn(B, 21),
        "click_left":  torch.randn(B, 3),
        "click_right": torch.randn(B, 3),
        "scroll":      torch.randn(B, 21),
        "keys":        torch.randn(B, 231),
        "done":        torch.randn(B, 1),
    }
    targets = {
        "dx_continuous":      torch.randn(B),     # for soft-label CE
        "dy_continuous":      torch.randn(B),
        "scroll_continuous":  torch.randn(B),
        "click_left":         torch.randint(0, 3, (B,)),
        "click_right":        torch.randint(0, 3, (B,)),
        "keys":               torch.randint(0, 3, (B, 77)),
        "done":               torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.ones(B)
    head_weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss_b0(head_logits, targets, head_weights, loss_mask)
    assert torch.isfinite(total)
    assert "click_left" in per_head
    assert "click_right" in per_head


def test_total_loss_b0_zero_mask_zeros_loss():
    """All-zero loss_mask should produce zero total loss."""
    from experiments.action_primitives.losses import total_loss_b0
    B = 4
    head_logits = {
        "dx":          torch.randn(B, 21),
        "dy":          torch.randn(B, 21),
        "click_left":  torch.randn(B, 3),
        "click_right": torch.randn(B, 3),
        "scroll":      torch.randn(B, 21),
        "keys":        torch.randn(B, 231),
        "done":        torch.randn(B, 1),
    }
    targets = {
        "dx_continuous":      torch.randn(B),
        "dy_continuous":      torch.randn(B),
        "scroll_continuous":  torch.randn(B),
        "click_left":         torch.randint(0, 3, (B,)),
        "click_right":        torch.randint(0, 3, (B,)),
        "keys":               torch.randint(0, 3, (B, 77)),
        "done":               torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.zeros(B)
    head_weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss_b0(head_logits, targets, head_weights, loss_mask)
    assert total.item() == 0.0


def test_soft_label_ce_works_with_bf16_logits():
    """Regression test: soft_label_ce must handle bf16 logits (used under autocast).

    Phase B0 first training run on HF Jobs failed because lower_w/upper_w were fp32
    while logits were bf16, causing scatter_ dtype mismatch. Fix casts weights
    to logits.dtype before scatter.
    """
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    logits_bf16 = torch.zeros(4, 21, dtype=torch.bfloat16)
    expert = torch.randn(4, dtype=torch.float32)
    loss = soft_label_ce(logits_bf16, expert, centers)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.bfloat16


def test_total_loss_b0_works_with_bf16_logits():
    """Regression test: total_loss_b0 must handle bf16 logits (used under autocast)."""
    from experiments.action_primitives.losses import total_loss_b0
    B = 4
    head_logits = {
        "dx":          torch.randn(B, 21, dtype=torch.bfloat16),
        "dy":          torch.randn(B, 21, dtype=torch.bfloat16),
        "click_left":  torch.randn(B, 3,  dtype=torch.bfloat16),
        "click_right": torch.randn(B, 3,  dtype=torch.bfloat16),
        "scroll":      torch.randn(B, 21, dtype=torch.bfloat16),
        "keys":        torch.randn(B, 231, dtype=torch.bfloat16),
        "done":        torch.randn(B, 1,  dtype=torch.bfloat16),
    }
    targets = {
        "dx_continuous":      torch.randn(B),
        "dy_continuous":      torch.randn(B),
        "scroll_continuous":  torch.randn(B),
        "click_left":         torch.randint(0, 3, (B,)),
        "click_right":        torch.randint(0, 3, (B,)),
        "keys":               torch.randint(0, 3, (B, 77)),
        "done":               torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.ones(B)
    head_weights = {n: 1.0 for n in head_logits}
    total, per_head = total_loss_b0(head_logits, targets, head_weights, loss_mask)
    assert torch.isfinite(total)


def test_focal_ce_class_weight_upweights_press():
    """class_weight=[1,5,5] makes a press misclassification produce ~5x the
    loss of an idle misclassification of equal logit margin."""
    import torch
    from experiments.action_primitives.losses import _focal_ce_masked

    # Sample 0: target=idle, predicted strongly toward press (wrong)
    # Sample 1: target=press, predicted strongly toward idle (wrong)
    # Same logit margin in both — only difference is target class.
    logits = torch.tensor([
        [0.1, 5.0, 0.1],
        [5.0, 0.1, 0.1],
    ])
    targets = torch.tensor([0, 1])
    loss_mask = torch.ones(2)

    unweighted = _focal_ce_masked(logits, targets, loss_mask, gamma=2.0)
    weighted = _focal_ce_masked(
        logits, targets, loss_mask, gamma=2.0,
        class_weight=torch.tensor([1.0, 5.0, 5.0]),
    )
    # With class_weight=[1,5,5], the press-target sample contributes 5x weight
    # while idle-target stays at 1x. So weighted > unweighted by a factor close
    # to (1 + 5) / (1 + 1) = 3x (roughly; focal weighting moderates this).
    assert float(weighted) > float(unweighted) * 1.5


def test_focal_ce_class_weight_none_matches_no_weight():
    """class_weight=None must reproduce the no-weight path exactly."""
    import torch
    from experiments.action_primitives.losses import _focal_ce_masked

    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loss_mask = torch.ones(8)
    a = _focal_ce_masked(logits, targets, loss_mask, gamma=2.0)
    b = _focal_ce_masked(logits, targets, loss_mask, gamma=2.0, class_weight=None)
    assert torch.allclose(a, b)


def test_total_loss_b0_zero_weight_zeros_head_contribution():
    """A head with weight=0 must not contribute to the total loss."""
    import torch
    from experiments.action_primitives.losses import total_loss_b0

    B = 4
    head_logits = {
        "dx": torch.randn(B, 21), "dy": torch.randn(B, 21),
        "click_left": torch.randn(B, 3), "click_right": torch.randn(B, 3),
        "scroll": torch.randn(B, 21),
        "keys": torch.randn(B, 77, 3),
        "done": torch.randn(B, 1),
    }
    targets = {
        "dx_continuous": torch.randn(B),
        "dy_continuous": torch.randn(B),
        "scroll_continuous": torch.randn(B),
        "click_left": torch.randint(0, 3, (B,)),
        "click_right": torch.randint(0, 3, (B,)),
        "keys": torch.randint(0, 3, (B, 77)),
        "done": torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.ones(B)

    weights_full = {n: 1.0 for n in head_logits}
    weights_no_aux = {n: 1.0 for n in head_logits if n not in ("scroll", "keys", "done")}
    weights_no_aux.update({"scroll": 0.0, "keys": 0.0, "done": 0.0})

    total_full, _ = total_loss_b0(head_logits, targets, weights_full, loss_mask)
    total_partial, _ = total_loss_b0(head_logits, targets, weights_no_aux, loss_mask)
    assert float(total_full) > float(total_partial) + 1e-4


def test_aux_target_loss_first_n_mask_excludes_later_frames():
    """Frames with episode_frame_idx >= first_n must contribute zero loss."""
    import torch
    from experiments.action_primitives.losses import aux_target_loss

    B = 6
    n_cells = 9
    logits = torch.randn(B, n_cells, requires_grad=True)
    target_cells = torch.tensor([0, 1, 2, 3, 4, 5])
    # All frames have idx >= 1 — out of scope when first_n=1
    frame_idx = torch.tensor([1, 2, 3, 4, 5, 6])
    loss = aux_target_loss(logits, target_cells, frame_idx, first_n=1)
    assert float(loss) == 0.0


def test_aux_target_loss_first_n_mask_includes_frame_zero():
    """Loss > 0 when at least one frame has idx < first_n."""
    import torch
    from experiments.action_primitives.losses import aux_target_loss

    B = 6
    n_cells = 9
    # Random logits give nonzero CE on average
    torch.manual_seed(0)
    logits = torch.randn(B, n_cells)
    target_cells = torch.tensor([0, 1, 2, 3, 4, 5])
    # Half in-scope, half out
    frame_idx = torch.tensor([0, 0, 0, 5, 5, 5])
    loss = aux_target_loss(logits, target_cells, frame_idx, first_n=1)
    assert float(loss) > 0.0


def test_aux_target_loss_gradient_flows_only_to_in_scope_samples():
    """Backprop through aux_target_loss should produce gradient only on the
    rows of `logits` that are in scope."""
    import torch
    from experiments.action_primitives.losses import aux_target_loss

    B = 6
    n_cells = 9
    torch.manual_seed(0)
    logits = torch.randn(B, n_cells, requires_grad=True)
    target_cells = torch.tensor([0, 1, 2, 3, 4, 5])
    frame_idx = torch.tensor([0, 0, 0, 5, 5, 5])
    loss = aux_target_loss(logits, target_cells, frame_idx, first_n=1)
    loss.backward()
    # Frames 0-2 (in scope) must have nonzero gradient; 3-5 must have zero.
    assert (logits.grad[:3].abs().sum() > 0).item()
    assert (logits.grad[3:].abs().sum() == 0).item()


def test_total_loss_b0_with_aux_target_includes_aux_in_per_head():
    """When aux_target_logits is provided, total_loss_b0 returns 'aux_target'
    in per_head."""
    import torch
    from experiments.action_primitives.losses import total_loss_b0

    B = 4
    head_logits = {
        "dx": torch.randn(B, 21), "dy": torch.randn(B, 21),
        "click_left": torch.randn(B, 3), "click_right": torch.randn(B, 3),
        "scroll": torch.randn(B, 21),
        "keys": torch.randn(B, 77, 3),
        "done": torch.randn(B, 1),
    }
    targets = {
        "dx_continuous": torch.randn(B),
        "dy_continuous": torch.randn(B),
        "scroll_continuous": torch.randn(B),
        "click_left": torch.randint(0, 3, (B,)),
        "click_right": torch.randint(0, 3, (B,)),
        "keys": torch.randint(0, 3, (B, 77)),
        "done": torch.randint(0, 2, (B,)).float(),
    }
    loss_mask = torch.ones(B)
    head_weights = {n: 1.0 for n in head_logits}
    head_weights["aux_target"] = 0.2

    aux_logits = torch.randn(B, 9)
    aux_cells = torch.randint(0, 9, (B,))
    aux_frame_idx = torch.zeros(B, dtype=torch.long)  # all in scope

    total, per_head = total_loss_b0(
        head_logits, targets, head_weights, loss_mask,
        aux_target_logits=aux_logits,
        aux_target_cells=aux_cells,
        aux_target_frame_idx=aux_frame_idx,
        aux_target_first_n=1,
    )
    assert "aux_target" in per_head
    assert torch.isfinite(per_head["aux_target"])
    assert torch.isfinite(total)
