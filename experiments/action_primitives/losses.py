"""Loss functions for Experiment 6 (Q2, Q3).

- Focal CE on all action heads (mouse bins, click, scroll, keys 3-way)
- Label smoothing on bin heads (uniform label smoothing across non-target bins)
- Idle-biased smoothing on keys (concentrate smoothing mass on idle neighbor)
- Per-class inverse-frequency weighting on keys (boosts rare keys)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.action_primitives.config import (
    KEY_STATE_IDLE,
    LOSS,
    NUM_KEYS,
)


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal cross-entropy (Lin et al. 2017 focal loss, softmax variant).

    logits: (N, C)    target: (N,) int64 class ids.
    """
    logp = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        C = logits.size(-1)
        smooth = torch.full_like(logp, label_smoothing / (C - 1))
        smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - label_smoothing)
        ce = -(smooth * logp).sum(dim=-1)
        p_true = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
    else:
        ce = F.nll_loss(logp, target, reduction="none")
        p_true = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma
    return (focal_weight * ce).mean()


def keys_focal_loss(
    logits_keys: torch.Tensor,     # (B, NUM_KEYS * 3)
    target_keys: torch.Tensor,     # (B, NUM_KEYS) int64 with 0=press, 1=release, 2=idle
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,  # (NUM_KEYS, 3) or None
    idle_smoothing: float = 0.05,
) -> torch.Tensor:
    """Per-key 3-way focal CE; each key treated independently.

    Handles 77 independent softmaxes in parallel by reshaping to (B*77, 3).
    """
    B = logits_keys.size(0)
    logits = logits_keys.view(B, NUM_KEYS, 3)
    logp = F.log_softmax(logits, dim=-1)  # (B, NUM_KEYS, 3)

    # Idle-biased smoothing: smoothing mass on the idle neighbor, not uniform
    smooth = torch.full_like(logp, 0.0)
    # Main target
    smooth.scatter_(-1, target_keys.unsqueeze(-1), 1.0 - idle_smoothing)
    # Add smoothing mass on idle (even if idle IS the target — no-op since scatter overwrote)
    idle_slot = smooth[..., KEY_STATE_IDLE]
    mask_target_not_idle = (target_keys != KEY_STATE_IDLE).float()
    smooth[..., KEY_STATE_IDLE] = idle_slot + idle_smoothing * mask_target_not_idle
    # For targets that ARE idle, leave full mass on idle (no redistribution)

    # When target IS idle, no redistribution ran but smoothed mass is still 1 - ls.
    # Restore full mass on idle so smoothed distribution sums to 1.0 for every sample.
    target_is_idle = (target_keys == KEY_STATE_IDLE)  # (B, NUM_KEYS) bool
    smooth[..., KEY_STATE_IDLE] = torch.where(
        target_is_idle,
        torch.ones_like(smooth[..., KEY_STATE_IDLE]),
        smooth[..., KEY_STATE_IDLE],
    )

    ce = -(smooth * logp).sum(dim=-1)  # (B, NUM_KEYS)

    # Focal weight
    p_true = logp.gather(-1, target_keys.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma

    # Per-class weighting (rare keys)
    if class_weights is not None:
        # class_weights: (NUM_KEYS, 3); pick based on target
        w = class_weights.gather(-1, target_keys.unsqueeze(-1)).squeeze(-1)  # (B, NUM_KEYS)
        focal_weight = focal_weight * w

    return (focal_weight * ce).mean()


def done_loss(logits_done: torch.Tensor, target_done: torch.Tensor) -> torch.Tensor:
    """Focal BCE for the 1-logit done head (Q8)."""
    # logits_done: (B, 1)  target_done: (B,) float
    logits = logits_done.squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, target_done.float(), reduction="none")
    p_true = torch.sigmoid(logits)
    # Focal: (1 - p_t)^gamma
    p_t = torch.where(target_done.bool(), p_true, 1 - p_true)
    focal_weight = (1.0 - p_t) ** LOSS.focal_gamma
    return (focal_weight * bce).mean()


def total_loss(
    head_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    head_weights: dict[str, float],
    class_weights_keys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted sum of per-head losses; return (total, per-head dict)."""
    per_head = {
        "dx":     focal_ce_loss(head_logits["dx"],     targets["dx"],     LOSS.focal_gamma, LOSS.label_smoothing_mouse),
        "dy":     focal_ce_loss(head_logits["dy"],     targets["dy"],     LOSS.focal_gamma, LOSS.label_smoothing_mouse),
        "click":  focal_ce_loss(head_logits["click"],  targets["click"],  LOSS.focal_gamma),
        "scroll": focal_ce_loss(head_logits["scroll"], targets["scroll"], LOSS.focal_gamma, LOSS.label_smoothing_mouse),
        "keys":   keys_focal_loss(head_logits["keys"], targets["keys"], LOSS.focal_gamma, class_weights_keys, LOSS.idle_smoothing_keys),
        "done":   done_loss(head_logits["done"], targets["done"]),
    }
    total = sum(head_weights[n] * per_head[n] for n in per_head)
    return total, per_head
