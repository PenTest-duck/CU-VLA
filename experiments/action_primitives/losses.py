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
        # class_weights: (NUM_KEYS, 3) — broadcast to (B, NUM_KEYS, 3) so
        # gather() ranks match. Pick the weight for the target class per (b, k).
        B = target_keys.size(0)
        w = (class_weights.unsqueeze(0).expand(B, -1, -1)
             .gather(-1, target_keys.unsqueeze(-1)).squeeze(-1))  # (B, NUM_KEYS)
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


def soft_label_ce(
    logits: torch.Tensor,         # (B, num_bins)
    expert_continuous: torch.Tensor,  # (B,) float — expert action in continuous units
    bin_centers: torch.Tensor,    # (num_bins,) float — sorted ascending
) -> torch.Tensor:
    """2-bin triangular soft-label cross-entropy.

    For each sample, find the two bins bracketing expert_continuous and assign
    weights linearly interpolated between them. Example: bins [0, 1, 4, 10],
    expert=2.5 → bracketing bins {1, 2}, weight on bin 1 = (4-2.5)/(4-1) = 0.5,
    weight on bin 2 = (2.5-1)/(4-1) = 0.5.
    """
    B = logits.size(0)
    num_bins = bin_centers.size(0)
    # For each sample, find the rightmost bin <= expert (lower bracket)
    # `searchsorted` with side="right" gives the index of the first bin > expert.
    upper_idx = torch.searchsorted(bin_centers, expert_continuous, right=True)
    upper_idx = torch.clamp(upper_idx, 1, num_bins - 1)
    lower_idx = upper_idx - 1
    lower_centers = bin_centers[lower_idx]
    upper_centers = bin_centers[upper_idx]
    span = upper_centers - lower_centers
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    upper_w = (expert_continuous - lower_centers) / span
    lower_w = 1.0 - upper_w
    # Cast weights to logits.dtype so scatter_ works under bf16 autocast
    # (logits are bf16; weights computed from fp32 expert_continuous + bin_centers).
    upper_w = torch.clamp(upper_w, 0.0, 1.0).to(logits.dtype)
    lower_w = torch.clamp(lower_w, 0.0, 1.0).to(logits.dtype)

    # Construct soft target (B, num_bins)
    soft_target = torch.zeros_like(logits)
    soft_target.scatter_(-1, lower_idx.unsqueeze(-1), lower_w.unsqueeze(-1))
    soft_target.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_w.unsqueeze(-1))

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_target * log_probs).sum(dim=-1).mean()
    return loss


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


# ============================================================================
# Phase B0: masked variants + total_loss_b0
# ============================================================================
from experiments.action_primitives.config import (  # noqa: E402
    MOUSE_BIN_CENTERS,
    SCROLL_BIN_CENTERS,
)


def _focal_ce_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    label_smoothing: float = 0.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample focal CE multiplied by loss_mask; mean over masked samples.

    If ``class_weight`` is provided, it must be a (C,) tensor; per-sample CE
    is multiplied by ``class_weight[target]``. Used in B0 attempt 2 to
    upweight click press/release (rare classes) over idle (~95% of frames).
    """
    logp = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        C = logits.size(-1)
        smooth = torch.full_like(logp, label_smoothing / (C - 1))
        smooth.scatter_(-1, target.unsqueeze(-1), 1.0 - label_smoothing)
        ce = -(smooth * logp).sum(dim=-1)
    else:
        ce = F.nll_loss(logp, target, reduction="none")
    p_true = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma
    per_sample = focal_weight * ce  # (B,)
    if class_weight is not None:
        per_sample = per_sample * class_weight.to(per_sample.device)[target]
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def _soft_label_ce_masked(
    logits: torch.Tensor,
    expert_continuous: torch.Tensor,
    bin_centers: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    B = logits.size(0)
    num_bins = bin_centers.size(0)
    upper_idx = torch.searchsorted(bin_centers, expert_continuous, right=True)
    upper_idx = torch.clamp(upper_idx, 1, num_bins - 1)
    lower_idx = upper_idx - 1
    lower_centers = bin_centers[lower_idx]
    upper_centers = bin_centers[upper_idx]
    span = upper_centers - lower_centers
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    # Cast weights to logits.dtype so scatter_ works under bf16 autocast
    # (logits are bf16; weights computed from fp32 expert_continuous + bin_centers).
    upper_w = torch.clamp((expert_continuous - lower_centers) / span, 0.0, 1.0).to(logits.dtype)
    lower_w = (1.0 - upper_w).to(logits.dtype)
    soft_target = torch.zeros_like(logits)
    soft_target.scatter_(-1, lower_idx.unsqueeze(-1), lower_w.unsqueeze(-1))
    soft_target.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_w.unsqueeze(-1))
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(soft_target * log_probs).sum(dim=-1)
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def _done_loss_masked(
    logits_done: torch.Tensor,
    target_done: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    logits = logits_done.squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, target_done.float(), reduction="none")
    p_true = torch.sigmoid(logits)
    p_t = torch.where(target_done.bool(), p_true, 1 - p_true)
    focal_weight = (1.0 - p_t) ** LOSS.focal_gamma
    per_sample = focal_weight * bce
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def _keys_focal_loss_masked(
    logits_keys: torch.Tensor,
    target_keys: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 2.0,
    idle_smoothing: float = 0.05,
) -> torch.Tensor:
    B = logits_keys.size(0)
    logits = logits_keys.view(B, NUM_KEYS, 3)
    logp = F.log_softmax(logits, dim=-1)
    smooth = torch.full_like(logp, 0.0)
    smooth.scatter_(-1, target_keys.unsqueeze(-1), 1.0 - idle_smoothing)
    idle_slot = smooth[..., KEY_STATE_IDLE]
    mask_target_not_idle = (target_keys != KEY_STATE_IDLE).float()
    smooth[..., KEY_STATE_IDLE] = idle_slot + idle_smoothing * mask_target_not_idle
    target_is_idle = (target_keys == KEY_STATE_IDLE)
    smooth[..., KEY_STATE_IDLE] = torch.where(
        target_is_idle,
        torch.ones_like(smooth[..., KEY_STATE_IDLE]),
        smooth[..., KEY_STATE_IDLE],
    )
    ce = -(smooth * logp).sum(dim=-1)  # (B, NUM_KEYS)
    p_true = logp.gather(-1, target_keys.unsqueeze(-1)).squeeze(-1).exp()
    focal_weight = (1.0 - p_true) ** gamma
    per_sample = (focal_weight * ce).mean(dim=-1)  # (B,)
    masked = per_sample * loss_mask
    n_active = loss_mask.sum().clamp(min=1)
    return masked.sum() / n_active


def total_loss_b0(
    head_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    head_weights: dict[str, float],
    loss_mask: torch.Tensor,            # (B,) float — 0 means skip, 1 means train
    bin_centers_mouse: torch.Tensor | None = None,
    bin_centers_scroll: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """B0 total loss: soft-CE on dx/dy/scroll, focal CE on click_left/click_right/keys, focal BCE on done."""
    if bin_centers_mouse is None:
        bin_centers_mouse = torch.tensor(MOUSE_BIN_CENTERS, device=head_logits["dx"].device)
    if bin_centers_scroll is None:
        bin_centers_scroll = torch.tensor(SCROLL_BIN_CENTERS, device=head_logits["scroll"].device)
    click_w = torch.tensor(LOSS.click_class_weight, device=head_logits["click_left"].device)
    per_head = {
        "dx":          _soft_label_ce_masked(head_logits["dx"], targets["dx_continuous"], bin_centers_mouse, loss_mask),
        "dy":          _soft_label_ce_masked(head_logits["dy"], targets["dy_continuous"], bin_centers_mouse, loss_mask),
        "click_left":  _focal_ce_masked(head_logits["click_left"], targets["click_left"], loss_mask, LOSS.focal_gamma, class_weight=click_w),
        "click_right": _focal_ce_masked(head_logits["click_right"], targets["click_right"], loss_mask, LOSS.focal_gamma, class_weight=click_w),
        "scroll":      _soft_label_ce_masked(head_logits["scroll"], targets["scroll_continuous"], bin_centers_scroll, loss_mask),
        "keys":        _keys_focal_loss_masked(head_logits["keys"], targets["keys"], loss_mask, LOSS.focal_gamma, LOSS.idle_smoothing_keys),
        "done":        _done_loss_masked(head_logits["done"], targets["done"], loss_mask),
    }
    total = sum(head_weights.get(n, 0.0) * per_head[n] for n in per_head)
    return total, per_head
