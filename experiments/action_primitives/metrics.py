"""Diagnostic metrics for Phase B0 training + eval."""
from __future__ import annotations

import torch


CLICK_BTN_CLASS_NAMES = ("idle", "press", "release")


def per_class_click_recall(
    preds: torch.Tensor,    # (N,) int 0..2
    targets: torch.Tensor,  # (N,) int 0..2
) -> dict[str, float]:
    """Per-class recall for the 3-way click button head."""
    out: dict[str, float] = {}
    for cls_idx, name in enumerate(CLICK_BTN_CLASS_NAMES):
        mask = targets == cls_idx
        n = int(mask.sum().item())
        if n == 0:
            out[f"recall_{name}"] = float("nan")
            continue
        correct = int(((preds == targets) & mask).sum().item())
        out[f"recall_{name}"] = correct / n
    return out


def soft_ce_diagnostics(
    logits: torch.Tensor,        # (N, num_bins)
    expert_continuous: torch.Tensor,  # (N,)
    bin_centers: torch.Tensor,   # (num_bins,)
) -> dict[str, float]:
    """Diagnostics for soft-CE failure modes."""
    probs = torch.softmax(logits, dim=-1)
    expected_value = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)  # (N,)
    # Sign accuracy
    sign_match = (torch.sign(expected_value) == torch.sign(expert_continuous))
    sign_acc = sign_match.float().mean().item()
    # Entropy
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    # Mass on wrong-sign bins
    expert_sign = torch.sign(expert_continuous)
    bin_signs = torch.sign(bin_centers).unsqueeze(0).expand_as(probs)
    wrong_sign_mass = torch.where(
        bin_signs != expert_sign.unsqueeze(-1), probs, torch.zeros_like(probs),
    ).sum(dim=-1).mean().item()
    # Expected-value endpoint error
    ev_l1 = (expected_value - expert_continuous).abs().mean().item()
    return {
        "sign_acc": sign_acc,
        "entropy_mean": entropy,
        "wrong_sign_mass_mean": wrong_sign_mass,
        "ev_l1_mean": ev_l1,
    }


def bin_10_frequency(
    pred_bins: torch.Tensor,         # (N,) int
    expert_continuous: torch.Tensor, # (N,) float
    bin_centers: torch.Tensor,       # (num_bins,)
    threshold_px: float = 5.0,
) -> float:
    """Fraction of frames where pred==bin 10 (zero) and |expert| > threshold_px."""
    significant_motion = expert_continuous.abs() > threshold_px
    if not significant_motion.any():
        return 0.0
    n_significant = int(significant_motion.sum().item())
    n_bin_10 = int(((pred_bins == 10) & significant_motion).sum().item())
    return n_bin_10 / n_significant
