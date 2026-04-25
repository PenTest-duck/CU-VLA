import torch
from experiments.action_primitives.metrics import (
    per_class_click_recall, soft_ce_diagnostics, bin_10_frequency,
)
from experiments.action_primitives.config import MOUSE_BIN_CENTERS


def test_per_class_click_recall():
    # 6 frames: 2 idle, 2 press, 2 release
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    # Predictions: idle correct on both, press correct 1/2, release correct 2/2
    preds = torch.tensor([0, 0, 1, 0, 2, 2])
    out = per_class_click_recall(preds, targets)
    assert out["recall_idle"] == 1.0
    assert out["recall_press"] == 0.5
    assert out["recall_release"] == 1.0


def test_soft_ce_diagnostics_returns_all_fields():
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    logits = torch.zeros(4, 21)  # uniform softmax → high entropy
    expert = torch.tensor([1.0, -1.0, 5.0, -5.0])
    out = soft_ce_diagnostics(logits, expert, centers)
    for key in ("sign_acc", "entropy_mean", "wrong_sign_mass_mean", "ev_l1_mean"):
        assert key in out


def test_bin_10_frequency_zero_when_no_significant_motion():
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    pred_bins = torch.tensor([10, 10, 10])  # all bin-10 (zero motion)
    expert = torch.tensor([0.1, -0.5, 0.0])  # all below 5px threshold
    freq = bin_10_frequency(pred_bins, expert, centers)
    assert freq == 0.0


def test_bin_10_frequency_positive_when_model_collapses():
    centers = torch.tensor(MOUSE_BIN_CENTERS, dtype=torch.float32)
    pred_bins = torch.tensor([10, 10, 5, 15])  # 2 of 4 are bin-10
    expert = torch.tensor([20.0, 30.0, 25.0, -25.0])  # all above 5px threshold
    freq = bin_10_frequency(pred_bins, expert, centers)
    assert freq == 0.5  # 2/4 frames have bin-10 prediction
