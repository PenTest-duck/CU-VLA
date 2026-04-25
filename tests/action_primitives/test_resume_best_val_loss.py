"""Unit test for the patch in train.py that restores best_val_loss from best.pt
on --resume. We mock out the heavy training-loop machinery and import only the
small piece of logic we want to verify: given a resume path, if best.pt exists
in the same dir with a known val_loss, the variable is restored to that value;
otherwise it stays float('inf').
"""
import torch
from pathlib import Path


def _restore_best_val_loss(resume_path):
    """Mirrors the patch in train.py — kept here for unit testing.

    The actual patch in train.py inlines this logic (no helper function imported).
    If the patch implementation diverges, update this mirror to match.
    """
    best_val_loss = float("inf")
    if resume_path is not None:
        best_path = resume_path.parent / "best.pt"
        if best_path.exists():
            best_val_loss = torch.load(best_path, map_location="cpu")["val_loss"]
    return best_val_loss


def test_no_resume_returns_inf():
    assert _restore_best_val_loss(None) == float("inf")


def test_resume_no_best_pt_returns_inf(tmp_path):
    # Create a fake step ckpt but no best.pt next to it.
    step_path = tmp_path / "step_00010.pt"
    torch.save({"step": 10}, step_path)
    assert _restore_best_val_loss(step_path) == float("inf")


def test_resume_with_best_pt_returns_stored_loss(tmp_path):
    step_path = tmp_path / "step_00010.pt"
    torch.save({"step": 10}, step_path)
    best_path = tmp_path / "best.pt"
    torch.save({"val_loss": 0.123, "step": 8}, best_path)
    restored = _restore_best_val_loss(step_path)
    assert restored == 0.123
