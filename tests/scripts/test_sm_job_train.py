"""Unit tests for the resume-detection logic in scripts/sm_job_train.py.

Only the pure filesystem-only function is tested here. SSM/git/subprocess
behavior is covered end-to-end by Task 11's smoke test.
"""
from pathlib import Path

from scripts.sm_job_train import _latest_step_ckpt


def test_no_ckpts_returns_none(tmp_path):
    assert _latest_step_ckpt(tmp_path) is None


def test_picks_highest_step_with_zero_padded_names(tmp_path):
    # 5-digit zero-padded names → lex sort == numeric sort.
    for name in ["step_00001.pt", "step_00010.pt", "step_00100.pt", "step_00099.pt"]:
        (tmp_path / name).touch()
    latest = _latest_step_ckpt(tmp_path)
    assert latest is not None
    assert latest.name == "step_00100.pt"


def test_ignores_best_pt_and_final_pt(tmp_path):
    (tmp_path / "step_00010.pt").touch()
    (tmp_path / "best.pt").touch()
    (tmp_path / "final.pt").touch()
    latest = _latest_step_ckpt(tmp_path)
    assert latest is not None
    assert latest.name == "step_00010.pt"


def test_nonexistent_dir_returns_none(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    assert _latest_step_ckpt(nonexistent) is None
