"""Unit tests for scripts/launch_sm_job.py argument parsing + branch validation."""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from scripts.launch_sm_job import (
    parse_args,
    validate_remote_branch,
    build_job_name,
    DEFAULT_REPO_URL,
)


def test_parse_args_required_train_module():
    args = parse_args(["--train-module", "experiments.foo.train"])
    assert args.train_module == "experiments.foo.train"
    assert args.spot is True  # default
    assert args.detach is False
    assert args.instance_type == "ml.g6e.xlarge"  # default


def test_parse_args_no_spot_flag():
    args = parse_args(["--train-module", "x.y", "--no-spot"])
    assert args.spot is False


def test_parse_args_train_passthrough_args():
    # Anything after `--` is forwarded to the train module.
    args = parse_args(["--train-module", "x.y", "--", "--epochs", "5", "--lr", "1e-4"])
    assert args.train_args == ["--epochs", "5", "--lr", "1e-4"]


def test_parse_args_branch_default_is_main():
    args = parse_args(["--train-module", "x.y"])
    assert args.branch == "main"


def test_build_job_name_format():
    name = build_job_name(experiment="exp6", phase="b0", now_iso="20260427T093142")
    assert name == "cu-vla-exp6-phaseb0-20260427-093142"


def test_build_job_name_no_phase():
    name = build_job_name(experiment="exp6", phase=None, now_iso="20260427T093142")
    assert name == "cu-vla-exp6-20260427-093142"


@patch("scripts.launch_sm_job.subprocess.run")
def test_validate_remote_branch_passes_for_existing(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout=b"abc123\trefs/heads/feat/exp6-phase-b0\n")
    validate_remote_branch("https://github.com/PenTest-duck/CU-VLA.git", "feat/exp6-phase-b0")
    mock_run.assert_called_once()


@patch("scripts.launch_sm_job.subprocess.run")
def test_validate_remote_branch_raises_for_missing(mock_run):
    mock_run.return_value = MagicMock(returncode=2, stdout=b"")
    with pytest.raises(SystemExit):
        validate_remote_branch("https://github.com/PenTest-duck/CU-VLA.git", "nonexistent-branch")
