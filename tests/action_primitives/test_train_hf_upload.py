"""Tests for the _HFUploader hardening (post B0 attempt 2 ckpt loss).

Covers:
- Successful upload resets the consecutive-failure counter
- Single transient failures don't kill the run
- N=3 consecutive failures raise RuntimeError
- Successful upload after a transient failure resets the counter (so a single
  network blip every 5 steps doesn't accumulate to a kill)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.action_primitives.train import _HFUploader


@pytest.fixture
def fake_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    return p


def test_uploader_successful_upload_resets_counter(fake_ckpt: Path) -> None:
    u = _HFUploader("test/repo")
    u.consecutive_failures = 2  # simulate prior failures
    with patch("huggingface_hub.upload_file") as mock_upload:
        mock_upload.return_value = None
        u.upload(fake_ckpt)
    assert u.consecutive_failures == 0


def test_uploader_single_transient_failure_does_not_raise(fake_ckpt: Path) -> None:
    """A single failure should be logged and counted but not raise."""
    u = _HFUploader("test/repo")
    with patch("huggingface_hub.upload_file") as mock_upload:
        mock_upload.side_effect = ConnectionError("transient network blip")
        u.upload(fake_ckpt)  # should not raise
    assert u.consecutive_failures == 1


def test_uploader_three_consecutive_failures_raises(fake_ckpt: Path) -> None:
    """N=3 consecutive failures must surface as RuntimeError."""
    u = _HFUploader("test/repo")
    with patch("huggingface_hub.upload_file") as mock_upload:
        mock_upload.side_effect = ConnectionError("HF down")
        u.upload(fake_ckpt)
        u.upload(fake_ckpt)
        with pytest.raises(RuntimeError, match="3 consecutive times"):
            u.upload(fake_ckpt)
    assert u.consecutive_failures == 3


def test_uploader_success_between_failures_resets(fake_ckpt: Path) -> None:
    """A success between failures must reset the counter so the run survives
    occasional blips."""
    u = _HFUploader("test/repo")
    with patch("huggingface_hub.upload_file") as mock_upload:
        # Fail, fail, succeed, fail, fail — counter should be at 2 after the
        # last fail, never hitting the kill threshold.
        mock_upload.side_effect = [
            ConnectionError("blip 1"),
            ConnectionError("blip 2"),
            None,                      # success — resets counter
            ConnectionError("blip 3"),
            ConnectionError("blip 4"),
        ]
        u.upload(fake_ckpt)
        u.upload(fake_ckpt)
        u.upload(fake_ckpt)            # success
        u.upload(fake_ckpt)
        u.upload(fake_ckpt)            # 2nd consecutive fail post-success
    assert u.consecutive_failures == 2  # NOT 4


def test_uploader_nonzero_initial_repo_passes_through(fake_ckpt: Path) -> None:
    """Repo id is forwarded to upload_file unmodified."""
    u = _HFUploader("PenTest-duck/cu-vla-something-v3")
    with patch("huggingface_hub.upload_file") as mock_upload:
        mock_upload.return_value = None
        u.upload(fake_ckpt)
    call_kwargs = mock_upload.call_args.kwargs
    assert call_kwargs["repo_id"] == "PenTest-duck/cu-vla-something-v3"
    assert call_kwargs["repo_type"] == "model"
    assert call_kwargs["path_in_repo"] == fake_ckpt.name
