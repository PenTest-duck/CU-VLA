"""Integration smoke test: generate tiny dataset, train 1 epoch, save final.pt,
resume from final.pt, train 1 more epoch. Verifies resume wires up correctly.

Marked slow — SigLIP2 forward on CPU is expensive. On M1 M-series or with CUDA
this runs in ~2-10 minutes. On a fresh machine, SigLIP2 weights (~400 MB) will
download on first invocation.

NOTE: In practice this is typically only exercised on GPU in the HF Jobs
environment where a full training run is happening anyway — CPU runtime on
a laptop can reach several hours.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_resume_smoke(tmp_path):
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "ckpt"

    # 1. Generate 2 episodes (minimal; the training loop cycles via infinite_loader).
    subprocess.run(
        [sys.executable, "-m", "experiments.action_primitives.generate_data",
         "-n", "2", "-o", str(data_dir), "--shard-size", "2"],
        check=True,
    )

    # 2. Train for 1 epoch — produces final.pt at step=1 (tiny dataset, 1 step/epoch).
    subprocess.run(
        [sys.executable, "-m", "experiments.action_primitives.train",
         "--data-dir", str(data_dir), "--epochs", "1",
         "--device", "cpu", "--out-dir", str(ckpt_dir),
         "--wandb-mode", "disabled"],
        check=True,
    )

    # 3. Assert final.pt was produced.
    final_ckpt = ckpt_dir / "final.pt"
    assert final_ckpt.exists(), f"expected {final_ckpt} to exist after first training run"

    # 4. Resume from final.pt, train 1 more epoch — expects loop to advance step
    #    from 1 → 2 and overwrite final.pt.
    subprocess.run(
        [sys.executable, "-m", "experiments.action_primitives.train",
         "--data-dir", str(data_dir), "--epochs", "2",
         "--device", "cpu", "--out-dir", str(ckpt_dir),
         "--wandb-mode", "disabled",
         "--resume", str(final_ckpt)],
        check=True,
    )

    # 5. final.pt should still exist (overwritten with new state).
    assert final_ckpt.exists()

    # 6. Verify the resumed checkpoint has a larger step count than the original.
    import torch
    ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    assert ckpt["step"] >= 2, f"expected step >= 2 after resume run, got {ckpt['step']}"
