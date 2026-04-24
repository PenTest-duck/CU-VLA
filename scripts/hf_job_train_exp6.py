# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "torch>=2.6,<2.11",
#   "torchvision>=0.21,<0.22",
#   "transformers>=4.50",
#   "datasets>=3.0",
#   "peft>=0.10",
#   "accelerate",
#   "pygame-ce>=2.5",
#   "wandb",
#   "huggingface_hub>=0.30",
#   "pillow",
#   "numpy>=2.0",
# ]
# ///
"""HF Jobs training entrypoint for Experiment 6 Phase A.

Clones the CU-VLA repo, downloads the dataset from HF Hub (if --hf-data-repo set),
and runs experiments.action_primitives.train.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    repo_url = os.environ.get("CU_VLA_REPO_URL", "https://github.com/PenTest-duck/CU-VLA.git")
    branch = os.environ.get("CU_VLA_BRANCH", "feat/exp6-phase-a")
    workdir = "/workspace/CU-VLA"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, workdir],
        check=True,
    )
    os.chdir(workdir)
    # No `pip install -e .` — the UV-managed env in HF Jobs lacks pip and all
    # project deps are declared in this script's header. cwd + sys.path lets
    # the `experiments.action_primitives.*` package import without install.
    sys.path.insert(0, workdir)
    # Silence per-batch progress bars from huggingface_hub (snapshot_download)
    # and datasets (load_dataset + ds.filter). They spam the HF Jobs log with
    # hundreds of `Filter: 37%|... ` lines per run.
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

    # -u: unbuffered stdout/stderr so `hf jobs logs` sees print() output in real
    # time instead of in end-of-run batches.
    result = subprocess.run(
        [sys.executable, "-u", "-m", "experiments.action_primitives.train", *sys.argv[1:]],
        cwd=workdir,
        env=env,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
