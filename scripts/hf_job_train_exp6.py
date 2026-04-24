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
    workdir = "/workspace/CU-VLA"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, workdir], check=True)
    os.chdir(workdir)
    # No `pip install -e .` — the UV-managed env in HF Jobs lacks pip and all
    # project deps are declared in this script's header. cwd + sys.path lets
    # the `experiments.action_primitives.*` package import without install.
    sys.path.insert(0, workdir)
    result = subprocess.run(
        [sys.executable, "-m", "experiments.action_primitives.train", *sys.argv[1:]],
        cwd=workdir,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
