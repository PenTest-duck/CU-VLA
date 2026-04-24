# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "datasets",
#   "peft",
#   "accelerate",
#   "pygame",
#   "wandb",
#   "huggingface_hub",
#   "pillow",
#   "numpy",
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
    # Clone repo
    repo_url = os.environ.get("CU_VLA_REPO_URL", "https://github.com/PenTest-duck/CU-VLA.git")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, "/workspace/CU-VLA"], check=True)
    os.chdir("/workspace/CU-VLA")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    # Forward all args to the training script
    subprocess.run([sys.executable, "-m", "experiments.action_primitives.train", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
