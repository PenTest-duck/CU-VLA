# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.11",
#     "torchvision>=0.22",
#     "transformers>=4.52",
#     "numpy>=2.0",
#     "h5py>=3.12",
#     "huggingface-hub>=0.30",
#     "matplotlib>=3.10",
# ]
# ///
"""Self-contained training script for HuggingFace Jobs.

Clones the repo, installs it, and runs train.py with forwarded args.

Usage (via hf CLI):
    hf jobs uv run scripts/hf_job_train.py \
      --flavor t4-medium --timeout 4h --secrets HF_TOKEN \
      -- --backbone resnet18 --chunk-size 10 \
         --hf-data-repo PenTest-duck/cu-vla-data \
         --hf-upload-repo PenTest-duck/cu-vla-checkpoints
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/PenTest-duck/CU-VLA.git"
WORKDIR = "/tmp/cu-vla"


def main() -> None:
    # Clone repo
    if not os.path.exists(WORKDIR):
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", REPO_URL, WORKDIR], check=True)

    os.chdir(WORKDIR)

    # Add repo root to Python path so our modules are importable
    sys.path.insert(0, WORKDIR)

    # Forward all CLI args to train.py, adding --device cuda
    # Filter out non-breaking spaces that can sneak in from copy-paste
    train_args = [a for a in sys.argv[1:] if a.strip()]
    if "--device" not in train_args:
        train_args.extend(["--device", "cuda"])

    print(f"Running train.py with args: {train_args}")
    cmd = [sys.executable, "experiments/act_drag_label/train.py"] + train_args
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
