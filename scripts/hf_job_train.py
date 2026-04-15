# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "torch>=2.6,<2.11",
#     "torchvision>=0.21,<0.22",
#     "transformers>=4.50",
#     "numpy>=2.0",
#     "datasets>=3.0",
#     "Pillow>=10.0",
#     "huggingface-hub>=0.30",
#     "matplotlib>=3.10",
#     "psutil>=5.9",
# ]
# ///
"""Self-contained training script for HuggingFace Jobs.

Clones the repo, installs it, and runs train.py with forwarded args.
The dataset is loaded via load_dataset() inside train.py — no volume
mounting or snapshot_download needed.

Launch via the companion script:
    uv run python scripts/launch_hf_job.py \
      --flavor t4-medium --timeout 4h \
      -- --backbone resnet18 --chunk-size 10
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/PenTest-duck/CU-VLA.git"
WORKDIR = "/tmp/cu-vla"

EXPERIMENTS = {
    "exp2": {
        "train_script": "experiments/act_drag_label/train.py",
        "default_data_repo": "PenTest-duck/cu-vla-data",
    },
    "exp3": {
        "train_script": "experiments/miniwob_pygame/train.py",
        "default_data_repo": "PenTest-duck/cu-vla-exp3-data",
    },
    "exp5": {
        "train_script": "experiments/mini_editor/train.py",
        "default_data_repo": "PenTest-duck/cu-vla-exp5-data",
    },
}


def main() -> None:
    # Parse --experiment flag before forwarding remaining args
    train_args = [a for a in sys.argv[1:] if a.strip()]

    experiment = "exp3"  # default to latest
    if "--experiment" in train_args:
        idx = train_args.index("--experiment")
        experiment = train_args[idx + 1]
        train_args = train_args[:idx] + train_args[idx + 2:]

    if experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment}. Options: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    exp = EXPERIMENTS[experiment]

    # Clone repo
    if not os.path.exists(WORKDIR):
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", REPO_URL, WORKDIR], check=True)

    os.chdir(WORKDIR)
    sys.path.insert(0, WORKDIR)

    # Add --device cuda if not specified
    if "--device" not in train_args:
        train_args.extend(["--device", "cuda"])

    # Default to loading data from HF Hub if no data source specified
    if "--hf-data-repo" not in train_args and "--data-dir" not in train_args:
        train_args.extend(["--hf-data-repo", exp["default_data_repo"]])

    print(f"Experiment: {experiment}")
    print(f"Running {exp['train_script']} with args: {train_args}")
    cmd = [sys.executable, exp["train_script"]] + train_args
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
