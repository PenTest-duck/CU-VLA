"""Upload/download data and checkpoints to/from HuggingFace Hub.

Usage:
    uv run python experiments/miniwob_pygame/hf_sync.py upload-data --repo PenTest-duck/cu-vla-exp3-data
    uv run python experiments/miniwob_pygame/hf_sync.py download-data --repo PenTest-duck/cu-vla-exp3-data
    uv run python experiments/miniwob_pygame/hf_sync.py upload-checkpoints --repo PenTest-duck/cu-vla-exp3-checkpoints
    uv run python experiments/miniwob_pygame/hf_sync.py download-checkpoints --repo PenTest-duck/cu-vla-exp3-checkpoints

Data layout: data/{task_name}/{shard}/episode_*.hdf5

Requires: hf auth login (or HF_TOKEN env var)
"""

import argparse
import os

from huggingface_hub import HfApi, snapshot_download


HF_USER = "PenTest-duck"
DEFAULT_DATA_REPO = f"{HF_USER}/cu-vla-exp3-data"
DEFAULT_CHECKPOINTS_REPO = f"{HF_USER}/cu-vla-exp3-checkpoints"

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")


def _discover_tasks(local_dir: str) -> list[str]:
    """List task subdirectories in the data directory."""
    if not os.path.isdir(local_dir):
        return []
    return sorted(
        d
        for d in os.listdir(local_dir)
        if os.path.isdir(os.path.join(local_dir, d)) and not d.startswith(".")
    )


def upload_data(
    local_dir: str,
    repo: str,
    task_names: list[str] | None = None,
) -> None:
    """Upload data for specified tasks (or all) to HF Hub.

    Each task lives in data/{task_name}/{shard}/episode_*.hdf5.
    """
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)

    available = _discover_tasks(local_dir)
    if not available:
        print(f"No task directories found in {local_dir}")
        return

    tasks = task_names if task_names else available
    missing = set(tasks) - set(available)
    if missing:
        print(f"Warning: tasks not found locally, skipping: {sorted(missing)}")
        tasks = [t for t in tasks if t in available]

    for task in tasks:
        task_path = os.path.join(local_dir, task)
        print(f"Uploading {task} from {task_path} to {repo}/data/{task} ...")
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=task_path,
            path_in_repo=f"data/{task}",
        )
        print(f"  Done: {task}")

    print(f"All uploads complete ({len(tasks)} tasks).")


def download_data(
    repo: str,
    local_dir: str,
    task_names: list[str] | None = None,
) -> None:
    """Download data from HF Hub for specified tasks (or all)."""
    os.makedirs(local_dir, exist_ok=True)

    if task_names:
        # Download only specific task directories
        allow_patterns = [f"data/{t}/**" for t in task_names]
    else:
        allow_patterns = ["data/**"]

    print(f"Downloading from {repo} to {local_dir} ...")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    print("Done.")


def upload_checkpoints(
    checkpoint_dir: str,
    repo: str,
    subfolder: str | None = None,
) -> None:
    """Upload model checkpoints to HF Hub."""
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)

    path_in_repo = subfolder or ""
    print(f"Uploading checkpoints from {checkpoint_dir} to {repo}/{path_in_repo} ...")
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=checkpoint_dir,
        path_in_repo=path_in_repo,
    )
    print("Done.")


def download_checkpoints(
    repo: str,
    local_dir: str,
    subfolder: str | None = None,
) -> None:
    """Download model checkpoints from HF Hub."""
    os.makedirs(local_dir, exist_ok=True)

    allow_patterns = [f"{subfolder}/**"] if subfolder else None
    print(f"Downloading checkpoints from {repo} to {local_dir} ...")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync MiniWoB-Pygame data/checkpoints with HuggingFace Hub"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload-data
    p = sub.add_parser("upload-data", help="Upload task data to HF Hub")
    p.add_argument("--repo", default=DEFAULT_DATA_REPO)
    p.add_argument("--local-dir", default=DATA_DIR)
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to upload (default: all)",
    )

    # download-data
    p = sub.add_parser("download-data", help="Download task data from HF Hub")
    p.add_argument("--repo", default=DEFAULT_DATA_REPO)
    p.add_argument("--local-dir", default=DATA_DIR)
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to download (default: all)",
    )

    # upload-checkpoints
    p = sub.add_parser("upload-checkpoints", help="Upload checkpoints to HF Hub")
    p.add_argument("--repo", default=DEFAULT_CHECKPOINTS_REPO)
    p.add_argument("--checkpoint-dir", default=CHECKPOINTS_DIR)
    p.add_argument("--subfolder", default=None, help="Subfolder in HF repo")

    # download-checkpoints
    p = sub.add_parser("download-checkpoints", help="Download checkpoints from HF Hub")
    p.add_argument("--repo", default=DEFAULT_CHECKPOINTS_REPO)
    p.add_argument("--local-dir", default=CHECKPOINTS_DIR)
    p.add_argument("--subfolder", default=None, help="Subfolder in HF repo")

    args = parser.parse_args()

    if args.command == "upload-data":
        upload_data(args.local_dir, args.repo, args.tasks)
    elif args.command == "download-data":
        download_data(args.repo, args.local_dir, args.tasks)
    elif args.command == "upload-checkpoints":
        upload_checkpoints(args.checkpoint_dir, args.repo, args.subfolder)
    elif args.command == "download-checkpoints":
        download_checkpoints(args.repo, args.local_dir, args.subfolder)
