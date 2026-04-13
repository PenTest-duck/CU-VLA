"""Upload/download data and checkpoints to/from HuggingFace Hub.

Usage:
    uv run python experiments/miniwob_pygame/hf_sync.py upload-data --repo PenTest-duck/cu-vla-exp3-data
    uv run python experiments/miniwob_pygame/hf_sync.py download-data --repo PenTest-duck/cu-vla-exp3-data
    uv run python experiments/miniwob_pygame/hf_sync.py upload-checkpoints --repo PenTest-duck/cu-vla-exp3-checkpoints
    uv run python experiments/miniwob_pygame/hf_sync.py download-checkpoints --repo PenTest-duck/cu-vla-exp3-checkpoints

Data layout: data/{task_name}/ (Arrow datasets via save_to_disk)

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
    """List task subdirectories that contain Arrow datasets."""
    if not os.path.isdir(local_dir):
        return []
    tasks = []
    for d in sorted(os.listdir(local_dir)):
        task_dir = os.path.join(local_dir, d)
        if os.path.isdir(task_dir) and (
            os.path.exists(os.path.join(task_dir, "dataset_info.json"))
            or os.path.exists(os.path.join(task_dir, "state.json"))
        ):
            tasks.append(d)
    return tasks


def upload_data(
    local_dir: str,
    repo: str,
    task_names: list[str] | None = None,
    num_shards: int = 10,
) -> None:
    """Upload task datasets to HF Hub using push_to_hub with config_name per task."""
    from datasets import load_from_disk

    available = _discover_tasks(local_dir)
    if not available:
        print(f"No task datasets found in {local_dir}")
        return

    tasks = task_names if task_names else available
    missing = set(tasks) - set(available)
    if missing:
        print(f"Warning: tasks not found locally, skipping: {sorted(missing)}")
        tasks = [t for t in tasks if t in available]

    for task in tasks:
        task_dir = os.path.join(local_dir, task)
        print(f"Loading {task} from {task_dir} ...")
        ds = load_from_disk(task_dir)
        print(f"  {len(ds)} rows. Pushing to {repo} (config={task}) ...")
        ds.push_to_hub(repo, config_name=task, num_shards=num_shards)
        print(f"  Done: {task}")

    print(f"All uploads complete ({len(tasks)} tasks).")


def download_data(
    repo: str,
    local_dir: str,
    task_names: list[str] | None = None,
) -> None:
    """Download task datasets from HF Hub."""
    from datasets import load_dataset

    os.makedirs(local_dir, exist_ok=True)

    if task_names:
        configs = task_names
    else:
        # Download entire repo and let HF handle configs
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(repo)
        print(f"Found configs: {configs}")

    for config in configs:
        print(f"Downloading {config} from {repo} ...")
        ds = load_dataset(repo, name=config, split="train")
        dest = os.path.join(local_dir, config)
        os.makedirs(dest, exist_ok=True)
        ds.save_to_disk(dest)
        n_eps = len(set(ds["episode_id"]))
        print(f"  Done: {n_eps} episodes saved to {dest}")

    print("All downloads complete.")


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
    p.add_argument("--num-shards", type=int, default=10)

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
        upload_data(args.local_dir, args.repo, args.tasks, args.num_shards)
    elif args.command == "download-data":
        download_data(args.repo, args.local_dir, args.tasks)
    elif args.command == "upload-checkpoints":
        upload_checkpoints(args.checkpoint_dir, args.repo, args.subfolder)
    elif args.command == "download-checkpoints":
        download_checkpoints(args.repo, args.local_dir, args.subfolder)
