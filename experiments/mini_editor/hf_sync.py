"""Upload/download data and checkpoints to/from HuggingFace Hub.

Usage:
    uv run python -m experiments.mini_editor.hf_sync upload-data --repo PenTest-duck/cu-vla-mini-editor
    uv run python -m experiments.mini_editor.hf_sync download-data --repo PenTest-duck/cu-vla-mini-editor
    uv run python -m experiments.mini_editor.hf_sync upload-checkpoints --repo PenTest-duck/cu-vla-mini-editor-checkpoints
    uv run python -m experiments.mini_editor.hf_sync download-checkpoints --repo PenTest-duck/cu-vla-mini-editor-checkpoints

Data layout: data/mini_editor/ (Arrow dataset via save_to_disk)

Requires: hf auth login (or HF_TOKEN env var)
"""

import argparse
import os

from huggingface_hub import HfApi, snapshot_download


HF_USER = "PenTest-duck"
DEFAULT_DATA_REPO = f"{HF_USER}/cu-vla-mini-editor"
DEFAULT_CHECKPOINTS_REPO = f"{HF_USER}/cu-vla-mini-editor-checkpoints"

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")


def upload_data(
    local_dir: str,
    repo: str,
    num_shards: int = 10,
) -> None:
    """Upload dataset to HF Hub."""
    from datasets import load_from_disk

    if not os.path.isdir(local_dir):
        print(f"No dataset found at {local_dir}")
        return

    print(f"Loading dataset from {local_dir} ...")
    ds = load_from_disk(local_dir)
    print(f"  {len(ds)} rows. Pushing to {repo} ...")
    ds.push_to_hub(repo, num_shards=num_shards)
    print("Done.")


def download_data(
    repo: str,
    local_dir: str,
) -> None:
    """Download dataset from HF Hub."""
    from datasets import load_dataset

    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading dataset from {repo} ...")
    ds = load_dataset(repo, split="train")
    ds.save_to_disk(local_dir)
    n_eps = len(set(ds["episode_id"]))
    print(f"  Done: {n_eps} episodes saved to {local_dir}")


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
        description="Sync Mini Editor data/checkpoints with HuggingFace Hub"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload-data
    p = sub.add_parser("upload-data", help="Upload data to HF Hub")
    p.add_argument("--repo", default=DEFAULT_DATA_REPO)
    p.add_argument("--local-dir", default=DATA_DIR)
    p.add_argument("--num-shards", type=int, default=10)

    # download-data
    p = sub.add_parser("download-data", help="Download data from HF Hub")
    p.add_argument("--repo", default=DEFAULT_DATA_REPO)
    p.add_argument("--local-dir", default=DATA_DIR)

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
        upload_data(args.local_dir, args.repo, args.num_shards)
    elif args.command == "download-data":
        download_data(args.repo, args.local_dir)
    elif args.command == "upload-checkpoints":
        upload_checkpoints(args.checkpoint_dir, args.repo, args.subfolder)
    elif args.command == "download-checkpoints":
        download_checkpoints(args.repo, args.local_dir, args.subfolder)
