"""Upload/download data and checkpoints to/from HuggingFace Hub.

Usage:
    uv run python experiments/act_drag_label/hf_sync.py upload-data --repo user/cu-vla-data
    uv run python experiments/act_drag_label/hf_sync.py download-data --repo user/cu-vla-data
    uv run python experiments/act_drag_label/hf_sync.py upload-checkpoints --repo user/cu-vla-checkpoints
    uv run python experiments/act_drag_label/hf_sync.py download-checkpoints --repo user/cu-vla-checkpoints

Requires: huggingface-cli login (or HF_TOKEN env var)
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download


BASE = os.path.join(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
CHECKPOINTS_DIR = os.path.join(BASE, "checkpoints")


def upload_data(repo: str) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    print(f"Uploading data from {DATA_DIR} to {repo} ...")
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=DATA_DIR,
    )
    print("Done.")


def download_data(repo: str, local_dir: str | None = None) -> None:
    dest = local_dir or DATA_DIR
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading data from {repo} to {dest} ...")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=dest,
    )
    n_files = len([f for f in os.listdir(dest) if f.endswith(".hdf5")])
    print(f"Done. {n_files} episodes downloaded.")


def upload_checkpoints(repo: str) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    print(f"Uploading checkpoints from {CHECKPOINTS_DIR} to {repo} ...")
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=CHECKPOINTS_DIR,
    )
    print("Done.")


def download_checkpoints(repo: str, local_dir: str | None = None) -> None:
    dest = local_dir or CHECKPOINTS_DIR
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading checkpoints from {repo} to {dest} ...")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=dest,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync data/checkpoints with HuggingFace Hub")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("upload-data", help="Upload dataset to HF Hub")
    p.add_argument("--repo", required=True, help="HF repo id (e.g. user/cu-vla-data)")

    p = sub.add_parser("download-data", help="Download dataset from HF Hub")
    p.add_argument("--repo", required=True)
    p.add_argument("--local-dir", default=None)

    p = sub.add_parser("upload-checkpoints", help="Upload checkpoints to HF Hub")
    p.add_argument("--repo", required=True, help="HF repo id (e.g. user/cu-vla-checkpoints)")

    p = sub.add_parser("download-checkpoints", help="Download checkpoints from HF Hub")
    p.add_argument("--repo", required=True)
    p.add_argument("--local-dir", default=None)

    args = parser.parse_args()

    if args.command == "upload-data":
        upload_data(args.repo)
    elif args.command == "download-data":
        download_data(args.repo, getattr(args, "local_dir", None))
    elif args.command == "upload-checkpoints":
        upload_checkpoints(args.repo)
    elif args.command == "download-checkpoints":
        download_checkpoints(args.repo, getattr(args, "local_dir", None))
