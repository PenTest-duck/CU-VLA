"""HF Hub sync helpers for Experiment 6 Phase A."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def upload_parquet_dir(local_dir: Path, repo_id: str, repo_type: str = "dataset") -> None:
    api = HfApi()
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type=repo_type)


def download_hf_dataset(repo_id: str, local_dir: str = "data/hf-download") -> Path:
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
    return Path(path)
