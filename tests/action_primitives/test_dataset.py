"""Dataset unit tests. Requires generating a tiny dataset first."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Generate 10 episodes into a temp parquet directory."""
    from experiments.action_primitives.generate_data import generate_all

    d = tmp_path_factory.mktemp("phase-a-tiny")
    generate_all(n_episodes=10, out_dir=d, shard_size=5)
    return d


def test_dataset_returns_per_episode_dict(tiny_dataset):
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    ds = PhaseAEpisodeDataset(tiny_dataset, split="train")
    # 80% of 10 = 8 episodes (bucket 0..7)
    assert len(ds) > 0
    ep = ds[0]
    assert "images" in ep
    assert "proprio" in ep
    assert "history" in ep
    assert ep["proprio"].shape[-1] == 83
    assert ep["history"].shape[-1] == 223
    T = len(ep["images"])
    assert ep["dx_bins"].shape == (T,)
    assert ep["key_events"].shape == (T, 77)


def test_dataset_splits_disjoint_by_episode_id(tiny_dataset):
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    tr = PhaseAEpisodeDataset(tiny_dataset, split="train")
    val = PhaseAEpisodeDataset(tiny_dataset, split="val")
    tr_ids = set(tr.episode_ids)
    val_ids = set(val.episode_ids)
    assert not (tr_ids & val_ids)
