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


def test_dataset_returns_per_episode_dict_preprocessed(tiny_dataset):
    """Default preprocess=True: returns SigLIP2 pre-processed vision tensors."""
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    ds = PhaseAEpisodeDataset(tiny_dataset, split="train")
    assert len(ds) > 0
    ep = ds[0]
    # Vision side: tensors, not PIL.
    assert "images" not in ep
    for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes"):
        assert k in ep, f"missing preprocessed vision key {k}"
    # All three tensors share the same leading T dimension.
    T = ep["pixel_values"].shape[0]
    assert ep["pixel_attention_mask"].shape[0] == T
    assert ep["spatial_shapes"].shape == (T, 2)
    # Non-vision keys unchanged.
    assert ep["proprio"].shape == (T, 83)
    assert ep["history"].shape[-1] == 223
    assert ep["history"].shape[0] == T
    assert ep["dx_bins"].shape == (T,)
    assert ep["key_events"].shape == (T, 77)


def test_dataset_returns_per_episode_dict_raw_pil(tiny_dataset):
    """preprocess=False gives raw PIL images (used by smoke/inspect paths)."""
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    ds = PhaseAEpisodeDataset(tiny_dataset, split="train", preprocess=False)
    ep = ds[0]
    assert "images" in ep
    assert "pixel_values" not in ep
    T = len(ep["images"])
    assert ep["proprio"].shape[0] == T


def test_dataset_splits_disjoint_by_episode_id(tiny_dataset):
    from experiments.action_primitives.dataset import PhaseAEpisodeDataset

    tr = PhaseAEpisodeDataset(tiny_dataset, split="train")
    val = PhaseAEpisodeDataset(tiny_dataset, split="val")
    tr_ids = set(tr.episode_ids)
    val_ids = set(val.episode_ids)
    assert not (tr_ids & val_ids)
