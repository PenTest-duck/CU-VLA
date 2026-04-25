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


# ---------------------------------------------------------------------------
# Phase B0 dataset tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_b0_dataset(tmp_path_factory):
    """Generate a tiny B0 dataset (2 episodes via the multiproc generator)."""
    from experiments.action_primitives.generate_data import generate_dataset_multiproc

    out_dir = tmp_path_factory.mktemp("phase-b0-tiny")
    generate_dataset_multiproc(
        n_episodes=2,
        output_dir=out_dir,
        n_workers=1,
        seed=0,
        episodes_per_shard=10,
    )
    return out_dir


def test_b0_dataset_yields_b0_fields(tiny_b0_dataset):
    """B0 dataset returns image, proprio, history, action_label dict, loss_mask, instruction."""
    from experiments.action_primitives.dataset import PhaseB0EpisodeDataset

    ds = PhaseB0EpisodeDataset(data_dir=tiny_b0_dataset, split="all", preprocess=False)
    assert len(ds) == 2  # 2 episodes
    sample = ds[0]
    for field in (
        "images", "proprio", "action_history", "action_label",
        "loss_mask", "instruction",
    ):
        assert field in sample, f"missing field: {field}"
    # action_label should have separate click_left, click_right targets
    assert "click_left" in sample["action_label"]
    assert "click_right" in sample["action_label"]


def test_b0_dataset_field_shapes_and_types(tiny_b0_dataset):
    from experiments.action_primitives.dataset import PhaseB0EpisodeDataset

    ds = PhaseB0EpisodeDataset(data_dir=tiny_b0_dataset, split="all", preprocess=False)
    sample = ds[0]
    T = len(sample["images"])
    assert T > 0

    # proprio shape (T, 83) — same as Phase A
    assert sample["proprio"].shape == (T, 83)

    # action_history (T, K, 223)
    assert sample["action_history"].dim() == 3
    assert sample["action_history"].shape[0] == T
    assert sample["action_history"].shape[-1] == 223

    # loss_mask (T,) float
    assert sample["loss_mask"].shape == (T,)
    assert sample["loss_mask"].dtype == torch.float32

    # instruction is a string (per-episode)
    assert isinstance(sample["instruction"], str)
    assert len(sample["instruction"]) > 0

    # action_label fields all (T,) long, with click split into two 3-way labels
    label = sample["action_label"]
    for k in ("dx_bins", "dy_bins", "click_left", "click_right", "scroll_bins", "dones"):
        assert k in label, f"missing label key {k}"
        assert label[k].shape == (T,)
        assert label[k].dtype == torch.long
    # key_events label is (T, NUM_KEYS=77)
    assert label["key_events"].shape == (T, 77)
    assert label["key_events"].dtype == torch.long

    # click_left / click_right values must be in {0,1,2}
    assert int(label["click_left"].min()) >= 0
    assert int(label["click_left"].max()) <= 2
    assert int(label["click_right"].min()) >= 0
    assert int(label["click_right"].max()) <= 2


def test_b0_click_split_logic():
    """Direct unit test of the 5-way → (left, right) 3-way split."""
    from experiments.action_primitives.dataset import _split_click_5way

    assert _split_click_5way(0) == (0, 0)  # idle
    assert _split_click_5way(1) == (1, 0)  # L_press
    assert _split_click_5way(2) == (2, 0)  # L_release
    assert _split_click_5way(3) == (0, 1)  # R_press
    assert _split_click_5way(4) == (0, 2)  # R_release
