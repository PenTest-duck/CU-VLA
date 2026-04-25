import pandas as pd
import pytest

from experiments.action_primitives.generate_data import generate_dataset_multiproc


def test_multiproc_generates_n_episodes_to_shards(tmp_path):
    out_dir = tmp_path / "test_b0_dataset"
    out_dir.mkdir()
    generate_dataset_multiproc(
        n_episodes=20, output_dir=out_dir, n_workers=2, seed=0,
        episodes_per_shard=10,
    )
    shards = sorted(out_dir.glob("shard_*.parquet"))
    assert len(shards) >= 2  # 20 eps / 10 per shard = 2 shards
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    assert df["episode_id"].nunique() == 20


def test_multiproc_rows_contain_b0_fields(tmp_path):
    out_dir = tmp_path / "test_b0_fields"
    out_dir.mkdir()
    generate_dataset_multiproc(
        n_episodes=5, output_dir=out_dir, n_workers=1, seed=0,
        episodes_per_shard=10,
    )
    shards = sorted(out_dir.glob("shard_*.parquet"))
    df = pd.read_parquet(shards[0])
    for field in ("instruction", "target_button_id", "n_buttons",
                  "composite_tier", "is_adversarial", "is_scenario_error",
                  "loss_mask", "is_dart_noisy_frame"):
        assert field in df.columns, f"missing field: {field}"
