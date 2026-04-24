"""Phase A batched episode generator.

Generates N L-click episodes and writes a HuggingFace `datasets`-compatible
parquet shard set.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

from experiments.action_primitives.config import PHASE_A_DATA
from experiments.action_primitives.generator import generate_one_episode


FEATURES = Features({
    "episode_id": Value("int64"),
    "frame_idx": Value("int64"),
    "image_bytes": Value("binary"),
    "primitive_type": Value("string"),
    "theme": Value("string"),
    "tempo": Value("string"),
    "target_bbox_x": Value("int64"),
    "target_bbox_y": Value("int64"),
    "target_bbox_w": Value("int64"),
    "target_bbox_h": Value("int64"),
    "done_gt": Value("int8"),
    "action_dx": Value("float32"),
    "action_dy": Value("float32"),
    "action_click": Value("int8"),
    "action_scroll": Value("float32"),
    "action_key_events": Sequence(Value("int8"), length=77),
    "cursor_x": Value("float32"),
    "cursor_y": Value("float32"),
    "held_keys": Sequence(Value("int8"), length=77),
    "held_mouse": Sequence(Value("int8"), length=3),
    "capslock": Value("int8"),
})


def generate_all(n_episodes: int, out_dir: Path, shard_size: int = 500) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_rows: list[dict] = []
    shard_idx = 0
    t0 = time.time()
    for i in range(n_episodes):
        rows = generate_one_episode(episode_id=i, seed=i)
        shard_rows.extend(rows)
        if (i + 1) % shard_size == 0 or (i + 1) == n_episodes:
            ds = Dataset.from_list(shard_rows, features=FEATURES)
            shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
            ds.to_parquet(shard_path)
            print(f"[shard {shard_idx}] episodes {i + 1 - len(shard_rows) // 30 + 1}..{i + 1}  frames={len(shard_rows)}  → {shard_path}")
            shard_rows = []
            shard_idx += 1
    elapsed = time.time() - t0
    print(f"Generated {n_episodes} episodes in {elapsed:.1f}s  ({n_episodes / elapsed:.2f} eps/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=PHASE_A_DATA.n_episodes)
    parser.add_argument("-o", "--out-dir", type=str, default="data/phase-a-lclick")
    parser.add_argument("--shard-size", type=int, default=500, help="Episodes per parquet shard")
    args = parser.parse_args()
    generate_all(args.n_episodes, Path(args.out_dir), args.shard_size)


if __name__ == "__main__":
    main()
