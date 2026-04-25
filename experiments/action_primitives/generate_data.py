"""Phase A batched episode generator.

Generates N L-click episodes and writes a HuggingFace `datasets`-compatible
parquet shard set.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
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
    "env_done_frame": Value("int64"),
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


def _gen_one_worker(episode_id: int) -> list[dict]:
    """Module-level worker for multiprocessing.Pool (must be picklable by name)."""
    return generate_one_episode(episode_id=episode_id, seed=episode_id)


def generate_all(n_episodes: int, out_dir: Path, shard_size: int = 500, workers: int = 1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Idempotency guard BEFORE dispatching anything
    existing = list(out_dir.glob("shard_*.parquet"))
    if existing:
        raise RuntimeError(
            f"{out_dir} already contains {len(existing)} shard(s). "
            f"Delete them before re-running, or pass a different --out-dir."
        )

    t0 = time.time()
    shard_rows: list[dict] = []
    shard_idx = 0
    shard_first_ep = 0  # explicit tracking — episodes can finish early so
                         # `len(shard_rows) // max_frames_lclick` would be wrong
    pool: mp.pool.Pool | None = None

    if workers == 1:
        row_stream = (generate_one_episode(episode_id=i, seed=i) for i in range(n_episodes))
    else:
        # imap preserves order and streams results — important so shard writes are deterministic
        pool = mp.Pool(processes=workers)
        row_stream = pool.imap(_gen_one_worker, range(n_episodes))

    for i, rows in enumerate(row_stream):
        shard_rows.extend(rows)
        if (i + 1) % shard_size == 0 or (i + 1) == n_episodes:
            ds = Dataset.from_list(shard_rows, features=FEATURES)
            shard_path = out_dir / f"shard_{shard_idx:04d}.parquet"
            ds.to_parquet(shard_path)
            print(f"[shard {shard_idx}] episodes {shard_first_ep}..{i}  frames={len(shard_rows)}  → {shard_path}")
            shard_rows = []
            shard_idx += 1
            shard_first_ep = i + 1

    if pool is not None:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f"Generated {n_episodes} episodes in {elapsed:.1f}s  ({n_episodes / elapsed:.2f} eps/s)  workers={workers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=PHASE_A_DATA.n_episodes)
    parser.add_argument("-o", "--out-dir", type=str, default="data/phase-a-lclick")
    parser.add_argument("--shard-size", type=int, default=500, help="Episodes per parquet shard")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes (1 = serial). "
            "On M1 MacBook: try --workers 4 (P-core count) for ~4x speedup."
        ),
    )
    args = parser.parse_args()
    generate_all(args.n_episodes, Path(args.out_dir), args.shard_size, args.workers)


if __name__ == "__main__":
    main()
