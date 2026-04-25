"""Phase A / Phase B0 batched episode generator.

Phase A: ``generate_all`` writes a HuggingFace ``datasets``-compatible parquet
shard set for L-click episodes.

Phase B0: ``generate_dataset_multiproc`` is the new worker-writes-shards
multiproc driver — each worker owns an episode-id range and writes its own
parquet shards directly to disk, eliminating IPC for episode payloads (the
bottleneck observed in Phase A's ``Pool.imap`` design).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Sequence, Value

from experiments.action_primitives.config import PHASE_A_DATA
from experiments.action_primitives.generator import (
    generate_one_b0_episode,
    generate_one_episode,
)


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


# ---------------------------------------------------------------------------
# Phase B0 worker-writes-shards multiproc generator
# ---------------------------------------------------------------------------


def _flush_shard(buffer: list[dict], output_dir: Path, worker_id: int, shard_idx: int) -> None:
    df = pd.DataFrame(buffer)
    out_path = output_dir / f"shard_w{worker_id:02d}_s{shard_idx:04d}.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)


def _worker_run_episodes(
    worker_id: int,
    episode_id_start: int,
    n_episodes_for_worker: int,
    output_dir: Path,
    base_seed: int,
    episodes_per_shard: int,
    progress_every: int = 25,
) -> tuple[int, int]:
    """Run a worker that generates n episodes and writes them as parquet shards.

    Each worker initializes its own pygame instance ONCE on first call to
    ``generate_one_b0_episode`` (which constructs ``LClickEnv``, which
    initializes pygame). Subsequent episodes reuse the same pygame state.

    Prints periodic progress (every ``progress_every`` episodes) with
    ``flush=True`` so output is visible in real-time across multiproc workers.

    Returns ``(worker_id, n_episodes_written)``.
    """
    import time
    t_start = time.time()
    print(f"[w{worker_id}] starting: {n_episodes_for_worker} eps "
          f"(ids {episode_id_start}..{episode_id_start + n_episodes_for_worker - 1})",
          flush=True)
    buffer: list[dict] = []
    shard_idx = 0
    episodes_in_shard = 0
    n_done = 0
    for offset in range(n_episodes_for_worker):
        episode_id = episode_id_start + offset
        seed = base_seed + episode_id
        rows = generate_one_b0_episode(episode_id=episode_id, seed=seed)
        buffer.extend(rows)
        episodes_in_shard += 1
        n_done += 1
        if n_done % progress_every == 0:
            elapsed = time.time() - t_start
            eps_per_s = n_done / elapsed if elapsed > 0 else 0.0
            eta = (n_episodes_for_worker - n_done) / eps_per_s if eps_per_s > 0 else float("inf")
            print(f"[w{worker_id}] {n_done}/{n_episodes_for_worker} eps "
                  f"({eps_per_s:.1f} eps/s, ETA {eta:.0f}s)", flush=True)
        if episodes_in_shard >= episodes_per_shard:
            _flush_shard(buffer, output_dir, worker_id, shard_idx)
            buffer = []
            episodes_in_shard = 0
            shard_idx += 1
    if buffer:
        _flush_shard(buffer, output_dir, worker_id, shard_idx)
    elapsed = time.time() - t_start
    print(f"[w{worker_id}] done: {n_done} eps in {elapsed:.1f}s "
          f"({n_done/elapsed:.1f} eps/s)", flush=True)
    return worker_id, n_done


def generate_dataset_multiproc(
    n_episodes: int,
    output_dir: Path,
    n_workers: int = 4,
    seed: int = 0,
    episodes_per_shard: int = 200,
) -> None:
    """Generate ``n_episodes`` B0 episodes via worker-writes-shards multiproc.

    No IPC for episode payloads — each worker writes its own parquet shards.
    Episode ids are partitioned across workers so there is no overlap.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eps_per_worker = n_episodes // n_workers
    remainder = n_episodes % n_workers

    args_list: list[tuple] = []
    cursor = 0
    for w in range(n_workers):
        n_for_w = eps_per_worker + (1 if w < remainder else 0)
        args_list.append((w, cursor, n_for_w, output_dir, seed, episodes_per_shard))
        cursor += n_for_w

    if n_workers == 1:
        # Skip multiprocessing for single-worker case — useful for tests/debug
        for args in args_list:
            _worker_run_episodes(*args)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.starmap(_worker_run_episodes, args_list)
            print(f"Workers completed: {results}")


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


def _main_b0() -> None:
    """CLI entry for B0 multiproc generation (worker-writes-shards)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=10000)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/phase-b0-lclick"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-shard", type=int, default=200)
    args = parser.parse_args()
    print(f"Generating {args.n_episodes} eps with {args.workers} workers → {args.output_dir}")
    t0 = time.time()
    generate_dataset_multiproc(
        n_episodes=args.n_episodes, output_dir=args.output_dir,
        n_workers=args.workers, seed=args.seed,
        episodes_per_shard=args.episodes_per_shard,
    )
    elapsed = time.time() - t0
    eps_per_sec = args.n_episodes / elapsed
    print(f"Done in {elapsed:.1f}s ({eps_per_sec:.1f} eps/s)")


if __name__ == "__main__":
    _main_b0()
