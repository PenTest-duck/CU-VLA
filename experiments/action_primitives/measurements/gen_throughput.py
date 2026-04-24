"""Spike E — Pygame generation throughput measurement.

Validates Q7's "≥200 eps/sec" claim. Generates 1K L-click episodes
and reports eps/s, frames/s, per-episode storage.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.action_primitives.generator import generate_one_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-episodes", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()
    total_frames = 0
    total_bytes = 0
    for i in range(args.n_episodes):
        rows = generate_one_episode(episode_id=i, seed=i)
        total_frames += len(rows)
        total_bytes += sum(len(r["image_bytes"]) for r in rows)
    elapsed = time.time() - t0
    eps_per_s = args.n_episodes / elapsed
    frames_per_s = total_frames / elapsed
    avg_bytes_per_ep = total_bytes / args.n_episodes
    print(f"Episodes:          {args.n_episodes}")
    print(f"Total frames:      {total_frames}")
    print(f"Wall-clock:        {elapsed:.2f}s")
    print(f"eps/s:             {eps_per_s:.2f}  (target: ≥200)")
    print(f"frames/s:          {frames_per_s:.1f}")
    print(f"Avg bytes/episode: {avg_bytes_per_ep / 1024:.1f} KB")
    print(f"Projected 24500-ep dataset size: {24500 * avg_bytes_per_ep / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
