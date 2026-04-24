"""Spike C — M1 closed-loop eval timing.

Runs N rollouts on M1 MPS with the Spike B checkpoint, reports per-frame
wall-clock breakdown (encoder, trunk, env, total) and aggregate eps/rollout.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.action_primitives.evaluate import load_model, rollout_one_episode
from experiments.action_primitives.env import LClickEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--out", type=str, default="docs/experiments/6-action-primitives-phase-a-results/spike-c-m1-eval-timing.json")
    args = parser.parse_args()

    assert torch.backends.mps.is_available(), "run on M1 with MPS available"
    device = "mps"
    model = load_model(args.checkpoint, device)

    # Warmup
    env = LClickEnv(seed=0)
    for _ in range(5):
        rollout_one_episode(model, env, device, max_frames=5)

    # Measure
    wall_times = []
    frame_counts = []
    t0_total = time.time()
    for i in range(args.n_rollouts):
        env = LClickEnv(seed=5000 + i)
        t0 = time.time()
        res = rollout_one_episode(model, env, device)
        wall_times.append(time.time() - t0)
        frame_counts.append(res["frames"])
    total_wall = time.time() - t0_total

    per_rollout_mean = float(np.mean(wall_times))
    per_rollout_median = float(np.median(wall_times))
    frames_total = int(np.sum(frame_counts))
    per_frame_mean_ms = total_wall / frames_total * 1000

    result = {
        "n_rollouts": args.n_rollouts,
        "total_wall_s": total_wall,
        "per_rollout_mean_s": per_rollout_mean,
        "per_rollout_median_s": per_rollout_median,
        "frames_total": frames_total,
        "per_frame_mean_ms": per_frame_mean_ms,
        "eff_hz": 1000.0 / per_frame_mean_ms,
    }
    print(json.dumps(result, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
