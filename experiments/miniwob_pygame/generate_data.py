"""Multi-task expert demonstration generation.

Loops over requested tasks, runs expert episodes, and saves HDF5 files
with the same sharding layout as Experiment 2:

    data/{task_name}/{shard:03d}/episode_{ep:05d}.hdf5

Usage:
    uv run python experiments/miniwob_pygame/generate_data.py \
        --tasks click-target drag-to-zone -n 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np

# Allow running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.miniwob_pygame.config import NUM_KEYS, TASK_NAMES, TRAIN
from experiments.miniwob_pygame.task_registry import get_env_class, get_expert_fn


def _save_episode(
    path: str,
    observations: list[np.ndarray],
    actions: list[dict],
    task_name: str,
    success: bool,
) -> None:
    """Save a single episode to an HDF5 file."""
    n_steps = len(actions)

    # Stack observations: (T, H, W, 3) uint8
    obs_array = np.stack([o["screenshot"] for o in observations], axis=0).astype(np.uint8)

    # Cursor positions: (T, 2) float32
    cursor_array = np.stack(
        [np.array(o["cursor_pos"], dtype=np.float32) for o in observations], axis=0
    )

    # Actions
    dx = np.array([a["dx"] for a in actions], dtype=np.float32)
    dy = np.array([a["dy"] for a in actions], dtype=np.float32)
    mouse_left = np.array([a["mouse_left"] for a in actions], dtype=np.int8)
    keys_held = np.zeros((n_steps, NUM_KEYS), dtype=np.int8)
    for t, a in enumerate(actions):
        keys_held[t] = a["keys_held"]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("observations", data=obs_array, compression="gzip", compression_opts=4)
        f.create_dataset("cursor_positions", data=cursor_array)
        f.create_dataset("actions_dx", data=dx)
        f.create_dataset("actions_dy", data=dy)
        f.create_dataset("actions_mouse_left", data=mouse_left)
        f.create_dataset("actions_keys_held", data=keys_held, compression="gzip", compression_opts=4)
        f.attrs["task_name"] = task_name
        f.attrs["success"] = success
        f.attrs["num_steps"] = n_steps


def generate_task(
    task_name: str,
    num_episodes: int,
    output_dir: str,
    seed: int,
) -> dict:
    """Generate expert demonstrations for a single task.

    Returns:
        Summary dict with success_count, total_steps, elapsed_s.
    """
    env_cls = get_env_class(task_name)
    expert_fn = get_expert_fn(task_name)

    env = env_cls(visual=False)
    rng = np.random.default_rng(seed)

    task_dir = os.path.join(output_dir, task_name)
    success_count = 0
    total_steps = 0
    t0 = time.perf_counter()

    for ep in range(num_episodes):
        observations, actions, info = expert_fn(env, rng, seed=seed + ep)

        if not actions:
            continue

        success = bool(info.get("success", False))
        if success:
            success_count += 1
        total_steps += len(actions)

        # Shard: 1000 episodes per directory
        shard = ep // 1000
        shard_dir = os.path.join(task_dir, f"{shard:03d}")
        path = os.path.join(shard_dir, f"episode_{ep:05d}.hdf5")
        _save_episode(path, observations, actions, task_name, success)

        if (ep + 1) % 200 == 0 or ep == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{task_name}] [{ep+1:5d}/{num_episodes}] "
                f"steps={total_steps}, successes={success_count}/{ep+1}, "
                f"elapsed={elapsed:.1f}s"
            )

    env.close()

    elapsed = time.perf_counter() - t0
    return {
        "success_count": success_count,
        "total_steps": total_steps,
        "elapsed_s": elapsed,
    }


def generate(
    tasks: list[str],
    num_episodes: int,
    output_dir: str,
    seed: int,
) -> None:
    """Generate expert demonstrations for all requested tasks."""
    print(f"Generating {num_episodes} episodes each for {len(tasks)} task(s):")
    for t in tasks:
        print(f"  - {t}")
    print()

    overall_t0 = time.perf_counter()
    summaries: dict[str, dict] = {}

    for task_name in tasks:
        print(f"=== {task_name} ===")
        summary = generate_task(task_name, num_episodes, output_dir, seed)
        summaries[task_name] = summary

        rate = summary["success_count"] / num_episodes * 100 if num_episodes > 0 else 0.0
        avg_steps = summary["total_steps"] / num_episodes if num_episodes > 0 else 0.0
        print(
            f"  Done: {summary['success_count']}/{num_episodes} success ({rate:.1f}%), "
            f"avg steps={avg_steps:.1f}, time={summary['elapsed_s']:.1f}s\n"
        )

    overall_elapsed = time.perf_counter() - overall_t0
    print("=" * 60)
    print(f"All tasks complete in {overall_elapsed:.1f}s")
    print(f"{'Task':<20} {'Success':>10} {'Avg Steps':>10} {'Time (s)':>10}")
    print("-" * 60)
    for task_name, s in summaries.items():
        rate = s["success_count"] / num_episodes * 100 if num_episodes > 0 else 0.0
        avg = s["total_steps"] / num_episodes if num_episodes > 0 else 0.0
        print(f"{task_name:<20} {rate:>9.1f}% {avg:>10.1f} {s['elapsed_s']:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-task expert demonstrations (HDF5)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASK_NAMES,
        choices=TASK_NAMES,
        help="Task(s) to generate data for (default: all)",
    )
    parser.add_argument(
        "-n", "--num-episodes",
        type=int,
        default=TRAIN.num_episodes_per_task,
        help=f"Episodes per task (default: {TRAIN.num_episodes_per_task})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/miniwob_pygame/data)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")

    generate(
        tasks=args.tasks,
        num_episodes=args.num_episodes,
        output_dir=output_dir,
        seed=args.seed,
    )
