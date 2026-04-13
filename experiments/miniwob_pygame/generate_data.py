"""Multi-task expert demonstration generation.

Generates expert episodes and saves as HF datasets (Arrow/Parquet),
one dataset directory per task: data/{task_name}/

Usage:
    uv run python experiments/miniwob_pygame/generate_data.py \
        --tasks click-target drag-to-zone -n 100
    uv run python experiments/miniwob_pygame/generate_data.py \
        --tasks click-target -n 5000 --push-to-hub PenTest-duck/cu-vla-exp3-data
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

# Allow running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.miniwob_pygame.config import NUM_KEYS, TASK_NAMES, TRAIN
from experiments.miniwob_pygame.task_registry import get_env_class, get_expert_fn


FEATURES = Features({
    "episode_id": Value("int32"),
    "timestep": Value("int32"),
    "image": Image(),  # PIL Image, stored as compressed bytes
    "cursor_x": Value("float32"),  # normalized [0,1]
    "cursor_y": Value("float32"),  # normalized [0,1]
    "action_dx": Value("float32"),
    "action_dy": Value("float32"),
    "action_mouse_left": Value("int8"),
    "action_keys_held": Sequence(Value("int8"), length=NUM_KEYS),
    "episode_length": Value("int32"),
    "task_name": Value("string"),
    "success": Value("bool"),
})


def _task_episode_generator(
    task_name: str,
    num_episodes: int,
    seed: int,
):
    """Yield one dict per timestep across all episodes for a single task."""
    env_cls = get_env_class(task_name)
    expert_fn = get_expert_fn(task_name)

    env = env_cls(visual=False)
    rng = np.random.default_rng(seed)

    total_frames = 0
    successes = 0
    t0 = time.perf_counter()

    for ep in range(num_episodes):
        observations, actions, info = expert_fn(env, rng, seed=seed + ep)

        if not actions:
            continue

        n_steps = len(actions)
        total_frames += n_steps

        success = bool(info.get("success", False))
        if success:
            successes += 1

        for t in range(n_steps):
            obs = observations[t]
            yield {
                "episode_id": ep,
                "timestep": t,
                "image": PILImage.fromarray(obs["screenshot"]),
                "cursor_x": float(obs["cursor_pos"][0]),
                "cursor_y": float(obs["cursor_pos"][1]),
                "action_dx": float(actions[t]["dx"]),
                "action_dy": float(actions[t]["dy"]),
                "action_mouse_left": int(actions[t]["mouse_left"]),
                "action_keys_held": list(int(x) for x in actions[t]["keys_held"]),
                "episode_length": n_steps,
                "task_name": task_name,
                "success": success,
            }

        if (ep + 1) % 200 == 0 or ep == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{task_name}] [{ep+1:5d}/{num_episodes}] "
                f"frames={total_frames}, successes={successes}/{ep+1}, "
                f"elapsed={elapsed:.1f}s"
            )

    env.close()

    elapsed = time.perf_counter() - t0
    rate = successes / num_episodes * 100 if num_episodes > 0 else 0.0
    avg_frames = total_frames / num_episodes if num_episodes > 0 else 0.0
    print(
        f"  [{task_name}] Done: {num_episodes} episodes, {total_frames} frames, "
        f"success={rate:.1f}%, avg frames/ep={avg_frames:.1f}, "
        f"time={elapsed:.1f}s"
    )


def generate(
    tasks: list[str],
    num_episodes: int,
    output_dir: str,
    seed: int,
    num_shards: int = 10,
    push_to_hub: str | None = None,
) -> None:
    """Generate expert demonstrations for all requested tasks."""
    print(f"Generating {num_episodes} episodes each for {len(tasks)} task(s):")
    for t in tasks:
        print(f"  - {t}")
    print()

    overall_t0 = time.perf_counter()

    for task_name in tasks:
        print(f"=== {task_name} ===")

        ds = Dataset.from_generator(
            _task_episode_generator,
            features=FEATURES,
            gen_kwargs={
                "task_name": task_name,
                "num_episodes": num_episodes,
                "seed": seed,
            },
        )

        print(f"  Dataset: {ds}")

        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        print(f"  Saving to {task_dir} ({num_shards} shards)...")
        ds.save_to_disk(task_dir, num_shards=num_shards)
        print(f"  Saved.\n")

        if push_to_hub:
            print(f"  Pushing to {push_to_hub} (config={task_name})...")
            ds.push_to_hub(push_to_hub, config_name=task_name, num_shards=num_shards)
            print(f"  Pushed.\n")

    overall_elapsed = time.perf_counter() - overall_t0
    print("=" * 60)
    print(f"All tasks complete in {overall_elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-task expert demonstrations (HF datasets)"
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
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of Arrow shards per task (default: 10)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HF dataset repo to push to (e.g. PenTest-duck/cu-vla-exp3-data)",
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
        num_shards=args.num_shards,
        push_to_hub=args.push_to_hub,
    )
