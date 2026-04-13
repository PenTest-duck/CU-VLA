"""Generate expert demonstration episodes and save as HF datasets (parquet)."""

import argparse
import os
import sys
import time

import numpy as np
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.act_drag_label.config import TRAIN
from experiments.act_drag_label.env import DragLabelEnv
from experiments.act_drag_label.expert import run_episode


FEATURES = Features(
    {
        "episode_id": Value("int32"),
        "timestep": Value("int32"),
        "image": Image(),
        "action_dx": Value("float32"),
        "action_dy": Value("float32"),
        "action_click": Value("int8"),
        "action_key": Value("int8"),
        "episode_length": Value("int32"),
        "num_shapes": Value("int32"),
        "success": Value("bool"),
        "shapes_completed": Value("int32"),
    }
)


def _episode_generator(
    num_episodes: int, num_shapes: int, seed: int
):
    """Yield one dict per timestep across all episodes."""
    env = DragLabelEnv(visual=False, num_shapes=num_shapes)
    rng = np.random.default_rng(seed)

    total_frames = 0
    successes = 0
    t0 = time.perf_counter()

    for ep in range(num_episodes):
        observations, actions, info = run_episode(env, seed=ep, rng=rng)

        if not info:
            continue

        n_steps = len(actions)
        total_frames += n_steps

        if info.get("success"):
            successes += 1

        ep_success = info.get("success", False)
        ep_shapes = info.get("shapes_completed", 0)

        for t in range(n_steps):
            yield {
                "episode_id": ep,
                "timestep": t,
                "image": PILImage.fromarray(observations[t]),
                "action_dx": float(actions[t]["dx"]),
                "action_dy": float(actions[t]["dy"]),
                "action_click": int(actions[t]["click"]),
                "action_key": int(actions[t]["key"]),
                "episode_length": n_steps,
                "num_shapes": num_shapes,
                "success": ep_success,
                "shapes_completed": ep_shapes,
            }

        if (ep + 1) % 200 == 0 or ep == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{ep+1:5d}/{num_episodes}] "
                f"frames={total_frames}, successes={successes}/{ep+1}, "
                f"elapsed={elapsed:.1f}s"
            )

    env.close()

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {num_episodes} episodes, {total_frames} total frames")
    print(f"Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Mean frames/episode: {total_frames/num_episodes:.1f}")
    print(f"Time: {elapsed:.1f}s ({num_episodes/elapsed:.0f} episodes/s)")


def generate(
    num_episodes: int = TRAIN.num_episodes,
    num_shapes: int = 1,
    output_dir: str | None = None,
    seed: int = 0,
    num_shards: int = 10,
    push_to_hub: str | None = None,
) -> None:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")

    print(f"Generating {num_episodes} episodes...")
    ds = Dataset.from_generator(
        _episode_generator,
        features=FEATURES,
        gen_kwargs={
            "num_episodes": num_episodes,
            "num_shapes": num_shapes,
            "seed": seed,
        },
    )

    print(f"Dataset: {ds}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir} ({num_shards} shards)...")
    ds.save_to_disk(output_dir, num_shards=num_shards)
    print(f"Saved to: {output_dir}")

    if push_to_hub:
        print(f"Pushing to {push_to_hub}...")
        ds.push_to_hub(push_to_hub, num_shards=num_shards)
        print("Pushed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert demonstrations")
    parser.add_argument("-n", "--num-episodes", type=int, default=TRAIN.num_episodes)
    parser.add_argument("--num-shapes", type=int, default=1)
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF dataset repo to push to (e.g. PenTest-duck/cu-vla-data)")
    args = parser.parse_args()

    generate(
        num_episodes=args.num_episodes,
        num_shapes=args.num_shapes,
        output_dir=args.output_dir,
        seed=args.seed,
        num_shards=args.num_shards,
        push_to_hub=args.push_to_hub,
    )
