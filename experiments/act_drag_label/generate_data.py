"""Generate expert demonstration episodes and save as HDF5 files."""

import argparse
import os
import sys
import time

import h5py
import numpy as np

# Allow running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.act_drag_label.config import TRAIN
from experiments.act_drag_label.env import DragLabelEnv
from experiments.act_drag_label.expert import run_episode


def generate(
    num_episodes: int = TRAIN.num_episodes,
    num_shapes: int = 1,
    output_dir: str | None = None,
    seed: int = 0,
) -> None:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

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

        # Stack observations and actions
        obs_array = np.stack(observations)  # (T, 224, 224, 3)
        dx_array = np.array([a["dx"] for a in actions], dtype=np.float32)
        dy_array = np.array([a["dy"] for a in actions], dtype=np.float32)
        click_array = np.array([a["click"] for a in actions], dtype=np.int8)
        key_array = np.array([a["key"] for a in actions], dtype=np.int8)

        # Save to HDF5
        path = os.path.join(output_dir, f"episode_{ep:05d}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "observations", data=obs_array, compression="gzip", compression_opts=4
            )
            f.create_dataset("actions_dx", data=dx_array)
            f.create_dataset("actions_dy", data=dy_array)
            f.create_dataset("actions_click", data=click_array)
            f.create_dataset("actions_key", data=key_array)

            # Metadata
            f.attrs["num_shapes"] = num_shapes
            f.attrs["success"] = info.get("success", False)
            f.attrs["num_steps"] = n_steps
            f.attrs["shapes_completed"] = info.get("shapes_completed", 0)

        if (ep + 1) % 200 == 0 or ep == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{ep+1:5d}/{num_episodes}] "
                f"frames={total_frames}, successes={successes}/{ep+1}, "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    env.close()

    print(f"\nDone: {num_episodes} episodes, {total_frames} total frames")
    print(f"Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Mean frames/episode: {total_frames/num_episodes:.1f}")
    print(f"Time: {elapsed:.1f}s ({num_episodes/elapsed:.0f} episodes/s)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert demonstrations")
    parser.add_argument("-n", "--num-episodes", type=int, default=TRAIN.num_episodes)
    parser.add_argument("--num-shapes", type=int, default=1)
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        num_episodes=args.num_episodes,
        num_shapes=args.num_shapes,
        output_dir=args.output_dir,
        seed=args.seed,
    )
