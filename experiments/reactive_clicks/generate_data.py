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

from experiments.reactive_clicks.config import TRAIN
from experiments.reactive_clicks.env import ReactiveClicksEnv
from experiments.reactive_clicks.expert import run_episode


def generate(
    num_episodes: int = TRAIN.num_episodes,
    output_dir: str | None = None,
    seed: int = 0,
) -> None:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    env = ReactiveClicksEnv(visual=False)
    rng = np.random.default_rng(seed)

    total_frames = 0
    hits = 0
    t0 = time.perf_counter()

    for ep in range(num_episodes):
        observations, actions, info = run_episode(env, seed=ep, rng=rng)

        if not info:
            continue

        n_steps = len(actions)
        total_frames += n_steps

        if info.get("hit"):
            hits += 1

        # Stack observations and actions
        obs_array = np.stack(observations)  # (T, 128, 128, 3)
        dx_array = np.array([a["dx"] for a in actions], dtype=np.float32)
        dy_array = np.array([a["dy"] for a in actions], dtype=np.float32)
        btn_array = np.array([a["btn"] for a in actions], dtype=np.int8)

        # Save to HDF5
        path = os.path.join(output_dir, f"episode_{ep:04d}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "observations", data=obs_array, compression="gzip", compression_opts=4
            )
            f.create_dataset("actions_dx", data=dx_array)
            f.create_dataset("actions_dy", data=dy_array)
            f.create_dataset("actions_btn", data=btn_array)

            # Metadata
            f.attrs["reaction_time_ms"] = info.get("reaction_time_s", 0) * 1000
            f.attrs["circle_x"] = info.get("circle_x", 0)
            f.attrs["circle_y"] = info.get("circle_y", 0)
            f.attrs["circle_radius"] = info.get("circle_radius", 0)
            f.attrs["hit"] = info.get("hit", False)
            f.attrs["num_steps"] = n_steps

        if (ep + 1) % 100 == 0 or ep == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{ep+1:4d}/{num_episodes}] "
                f"frames={total_frames}, hits={hits}/{ep+1}, "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    env.close()

    print(f"\nDone: {num_episodes} episodes, {total_frames} total frames")
    print(f"Hit rate: {hits}/{num_episodes} ({hits/num_episodes*100:.1f}%)")
    print(f"Mean frames/episode: {total_frames/num_episodes:.1f}")
    print(f"Time: {elapsed:.1f}s ({num_episodes/elapsed:.0f} episodes/s)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert demonstrations")
    parser.add_argument("-n", "--num-episodes", type=int, default=TRAIN.num_episodes)
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
    )
