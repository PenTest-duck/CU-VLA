"""One-shot migration: convert existing HDF5 episodes to HF datasets parquet format.

Usage:
    uv run python scripts/migrate_hdf5_to_parquet.py
    uv run python scripts/migrate_hdf5_to_parquet.py --push-to-hub PenTest-duck/cu-vla-data
"""

import argparse
import glob
import os

import h5py
import numpy as np
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage


DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "act_drag_label", "data"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "act_drag_label", "data_parquet"
)


def _row_generator(input_dir: str):
    """Yield one dict per timestep across all HDF5 episodes."""
    episode_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "episode_*.hdf5"), recursive=True)
    )
    print(f"Found {len(episode_files)} HDF5 episodes in {input_dir}")

    total_rows = 0
    for i, path in enumerate(episode_files):
        ep_id = int(
            os.path.basename(path).replace("episode_", "").replace(".hdf5", "")
        )

        with h5py.File(path, "r") as f:
            obs = f["observations"][:]
            dx = f["actions_dx"][:]
            dy = f["actions_dy"][:]
            click = f["actions_click"][:]
            key = f["actions_key"][:]
            n_steps = len(dx)
            num_shapes = int(f.attrs["num_shapes"])
            success = bool(f.attrs["success"])
            shapes_completed = int(f.attrs["shapes_completed"])

        for t in range(n_steps):
            yield {
                "episode_id": ep_id,
                "timestep": t,
                "image": PILImage.fromarray(obs[t]),
                "action_dx": float(dx[t]),
                "action_dy": float(dy[t]),
                "action_click": int(click[t]),
                "action_key": int(key[t]),
                "episode_length": n_steps,
                "num_shapes": num_shapes,
                "success": success,
                "shapes_completed": shapes_completed,
            }
        total_rows += n_steps

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1}/{len(episode_files)}] rows so far: {total_rows}")

    print(f"Total: {total_rows} rows from {len(episode_files)} episodes")


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


def main():
    parser = argparse.ArgumentParser(description="Migrate HDF5 episodes to parquet")
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT, help="Directory with HDF5 episodes"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT, help="Output directory for parquet"
    )
    parser.add_argument(
        "--push-to-hub", default=None, help="HF dataset repo to push to"
    )
    parser.add_argument(
        "--num-shards", type=int, default=10, help="Number of parquet shards"
    )
    args = parser.parse_args()

    print("Building dataset from HDF5 episodes...")
    ds = Dataset.from_generator(
        _row_generator,
        features=FEATURES,
        gen_kwargs={"input_dir": args.input_dir},
    )
    print(f"Dataset: {ds}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {args.output_dir} ({args.num_shards} shards)...")
    ds.save_to_disk(args.output_dir, num_shards=args.num_shards)
    print("Saved.")

    if args.push_to_hub:
        print(f"Pushing to {args.push_to_hub}...")
        ds.push_to_hub(args.push_to_hub, num_shards=args.num_shards)
        print("Pushed.")


if __name__ == "__main__":
    main()
