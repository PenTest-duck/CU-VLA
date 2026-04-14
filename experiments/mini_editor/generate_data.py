"""Data generation pipeline for Experiment 5: Mini Text Editor.

Generates expert demonstration episodes and saves as HuggingFace Dataset
in Parquet format. Uses multiprocessing for parallel episode generation.

Usage:
    uv run python -m experiments.mini_editor.generate_data -n 10000
    uv run python -m experiments.mini_editor.generate_data -n 10000 -w 8 --push-to-hub user/repo
"""

from __future__ import annotations

import argparse
import io
import os
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets
from PIL import Image as PILImage

from .config import DATA, NUM_KEYS
from .corpus import extract_words, load_corpus, make_passage
from .instructions import generate_instruction


FEATURES = Features({
    "episode_id": Value("int32"),
    "timestep": Value("int32"),
    "image": Image(),
    "instruction": Value("string"),
    "action_dx": Value("float32"),
    "action_dy": Value("float32"),
    "action_mouse_left": Value("int8"),
    "action_keys_held": Sequence(Value("int8"), length=NUM_KEYS),
    "proprio": Sequence(Value("float32"), length=56),
    "operation_type": Value("string"),
    "target_word": Value("string"),
    "expected_text": Value("string"),
    "initial_text": Value("string"),
    "episode_length": Value("int32"),
    "success": Value("bool"),
})


def _screenshot_to_jpeg_bytes(screenshot: np.ndarray) -> bytes:
    """Convert (H, W, 3) uint8 array to JPEG bytes."""
    img = PILImage.fromarray(screenshot)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=DATA.jpeg_quality)
    return buf.getvalue()


def _generate_shard(args: tuple) -> str:
    """Generate a shard of episodes in a worker process.

    Each worker has its own Pygame instance and writes its shard to disk.
    Returns the path to the saved shard directory.
    """
    shard_id, ep_start, ep_end, seed, output_dir = args

    # Import here so each process gets its own Pygame init
    from .env import MiniEditorEnv
    from .expert import generate_episode_trajectory, run_episode

    corpus = load_corpus()
    env = MiniEditorEnv(visual=False)
    rng = np.random.default_rng(seed + shard_id)

    rows: list[dict] = []
    op_counts: Counter = Counter()
    success_count = 0
    total_steps = 0
    t0 = time.time()

    for ep_id in range(ep_start, ep_end):
        passage = None
        for _ in range(100):
            passage = make_passage(corpus, rng)
            if passage is not None:
                break
        if passage is None:
            continue

        words = extract_words(passage)
        if len(words) < 4:
            continue

        inst = generate_instruction(passage, words, rng)
        op_counts[inst.operation] += 1

        # Generate expert trajectory
        env.reset(passage, seed=int(rng.integers(0, 2**31)))
        env.set_expected_text(inst.expected_text)
        traj = generate_episode_trajectory(env, inst, rng)

        # Replay to record observations
        env.reset(passage, seed=int(rng.integers(0, 2**31)))
        env.set_expected_text(inst.expected_text)
        observations, actions, info = run_episode(env, traj)

        success = env.text == inst.expected_text
        if success:
            success_count += 1
        ep_len = len(actions)
        total_steps += ep_len

        for t, (obs, act) in enumerate(zip(observations, actions)):
            rows.append({
                "episode_id": ep_id,
                "timestep": t,
                "image": _screenshot_to_jpeg_bytes(obs["screenshot"]),
                "instruction": inst.instruction_text,
                "action_dx": act["dx"],
                "action_dy": act["dy"],
                "action_mouse_left": act["mouse_left"],
                "action_keys_held": act["keys_held"],
                "proprio": obs["proprio"].tolist(),
                "operation_type": inst.operation,
                "target_word": inst.target_word,
                "expected_text": inst.expected_text,
                "initial_text": passage,
                "episode_length": ep_len,
                "success": success,
            })

        num_done = ep_id - ep_start + 1
        if num_done % 100 == 0:
            elapsed = time.time() - t0
            eps_per_sec = num_done / elapsed
            print(
                f"  [shard {shard_id}] {num_done}/{ep_end - ep_start} "
                f"({eps_per_sec:.1f} ep/s) | "
                f"success={success_count}/{num_done}"
            )

    elapsed = time.time() - t0
    num_eps = ep_end - ep_start
    print(
        f"  [shard {shard_id}] done: {num_eps} episodes, "
        f"{total_steps} frames in {elapsed:.1f}s "
        f"({num_eps / elapsed:.1f} ep/s) | "
        f"success={success_count}/{num_eps} | "
        f"ops={dict(op_counts)}"
    )

    # Save shard to disk
    shard_dir = os.path.join(output_dir, f"_shard_{shard_id}")
    ds = Dataset.from_dict(
        {k: [r[k] for r in rows] for k in FEATURES},
        features=FEATURES,
    )
    ds.save_to_disk(shard_dir)
    return shard_dir


def generate(
    num_episodes: int = DATA.num_episodes,
    output_dir: str | None = None,
    seed: int = 0,
    num_shards: int = DATA.num_shards,
    num_workers: int | None = None,
    push_to_hub: str | None = None,
) -> None:
    """Generate dataset using multiprocessing.

    Each worker generates a shard of episodes in parallel, then shards
    are concatenated into the final dataset.
    """
    if output_dir is None:
        output_dir = DATA.output_dir
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    # Clamp workers to not exceed episodes
    num_workers = min(num_workers, num_episodes)

    print(f"Generating {num_episodes} episodes (seed={seed}, workers={num_workers})")
    print(f"Output: {output_dir}")

    # Split episodes across workers
    eps_per_worker = num_episodes // num_workers
    remainder = num_episodes % num_workers
    shard_args = []
    ep_cursor = 0
    for i in range(num_workers):
        n = eps_per_worker + (1 if i < remainder else 0)
        shard_args.append((i, ep_cursor, ep_cursor + n, seed, output_dir))
        ep_cursor += n

    t0 = time.time()

    if num_workers == 1:
        # Single process — no pool overhead
        shard_dirs = [_generate_shard(shard_args[0])]
    else:
        with Pool(num_workers) as pool:
            shard_dirs = pool.map(_generate_shard, shard_args)

    # Concatenate shards
    print(f"\nConcatenating {len(shard_dirs)} shards...")
    shards = [Dataset.load_from_disk(d) for d in shard_dirs]
    ds = concatenate_datasets(shards)

    # Save final dataset
    print(f"Saving to {output_dir} ({num_shards} shards)...")
    ds.save_to_disk(output_dir, num_shards=num_shards)

    # Clean up temp shard dirs
    import shutil
    for d in shard_dirs:
        shutil.rmtree(d, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"Done: {len(ds)} rows in {elapsed:.1f}s")
    print(f"Effective speed: {num_episodes / elapsed:.1f} ep/s")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {push_to_hub}")
        ds.push_to_hub(push_to_hub, num_shards=num_shards)
        print("Pushed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mini editor expert demos")
    parser.add_argument("--num-episodes", "-n", type=int, default=DATA.num_episodes)
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=DATA.num_shards)
    parser.add_argument("--num-workers", "-w", type=int, default=None,
                        help=f"Parallel workers (default: min(cpu_count, 8) = {min(cpu_count(), 8)})")
    parser.add_argument("--push-to-hub", type=str, default=None)
    args = parser.parse_args()

    generate(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        num_shards=args.num_shards,
        num_workers=args.num_workers,
        push_to_hub=args.push_to_hub,
    )
