"""Data generation pipeline for Experiment 5: Mini Text Editor.

Generates expert demonstration episodes and saves as HuggingFace Dataset
in Parquet format.

Usage:
    uv run python experiments/mini_editor/generate_data.py --num-episodes 100
    uv run python experiments/mini_editor/generate_data.py --num-episodes 10000 --push-to-hub user/repo
"""

from __future__ import annotations

import argparse
import io
import time
from collections import Counter

import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from .config import DATA, NUM_KEYS
from .corpus import extract_words, load_corpus, make_passage
from .env import MiniEditorEnv
from .expert import generate_episode_trajectory, run_episode
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


def _episode_generator(
    num_episodes: int,
    seed: int,
):
    """Yield one dict per timestep across all episodes."""
    corpus = load_corpus()
    print(f"Loaded corpus: {len(corpus)} sentences")

    env = MiniEditorEnv(visual=False)

    rng = np.random.default_rng(seed)
    op_counts: Counter = Counter()
    success_count = 0
    total_steps = 0
    t0 = time.time()

    for ep_id in range(num_episodes):
        # Sample a passage
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

        # Generate instruction
        inst = generate_instruction(passage, words, rng)
        op_counts[inst.operation] += 1

        # Generate expert trajectory
        env.reset(passage, seed=int(rng.integers(0, 2**31)))
        env.set_expected_text(inst.expected_text)
        traj = generate_episode_trajectory(env, inst, rng)

        # Replay through env to record observations
        env.reset(passage, seed=int(rng.integers(0, 2**31)))
        env.set_expected_text(inst.expected_text)
        observations, actions, info = run_episode(env, traj)

        success = env.text == inst.expected_text
        if success:
            success_count += 1
        ep_len = len(actions)
        total_steps += ep_len

        # Yield one row per timestep
        for t, (obs, act) in enumerate(zip(observations, actions)):
            yield {
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
            }

        if (ep_id + 1) % 100 == 0 or ep_id == num_episodes - 1:
            elapsed = time.time() - t0
            eps_per_sec = (ep_id + 1) / elapsed
            print(
                f"  [{ep_id + 1:>6d}/{num_episodes}] "
                f"{eps_per_sec:.1f} ep/s | "
                f"success={success_count}/{ep_id + 1} | "
                f"avg_steps={total_steps / (ep_id + 1):.0f}"
            )

    elapsed = time.time() - t0
    print(f"\nDone: {num_episodes} episodes in {elapsed:.1f}s")
    print(f"Success rate: {success_count}/{num_episodes} ({100 * success_count / max(num_episodes, 1):.1f}%)")
    print(f"Avg episode length: {total_steps / max(num_episodes, 1):.0f} steps")
    print(f"Operations: {dict(op_counts)}")
    print(f"Total frames: {total_steps}")


def generate(
    num_episodes: int = DATA.num_episodes,
    output_dir: str | None = None,
    seed: int = 0,
    num_shards: int = DATA.num_shards,
    push_to_hub: str | None = None,
) -> None:
    """Main entry point: generate dataset and save."""
    if output_dir is None:
        output_dir = DATA.output_dir
    print(f"Generating {num_episodes} episodes (seed={seed})")
    print(f"Output: {output_dir}")

    ds = Dataset.from_generator(
        _episode_generator,
        features=FEATURES,
        gen_kwargs={"num_episodes": num_episodes, "seed": seed},
    )

    print(f"\nSaving to {output_dir} ({num_shards} shards)...")
    ds.save_to_disk(output_dir, num_shards=num_shards)
    print(f"Saved: {len(ds)} rows")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {push_to_hub}")
        ds.push_to_hub(push_to_hub, num_shards=num_shards)
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mini editor expert demos")
    parser.add_argument("--num-episodes", "-n", type=int, default=DATA.num_episodes)
    parser.add_argument("--output-dir", "-o", type=str, default=DATA.output_dir)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=DATA.num_shards)
    parser.add_argument("--push-to-hub", type=str, default=None)
    args = parser.parse_args()

    generate(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        num_shards=args.num_shards,
        push_to_hub=args.push_to_hub,
    )
