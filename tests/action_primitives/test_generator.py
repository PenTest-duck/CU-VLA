"""Unit tests for episode generator."""

import pytest

from experiments.action_primitives.config import ENV, NUM_KEYS
from experiments.action_primitives.generator import generate_one_episode


def test_generate_one_episode_returns_fixed_window():
    rows = generate_one_episode(episode_id=0, seed=0)
    assert len(rows) == ENV.max_frames_lclick


def test_generate_one_episode_schema_consistency():
    rows = generate_one_episode(episode_id=0, seed=0)
    required = {"episode_id", "frame_idx", "image_bytes", "action_dx", "action_dy",
                "action_click", "action_scroll", "action_key_events", "cursor_x",
                "cursor_y", "held_keys", "held_mouse", "capslock", "done_gt",
                "target_bbox_x", "primitive_type", "theme", "tempo"}
    for row in rows:
        assert required.issubset(row.keys())
        assert len(row["action_key_events"]) == NUM_KEYS
        assert len(row["held_keys"]) == NUM_KEYS
        assert len(row["held_mouse"]) == 3


def test_generate_one_episode_done_monotonic():
    rows = generate_one_episode(episode_id=0, seed=0)
    seen_done = False
    for row in rows:
        if row["done_gt"] == 1:
            seen_done = True
        else:
            # Once done_gt flips to 1 it should stay 1 for the rest of the episode
            assert not seen_done, f"done_gt went 1→0 at frame {row['frame_idx']}"


def test_generate_one_episode_deterministic():
    r1 = generate_one_episode(episode_id=0, seed=42)
    r2 = generate_one_episode(episode_id=0, seed=42)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a["action_dx"] == b["action_dx"]
        assert a["action_click"] == b["action_click"]
