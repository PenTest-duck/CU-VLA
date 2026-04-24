"""Unit tests for episode generator."""

from pathlib import Path

import pytest

from experiments.action_primitives.config import ENV, NUM_KEYS
from experiments.action_primitives.generate_data import generate_all
from experiments.action_primitives.generator import generate_one_episode


def test_generate_one_episode_returns_fixed_window():
    rows = generate_one_episode(episode_id=0, seed=0)
    assert len(rows) == ENV.max_frames_lclick


def test_generate_one_episode_schema_consistency():
    rows = generate_one_episode(episode_id=0, seed=0)
    required = {"episode_id", "frame_idx", "image_bytes", "action_dx", "action_dy",
                "action_click", "action_scroll", "action_key_events", "cursor_x",
                "cursor_y", "held_keys", "held_mouse", "capslock", "done_gt",
                "env_done_frame", "target_bbox_x", "primitive_type", "theme", "tempo"}
    for row in rows:
        assert required.issubset(row.keys())
        assert len(row["action_key_events"]) == NUM_KEYS
        assert len(row["held_keys"]) == NUM_KEYS
        assert len(row["held_mouse"]) == 3
    # env_done_frame is episode-level metadata — must be identical on every row
    env_done_values = {row["env_done_frame"] for row in rows}
    assert len(env_done_values) == 1


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


def test_generate_one_episode_env_done_aligns_with_expert():
    """env_done_frame should be populated and within 2 frames of expert done_frame."""
    rows = generate_one_episode(episode_id=0, seed=0)
    env_done_frame = rows[0]["env_done_frame"]
    # Expert should successfully click for a vanilla seed
    assert env_done_frame != -1, f"env never signalled success for seed=0 (env_done_frame={env_done_frame})"
    # Find expert done_frame (first frame where done_gt == 1)
    done_frames = [r["frame_idx"] for r in rows if r["done_gt"] == 1]
    assert done_frames, "done_gt never flipped to 1"
    expert_done_frame = done_frames[0]
    assert abs(env_done_frame - expert_done_frame) <= 2, (
        f"env_done_frame={env_done_frame} drifts from expert_done_frame={expert_done_frame}"
    )


def test_generator_warns_on_drift():
    """When the expert never successfully clicks (target_center outside target_bbox), env
    never signals success and the generator should emit a warning.

    We synthesize this cheaply by calling generate_one_episode with an explicit
    max_frames too small to allow the expert to reach target + settle + press/release.
    That forces env_done_frame to remain None → sentinel -1 → warning fires.
    """
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # max_frames=2 is not enough to move+settle+press+release for any tempo
        rows = generate_one_episode(episode_id=0, seed=0, max_frames=2)
    assert rows[0]["env_done_frame"] == -1
    assert any("label noise" in str(w.message).lower() for w in captured), (
        f"expected drift warning, got: {[str(w.message) for w in captured]}"
    )


def test_generate_all_refuses_to_overwrite_existing_shards(tmp_path: Path):
    """Re-running generate_all with existing shards in out_dir must raise RuntimeError."""
    # First call writes shards successfully
    generate_all(n_episodes=2, out_dir=tmp_path, shard_size=1)
    # Second call with same out_dir must refuse
    with pytest.raises(RuntimeError, match="already contains"):
        generate_all(n_episodes=2, out_dir=tmp_path, shard_size=1)
