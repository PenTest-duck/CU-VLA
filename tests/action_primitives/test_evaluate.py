import pandas as pd
from experiments.action_primitives.evaluate import filter_eval_split


def test_filter_phase_a_holdout():
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 1, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 2, "n_buttons": 1, "is_scenario_error": 1, "is_adversarial": 0, "composite_tier": 1},
    ])
    out = filter_eval_split(df, slice_name="phase_a_holdout")
    assert set(out["episode_id"]) == {0}


def test_filter_multi_btn_generic():
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 1, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 2, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 2},
        {"episode_id": 3, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 1, "composite_tier": 1},
    ])
    out = filter_eval_split(df, slice_name="multi_btn_generic")
    assert set(out["episode_id"]) == {1}


def test_filter_multi_btn_composite():
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 2},
        {"episode_id": 2, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 3},
    ])
    out = filter_eval_split(df, slice_name="multi_btn_composite")
    assert set(out["episode_id"]) == {1, 2}


def test_filter_scenario_recovery():
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 1, "is_adversarial": 0, "composite_tier": 1},
    ])
    out = filter_eval_split(df, slice_name="scenario_recovery")
    assert set(out["episode_id"]) == {1}


def test_filter_adversarial():
    df = pd.DataFrame([
        {"episode_id": 0, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1},
        {"episode_id": 1, "n_buttons": 3, "is_scenario_error": 0, "is_adversarial": 1, "composite_tier": 2},
    ])
    out = filter_eval_split(df, slice_name="adversarial")
    assert set(out["episode_id"]) == {1}


def test_filter_unknown_slice_raises():
    import pytest
    df = pd.DataFrame([{"episode_id": 0, "n_buttons": 1, "is_scenario_error": 0, "is_adversarial": 0, "composite_tier": 1}])
    with pytest.raises(ValueError, match="slice"):
        filter_eval_split(df, slice_name="nonexistent")


def test_adversarial_tier_color_ambiguous():
    """A scene where 2 buttons share color but differ on shape → color ambiguous; shape disambiguates."""
    from experiments.action_primitives.evaluate import classify_adversarial_tier
    from experiments.action_primitives.scene import Scene, Button
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "square", "small", (1, 0), 100, 10, 40, 40),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    target_id = 0
    used_attrs = ("shape",)  # shape disambiguates since color is shared
    tier = classify_adversarial_tier(scene, target_id, used_attrs)
    assert tier == "color-ambiguous"


def test_adversarial_tier_two_attr_needed():
    """Two buttons share color and shape; size disambiguates → not single-ambiguous, but used_attrs has size+something else."""
    from experiments.action_primitives.evaluate import classify_adversarial_tier
    from experiments.action_primitives.scene import Scene, Button
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "circle", "large", (1, 0), 100, 10, 100, 100),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    target_id = 0
    # used_attrs has 2 entries (composite needed) → 2-attr-needed
    tier = classify_adversarial_tier(scene, target_id, ("color", "size"))
    assert tier == "2-attr-needed"


def test_adversarial_tier_three_attr_needed():
    """Three-attribute composite required."""
    from experiments.action_primitives.evaluate import classify_adversarial_tier
    from experiments.action_primitives.scene import Scene, Button
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "square", "small", (1, 0), 100, 10, 40, 40),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    tier = classify_adversarial_tier(scene, 0, ("color", "shape", "size"))
    assert tier == "3-attr-needed"


def test_wrong_direction_first_3_frames_heading_away():
    from experiments.action_primitives.evaluate import compute_wrong_direction_first_3_frames
    cursor_xys = [(100.0, 100.0), (90.0, 90.0), (80.0, 80.0)]  # heading away from (200, 200)
    target_xy = (200.0, 200.0)
    out = compute_wrong_direction_first_3_frames(cursor_xys, target_xy)
    assert out is True


def test_wrong_direction_first_3_frames_heading_toward():
    from experiments.action_primitives.evaluate import compute_wrong_direction_first_3_frames
    cursor_xys = [(100.0, 100.0), (130.0, 130.0), (160.0, 160.0)]  # heading toward (200, 200)
    target_xy = (200.0, 200.0)
    out = compute_wrong_direction_first_3_frames(cursor_xys, target_xy)
    assert out is False


def test_wrong_direction_first_3_frames_short_rollout():
    """Rollouts shorter than 3 frames should not flag as wrong-direction (insufficient data)."""
    from experiments.action_primitives.evaluate import compute_wrong_direction_first_3_frames
    cursor_xys = [(100.0, 100.0), (90.0, 90.0)]  # only 2 frames
    target_xy = (200.0, 200.0)
    out = compute_wrong_direction_first_3_frames(cursor_xys, target_xy)
    assert out is False


def test_build_zero_instruction_embedding():
    """Zero-instruction probe creates an all-zero text embedding tensor of the requested dim."""
    import torch
    from experiments.action_primitives.evaluate import build_zero_instruction_embedding
    out = build_zero_instruction_embedding(emb_dim=768)
    assert out.shape == (1, 768)
    assert torch.allclose(out, torch.zeros_like(out))


def test_build_shuffled_instruction():
    """Shuffled probe picks a different instruction from val set."""
    import numpy as np
    from experiments.action_primitives.evaluate import build_shuffled_instruction
    rng = np.random.default_rng(42)
    val_instructions = [
        "click the red button",
        "tap the blue square",
        "press the small green circle",
    ]
    out = build_shuffled_instruction(rng, val_instructions)
    assert out in val_instructions


def test_build_shuffled_instruction_with_exclude():
    """Shuffled probe should avoid the original instruction when exclude is provided."""
    import numpy as np
    from experiments.action_primitives.evaluate import build_shuffled_instruction
    rng = np.random.default_rng(0)
    val_instructions = ["click the red button", "tap the blue square"]
    # Run several times to make sure the excluded one never appears
    for _ in range(20):
        out = build_shuffled_instruction(rng, val_instructions, exclude="click the red button")
        assert out == "tap the blue square"


def test_build_wrong_instruction_targets_different_button():
    """Wrong-instruction probe constructs an instruction for a DIFFERENT button."""
    import numpy as np
    from experiments.action_primitives.evaluate import build_wrong_instruction
    from experiments.action_primitives.scene import Scene, Button
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "blue", "square", "large", (1, 0), 100, 10, 100, 100),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    rng = np.random.default_rng(0)
    out = build_wrong_instruction(scene, target_id=0, rng=rng)
    # The instruction should describe button 1 (the wrong target), not button 0
    assert "blue" in out or "square" in out or "large" in out
    assert "red" not in out  # the correct target's color shouldn't be mentioned


def test_build_wrong_instruction_single_button_returns_empty():
    """For 1-button scenes there's no other button to target — returns empty string."""
    import numpy as np
    from experiments.action_primitives.evaluate import build_wrong_instruction
    from experiments.action_primitives.scene import Scene, Button
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    rng = np.random.default_rng(0)
    out = build_wrong_instruction(scene, target_id=0, rng=rng)
    assert out == ""


def test_eval_b0_mode_constructs_scene_from_parquet(tmp_path):
    """Smoke test: B0 eval mode should reconstruct Scene from parquet metadata.

    Generates a tiny B0 dataset, loads it via load_episode_metadata, verifies
    the per-episode metadata has the fields B0 eval needs, and verifies
    scene reconstruction via the generator's deterministic seed.
    """
    from pathlib import Path
    from experiments.action_primitives.generate_data import generate_dataset_multiproc
    from experiments.action_primitives.evaluate import (
        load_episode_metadata, reconstruct_scene_for_episode,
    )

    out_dir = tmp_path / "tiny"
    out_dir.mkdir()
    base_seed = 0
    generate_dataset_multiproc(
        n_episodes=2, output_dir=out_dir, n_workers=1, seed=base_seed,
        episodes_per_shard=10,
    )

    # Verify metadata fields exist for B0 reconstruction (read raw shards).
    import pandas as pd
    df = pd.concat([pd.read_parquet(p) for p in Path(out_dir).glob("*.parquet")])
    assert "instruction" in df.columns
    assert "target_button_id" in df.columns
    assert "n_buttons" in df.columns
    assert "composite_tier" in df.columns
    assert "is_adversarial" in df.columns
    assert "is_scenario_error" in df.columns

    # load_episode_metadata("all") returns one row per episode regardless of split bucket.
    meta = load_episode_metadata(out_dir, split="all")
    assert len(meta) == 2
    assert set(meta.columns) >= {
        "episode_id", "instruction", "target_button_id", "n_buttons",
        "composite_tier", "is_adversarial", "is_scenario_error",
    }

    # Scene reconstruction must match what the generator produced.
    for _, row in meta.iterrows():
        eid = int(row["episode_id"])
        scene = reconstruct_scene_for_episode(eid, base_seed=base_seed)
        # The reconstructed scene's button count must match the parquet's n_buttons,
        # and target_button_id must be a valid index into scene.buttons.
        assert len(scene.buttons) == int(row["n_buttons"]), (
            f"Scene reconstruction skew for eid={eid}: "
            f"reconstructed {len(scene.buttons)} buttons, parquet has {row['n_buttons']}"
        )
        assert 0 <= int(row["target_button_id"]) < len(scene.buttons)


def test_decode_b0_click_idle_when_both_idle():
    """Both heads predict idle => return legacy idle (0)."""
    import torch
    from experiments.action_primitives.evaluate import _decode_b0_click
    logits = {
        "click_left": torch.tensor([[5.0, 0.0, 0.0]]),
        "click_right": torch.tensor([[5.0, 0.0, 0.0]]),
    }
    assert _decode_b0_click(logits) == 0


def test_decode_b0_click_left_press():
    """Only left predicts press => return L_press (1)."""
    import torch
    from experiments.action_primitives.evaluate import _decode_b0_click
    logits = {
        "click_left": torch.tensor([[0.0, 5.0, 0.0]]),
        "click_right": torch.tensor([[5.0, 0.0, 0.0]]),
    }
    assert _decode_b0_click(logits) == 1


def test_decode_b0_click_right_release():
    """Only right predicts release => return R_release (4)."""
    import torch
    from experiments.action_primitives.evaluate import _decode_b0_click
    logits = {
        "click_left": torch.tensor([[5.0, 0.0, 0.0]]),
        "click_right": torch.tensor([[0.0, 0.0, 5.0]]),
    }
    assert _decode_b0_click(logits) == 4


def test_is_b0_model_detects_b0_heads():
    """Regression test: _is_b0_model must check 'head_click_left' / 'head_click_right'
    (the actual ModuleDict keys per heads.py prefix convention) — not the bare names.

    First B0 visual eval crashed with KeyError: 'click' because the detector
    returned False on a real B0 model and fell through to the Phase A path.
    """
    from experiments.action_primitives.evaluate import _is_b0_model
    from experiments.action_primitives.heads import ActionHeads
    # Build the real ActionHeads — it uses the head_<name> prefix
    heads_module = ActionHeads()
    # Mock model with .heads attribute pointing to ActionHeads
    class MockModel:
        def __init__(self, heads):
            self.heads = heads
    model = MockModel(heads_module)
    assert _is_b0_model(model) is True, (
        f"Expected B0 detection, got False. heads_module.heads keys: {list(heads_module.heads.keys())}"
    )
