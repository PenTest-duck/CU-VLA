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
