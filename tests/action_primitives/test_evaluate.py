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
