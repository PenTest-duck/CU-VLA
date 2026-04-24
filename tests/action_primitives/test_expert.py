"""Unit tests for LClickExpert."""

import numpy as np
import pytest

from experiments.action_primitives.expert import LClickExpert, LClickExpertConfig, TEMPO_PROFILES


def test_expert_reaches_target_and_clicks():
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo="normal", seed=0),
        cursor_xy=(100.0, 100.0),
        target_center=(400.0, 300.0),
    )
    actions = list(expert)
    # Last two should be press then release
    assert actions[-2].click == 1
    assert actions[-1].click == 2


def test_expert_respects_mouse_cap():
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo="superhuman", seed=0),
        cursor_xy=(10.0, 10.0),
        target_center=(700.0, 440.0),
    )
    for a in expert:
        assert abs(a.dx) <= 100.0 + 1e-5
        assert abs(a.dy) <= 100.0 + 1e-5


@pytest.mark.parametrize("tempo", list(TEMPO_PROFILES.keys()))
def test_expert_all_tempos_terminate(tempo):
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo=tempo, seed=0),
        cursor_xy=(0.0, 0.0),
        target_center=(360.0, 225.0),
    )
    actions = list(expert)
    assert len(actions) > 0
    assert actions[-1].click == 2


def test_expert_no_spurious_keys():
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo="normal", seed=0),
        cursor_xy=(100.0, 100.0),
        target_center=(400.0, 300.0),
    )
    for a in expert:
        # Idle == 2 for all 77 keys
        assert np.all(a.key_events == 2)
