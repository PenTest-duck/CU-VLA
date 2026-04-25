"""Unit tests for LClickExpert."""

import numpy as np
import pytest

from experiments.action_primitives.expert import LClickExpert, LClickExpertConfig, TEMPO_PROFILES
from experiments.action_primitives.scene import generate_scene
from experiments.action_primitives.expert import InstructionAwareLClickExpert


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


@pytest.mark.parametrize("tempo", list(TEMPO_PROFILES.keys()))
def test_expert_presses_while_cursor_on_target(tempo):
    """At press frame, the cursor must be within the target rect.

    This is the load-bearing success condition for T8 data generation —
    if experts press off-target, the env rejects the click and training
    receives failed demonstrations.
    """
    cursor_start = (100.0, 100.0)
    target_center = (400.0, 300.0)
    # Button radius: 20px — tighter than the env's minimum button size (40x30)
    # so this test is stricter than any real button hit-box.
    tolerance = 20.0
    expert = LClickExpert(
        cfg=LClickExpertConfig(tempo=tempo, seed=0),
        cursor_xy=cursor_start,
        target_center=target_center,
    )
    cursor = np.array(cursor_start, dtype=np.float64)
    press_cursor = None
    for a in expert:
        cursor = cursor + np.array([a.dx, a.dy])
        if a.click == 1:  # L_press
            press_cursor = cursor.copy()
    assert press_cursor is not None, f"{tempo}: expert never pressed"
    err = np.linalg.norm(press_cursor - np.array(target_center))
    assert err < tolerance, (
        f"{tempo}: cursor was {err:.2f}px from target at press "
        f"(cursor={press_cursor}, target={target_center})"
    )


def test_instruction_aware_expert_targets_specified_button():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=3)
    target_id = 1
    target = scene.buttons[target_id]
    cx, cy = 10.0, 10.0
    cfg = LClickExpertConfig(tempo="normal", seed=0)
    expert = InstructionAwareLClickExpert(
        cfg=cfg, scene=scene, target_button_id=target_id, cursor_xy=(cx, cy),
    )
    # First action's dx, dy should head toward target
    a0 = next(iter(expert))
    target_dx = target.center()[0] - cx
    target_dy = target.center()[1] - cy
    assert np.sign(a0.dx) == np.sign(target_dx)
    assert np.sign(a0.dy) == np.sign(target_dy)


def test_expert_re_query_from_arbitrary_state():
    """Re-querying the expert from a new cursor position should produce a fresh trajectory."""
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=2)
    target_id = 0
    cfg = LClickExpertConfig(tempo="normal", seed=0)
    expert1 = InstructionAwareLClickExpert(cfg, scene, target_id, cursor_xy=(10.0, 10.0))
    expert2 = InstructionAwareLClickExpert(cfg, scene, target_id, cursor_xy=(700.0, 400.0))
    a1 = next(iter(expert1))
    a2 = next(iter(expert2))
    # Two starting positions on opposite sides → different signs in at least one of dx, dy
    assert (np.sign(a1.dx) != np.sign(a2.dx)) or (np.sign(a1.dy) != np.sign(a2.dy))
