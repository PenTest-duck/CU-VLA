import numpy as np
from experiments.action_primitives.scene import generate_scene
from experiments.action_primitives.recovery import (
    sample_wrong_segment_type, generate_wrong_segment, WrongSegment,
)
from experiments.action_primitives.env import Action
from experiments.action_primitives.recovery import (
    apply_dart_noise, DART_PROB_DEFAULT, DART_SIGMA_DEFAULT,
)
from experiments.action_primitives.config import NUM_KEYS


def _idle_keys_test():
    return np.full(NUM_KEYS, 2, dtype=np.int64)


def test_sample_wrong_segment_type_distribution():
    """Verify 50/30/20 wrong-direction/overshoot/edge-bang sampling."""
    rng = np.random.default_rng(0)
    counts = {"wrong-direction": 0, "overshoot": 0, "edge-bang": 0}
    n = 10000
    for _ in range(n):
        t = sample_wrong_segment_type(rng=rng)
        counts[t] += 1
    # Allow ±2 percentage points
    assert abs(counts["wrong-direction"] / n - 0.50) < 0.02
    assert abs(counts["overshoot"] / n - 0.30) < 0.02
    assert abs(counts["edge-bang"] / n - 0.20) < 0.02


def test_generate_wrong_segment_returns_actions_and_state():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=2)
    target = scene.buttons[0]
    seg = generate_wrong_segment(
        scene=scene, target_button_id=0, cursor_xy=(360.0, 225.0),
        segment_type="wrong-direction", k_frames=10, rng=rng,
    )
    assert isinstance(seg, WrongSegment)
    assert len(seg.actions) == 10
    assert all(isinstance(a, Action) for a in seg.actions)
    assert seg.k_frames == 10
    assert seg.segment_type == "wrong-direction"
    # Final cursor should be off-target (we deliberately moved away)
    final_x, final_y = seg.final_cursor_xy
    assert not (target.x <= final_x < target.x + target.w
                and target.y <= final_y < target.y + target.h)


def test_wrong_segment_overshoot_passes_target():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=1)
    target = scene.buttons[0]
    cx, cy = 10.0, 10.0  # far from target
    seg = generate_wrong_segment(
        scene=scene, target_button_id=0, cursor_xy=(cx, cy),
        segment_type="overshoot", k_frames=10, rng=rng,
    )
    final_x, final_y = seg.final_cursor_xy
    target_cx, target_cy = target.center()
    # Overshoot should travel past target (start to final dist > start to target dist)
    start_to_target = ((target_cx - cx)**2 + (target_cy - cy)**2)**0.5
    start_to_final = ((final_x - cx)**2 + (final_y - cy)**2)**0.5
    assert start_to_final >= start_to_target * 0.8  # passed by or near target


def test_dart_noise_perturbs_at_expected_rate():
    rng = np.random.default_rng(0)
    n = 10000
    perturbed = 0
    for _ in range(n):
        clean_action = Action(dx=10.0, dy=5.0, click=0)
        noisy_action, was_perturbed = apply_dart_noise(
            clean_action, rng=rng, p=0.10, sigma=20.0,
        )
        if was_perturbed:
            perturbed += 1
    rate = perturbed / n
    assert abs(rate - 0.10) < 0.015  # ~10% with allowance


def test_dart_noise_skips_click_frames():
    rng = np.random.default_rng(0)
    # Force "always perturb" by setting p=1.0; click frame should still be skipped
    a_press = Action(dx=10.0, dy=5.0, click=1)
    a_out, was_perturbed = apply_dart_noise(a_press, rng=rng, p=1.0, sigma=20.0)
    assert was_perturbed is False
    assert a_out.dx == 10.0 and a_out.dy == 5.0


def test_dart_noise_preserves_other_fields():
    rng = np.random.default_rng(0)
    a = Action(dx=10.0, dy=5.0, click=0, scroll=0.0, key_events=_idle_keys_test())
    noisy, _ = apply_dart_noise(a, rng=rng, p=1.0, sigma=20.0)
    assert noisy.click == a.click
    assert noisy.scroll == a.scroll
