"""Fitts's Law scripted expert for the drag-and-label task.

Generates realistic demonstration trajectories with:
- Fitts's Law timing: MT = a + b * log2(D/W + 1)
- Bell-shaped velocity profile (minimum-jerk-inspired)
- Gaussian noise on path for variability
- Nearest-first shape ordering heuristic
- Three phases per shape: navigate+grab, drag+drop, type label
"""

import math

import numpy as np

from .config import ENV, ACTION, EXPERT


def _fitts_trajectory(
    cx: float,
    cy: float,
    tx: float,
    ty: float,
    target_width: float,
    rng: np.random.Generator,
    click_held: bool = False,
) -> list[dict]:
    """Generate a Fitts's Law mouse movement from (cx, cy) to (tx, ty).

    Args:
        cx, cy: Current cursor position.
        tx, ty: Target position.
        target_width: Width of the target (for Fitts's Law index of difficulty).
        rng: Numpy random generator.
        click_held: If True, all actions have click=1 (dragging).

    Returns:
        List of action dicts with keys dx, dy, click, key.
    """
    click_val = 1 if click_held else 0

    # Pick a random target point near the center
    angle = rng.uniform(0, 2 * math.pi)
    r = target_width * 0.3 * math.sqrt(rng.uniform(0, 1))
    target_x = tx + r * math.cos(angle)
    target_y = ty + r * math.sin(angle)

    distance = math.hypot(target_x - cx, target_y - cy)
    if distance < 1.0:
        return []

    # Fitts's Law: movement time in seconds
    movement_time = EXPERT.fitts_a + EXPERT.fitts_b * math.log2(
        distance / target_width + 1
    )
    num_steps = max(2, round(movement_time * ENV.control_hz))

    # Bell-shaped velocity profile (minimum-jerk-inspired)
    t = np.linspace(0, 1, num_steps, endpoint=False) + 0.5 / num_steps
    velocity_profile = 30.0 * t**2 * (1.0 - t) ** 2
    velocity_profile /= velocity_profile.sum()

    # Per-step displacements with online correction
    actions = []
    cur_x, cur_y = cx, cy
    cum_frac = np.cumsum(velocity_profile)
    max_delta = ACTION.max_delta_px

    for i in range(num_steps):
        remaining_x = target_x - cur_x
        remaining_y = target_y - cur_y

        covered_so_far = cum_frac[i - 1] if i > 0 else 0.0
        remaining_frac = 1.0 - covered_so_far
        step_frac = (
            velocity_profile[i] / remaining_frac if remaining_frac > 1e-6 else 1.0
        )

        dx = remaining_x * step_frac + rng.normal(
            0, EXPERT.noise_std * velocity_profile[i]
        )
        dy = remaining_y * step_frac + rng.normal(
            0, EXPERT.noise_std * velocity_profile[i]
        )

        dx = float(np.clip(dx, -max_delta, max_delta))
        dy = float(np.clip(dy, -max_delta, max_delta))

        actions.append({"dx": dx, "dy": dy, "click": click_val, "key": 0})
        cur_x = float(np.clip(cur_x + dx, 0, ENV.window_size - 1))
        cur_y = float(np.clip(cur_y + dy, 0, ENV.window_size - 1))

    return actions


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    shapes: list[dict],
    zones: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Generate a full expert trajectory for all shapes using nearest-first heuristic.

    For each shape (nearest remaining first):
      1. Navigate to shape center
      2. Mouse down (grab)
      3. Drag to matching zone center
      4. Mouse up (drop)
      5. Pause, then type label
      6. Inter-shape pause if more shapes remain

    Args:
        cursor_x, cursor_y: Starting cursor position.
        shapes: List of shape dicts from env.
        zones: List of zone dicts from env.
        rng: Numpy random generator.

    Returns:
        List of action dicts.
    """
    # Build zone lookup by color
    zone_by_color = {}
    for zone in zones:
        zone_by_color[zone["color"]] = zone

    # Determine processing order: nearest-first
    remaining = list(range(len(shapes)))
    cx, cy = cursor_x, cursor_y
    actions = []

    while remaining:
        # Find nearest remaining shape
        best_idx = None
        best_dist = float("inf")
        for idx in remaining:
            s = shapes[idx]
            sx = s["x"] + s["width"] / 2
            sy = s["y"] + s["height"] / 2
            dist = math.hypot(sx - cx, sy - cy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        remaining.remove(best_idx)

        shape = shapes[best_idx]
        zone = zone_by_color[shape["color"]]

        # Shape center
        shape_cx = shape["x"] + shape["width"] / 2
        shape_cy = shape["y"] + shape["height"] / 2

        # Zone center
        zone_cx = zone["x"] + zone["width"] / 2
        zone_cy = zone["y"] + zone["height"] / 2

        # 1. Navigate to shape center
        move_actions = _fitts_trajectory(
            cx, cy, shape_cx, shape_cy, shape["width"], rng, click_held=False
        )
        actions.extend(move_actions)
        for a in move_actions:
            cx = float(np.clip(cx + a["dx"], 0, ENV.window_size - 1))
            cy = float(np.clip(cy + a["dy"], 0, ENV.window_size - 1))

        # 2. Mouse down (grab)
        actions.append({"dx": 0.0, "dy": 0.0, "click": 1, "key": 0})

        # 3. Drag to matching zone center
        drag_actions = _fitts_trajectory(
            cx, cy, zone_cx, zone_cy, zone["width"], rng, click_held=True
        )
        actions.extend(drag_actions)
        for a in drag_actions:
            cx = float(np.clip(cx + a["dx"], 0, ENV.window_size - 1))
            cy = float(np.clip(cy + a["dy"], 0, ENV.window_size - 1))

        # 4. Mouse up (drop)
        actions.append({"dx": 0.0, "dy": 0.0, "click": 0, "key": 0})

        # 5. Pause then type label
        pause_steps = rng.integers(EXPERT.pause_min, EXPERT.pause_max + 1)
        for _ in range(pause_steps):
            actions.append({"dx": 0.0, "dy": 0.0, "click": 0, "key": 0})

        # 6. Type label: key_down (char index) then key_up (0) for each char
        for ch in shape["label"]:
            char_index = ord(ch) - ord("A") + 1
            actions.append({"dx": 0.0, "dy": 0.0, "click": 0, "key": char_index})
            actions.append({"dx": 0.0, "dy": 0.0, "click": 0, "key": 0})

        # 7. Inter-shape pause if more shapes remain
        if remaining:
            inter_pause = rng.integers(
                EXPERT.inter_shape_pause_min, EXPERT.inter_shape_pause_max + 1
            )
            for _ in range(inter_pause):
                actions.append({"dx": 0.0, "dy": 0.0, "click": 0, "key": 0})

    return actions


def run_episode(
    env,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[dict], list[dict], dict]:
    """Run a full expert episode in the environment.

    Args:
        env: DragLabelEnv instance.
        seed: Random seed for env reset and trajectory generation.
        rng: Optional pre-seeded random generator.

    Returns:
        observations: List of numpy observation arrays.
        actions: List of action dicts.
        states: List of state dicts with cursor_x, cursor_y, click_state, key_state
                — the env state when each action was chosen.
        final_info: Info dict from the terminal step.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    obs = env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, env.shapes, env.zones, rng=rng
    )

    def _capture_state():
        cx, cy = env.cursor_pos
        return {
            "cursor_x": cx,
            "cursor_y": cy,
            "click_state": env._click_state,
            "key_state": env._current_key,
        }

    observations = [obs]
    states = [_capture_state()]
    actions = []
    final_info = {}

    for action in trajectory:
        obs, done, info = env.step(action)
        observations.append(obs)
        states.append(_capture_state())
        actions.append(action)
        if done:
            final_info = info
            break

    if not final_info.get("success"):
        final_info = info if trajectory else {}

    # Trim to T observations for T actions: obs[t] is the state when action[t] was chosen.
    # The terminal observation (after last action) is discarded — it has no action pair.
    observations = observations[:len(actions)]
    states = states[:len(actions)]
    return observations, actions, states, final_info
