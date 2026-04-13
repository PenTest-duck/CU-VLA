"""Shared expert utilities for MiniWoB-Pygame tasks.

Provides Fitts's Law trajectory generation, noop/pause helpers,
cursor simulation, and episode replay.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import ACTION, ENV, EXPERT, NUM_KEYS


def noop_action() -> dict:
    """Return an idle action with no mouse or key presses."""
    return {
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 0,
        "keys_held": [0] * NUM_KEYS,
    }


def pause_actions(rng: np.random.Generator) -> list[dict]:
    """Generate EXPERT.pause_min to pause_max noop actions."""
    n = int(rng.integers(EXPERT.pause_min, EXPERT.pause_max + 1))
    return [noop_action() for _ in range(n)]


def fitts_trajectory(
    cx: float,
    cy: float,
    tx: float,
    ty: float,
    target_width: float,
    rng: np.random.Generator,
    mouse_held: bool = False,
) -> list[dict]:
    """Generate a Fitts's Law mouse movement from (cx, cy) to (tx, ty).

    Args:
        cx, cy: Current cursor position.
        tx, ty: Target position.
        target_width: Width of the target (for Fitts's Law index of difficulty).
        rng: Numpy random generator.
        mouse_held: If True, all actions have mouse_left=1 (dragging).

    Returns:
        List of action dicts with keys dx, dy, mouse_left, keys_held.
    """
    mouse_val = 1 if mouse_held else 0

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

        actions.append({
            "dx": dx,
            "dy": dy,
            "mouse_left": mouse_val,
            "keys_held": [0] * NUM_KEYS,
        })
        cur_x = float(np.clip(cur_x + dx, 0, ENV.window_size - 1))
        cur_y = float(np.clip(cur_y + dy, 0, ENV.window_size - 1))

    return actions


def simulate_cursor(
    actions: list[dict], start_x: float, start_y: float
) -> tuple[float, float]:
    """Simulate cursor position after replaying a sequence of actions.

    Args:
        actions: List of action dicts with dx, dy keys.
        start_x, start_y: Starting cursor position.

    Returns:
        (final_x, final_y) after applying all actions with clamping.
    """
    ws = ENV.window_size
    x, y = start_x, start_y
    for a in actions:
        x = float(np.clip(x + a["dx"], 0, ws - 1))
        y = float(np.clip(y + a["dy"], 0, ws - 1))
    return x, y


def run_episode(
    env,
    trajectory: list[dict],
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Replay a pre-computed trajectory in an already-reset environment.

    Args:
        env: A BaseTaskEnv instance that has already been reset().
        trajectory: List of action dicts to replay.

    Returns:
        observations: List of screenshot arrays (obs[t] = state when action[t] chosen).
        actions: List of action dicts that were actually executed.
        final_info: Info dict from the terminal step.
    """
    # Capture initial observation
    obs = env._get_observation()
    observations = [obs]
    actions = []
    final_info: dict = {}

    for action in trajectory:
        obs, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        if done:
            final_info = info
            break

    if not final_info and actions:
        final_info = info  # type: ignore[possibly-undefined]

    # Trim: obs[t] is the state when action[t] was chosen
    observations = observations[: len(actions)]
    return observations, actions, final_info
