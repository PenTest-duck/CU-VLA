"""Fitts's Law scripted expert for generating demonstration trajectories.

Generates realistic mouse movement with:
- Fitts's Law timing: MT = a + b * log2(D/W + 1)
- Bell-shaped velocity profile (acceleration then deceleration)
- Gaussian noise on path for variability
- Click on target arrival (mouse_down then mouse_up next frame)

Actions are continuous float pixel deltas, clamped to [-max_delta_px, +max_delta_px].
"""

import math

import numpy as np

from .config import ENV, ACTION, EXPERT
from .env import BTN_NONE, BTN_DOWN, BTN_UP


def generate_trajectory(
    cursor_x: float,
    cursor_y: float,
    circle_x: int,
    circle_y: int,
    circle_radius: int,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate a full expert trajectory from cursor to circle click.

    Returns:
        List of action dicts with 'dx' (float px), 'dy' (float px), 'btn' (int).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pick a random target point within the circle (not always center)
    angle = rng.uniform(0, 2 * math.pi)
    r = circle_radius * 0.6 * math.sqrt(rng.uniform(0, 1))
    target_x = circle_x + r * math.cos(angle)
    target_y = circle_y + r * math.sin(angle)

    # Fitts's Law: movement time in seconds
    distance = math.hypot(target_x - cursor_x, target_y - cursor_y)
    width = circle_radius * 2
    if distance < 1.0:
        return [
            {"dx": 0.0, "dy": 0.0, "btn": BTN_DOWN},
            {"dx": 0.0, "dy": 0.0, "btn": BTN_UP},
        ]

    movement_time = EXPERT.fitts_a + EXPERT.fitts_b * math.log2(distance / width + 1)
    num_steps = max(2, round(movement_time * ENV.control_hz))

    # Bell-shaped velocity profile (minimum-jerk-inspired)
    t = np.linspace(0, 1, num_steps, endpoint=False) + 0.5 / num_steps
    velocity_profile = 30.0 * t**2 * (1.0 - t) ** 2
    velocity_profile /= velocity_profile.sum()

    # Per-step displacements with online correction
    actions = []
    cx, cy = cursor_x, cursor_y
    cum_frac = np.cumsum(velocity_profile)
    max_delta = ACTION.max_delta_px

    for i in range(num_steps):
        remaining_x = target_x - cx
        remaining_y = target_y - cy

        covered_so_far = cum_frac[i - 1] if i > 0 else 0.0
        remaining_frac = 1.0 - covered_so_far
        step_frac = velocity_profile[i] / remaining_frac if remaining_frac > 1e-6 else 1.0

        dx = remaining_x * step_frac + rng.normal(0, EXPERT.noise_std * velocity_profile[i])
        dy = remaining_y * step_frac + rng.normal(0, EXPERT.noise_std * velocity_profile[i])

        # Clamp to max delta
        dx = float(np.clip(dx, -max_delta, max_delta))
        dy = float(np.clip(dy, -max_delta, max_delta))

        actions.append({"dx": dx, "dy": dy, "btn": BTN_NONE})
        cx += dx
        cy += dy

    # Click: mouse_down then mouse_up
    actions.append({"dx": 0.0, "dy": 0.0, "btn": BTN_DOWN})
    actions.append({"dx": 0.0, "dy": 0.0, "btn": BTN_UP})

    return actions


def run_episode(
    env,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Run a full expert episode in the environment.

    Returns:
        observations: List of numpy observation arrays.
        actions: List of action dicts.
        final_info: Info dict from the terminal step.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    obs = env.reset(seed=seed)
    trajectory = generate_trajectory(
        *env.cursor_pos, *env.circle_pos, env.circle_radius, rng=rng
    )

    observations = [obs]
    actions = []
    final_info = {}

    for action in trajectory:
        obs, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        if done:
            final_info = info
            break

    if not done:
        click = {"dx": 0.0, "dy": 0.0, "btn": BTN_DOWN}
        obs, done, info = env.step(click)
        observations.append(obs)
        actions.append(click)
        final_info = info

    observations = observations[: len(actions)]
    return observations, actions, final_info
