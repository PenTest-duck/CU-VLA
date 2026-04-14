"""Expert movement primitives with human-like variance for Experiment 5."""

from __future__ import annotations

import math

import numpy as np

from .config import (
    NUM_KEYS,
    KEY_LSHIFT,
    KEY_RSHIFT,
    KEY_SPACE,
    KEY_DELETE,
    KEY_RETURN,
    KEY_TAB,
    ACTION,
    ENV,
    EXPERT,
    COMMON_BIGRAMS,
    CHAR_TO_KEY,
    char_to_key_action,
    shift_key_for,
)


def noop_action() -> dict:
    """Return an idle action: zero deltas, nothing pressed."""
    return {"dx": 0.0, "dy": 0.0, "mouse_left": 0, "keys_held": [0] * NUM_KEYS}


def pause_actions(rng: np.random.Generator, lo: int, hi: int) -> list[dict]:
    """Generate lo..hi noop actions (inclusive)."""
    n = int(rng.integers(lo, hi + 1))
    return [noop_action() for _ in range(n)]


def _quadratic_bezier(p0: float, p1: float, p2: float, t: float) -> float:
    """Evaluate quadratic bezier at parameter t."""
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def fitts_trajectory_human(
    cx: float,
    cy: float,
    tx: float,
    ty: float,
    target_width: float,
    rng: np.random.Generator,
    mouse_held: bool = False,
    held_keys: list[int] | None = None,
) -> list[dict]:
    """Fitts's Law trajectory from (cx,cy) to (tx,ty) with human-like variance.

    Base: Same as experiments/miniwob_pygame/experts/common.py fitts_trajectory()
    with added curvature, overshoot, speed noise, and micro-jitter.
    """
    mouse_val = 1 if mouse_held else 0
    keys = held_keys if held_keys is not None else [0] * NUM_KEYS
    max_delta = ACTION.max_delta_px

    # Random target scatter near center
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

    # Bell-shaped velocity profile
    t_arr = np.linspace(0, 1, num_steps, endpoint=False) + 0.5 / num_steps
    velocity_profile = 30.0 * t_arr**2 * (1.0 - t_arr) ** 2
    velocity_profile /= velocity_profile.sum()

    # --- Curvature: quadratic bezier control point ---
    curvature_frac = rng.uniform(
        EXPERT.curvature_frac_lo, EXPERT.curvature_frac_hi
    )
    # Perpendicular direction (randomly left or right)
    dx_line = target_x - cx
    dy_line = target_y - cy
    perp_x = -dy_line
    perp_y = dx_line
    perp_len = math.hypot(perp_x, perp_y)
    if perp_len > 1e-6:
        perp_x /= perp_len
        perp_y /= perp_len
    sign = 1.0 if rng.random() < 0.5 else -1.0
    ctrl_x = (cx + target_x) / 2 + sign * curvature_frac * distance * perp_x
    ctrl_y = (cy + target_y) / 2 + sign * curvature_frac * distance * perp_y

    # Compute waypoints along the bezier curve
    cum_frac = np.cumsum(velocity_profile)

    actions: list[dict] = []
    cur_x, cur_y = cx, cy

    for i in range(num_steps):
        # Target position at this step along the bezier
        covered_so_far = cum_frac[i - 1] if i > 0 else 0.0
        remaining_frac = 1.0 - covered_so_far
        step_frac = (
            velocity_profile[i] / remaining_frac if remaining_frac > 1e-6 else 1.0
        )

        # Where we want to be after this step (bezier position at cum_frac[i])
        bez_x = _quadratic_bezier(cx, ctrl_x, target_x, cum_frac[i])
        bez_y = _quadratic_bezier(cy, ctrl_y, target_y, cum_frac[i])

        # Online correction: blend bezier target with remaining distance
        remaining_x = bez_x - cur_x
        remaining_y = bez_y - cur_y

        # If not the last step, use step_frac-based correction toward bezier
        # For the last step, go directly to the remaining bezier position
        if i < num_steps - 1:
            # Step toward the bezier waypoint
            dx = remaining_x + rng.normal(0, EXPERT.noise_std * velocity_profile[i])
            dy = remaining_y + rng.normal(0, EXPERT.noise_std * velocity_profile[i])
        else:
            # Final step: close the gap to the bezier endpoint (== target)
            dx = remaining_x
            dy = remaining_y

        # Speed noise
        speed_mult = 1.0 + rng.uniform(
            -EXPERT.speed_noise_frac, EXPERT.speed_noise_frac
        )
        dx *= speed_mult
        dy *= speed_mult

        # Micro-jitter
        dx += rng.normal(0, EXPERT.jitter_px)
        dy += rng.normal(0, EXPERT.jitter_px)

        dx = float(np.clip(dx, -max_delta, max_delta))
        dy = float(np.clip(dy, -max_delta, max_delta))

        actions.append({
            "dx": dx,
            "dy": dy,
            "mouse_left": mouse_val,
            "keys_held": list(keys),
        })
        cur_x = float(np.clip(cur_x + dx, 0, ENV.window_w - 1))
        cur_y = float(np.clip(cur_y + dy, 0, ENV.window_h - 1))

    # --- Overshoot submovement ---
    if rng.random() < EXPERT.overshoot_prob and len(actions) > 0:
        # Overshoot: continue past target by 5-20px
        overshoot_px = rng.uniform(EXPERT.overshoot_px_lo, EXPERT.overshoot_px_hi)
        # Direction of last movement
        last_dx = actions[-1]["dx"]
        last_dy = actions[-1]["dy"]
        last_len = math.hypot(last_dx, last_dy)
        if last_len > 1e-6:
            os_dx = last_dx / last_len * overshoot_px
            os_dy = last_dy / last_len * overshoot_px
        else:
            os_angle = rng.uniform(0, 2 * math.pi)
            os_dx = overshoot_px * math.cos(os_angle)
            os_dy = overshoot_px * math.sin(os_angle)

        # Overshoot step
        os_dx = float(np.clip(os_dx, -max_delta, max_delta))
        os_dy = float(np.clip(os_dy, -max_delta, max_delta))
        actions.append({
            "dx": os_dx,
            "dy": os_dy,
            "mouse_left": mouse_val,
            "keys_held": list(keys),
        })
        cur_x = float(np.clip(cur_x + os_dx, 0, ENV.window_w - 1))
        cur_y = float(np.clip(cur_y + os_dy, 0, ENV.window_h - 1))

        # Pause at overshoot
        pause_n = int(
            rng.integers(EXPERT.overshoot_pause_lo, EXPERT.overshoot_pause_hi + 1)
        )
        for _ in range(pause_n):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "mouse_left": mouse_val,
                "keys_held": list(keys),
            })

        # Correction trajectory back to target (short, straight)
        corr_dx = target_x - cur_x
        corr_dy = target_y - cur_y
        corr_dist = math.hypot(corr_dx, corr_dy)
        if corr_dist > 1.0:
            corr_steps = max(2, round(corr_dist / 10.0))
            for j in range(corr_steps):
                frac = 1.0 / (corr_steps - j)
                sdx = (target_x - cur_x) * frac
                sdy = (target_y - cur_y) * frac
                sdx += rng.normal(0, EXPERT.jitter_px)
                sdy += rng.normal(0, EXPERT.jitter_px)
                sdx = float(np.clip(sdx, -max_delta, max_delta))
                sdy = float(np.clip(sdy, -max_delta, max_delta))
                actions.append({
                    "dx": sdx,
                    "dy": sdy,
                    "mouse_left": mouse_val,
                    "keys_held": list(keys),
                })
                cur_x = float(np.clip(cur_x + sdx, 0, ENV.window_w - 1))
                cur_y = float(np.clip(cur_y + sdy, 0, ENV.window_h - 1))

    return actions


def click_actions(
    rng: np.random.Generator,
    held_keys: list[int] | None = None,
) -> list[dict]:
    """Generate a click sequence: dwell + mouse_down (1-4 frames) + mouse_up.

    1. Pre-click dwell: click_dwell_lo..click_dwell_hi noop frames
    2. Mouse down: click_duration_lo..click_duration_hi frames with mouse_left=1
    3. Mouse up: 1 frame with mouse_left=0
    """
    keys = held_keys if held_keys is not None else [0] * NUM_KEYS
    actions: list[dict] = []

    # Pre-click dwell
    dwell_n = int(
        rng.integers(EXPERT.click_dwell_lo, EXPERT.click_dwell_hi + 1)
    )
    for _ in range(dwell_n):
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": list(keys),
        })

    # Mouse down
    down_n = int(
        rng.integers(EXPERT.click_duration_lo, EXPERT.click_duration_hi + 1)
    )
    for _ in range(down_n):
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 1,
            "keys_held": list(keys),
        })

    # Mouse up
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 0,
        "keys_held": list(keys),
    })

    return actions


def type_string_actions(
    text: str,
    rng: np.random.Generator,
    speed_mult: float | None = None,
) -> list[dict]:
    """Generate typing actions for a string of characters.

    Handles shift for uppercase/symbols, inter-key intervals based on bigrams,
    and keeps shift held for consecutive shifted characters.
    """
    if speed_mult is None:
        speed_mult = float(
            rng.uniform(EXPERT.typing_speed_lo, EXPERT.typing_speed_hi)
        )

    actions: list[dict] = []
    prev_char: str | None = None
    prev_shifted = False

    for ci, ch in enumerate(text):
        key_index, needs_shift = char_to_key_action(ch)

        # Inter-key interval (skip before first character)
        if prev_char is not None:
            bigram = prev_char + ch
            if bigram.lower() in COMMON_BIGRAMS:
                iki_lo, iki_hi = EXPERT.iki_common_lo, EXPERT.iki_common_hi
            elif prev_char == " ":
                iki_lo, iki_hi = EXPERT.iki_space_lo, EXPERT.iki_space_hi
            else:
                iki_lo, iki_hi = EXPERT.iki_uncommon_lo, EXPERT.iki_uncommon_hi

            raw_frames = rng.integers(iki_lo, iki_hi + 1)
            iki_frames = max(0, round(float(raw_frames) * speed_mult))
            for _ in range(iki_frames):
                actions.append(noop_action())

            # Micro-pause
            if rng.random() < EXPERT.micro_pause_prob:
                mp_n = int(
                    rng.integers(EXPERT.micro_pause_lo, EXPERT.micro_pause_hi + 1)
                )
                for _ in range(mp_n):
                    actions.append(noop_action())

        if needs_shift:
            shift_idx = shift_key_for(key_index)

            # If previous char was also shifted (with same shift key), keep it held
            if not prev_shifted:
                # Shift down: lead frames
                lead_n = int(
                    rng.integers(EXPERT.shift_lead_lo, EXPERT.shift_lead_hi + 1)
                )
                for _ in range(lead_n):
                    k = [0] * NUM_KEYS
                    k[shift_idx] = 1
                    actions.append({
                        "dx": 0.0,
                        "dy": 0.0,
                        "mouse_left": 0,
                        "keys_held": k,
                    })

            # Key down: 1 frame with shift + character key
            k = [0] * NUM_KEYS
            k[shift_idx] = 1
            k[key_index] = 1
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "mouse_left": 0,
                "keys_held": k,
            })

            # Key up: 1 frame with just shift held
            k = [0] * NUM_KEYS
            k[shift_idx] = 1
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "mouse_left": 0,
                "keys_held": k,
            })

            # Check if next char also needs shift — if so, keep shift held
            next_shifted = False
            if ci + 1 < len(text):
                next_key, next_needs_shift = char_to_key_action(text[ci + 1])
                if next_needs_shift:
                    next_shifted = True

            if not next_shifted:
                # Shift up: lag frames then release
                lag_n = int(
                    rng.integers(EXPERT.shift_lag_lo, EXPERT.shift_lag_hi + 1)
                )
                for _ in range(lag_n):
                    actions.append(noop_action())

            prev_shifted = True
        else:
            # Release shift if it was held from previous char
            prev_shifted = False

            # Key down: 1 frame with character key held
            k = [0] * NUM_KEYS
            k[key_index] = 1
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "mouse_left": 0,
                "keys_held": k,
            })

            # Key up: 1 frame with nothing held
            actions.append(noop_action())

        prev_char = ch

    return actions


def shift_click_actions(
    shift_key: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate shift+click: shift_down -> mouse_down+up (shift held) -> shift_up.

    1. Shift down: 1-2 frames with just shift held
    2. Mouse down: 1-4 frames with shift + mouse_left held
    3. Mouse up: 1 frame with just shift held
    4. Shift up: 1 frame noop
    """
    actions: list[dict] = []

    # Shift down: 1-2 frames
    shift_lead = int(rng.integers(1, 3))  # 1 or 2
    for _ in range(shift_lead):
        k = [0] * NUM_KEYS
        k[shift_key] = 1
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 0,
            "keys_held": k,
        })

    # Mouse down: 1-4 frames with shift + mouse
    down_n = int(
        rng.integers(EXPERT.click_duration_lo, EXPERT.click_duration_hi + 1)
    )
    for _ in range(down_n):
        k = [0] * NUM_KEYS
        k[shift_key] = 1
        actions.append({
            "dx": 0.0,
            "dy": 0.0,
            "mouse_left": 1,
            "keys_held": k,
        })

    # Mouse up: 1 frame with just shift held
    k = [0] * NUM_KEYS
    k[shift_key] = 1
    actions.append({
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 0,
        "keys_held": k,
    })

    # Shift up: 1 frame noop
    actions.append(noop_action())

    return actions
