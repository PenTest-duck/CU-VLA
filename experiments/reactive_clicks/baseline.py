"""Deterministic baseline controller: color threshold + centroid beeline.

No learning. Detects the red circle via color thresholding on the observation,
computes its centroid, and outputs a delta toward the centroid each step.
Clicks when the cursor overlaps the detected region.

Actions are continuous float pixel deltas.
"""

import numpy as np

from .config import ENV, ACTION
from .env import BTN_NONE, BTN_DOWN, BTN_UP


class BaselineController:
    """Threshold + centroid beeline controller."""

    def __init__(self, click_threshold_px: float = 5.0):
        self.click_threshold = click_threshold_px
        self._clicked = False

    def reset(self) -> None:
        self._clicked = False

    def act(self, obs: np.ndarray) -> dict:
        if self._clicked:
            self._clicked = False
            return {"dx": 0.0, "dy": 0.0, "btn": BTN_UP}

        # Detect red pixels: R > 200, G < 50, B < 50
        red_mask = (obs[:, :, 0] > 200) & (obs[:, :, 1] < 50) & (obs[:, :, 2] < 50)

        # Detect cursor (white pixels): all channels > 200
        cursor_mask = (obs[:, :, 0] > 200) & (obs[:, :, 1] > 200) & (obs[:, :, 2] > 200)

        red_ys, red_xs = np.where(red_mask)
        if len(red_xs) == 0:
            return {"dx": 0.0, "dy": 0.0, "btn": BTN_NONE}

        target_x = red_xs.mean()
        target_y = red_ys.mean()

        cursor_ys, cursor_xs = np.where(cursor_mask)
        if len(cursor_xs) == 0:
            return {"dx": 0.0, "dy": 0.0, "btn": BTN_NONE}

        cursor_x = cursor_xs.mean()
        cursor_y = cursor_ys.mean()

        # Delta in observation space, scaled to native coords
        scale = ENV.window_size / ENV.obs_size
        dx_native = (target_x - cursor_x) * scale
        dy_native = (target_y - cursor_y) * scale
        distance_native = np.hypot(dx_native, dy_native)

        if distance_native < self.click_threshold + ENV.cursor_radius:
            self._clicked = True
            return {"dx": 0.0, "dy": 0.0, "btn": BTN_DOWN}

        # Clamp to max delta
        dx = float(np.clip(dx_native, -ACTION.max_delta_px, ACTION.max_delta_px))
        dy = float(np.clip(dy_native, -ACTION.max_delta_px, ACTION.max_delta_px))

        return {"dx": dx, "dy": dy, "btn": BTN_NONE}
