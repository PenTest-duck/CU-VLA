"""Tests for BaseTaskEnv."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.miniwob_pygame.base_env import BaseTaskEnv
from experiments.miniwob_pygame.config import (
    ACTION,
    ENV,
    KEY_A,
    KEY_CTRL,
    NUM_KEYS,
)


# ------------------------------------------------------------------
# Minimal concrete subclass for testing
# ------------------------------------------------------------------


class ConcreteTask(BaseTaskEnv):
    task_name = "test-task"

    def _setup_task(self, rng: np.random.Generator) -> None:
        self.task_instruction = "Test instruction"

    def _check_success(self) -> tuple[bool, dict]:
        return (False, {})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _noop(**overrides) -> dict:
    """Return a default idle action dict with optional overrides."""
    action = {
        "dx": 0.0,
        "dy": 0.0,
        "mouse_left": 0,
        "keys_held": [0] * NUM_KEYS,
    }
    action.update(overrides)
    return action


@pytest.fixture
def env():
    e = ConcreteTask(visual=False)
    yield e
    e.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_reset_returns_obs_dict(env: ConcreteTask):
    obs = env.reset(seed=42)
    assert "screenshot" in obs
    assert "cursor_pos" in obs
    assert obs["screenshot"].shape == (ENV.obs_size, ENV.obs_size, 3)
    assert obs["screenshot"].dtype == np.uint8
    assert obs["cursor_pos"].shape == (2,)
    assert obs["cursor_pos"].dtype == np.float32
    assert 0.0 <= obs["cursor_pos"][0] <= 1.0
    assert 0.0 <= obs["cursor_pos"][1] <= 1.0


def test_step_applies_cursor_delta(env: ConcreteTask):
    env.reset(seed=42)
    old_x, old_y = env.cursor_pos
    action = _noop(dx=10.0, dy=-5.0)
    env.step(action)
    new_x, new_y = env.cursor_pos
    assert abs(new_x - (old_x + 10.0)) < 1e-3
    assert abs(new_y - (old_y - 5.0)) < 1e-3


def test_step_clamps_delta(env: ConcreteTask):
    env.reset(seed=42)
    old_x, old_y = env.cursor_pos
    action = _noop(dx=999.0, dy=-999.0)
    env.step(action)
    new_x, new_y = env.cursor_pos
    max_d = ACTION.max_delta_px
    # Delta should have been clamped to max_delta
    expected_x = np.clip(old_x + max_d, 0, ENV.window_size - 1)
    expected_y = np.clip(old_y - max_d, 0, ENV.window_size - 1)
    assert abs(new_x - expected_x) < 1e-3
    assert abs(new_y - expected_y) < 1e-3


def test_step_clamps_cursor_to_window(env: ConcreteTask):
    env.reset(seed=42)
    # Force cursor near top-left
    env._cursor_x = 5.0
    env._cursor_y = 5.0
    action = _noop(dx=-50.0, dy=-50.0)
    env.step(action)
    new_x, new_y = env.cursor_pos
    assert new_x >= 0.0
    assert new_y >= 0.0


def test_mouse_held_state_tracking(env: ConcreteTask):
    env.reset(seed=42)
    assert env._mouse_pressed is False

    # Press
    env.step(_noop(mouse_left=1))
    assert env._mouse_pressed is True

    # Hold
    env.step(_noop(mouse_left=1))
    assert env._mouse_pressed is True

    # Release
    env.step(_noop(mouse_left=0))
    assert env._mouse_pressed is False


def test_key_held_state_tracking(env: ConcreteTask):
    env.reset(seed=42)
    assert env._keys_held == [0] * NUM_KEYS

    # Press KEY_A + KEY_CTRL simultaneously
    keys = [0] * NUM_KEYS
    keys[KEY_A] = 1
    keys[KEY_CTRL] = 1
    env.step(_noop(keys_held=keys))
    assert env._keys_held[KEY_A] == 1
    assert env._keys_held[KEY_CTRL] == 1

    # Release only KEY_A
    keys2 = list(keys)
    keys2[KEY_A] = 0
    env.step(_noop(keys_held=keys2))
    assert env._keys_held[KEY_A] == 0
    assert env._keys_held[KEY_CTRL] == 1


def test_instruction_rendered_in_header(env: ConcreteTask):
    obs = env.reset(seed=42)
    screenshot = obs["screenshot"]
    # Compute the header row in observation coordinates
    scale = ENV.obs_size / ENV.window_size
    header_rows = int(ENV.instruction_bar_height * scale)
    row = header_rows // 2
    col = 5
    pixel = screenshot[row, col]
    # The instruction bg (50,50,50) should be brighter than the main bg (30,30,30)
    assert pixel[0] > 40, f"Expected header pixel R > 40, got {pixel[0]}"


def test_step_after_done_raises(env: ConcreteTask):
    env.reset(seed=42)
    env._done = True
    with pytest.raises(AssertionError):
        env.step(_noop())
