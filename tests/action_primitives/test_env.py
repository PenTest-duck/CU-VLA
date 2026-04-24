"""Unit tests for LClickEnv."""

import numpy as np
import pytest

from experiments.action_primitives.env import Action, LClickEnv


def test_env_reset_returns_valid_obs_and_info():
    env = LClickEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert "image" in obs and "proprio" in obs
    assert obs["image"].size == (720, 450)
    assert "target_bbox" in info
    x, y, w, h = info["target_bbox"]
    assert w > 0 and h > 0
    # Cursor should not start on target
    cx, cy = info["cursor_xy"]
    assert not (x <= cx <= x + w and y <= cy <= y + h)


def test_env_step_moves_cursor():
    env = LClickEnv(seed=0)
    obs, info = env.reset(seed=0)
    cx0, cy0 = info["cursor_xy"]
    action = Action(dx=10.0, dy=5.0)
    obs, done, info = env.step(action)
    cx1, cy1 = info["cursor_xy"]
    assert cx1 == cx0 + 10.0
    assert cy1 == cy0 + 5.0
    assert not done


def test_env_lclick_on_target_succeeds():
    env = LClickEnv(seed=7)
    obs, info = env.reset(seed=7)
    x, y, w, h = info["target_bbox"]
    tx, ty = x + w // 2, y + h // 2
    cx0, cy0 = info["cursor_xy"]
    # Move to target in one step (unrealistic but tests success detection)
    env.step(Action(dx=tx - cx0, dy=ty - cy0))
    env.step(Action(click=1))   # press
    obs, done, info = env.step(Action(click=2))  # release
    assert done is True
    assert info["success"] is True


def test_env_lclick_off_target_no_success():
    env = LClickEnv(seed=11)
    obs, info = env.reset(seed=11)
    x, y, w, h = info["target_bbox"]
    # Move far from target
    cx0, cy0 = info["cursor_xy"]
    env.step(Action(dx=0.0, dy=0.0))  # no-op, still off-target
    env.step(Action(click=1))
    obs, done, info = env.step(Action(click=2))
    assert done is False


def test_env_clamps_cursor_to_canvas():
    env = LClickEnv(seed=0)
    obs, info = env.reset(seed=0)
    # Try to move way off-screen
    obs, done, info = env.step(Action(dx=10000, dy=10000))
    cx, cy = info["cursor_xy"]
    assert 0 <= cx <= 720 - 1
    assert 0 <= cy <= 450 - 1
