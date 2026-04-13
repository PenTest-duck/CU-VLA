"""Tests for MiniWoB-Pygame task environments."""

from experiments.miniwob_pygame.config import NUM_KEYS
from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv


def _noop(**overrides):
    action = {"dx": 0.0, "dy": 0.0, "mouse_left": 0, "keys_held": [0] * NUM_KEYS}
    action.update(overrides)
    return action


class TestClickTarget:
    def test_reset_places_target(self):
        env = ClickTargetEnv()
        env.reset(seed=42)
        t = env._target
        assert "x" in t and "y" in t
        assert "width" in t and "height" in t
        assert "color" in t
        assert 40 <= t["width"] < 80
        assert 40 <= t["height"] < 80
        assert env.task_instruction.startswith("Click the ")
        env.close()

    def test_click_on_target_succeeds(self):
        env = ClickTargetEnv()
        env.reset(seed=42)
        t = env._target
        # Teleport cursor to target center
        target_cx = t["x"] + t["width"] / 2
        target_cy = t["y"] + t["height"] / 2
        env._cursor_x = target_cx
        env._cursor_y = target_cy
        # Press mouse
        _, done, info = env.step(_noop(mouse_left=1))
        assert not done
        # Release mouse -> click registered
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("success") is True
        env.close()

    def test_click_off_target_does_not_succeed(self):
        env = ClickTargetEnv()
        env.reset(seed=42)
        # Cursor at (0, 0) — well away from target (target y >= instruction_bar_height + 10)
        env._cursor_x = 0.0
        env._cursor_y = 0.0
        # Press + release
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert not done
        assert info.get("success") is not True
        env.close()

    def test_timeout(self):
        env = ClickTargetEnv()
        env.reset(seed=42)
        max_steps = env._get_max_steps()
        done = False
        info = {}
        for _ in range(max_steps):
            _, done, info = env.step(_noop())
            if done:
                break
        assert done
        assert info.get("timeout") is True
        env.close()

    def test_distractors_placed(self):
        env = ClickTargetEnv(num_distractors=3)
        env.reset(seed=42)
        assert len(env._distractors) == 3
        # Distractor colors differ from target
        for d in env._distractors:
            assert d["color"] != env._target["color"]
        env.close()
