"""Tests for MiniWoB-Pygame task environments."""

from experiments.miniwob_pygame.config import ENV, NUM_KEYS, char_to_key_index
from experiments.miniwob_pygame.tasks.click_sequence import ClickSequenceEnv
from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
from experiments.miniwob_pygame.tasks.drag_to_zone import DragToZoneEnv
from experiments.miniwob_pygame.tasks.type_field import TypeFieldEnv
from experiments.miniwob_pygame.tasks.use_slider import UseSliderEnv


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


class TestUseSlider:
    def test_slider_at_target_succeeds(self):
        env = UseSliderEnv()
        env.reset(seed=42)
        slider = env._slider
        assert slider is not None
        # Teleport slider value to target
        slider.value = slider.target_value
        slider.dragging = False
        # _check_success is called every step, so a noop triggers it
        _, done, info = env.step(_noop())
        assert done
        assert info.get("success") is True
        env.close()

    def test_slider_wrong_value_not_success(self):
        env = UseSliderEnv(tolerance=5.0)
        env.reset(seed=42)
        slider = env._slider
        assert slider is not None
        # Set value far from target
        slider.value = (slider.target_value + 50) % 100
        slider.dragging = False
        _, done, info = env.step(_noop())
        assert not done or info.get("success") is not True
        env.close()


class TestDragToZone:
    def test_drag_to_correct_zone_succeeds(self):
        env = DragToZoneEnv(num_shapes=1)
        env.reset(seed=42)
        shape = env.shapes[0]
        zone = env.zones[0]
        assert shape["color"] == zone["color"]

        # Teleport cursor to shape center and press (grab)
        env._cursor_x = shape["x"] + shape["width"] / 2
        env._cursor_y = shape["y"] + shape["height"] / 2
        _, done, _ = env.step(_noop(mouse_left=1))
        assert not done
        assert shape["grabbed"] is True

        # Teleport cursor to zone center (while held)
        zone_cx = zone["x"] + zone["width"] / 2
        zone_cy = zone["y"] + zone["height"] / 2
        env._cursor_x = zone_cx
        env._cursor_y = zone_cy
        # Step with mouse held to update drag position
        _, done, _ = env.step(_noop(mouse_left=1))
        assert not done

        # Release mouse -> drop on correct zone
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("success") is True
        assert shape["dropped"] is True
        env.close()

    def test_drop_wrong_zone_fails(self):
        env = DragToZoneEnv(num_shapes=2)
        env.reset(seed=42)
        shape = env.shapes[0]
        # Find the zone that does NOT match shape's color
        wrong_zone = None
        for z in env.zones:
            if z["color"] != shape["color"]:
                wrong_zone = z
                break
        assert wrong_zone is not None, "Need a zone with different color"

        # Grab the shape
        env._cursor_x = shape["x"] + shape["width"] / 2
        env._cursor_y = shape["y"] + shape["height"] / 2
        env.step(_noop(mouse_left=1))

        # Drag to wrong zone center
        env._cursor_x = wrong_zone["x"] + wrong_zone["width"] / 2
        env._cursor_y = wrong_zone["y"] + wrong_zone["height"] / 2
        env.step(_noop(mouse_left=1))

        # Release on wrong zone
        _, done, info = env.step(_noop(mouse_left=0))
        assert not done
        assert shape["dropped"] is False
        env.close()


class TestClickSequence:
    def test_correct_order_succeeds(self):
        env = ClickSequenceEnv(num_buttons=3)
        env.reset(seed=42)
        target_order = env._target_order
        buttons = env._buttons

        # Build lookup by number
        btn_by_num = {b["number"]: b for b in buttons}

        done = False
        info = {}
        for number in target_order:
            btn = btn_by_num[number]
            # Teleport cursor to button center
            env._cursor_x = float(btn["x"] + btn["width"] / 2)
            env._cursor_y = float(btn["y"] + btn["height"] / 2)
            # Press
            _, done, info = env.step(_noop(mouse_left=1))
            if done:
                break
            # Release (click)
            _, done, info = env.step(_noop(mouse_left=0))
            if done:
                break

        assert done
        assert info.get("success") is True
        env.close()

    def test_wrong_order_fails(self):
        env = ClickSequenceEnv(num_buttons=3)
        env.reset(seed=42)
        target_order = env._target_order
        buttons = env._buttons

        # Build lookup by number
        btn_by_num = {b["number"]: b for b in buttons}

        # Find a button that is NOT the first expected
        expected_first = target_order[0]
        wrong_number = None
        for number in target_order:
            if number != expected_first:
                wrong_number = number
                break
        assert wrong_number is not None, "Need at least 2 distinct buttons"

        # Click the wrong button
        btn = btn_by_num[wrong_number]
        env._cursor_x = float(btn["x"] + btn["width"] / 2)
        env._cursor_y = float(btn["y"] + btn["height"] / 2)
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))

        assert done
        assert info.get("success") is not True
        env.close()


class TestTypeField:
    def test_correct_typing_succeeds(self):
        env = TypeFieldEnv()
        env.reset(seed=42)
        ti = env._text_input
        target = env._target_word
        # Teleport cursor to text input center and click to focus
        env._cursor_x = float(ti.x + ti.width / 2)
        env._cursor_y = float(ti.y + ti.height / 2)
        env.step(_noop(mouse_left=1))
        env.step(_noop(mouse_left=0))
        assert ti.focused
        # Type each character of the target word
        done = False
        info = {}
        for ch in target:
            key_idx = char_to_key_index(ch)
            keys = [0] * NUM_KEYS
            keys[key_idx] = 1
            _, done, info = env.step(_noop(keys_held=keys))
            if done:
                break
            # Release key
            _, done, info = env.step(_noop())
            if done:
                break
        assert done
        assert info.get("success") is True
        env.close()

    def test_wrong_character_fails(self):
        env = TypeFieldEnv()
        env.reset(seed=42)
        ti = env._text_input
        target = env._target_word
        # Focus the field
        env._cursor_x = float(ti.x + ti.width / 2)
        env._cursor_y = float(ti.y + ti.height / 2)
        env.step(_noop(mouse_left=1))
        env.step(_noop(mouse_left=0))
        # Type a character that is NOT the first char of target
        wrong_char = "Z" if target[0] != "Z" else "A"
        key_idx = char_to_key_index(wrong_char)
        keys = [0] * NUM_KEYS
        keys[key_idx] = 1
        _, done, info = env.step(_noop(keys_held=keys))
        if not done:
            # Release key triggers check
            _, done, info = env.step(_noop())
        assert done
        assert info.get("failure") == "wrong_key"
        env.close()
