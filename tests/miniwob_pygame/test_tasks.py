"""Tests for MiniWoB-Pygame task environments."""

from experiments.miniwob_pygame.config import (
    ENV, KEY_C, KEY_CTRL, KEY_V, NUM_KEYS, char_to_key_index,
)
from experiments.miniwob_pygame.tasks.click_sequence import ClickSequenceEnv
from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
from experiments.miniwob_pygame.tasks.copy_paste import CopyPasteEnv
from experiments.miniwob_pygame.tasks.drag_and_label import DragAndLabelEnv
from experiments.miniwob_pygame.tasks.drag_sort import DragSortEnv
from experiments.miniwob_pygame.tasks.drag_to_zone import DragToZoneEnv
from experiments.miniwob_pygame.tasks.draw_path import DrawPathEnv
from experiments.miniwob_pygame.tasks.form_fill import FormFillEnv
from experiments.miniwob_pygame.tasks.highlight_text import HighlightTextEnv
from experiments.miniwob_pygame.tasks.type_field import TypeFieldEnv
from experiments.miniwob_pygame.tasks.scroll_and_click import ScrollAndClickEnv
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


class TestHighlightText:
    def test_correct_highlight_succeeds(self):
        env = HighlightTextEnv()
        env.reset(seed=42)
        tb = env._text_block
        target = env._target_word
        assert tb is not None

        # Get word bounds and set highlight to exactly the target word
        bounds = tb.word_bounds(target)
        assert bounds is not None
        start_idx, end_idx = bounds
        tb.set_highlight(start_idx, end_idx)

        # Mark selection as made, then step to trigger check
        env._selection_made = True
        _, done, info = env.step(_noop())
        assert done
        assert info.get("success") is True
        env.close()

    def test_wrong_highlight_fails(self):
        env = HighlightTextEnv()
        env.reset(seed=42)
        tb = env._text_block
        target = env._target_word
        assert tb is not None

        # Find a word that is NOT the target
        words = tb.text.split()
        wrong_word = None
        for w in words:
            if w != target:
                wrong_word = w
                break
        assert wrong_word is not None, "Need at least 2 distinct words"

        # Highlight the wrong word
        bounds = tb.word_bounds(wrong_word)
        assert bounds is not None
        start_idx, end_idx = bounds
        tb.set_highlight(start_idx, end_idx)

        # Mark selection as made, then step to trigger check
        env._selection_made = True
        _, done, info = env.step(_noop())
        assert done
        assert info.get("failure") == "wrong_selection"
        assert info.get("expected") == target
        env.close()


class TestDrawPath:
    def test_good_path_succeeds(self):
        env = DrawPathEnv(path_type="line", distance_threshold=20.0)
        env.reset(seed=42)
        ref = env.reference_path
        assert len(ref) >= 20

        # Teleport cursor to first reference point and press mouse
        env._cursor_x, env._cursor_y = ref[0]
        env.step(_noop(mouse_left=1))

        # Trace closely along the reference path
        done = False
        info = {}
        for rx, ry in ref[1:]:
            env._cursor_x = rx
            env._cursor_y = ry
            _, done, info = env.step(_noop(mouse_left=1))
            if done:
                break

        if not done:
            # Release mouse -> triggers evaluation
            _, done, info = env.step(_noop(mouse_left=0))

        assert done
        assert info.get("success") is True
        assert info["mean_distance"] <= 20.0
        env.close()

    def test_no_drawing_timeout(self):
        env = DrawPathEnv(path_type="line")
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


class TestDragSort:
    def test_already_sorted(self):
        """If cards happen to start sorted, success is immediate."""
        env = DragSortEnv(num_cards=4)
        # Try seeds until we find one that starts sorted, or manually sort
        env.reset(seed=0)
        # Manually place cards in sorted order
        for card in env.cards:
            target_slot = card["value"] - 1
            card["slot_index"] = target_slot
            card["x"] = env.slots[target_slot] - card["width"] // 2
        # Step with noop — check_success should fire
        _, done, info = env.step(_noop())
        assert done
        assert info.get("success") is True
        env.close()

    def test_drag_swaps_cards(self):
        """Grabbing a card and dropping on another slot swaps the two cards."""
        env = DragSortEnv(num_cards=4)
        env.reset(seed=42)
        cards = env.cards

        # Pick two cards in different slots
        card_a = cards[0]
        card_b = cards[1]
        slot_a = card_a["slot_index"]
        slot_b = card_b["slot_index"]
        assert slot_a != slot_b

        # Teleport cursor to card_a center and grab
        env._cursor_x = float(card_a["x"] + card_a["width"] / 2)
        env._cursor_y = float(card_a["y"] + card_a["height"] / 2)
        env.step(_noop(mouse_left=1))
        assert env._grabbed_card == 0

        # Drag to card_b's slot center
        env._cursor_x = float(env.slots[slot_b])
        env._cursor_y = float(card_b["y"] + card_b["height"] / 2)
        env.step(_noop(mouse_left=1))

        # Release — should swap
        env.step(_noop(mouse_left=0))

        assert card_a["slot_index"] == slot_b, "card_a should now be in slot_b"
        assert card_b["slot_index"] == slot_a, "card_b should now be in slot_a"
        env.close()


class TestFormFill:
    def test_correct_fill_and_submit_succeeds(self):
        env = FormFillEnv(num_fields=2)
        env.reset(seed=42)
        fields = env._fields
        targets = env._target_values
        sx, sy, sw, sh = env._submit_rect

        # Focus and type each field
        for field, target in zip(fields, targets):
            env._cursor_x = float(field.x + field.width / 2)
            env._cursor_y = float(field.y + field.height / 2)
            env.step(_noop(mouse_left=1))
            env.step(_noop(mouse_left=0))
            assert field.focused

            for ch in target:
                key_idx = char_to_key_index(ch)
                keys = [0] * NUM_KEYS
                keys[key_idx] = 1
                env.step(_noop(keys_held=keys))
                env.step(_noop())

        # Click submit button
        env._cursor_x = float(sx + sw / 2)
        env._cursor_y = float(sy + sh / 2)
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("success") is True
        env.close()

    def test_submit_wrong_values_fails(self):
        env = FormFillEnv(num_fields=2)
        env.reset(seed=42)
        fields = env._fields
        targets = env._target_values
        sx, sy, sw, sh = env._submit_rect

        # Focus the first field and type a wrong value
        field = fields[0]
        env._cursor_x = float(field.x + field.width / 2)
        env._cursor_y = float(field.y + field.height / 2)
        env.step(_noop(mouse_left=1))
        env.step(_noop(mouse_left=0))

        wrong_word = "ZZZ"
        for ch in wrong_word:
            key_idx = char_to_key_index(ch)
            keys = [0] * NUM_KEYS
            keys[key_idx] = 1
            env.step(_noop(keys_held=keys))
            env.step(_noop())

        # Click submit without filling the second field correctly
        env._cursor_x = float(sx + sw / 2)
        env._cursor_y = float(sy + sh / 2)
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("failure") == "wrong_values"
        env.close()


class TestScrollAndClick:
    def test_click_target_item_succeeds(self):
        env = ScrollAndClickEnv(num_items=15)
        env.reset(seed=42)
        sl = env._scroll_list
        assert sl is not None
        target = env._target_item
        target_idx = env._target_index

        # Scroll so that target is visible: set scroll_offset directly
        target_y_in_content = target_idx * sl.item_height
        sl.scroll_offset = max(0.0, min(sl.max_scroll,
            target_y_in_content - sl.height / 2))

        # Find target item screen position
        item_screen_y = sl.y + target_idx * sl.item_height - sl.scroll_offset
        item_cx = sl.x + (sl.width - sl.scrollbar_width) / 2
        item_cy = item_screen_y + sl.item_height / 2

        # Teleport cursor to item and click
        env._cursor_x = float(item_cx)
        env._cursor_y = float(item_cy)
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("success") is True
        env.close()

    def test_click_wrong_item_fails(self):
        env = ScrollAndClickEnv(num_items=15)
        env.reset(seed=42)
        sl = env._scroll_list
        assert sl is not None
        target_idx = env._target_index

        # Pick a wrong item (first item, which is not the target since target
        # is from bottom half)
        wrong_idx = 0
        assert wrong_idx != target_idx

        # The first item is visible at scroll_offset=0
        item_screen_y = sl.y + wrong_idx * sl.item_height
        item_cx = sl.x + (sl.width - sl.scrollbar_width) / 2
        item_cy = item_screen_y + sl.item_height / 2

        # Teleport cursor and click
        env._cursor_x = float(item_cx)
        env._cursor_y = float(item_cy)
        env.step(_noop(mouse_left=1))
        _, done, info = env.step(_noop(mouse_left=0))
        assert done
        assert info.get("failure") == "wrong_item"
        env.close()


class TestDragAndLabel:
    def test_drag_drop_and_type_succeeds(self):
        env = DragAndLabelEnv(num_shapes=1)
        env.reset(seed=42)
        shape = env.shapes[0]
        zone = env.zones[0]
        assert shape["color"] == zone["color"]

        # Teleport cursor to shape center and grab
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
        _, done, _ = env.step(_noop(mouse_left=1))
        assert not done

        # Release mouse -> drop on correct zone
        _, done, _ = env.step(_noop(mouse_left=0))
        assert not done  # Not done yet — still need to type
        assert shape["dropped"] is True

        # Type label character by character
        label = shape["label"]
        for ch in label:
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
        assert shape["complete"] is True
        env.close()

    def test_wrong_key_fails(self):
        env = DragAndLabelEnv(num_shapes=1)
        env.reset(seed=42)
        shape = env.shapes[0]
        zone = env.zones[0]

        # Teleport and drag to zone
        env._cursor_x = shape["x"] + shape["width"] / 2
        env._cursor_y = shape["y"] + shape["height"] / 2
        env.step(_noop(mouse_left=1))
        env._cursor_x = zone["x"] + zone["width"] / 2
        env._cursor_y = zone["y"] + zone["height"] / 2
        env.step(_noop(mouse_left=1))
        env.step(_noop(mouse_left=0))
        assert shape["dropped"] is True

        # Type a wrong character (one that doesn't match the first char of label)
        label = shape["label"]
        wrong_char = "Z" if label[0] != "Z" else "A"
        key_idx = char_to_key_index(wrong_char)
        keys = [0] * NUM_KEYS
        keys[key_idx] = 1
        _, done, info = env.step(_noop(keys_held=keys))
        # The failure is set in _handle_key_down, check_success picks it up
        assert done
        assert info.get("failure") == "wrong_key"
        env.close()


class TestCopyPaste:
    def test_copy_paste_succeeds(self):
        """Manually highlight source, Ctrl+C, click field, Ctrl+V -> success."""
        env = CopyPasteEnv()
        env.reset(seed=42)
        tb = env._text_block
        ti = env._text_input
        source = env._source_text
        assert tb is not None
        assert ti is not None

        # Set highlight to the full source text
        bounds = tb.word_bounds(source)
        assert bounds is not None
        start_idx, end_idx = bounds
        tb.set_highlight(start_idx, end_idx)

        # Verify highlighted text matches source
        assert tb.highlighted_text() == source

        # Ctrl+C: press both keys simultaneously
        keys_ctrl_c = [0] * NUM_KEYS
        keys_ctrl_c[KEY_CTRL] = 1
        keys_ctrl_c[KEY_C] = 1
        _, done, _ = env.step(_noop(keys_held=keys_ctrl_c))
        assert not done
        assert env._clipboard == source

        # Release keys
        _, done, _ = env.step(_noop())
        assert not done

        # Click the TextInput to focus it
        env._cursor_x = float(ti.x + ti.width / 2)
        env._cursor_y = float(ti.y + ti.height / 2)
        env.step(_noop(mouse_left=1))
        _, done, _ = env.step(_noop(mouse_left=0))
        assert not done
        assert ti.focused

        # Ctrl+V: press both keys simultaneously
        keys_ctrl_v = [0] * NUM_KEYS
        keys_ctrl_v[KEY_CTRL] = 1
        keys_ctrl_v[KEY_V] = 1
        _, done, info = env.step(_noop(keys_held=keys_ctrl_v))
        assert done
        assert info.get("success") is True
        assert ti.text == source
        env.close()

    def test_paste_without_copy_empty(self):
        """Pasting without copying first leaves the field empty."""
        env = CopyPasteEnv()
        env.reset(seed=42)
        ti = env._text_input
        assert ti is not None

        # Click the TextInput to focus it
        env._cursor_x = float(ti.x + ti.width / 2)
        env._cursor_y = float(ti.y + ti.height / 2)
        env.step(_noop(mouse_left=1))
        env.step(_noop(mouse_left=0))
        assert ti.focused

        # Ctrl+V without prior Ctrl+C
        keys_ctrl_v = [0] * NUM_KEYS
        keys_ctrl_v[KEY_CTRL] = 1
        keys_ctrl_v[KEY_V] = 1
        _, done, _ = env.step(_noop(keys_held=keys_ctrl_v))
        assert not done
        assert ti.text == ""  # Nothing was copied, so paste is empty
        env.close()
