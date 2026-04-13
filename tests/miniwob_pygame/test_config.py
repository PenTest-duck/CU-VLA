"""Tests for experiments/miniwob_pygame/config.py."""

import pytest

from experiments.miniwob_pygame.config import (
    NUM_KEYS,
    KEY_A, KEY_B, KEY_Z,
    KEY_SPACE, KEY_ENTER, KEY_BACKSPACE, KEY_TAB,
    KEY_0, KEY_1, KEY_9,
    KEY_CTRL, KEY_SHIFT, KEY_ALT,
    TASK_NAMES, VOCAB,
    ENV, ACTION, MODEL, CHUNK, TRAIN, EVAL_CFG, EXPERT,
    char_to_key_index, key_index_to_char,
)


# ── Key index constants ─────────────────────────────────────────────────

class TestKeyConstants:
    def test_num_keys(self):
        assert NUM_KEYS == 43

    def test_letter_range(self):
        assert KEY_A == 0
        assert KEY_B == 1
        assert KEY_Z == 25

    def test_special_keys(self):
        assert KEY_SPACE == 26
        assert KEY_ENTER == 27
        assert KEY_BACKSPACE == 28
        assert KEY_TAB == 29

    def test_digit_range(self):
        assert KEY_0 == 30
        assert KEY_1 == 31
        assert KEY_9 == 39

    def test_modifier_keys(self):
        assert KEY_CTRL == 40
        assert KEY_SHIFT == 41
        assert KEY_ALT == 42


# ── Singleton defaults ───────────────────────────────────────────────────

class TestEnvConfig:
    def test_defaults(self):
        assert ENV.window_size == 400
        assert ENV.obs_size == 224
        assert ENV.bg_color == (30, 30, 30)
        assert ENV.control_hz == 30
        assert len(ENV.shape_colors) == 4

    def test_frozen(self):
        with pytest.raises(AttributeError):
            ENV.window_size = 999  # type: ignore[misc]


class TestActionConfig:
    def test_defaults(self):
        assert ACTION.max_delta_px == 50.0
        assert ACTION.num_keys == NUM_KEYS == 43


class TestModelConfig:
    def test_proprio_dim(self):
        # 2 (mouse_xy) + 1 (mouse_btn) + 43 (keys_held) = 46
        assert MODEL.proprio_dim == 46
        assert MODEL.proprio_dim == 2 + 1 + NUM_KEYS

    def test_backbone_feature_dims(self):
        assert "resnet18" in MODEL.backbone_feature_dims
        assert "dinov2-vits14" in MODEL.backbone_feature_dims
        assert "siglip2-base" in MODEL.backbone_feature_dims


# ── Task names ───────────────────────────────────────────────────────────

class TestTaskNames:
    def test_count(self):
        assert len(TASK_NAMES) == 12

    def test_first_and_last(self):
        assert TASK_NAMES[0] == "click-target"
        assert TASK_NAMES[-1] == "copy-paste"

    def test_unique(self):
        assert len(set(TASK_NAMES)) == len(TASK_NAMES)


# ── Vocab ────────────────────────────────────────────────────────────────

class TestVocab:
    def test_count(self):
        assert len(VOCAB) == 20

    def test_three_letter_uppercase(self):
        for word in VOCAB:
            assert len(word) == 3
            assert word == word.upper()


# ── Helper functions ─────────────────────────────────────────────────────

class TestCharToKeyIndex:
    def test_letters(self):
        assert char_to_key_index("A") == 0
        assert char_to_key_index("a") == 0  # case insensitive
        assert char_to_key_index("Z") == 25
        assert char_to_key_index("z") == 25

    def test_digits(self):
        assert char_to_key_index("0") == 30
        assert char_to_key_index("9") == 39

    def test_space(self):
        assert char_to_key_index(" ") == 26

    def test_unsupported(self):
        with pytest.raises(ValueError):
            char_to_key_index("!")
        with pytest.raises(ValueError):
            char_to_key_index("\n")

    def test_multi_char(self):
        with pytest.raises(ValueError):
            char_to_key_index("AB")


class TestKeyIndexToChar:
    def test_letters(self):
        assert key_index_to_char(0) == "A"
        assert key_index_to_char(25) == "Z"

    def test_space(self):
        assert key_index_to_char(26) == " "

    def test_digits(self):
        assert key_index_to_char(30) == "0"
        assert key_index_to_char(39) == "9"

    def test_non_printable(self):
        assert key_index_to_char(KEY_ENTER) is None
        assert key_index_to_char(KEY_BACKSPACE) is None
        assert key_index_to_char(KEY_TAB) is None
        assert key_index_to_char(KEY_CTRL) is None
        assert key_index_to_char(KEY_SHIFT) is None
        assert key_index_to_char(KEY_ALT) is None
