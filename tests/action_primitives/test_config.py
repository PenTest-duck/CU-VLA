"""Basic sanity checks on config constants."""

import numpy as np
import pytest

from experiments.action_primitives.config import (
    MOUSE_BIN_CENTERS,
    MOUSE_CAP_PX,
    NUM_BINS_MOUSE,
    SCROLL_BIN_CENTERS,
    NUM_BINS_SCROLL,
    TOTAL_LOGITS,
    HEAD_LOGITS,
)


def test_mouse_bin_centers_shape_and_symmetry():
    assert MOUSE_BIN_CENTERS.shape == (NUM_BINS_MOUSE,)
    # Symmetric around zero
    assert MOUSE_BIN_CENTERS[NUM_BINS_MOUSE // 2] == 0.0
    np.testing.assert_allclose(
        MOUSE_BIN_CENTERS[: NUM_BINS_MOUSE // 2],
        -MOUSE_BIN_CENTERS[NUM_BINS_MOUSE // 2 + 1 :][::-1],
        atol=1e-5,
    )


def test_mouse_bin_centers_cap():
    assert MOUSE_BIN_CENTERS[-1] == pytest.approx(MOUSE_CAP_PX)
    assert MOUSE_BIN_CENTERS[0] == pytest.approx(-MOUSE_CAP_PX)


def test_mouse_bin_centers_monotonic():
    assert np.all(np.diff(MOUSE_BIN_CENTERS) > 0)


def test_scroll_bin_centers_shape():
    assert SCROLL_BIN_CENTERS.shape == (NUM_BINS_SCROLL,)


def test_total_logits_matches_design():
    assert TOTAL_LOGITS == 300  # 21+21+5+21+231+1 per design Q1
    assert HEAD_LOGITS["keys"] == 231


def test_b0_attribute_palettes_defined():
    from experiments.action_primitives.config import (
        B0_COLORS, B0_SHAPES, B0_SIZES, B0_POSITION_GRID, B0_BG_COLORS,
    )
    assert len(B0_COLORS) == 10
    assert all(isinstance(c, tuple) and len(c) == 3 for c in B0_COLORS.values())
    assert set(B0_SHAPES) >= {"rect", "circle", "triangle", "square", "hexagon"}
    assert set(B0_SIZES) == {"small", "medium", "large"}
    # Each size value is a (min, max) tuple of ints with min < max
    for name, value in B0_SIZES.items():
        assert isinstance(value, tuple) and len(value) == 2, f"B0_SIZES[{name!r}] must be a 2-tuple"
        assert all(isinstance(v, int) for v in value), f"B0_SIZES[{name!r}] entries must be ints"
        assert value[0] < value[1], f"B0_SIZES[{name!r}] must have min < max"
    assert B0_POSITION_GRID == (3, 3)
    # Each background is a 3-tuple of ints in [0, 255]
    for bg in B0_BG_COLORS:
        assert isinstance(bg, tuple) and len(bg) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in bg)
