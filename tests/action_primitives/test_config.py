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
