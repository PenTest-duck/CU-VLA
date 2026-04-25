"""Tests for scene primitives (Phase B0)."""
import pytest
import numpy as np

from experiments.action_primitives.scene import (
    Button, DecorativeShape, Scene, compute_disambiguators, generate_scene, is_adversarial,
)
from experiments.action_primitives.config import ENV


def test_button_has_attributes():
    b = Button(
        button_id=0, color="red", shape="circle", size="medium",
        pos_zone=(1, 1), x=100, y=100, w=80, h=60,
    )
    assert b.color == "red"
    assert b.shape == "circle"
    assert b.size == "medium"
    assert b.pos_zone == (1, 1)
    assert (b.x, b.y, b.w, b.h) == (100, 100, 80, 60)
    assert b.is_clickable is True


def test_decorative_shape_is_not_clickable():
    d = DecorativeShape(
        shape="triangle", color="cyan", x=200, y=200, w=40, h=30,
    )
    assert d.is_clickable is False


def test_generate_scene_basic():
    rng = np.random.default_rng(42)
    scene = generate_scene(rng=rng)
    assert isinstance(scene, Scene)
    assert 1 <= len(scene.buttons) <= 6
    assert 0 <= len(scene.decorative_shapes) <= 3
    # All buttons have unique button_ids
    ids = [b.button_id for b in scene.buttons]
    assert len(ids) == len(set(ids))
    # All buttons fit on canvas
    for b in scene.buttons:
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= ENV.canvas_w
        assert b.y + b.h <= ENV.canvas_h


def test_generate_scene_seeded_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = generate_scene(rng=rng1)
    s2 = generate_scene(rng=rng2)
    assert len(s1.buttons) == len(s2.buttons)
    assert s1.bg_color == s2.bg_color


def test_generate_scene_n_buttons_override():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=4)
    assert len(scene.buttons) == 4


def test_button_rejects_invalid_color():
    with pytest.raises(ValueError, match="color"):
        Button(button_id=0, color="puce", shape="circle", size="medium",
               pos_zone=(1, 1), x=100, y=100, w=80, h=60)


def test_button_rejects_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        Button(button_id=0, color="red", shape="dodecahedron", size="medium",
               pos_zone=(1, 1), x=100, y=100, w=80, h=60)


def test_button_rejects_out_of_bounds_pos_zone():
    with pytest.raises(ValueError, match="pos_zone"):
        Button(button_id=0, color="red", shape="circle", size="medium",
               pos_zone=(5, 5), x=100, y=100, w=80, h=60)


def test_generate_scene_square_buttons_have_equal_wh():
    """All square buttons must have w == h (enforced by generator + Button validator)."""
    rng = np.random.default_rng(0)
    n_squares_seen = 0
    for _ in range(50):
        scene = generate_scene(rng=rng)
        for b in scene.buttons:
            if b.shape == "square":
                assert b.w == b.h, f"square button has w={b.w}, h={b.h}"
                n_squares_seen += 1
    assert n_squares_seen > 0, "test sample didn't include any squares (sample more)"


def test_button_rejects_square_with_non_equal_wh():
    with pytest.raises(ValueError, match="square"):
        Button(button_id=0, color="red", shape="square", size="medium",
               pos_zone=(1, 1), x=100, y=100, w=80, h=60)


def test_decorations_do_not_overlap_buttons():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scene = generate_scene(rng=rng)
        for d in scene.decorative_shapes:
            for b in scene.buttons:
                # AABB overlap test — must NOT overlap
                overlaps = (d.x < b.x + b.w and d.x + d.w > b.x and
                            d.y < b.y + b.h and d.y + d.h > b.y)
                assert not overlaps, (
                    f"decoration ({d.x},{d.y},{d.w}x{d.h}) overlaps "
                    f"button {b.button_id} ({b.x},{b.y},{b.w}x{b.h})"
                )


def test_generate_scene_rejects_invalid_n_buttons():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_buttons"):
        generate_scene(rng=rng, n_buttons=0)
    with pytest.raises(ValueError, match="n_buttons"):
        generate_scene(rng=rng, n_buttons=10)  # >9 (3x3 grid)


def test_compute_disambiguators_single_button():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=1)
    dis = compute_disambiguators(scene)
    # 1-button scene: any attribute trivially disambiguates
    assert len(dis) == 1
    assert "color" in dis[0] and "shape" in dis[0]


def test_is_adversarial_two_same_color_diff_shape():
    """Two red buttons differing in shape — not adversarial (shape disambiguates)."""
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "medium", (0, 0), 10, 10, 60, 60),
            Button(1, "red", "square", "medium", (1, 0), 100, 10, 60, 60),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    assert is_adversarial(scene) is False


def test_is_adversarial_all_unique_on_some_attribute():
    """Three buttons each with a unique color — not adversarial."""
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "medium", (0, 0), 10, 10, 60, 60),
            Button(1, "blue", "circle", "medium", (1, 0), 100, 10, 60, 60),
            Button(2, "green", "circle", "medium", (2, 0), 200, 10, 60, 60),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    # Each button has unique color → single-attribute disambiguates → not adversarial.
    assert is_adversarial(scene) is False


def test_is_adversarial_color_and_shape_shared():
    """Two red circles differing only in size+position — color and shape are
    shared (each ambiguous), so categorical disambiguation requires composite.
    Adversarial = True under the non-position definition.
    """
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "circle", "large", (1, 0), 100, 10, 100, 100),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    # Size DOES disambiguate (small vs large), so each button has a non-position
    # single-attr disambiguator → not adversarial under the new definition.
    assert is_adversarial(scene) is False


def test_is_adversarial_all_categorical_shared():
    """Two buttons sharing color, shape, AND size — only position differs.
    Adversarial under the non-position definition.
    """
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
            Button(1, "red", "circle", "small", (1, 0), 100, 10, 40, 40),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    # color/shape/size all shared → no non-position disambiguator → adversarial.
    assert is_adversarial(scene) is True


def test_is_adversarial_returns_false_for_single_button():
    """Trivial 1-button scene is never adversarial."""
    scene = Scene(
        buttons=(
            Button(0, "red", "circle", "small", (0, 0), 10, 10, 40, 40),
        ),
        decorative_shapes=(),
        bg_color=(245, 245, 248),
    )
    assert is_adversarial(scene) is False
