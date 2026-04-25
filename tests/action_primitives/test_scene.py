import pytest
from experiments.action_primitives.scene import Button, DecorativeShape


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


import numpy as np
from experiments.action_primitives.scene import generate_scene, Scene


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
    from experiments.action_primitives.config import ENV
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
