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
