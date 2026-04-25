import pytest
from experiments.action_primitives.instructions import (
    SINGLE_ATTR_TEMPLATES,
    DOUBLE_ATTR_TEMPLATES,
    TRIPLE_ATTR_TEMPLATES,
    render_template,
)


def test_template_registry_nonempty():
    assert len(SINGLE_ATTR_TEMPLATES) >= 3
    assert len(DOUBLE_ATTR_TEMPLATES) >= 3
    assert len(TRIPLE_ATTR_TEMPLATES) >= 3


def test_render_single_attr_color():
    template = "click the {color} button"
    out = render_template(template, color="red")
    assert out == "click the red button"


def test_render_uses_position_label():
    out = render_template("click the button in the {position}", position="top-left")
    assert "top-left" in out
