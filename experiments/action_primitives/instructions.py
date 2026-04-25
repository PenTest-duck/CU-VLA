"""Instruction templates and instruction rendering for Phase B0."""
from __future__ import annotations

# Single-attribute templates (60% of instructions)
SINGLE_ATTR_TEMPLATES: tuple[str, ...] = (
    "click the {color} button",
    "select the {color} one",
    "press the {color} button",
    "tap the {shape}",
    "click the {shape}",
    "press the {size} button",
    "select the {size} one",
    "click the button in the {position}",
    "tap the button on the {position}",
    "click the {position} button",
)

# Double-attribute templates (30%)
DOUBLE_ATTR_TEMPLATES: tuple[str, ...] = (
    "click the {color} {shape}",
    "select the {size} {color} button",
    "press the {color} {shape}",
    "tap the {color} button on the {position}",
    "click the {size} {shape}",
    "select the {shape} on the {position}",
)

# Triple-attribute templates (10%)
TRIPLE_ATTR_TEMPLATES: tuple[str, ...] = (
    "click the {size} {color} {shape}",
    "select the {color} {shape} on the {position}",
    "press the {size} {color} button on the {position}",
)

# Position labels (3x3 grid)
POSITION_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "top-left",     (1, 0): "top-center",    (2, 0): "top-right",
    (0, 1): "middle-left",  (1, 1): "center",        (2, 1): "middle-right",
    (0, 2): "bottom-left",  (1, 2): "bottom-center", (2, 2): "bottom-right",
}


def render_template(template: str, **kwargs: str) -> str:
    """Format a template with attribute slot values."""
    return template.format(**kwargs)
