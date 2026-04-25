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


import numpy as np
from experiments.action_primitives.scene import Scene, Button, generate_scene
from experiments.action_primitives.instructions import (
    generate_instruction,
    InstructionResult,
)


def test_generate_instruction_returns_unique_target():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scene = generate_scene(rng=rng)
        result = generate_instruction(scene, rng=rng)
        assert isinstance(result, InstructionResult)
        assert isinstance(result.instruction, str)
        assert len(result.instruction) > 0
        assert 0 <= result.target_button_id < len(scene.buttons)
        assert result.composite_tier in (1, 2, 3)


def test_generate_instruction_single_button_scene():
    rng = np.random.default_rng(0)
    scene = generate_scene(rng=rng, n_buttons=1)
    result = generate_instruction(scene, rng=rng)
    # 1-button scene: target_id must be 0; instruction can be generic
    assert result.target_button_id == 0


def test_generate_instruction_uniqueness_invariant():
    """The chosen attribute combination should uniquely identify the target."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        scene = generate_scene(rng=rng, n_buttons=4)
        result = generate_instruction(scene, rng=rng)
        target = scene.buttons[result.target_button_id]
        # The result's used_attrs must distinguish the target from all other buttons
        for j, other in enumerate(scene.buttons):
            if j == result.target_button_id:
                continue
            differs_on_any = any(
                _get_attr(target, a) != _get_attr(other, a) for a in result.used_attrs
            )
            assert differs_on_any, (
                f"Instruction {result.instruction!r} cannot distinguish target "
                f"button {result.target_button_id} from button {j}: used_attrs={result.used_attrs}"
            )


def _get_attr(b: Button, a: str):
    if a == "color": return b.color
    if a == "shape": return b.shape
    if a == "size": return b.size
    if a == "position": return b.pos_zone
    raise ValueError(a)


from experiments.action_primitives.instructions import inject_typo


def test_inject_typo_changes_string():
    rng = np.random.default_rng(0)
    s = "click the red button"
    out = inject_typo(s, rng=rng)
    assert out != s  # at least one char changed
    assert len(out) >= len(s) - 1  # bounded growth/shrinkage


def test_inject_typo_idempotent_at_zero_rate():
    """Calling with no perturbation should be a no-op."""
    rng = np.random.default_rng(0)
    s = "click the red button"
    out = inject_typo(s, rng=rng, n_changes=0)
    assert out == s
