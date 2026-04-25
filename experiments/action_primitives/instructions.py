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


from dataclasses import dataclass
from itertools import combinations

import numpy as np

from experiments.action_primitives.scene import (
    ATTRIBUTE_KEYS, Scene, compute_disambiguators, _attr_value,
)


@dataclass(frozen=True)
class InstructionResult:
    instruction: str
    target_button_id: int
    used_attrs: tuple[str, ...]
    composite_tier: int  # 1, 2, or 3


# Composite tier sampling probabilities
TIER_PROBS = {1: 0.6, 2: 0.3, 3: 0.1}


def _attr_value_human(button, attr: str) -> str:
    """Convert an attribute value to its human-readable form for templates."""
    val = _attr_value(button, attr)
    if attr == "position":
        return POSITION_LABELS[val]
    return val  # color/shape/size are already strings


def _candidate_attr_sets(scene: Scene, target_idx: int, k: int) -> list[tuple[str, ...]]:
    """All k-attribute combinations that uniquely identify the target."""
    target = scene.buttons[target_idx]
    out: list[tuple[str, ...]] = []
    for combo in combinations(ATTRIBUTE_KEYS, k):
        target_vals = tuple(_attr_value(target, a) for a in combo)
        if all(
            tuple(_attr_value(other, a) for a in combo) != target_vals
            for j, other in enumerate(scene.buttons) if j != target_idx
        ):
            out.append(combo)
    return out


def _matching_templates(used_attrs: tuple[str, ...]) -> list[str]:
    """Templates whose slots exactly match the used_attrs set."""
    if len(used_attrs) == 1:
        pool = SINGLE_ATTR_TEMPLATES
    elif len(used_attrs) == 2:
        pool = DOUBLE_ATTR_TEMPLATES
    else:
        pool = TRIPLE_ATTR_TEMPLATES
    used_set = set(used_attrs)
    out: list[str] = []
    for tpl in pool:
        # Slot names in template
        slots = {s for s in ("color", "shape", "size", "position") if "{" + s + "}" in tpl}
        if slots == used_set:
            out.append(tpl)
    return out


def generate_instruction(scene: Scene, rng: np.random.Generator) -> InstructionResult:
    """Sample a target + composite tier + template; render unique instruction."""
    n_buttons = len(scene.buttons)
    target_idx = int(rng.integers(0, n_buttons))

    if n_buttons == 1:
        # 1-button scene: trivially unique. Pick any single attribute or generic.
        target = scene.buttons[0]
        attr = ATTRIBUTE_KEYS[rng.integers(0, len(ATTRIBUTE_KEYS))]
        templates = _matching_templates((attr,))
        if not templates:
            # Fallback (shouldn't happen with current registry)
            templates = ["click the button"]
            instruction = templates[0]
            return InstructionResult(
                instruction=instruction, target_button_id=0,
                used_attrs=(), composite_tier=1,
            )
        tpl = templates[rng.integers(0, len(templates))]
        slot_vals = {a: _attr_value_human(target, a) for a in (attr,)}
        return InstructionResult(
            instruction=render_template(tpl, **slot_vals),
            target_button_id=0,
            used_attrs=(attr,),
            composite_tier=1,
        )

    # Sample composite tier weighted by TIER_PROBS, falling back to higher tier if needed
    tier_order = [1, 2, 3]
    weights = np.array([TIER_PROBS[t] for t in tier_order])
    weights = weights / weights.sum()
    primary_tier = int(rng.choice(tier_order, p=weights))

    # Try primary tier first; fall back to next tier if no unique attribute set exists
    used_attrs: tuple[str, ...] = ()
    chosen_tier = 0
    for tier in [primary_tier] + [t for t in tier_order if t != primary_tier]:
        candidates = _candidate_attr_sets(scene, target_idx, tier)
        # Filter further: only candidates whose template pool is non-empty
        candidates = [c for c in candidates if len(_matching_templates(c)) > 0]
        if candidates:
            used_attrs = candidates[rng.integers(0, len(candidates))]
            chosen_tier = tier
            break
    if chosen_tier == 0:
        raise RuntimeError(
            f"No uniqueness-preserving attribute combination found for scene "
            f"with {n_buttons} buttons targeting button {target_idx}."
        )

    target = scene.buttons[target_idx]
    templates = _matching_templates(used_attrs)
    tpl = templates[rng.integers(0, len(templates))]
    slot_vals = {a: _attr_value_human(target, a) for a in used_attrs}
    return InstructionResult(
        instruction=render_template(tpl, **slot_vals),
        target_button_id=target_idx,
        used_attrs=used_attrs,
        composite_tier=chosen_tier,
    )
