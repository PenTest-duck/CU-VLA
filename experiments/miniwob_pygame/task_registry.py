"""Central registry mapping task names to env classes and expert functions.

Uses lazy imports so that importing task_registry itself is lightweight —
actual env / expert modules are only loaded when needed.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

# (task_module, task_class, expert_module, expert_fn)
_TASK_MAP: dict[str, tuple[str, str, str, str]] = {
    "click-target": ("tasks.click_target", "ClickTargetEnv", "experts.click_target", "run_expert_episode"),
    "drag-to-zone": ("tasks.drag_to_zone", "DragToZoneEnv", "experts.drag_to_zone", "run_expert_episode"),
    "use-slider": ("tasks.use_slider", "UseSliderEnv", "experts.use_slider", "run_expert_episode"),
    "type-field": ("tasks.type_field", "TypeFieldEnv", "experts.type_field", "run_expert_episode"),
    "click-sequence": ("tasks.click_sequence", "ClickSequenceEnv", "experts.click_sequence", "run_expert_episode"),
    "draw-path": ("tasks.draw_path", "DrawPathEnv", "experts.draw_path", "run_expert_episode"),
    "highlight-text": ("tasks.highlight_text", "HighlightTextEnv", "experts.highlight_text", "run_expert_episode"),
    "drag-sort": ("tasks.drag_sort", "DragSortEnv", "experts.drag_sort", "run_expert_episode"),
    "form-fill": ("tasks.form_fill", "FormFillEnv", "experts.form_fill", "run_expert_episode"),
    "drag-and-label": ("tasks.drag_and_label", "DragAndLabelEnv", "experts.drag_and_label", "run_expert_episode"),
    "scroll-and-click": ("tasks.scroll_and_click", "ScrollAndClickEnv", "experts.scroll_and_click", "run_expert_episode"),
    "copy-paste": ("tasks.copy_paste", "CopyPasteEnv", "experts.copy_paste", "run_expert_episode"),
}


def get_env_class(task_name: str) -> type:
    """Dynamically import and return the env class for *task_name*.

    Raises:
        KeyError: If *task_name* is not in the registry.
    """
    task_mod, task_cls, _, _ = _TASK_MAP[task_name]
    mod = importlib.import_module(f".{task_mod}", package="experiments.miniwob_pygame")
    return getattr(mod, task_cls)


def get_expert_fn(task_name: str) -> Callable[..., Any]:
    """Dynamically import and return the expert function for *task_name*.

    Raises:
        KeyError: If *task_name* is not in the registry.
    """
    _, _, expert_mod, expert_fn = _TASK_MAP[task_name]
    mod = importlib.import_module(f".{expert_mod}", package="experiments.miniwob_pygame")
    return getattr(mod, expert_fn)
