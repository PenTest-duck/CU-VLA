"""Tests for the task registry and multi-task data generation."""

from __future__ import annotations

import pytest

from experiments.miniwob_pygame.config import TASK_NAMES
from experiments.miniwob_pygame.task_registry import get_env_class, get_expert_fn


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_task_registered(task_name: str) -> None:
    """Every task in TASK_NAMES resolves to a real env class and expert fn."""
    cls = get_env_class(task_name)
    assert cls is not None

    fn = get_expert_fn(task_name)
    assert callable(fn)


@pytest.mark.parametrize("task_name", TASK_NAMES[:4])  # Phase 1 only for speed
def test_env_can_reset(task_name: str) -> None:
    """Phase-1 envs can be instantiated and reset."""
    cls = get_env_class(task_name)
    env = cls()
    obs = env.reset(seed=0)
    assert "screenshot" in obs
    assert "cursor_pos" in obs
    env.close()


def test_unknown_task_raises() -> None:
    """Requesting a non-existent task raises KeyError."""
    with pytest.raises(KeyError):
        get_env_class("nonexistent-task")
    with pytest.raises(KeyError):
        get_expert_fn("nonexistent-task")
