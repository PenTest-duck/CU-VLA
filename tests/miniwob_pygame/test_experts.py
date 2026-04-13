"""Tests for MiniWoB-Pygame expert policies."""

import numpy as np

from experiments.miniwob_pygame.experts.click_sequence import (
    run_expert_episode as run_click_sequence_expert_episode,
)
from experiments.miniwob_pygame.experts.click_target import run_expert_episode
from experiments.miniwob_pygame.experts.drag_sort import (
    run_expert_episode as run_drag_sort_expert_episode,
)
from experiments.miniwob_pygame.experts.drag_to_zone import (
    run_expert_episode as run_drag_to_zone_expert_episode,
)
from experiments.miniwob_pygame.experts.draw_path import (
    run_expert_episode as run_draw_path_expert_episode,
)
from experiments.miniwob_pygame.experts.highlight_text import (
    run_expert_episode as run_highlight_text_expert_episode,
)
from experiments.miniwob_pygame.experts.type_field import (
    run_expert_episode as run_type_field_expert_episode,
)
from experiments.miniwob_pygame.experts.use_slider import (
    run_expert_episode as run_slider_expert_episode,
)
from experiments.miniwob_pygame.tasks.click_sequence import ClickSequenceEnv
from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
from experiments.miniwob_pygame.tasks.drag_sort import DragSortEnv
from experiments.miniwob_pygame.tasks.drag_to_zone import DragToZoneEnv
from experiments.miniwob_pygame.tasks.draw_path import DrawPathEnv
from experiments.miniwob_pygame.tasks.highlight_text import HighlightTextEnv
from experiments.miniwob_pygame.tasks.type_field import TypeFieldEnv
from experiments.miniwob_pygame.tasks.use_slider import UseSliderEnv


class TestClickSequenceExpert:
    def test_expert_completes_task(self):
        env = ClickSequenceEnv(num_buttons=4)
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_click_sequence_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 18, f"Expert only succeeded {successes}/{n_episodes} times"


class TestClickTargetExpert:
    def test_expert_completes_task(self):
        env = ClickTargetEnv()
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 18, f"Expert only succeeded {successes}/{n_episodes} times"


class TestUseSliderExpert:
    def test_expert_completes_task(self):
        env = UseSliderEnv()
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_slider_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 15, f"Expert only succeeded {successes}/{n_episodes} times"


class TestDragToZoneExpert:
    def test_expert_completes_task(self):
        env = DragToZoneEnv(num_shapes=1)
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_drag_to_zone_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 18, f"Expert only succeeded {successes}/{n_episodes} times"


class TestTypeFieldExpert:
    def test_expert_completes_task(self):
        env = TypeFieldEnv()
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_type_field_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 19, f"Expert only succeeded {successes}/{n_episodes} times"


class TestHighlightTextExpert:
    def test_expert_completes_task(self):
        env = HighlightTextEnv()
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_highlight_text_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 14, f"Expert only succeeded {successes}/{n_episodes} times"


class TestDrawPathExpert:
    def test_expert_completes_task(self):
        env = DrawPathEnv(path_type="line")
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_draw_path_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 14, f"Expert only succeeded {successes}/{n_episodes} times"


class TestDragSortExpert:
    def test_expert_completes_task(self):
        env = DragSortEnv(num_cards=4)
        successes = 0
        n_episodes = 20
        for i in range(n_episodes):
            rng = np.random.default_rng(seed=i)
            _, _, info = run_drag_sort_expert_episode(env, rng, seed=i)
            if info.get("success"):
                successes += 1
        env.close()
        assert successes >= 15, f"Expert only succeeded {successes}/{n_episodes} times"
