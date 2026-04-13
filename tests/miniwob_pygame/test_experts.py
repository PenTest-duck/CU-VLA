"""Tests for MiniWoB-Pygame expert policies."""

import numpy as np

from experiments.miniwob_pygame.experts.click_target import run_expert_episode
from experiments.miniwob_pygame.experts.use_slider import (
    run_expert_episode as run_slider_expert_episode,
)
from experiments.miniwob_pygame.tasks.click_target import ClickTargetEnv
from experiments.miniwob_pygame.tasks.use_slider import UseSliderEnv


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
