"""Notebook definitions for adaptation."""

from collections import deque
import numpy as np


def measure_adaptation_speed(
    agent,
    env,
    within_task_reference_episode_reward: float,
    threshold_fraction: float = 0.95,
    max_steps: int = 5000,
    eval_window: int = 100
) -> int:
    """
    Number of greedy-evaluation steps required to recover to a chosen fraction
    of the reference reward.

    FIX: the old implementation compared a rolling *per-step* reward (bounded
    near ±1) against ~95% of an *episode* reward (near 1000), which made recovery
    mathematically impossible. The reference is now normalized by env.max_steps.
    """
    reference_reward_per_step = within_task_reference_episode_reward / max(1, env.max_steps)
    target = threshold_fraction * reference_reward_per_step

    recent_rewards = deque(maxlen=eval_window)
    state = env.reset()

    for step in range(max_steps):
        action = agent.get_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        recent_rewards.append(float(reward))
        state = next_state if not done else env.reset()

        if len(recent_rewards) == eval_window and np.mean(recent_rewards) >= target:
            return step + 1

    return max_steps


print("Adaptation speed metric defined.")
