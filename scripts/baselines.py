"""Notebook definitions for baselines."""

from .utils import mean_std_ci95
import numpy as np
import random


class RandomBaseline:
    """Uniform random action baseline (including idle)."""
    def __init__(self, action_space: int):
        self.action_space = action_space

    def get_action(self, state, evaluate: bool = False) -> int:
        return random.randint(0, self.action_space - 1)


class GreedyMyopicBaseline:
    """
    Chooses the channel with the lowest observed occupancy over the state history.
    This is a non-learning domain heuristic.
    """
    def __init__(self, num_channels: int, history_length: int):
        self.num_channels = num_channels
        self.history_length = history_length

    def get_action(self, state, evaluate: bool = False) -> int:
        occ_means = []
        for c in range(self.num_channels):
            offset = c * 2 * self.history_length + self.history_length
            occ_hist = state[offset: offset + self.history_length]
            occ_means.append(np.mean(occ_hist))
        return int(np.argmin(occ_means))


def evaluate_baseline(baseline, environments, num_episodes: int = 200) -> dict:
    """
    Evaluates a non-learning baseline on all tasks with the exact same episode
    horizon and CRN metric definitions used for the proposed agent.
    """
    results = {
        'mean_rewards': [], 'reward_stds': [], 'reward_ci95_low': [],
        'reward_ci95_high': [], 'collision_rates': [], 'throughputs': [],
        'normalized_access_efficiency': []
    }

    for env in environments:
        ep_rewards = []
        total_steps = total_collisions = total_successes = total_attempts = 0

        for _ in range(num_episodes):
            state = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action = baseline.get_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                total_steps += 1

                if 0 <= action < env.num_channels:
                    total_attempts += 1
                    if reward == env.R_success:
                        total_successes += 1
                    elif reward == -env.R_collision:
                        total_collisions += 1

                state = next_state

            ep_rewards.append(ep_reward)

        mean, std, lo, hi = mean_std_ci95(ep_rewards)
        results['mean_rewards'].append(mean)
        results['reward_stds'].append(std)
        results['reward_ci95_low'].append(lo)
        results['reward_ci95_high'].append(hi)
        results['collision_rates'].append(
            total_collisions / total_attempts if total_attempts else 0.0
        )
        results['throughputs'].append(
            total_successes / total_steps if total_steps else 0.0
        )
        results['normalized_access_efficiency'].append(
            total_successes / total_attempts if total_attempts else 0.0
        )

    return results


print("Baseline agents and statistically aligned evaluator defined.")
