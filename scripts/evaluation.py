"""Notebook definitions for evaluation."""

from .utils import mean_std_ci95
import numpy as np


def evaluate_policy_on_env(agent, env, num_episodes=200):
    """
    Frozen greedy-policy evaluation on one CRN task.

    Returns episode-level reward uncertainty plus CRN coexistence metrics.
    No training state is changed.
    """
    episode_rewards = []
    total_steps = 0
    total_idle_actions = 0
    total_transmission_attempts = 0
    total_successful_transmissions = 0
    total_collisions = 0
    action_counts = np.zeros(env.action_space, dtype=np.int64)

    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)

            total_steps += 1
            ep_reward += reward
            action_counts[action] += 1

            if action == env.idle_action:
                total_idle_actions += 1
            elif 0 <= action < env.num_channels:
                total_transmission_attempts += 1
                if reward == env.R_success:
                    total_successful_transmissions += 1
                elif reward == -env.R_collision:
                    total_collisions += 1

            state = next_state

        episode_rewards.append(ep_reward)

    reward_mean, reward_std, ci_low, ci_high = mean_std_ci95(episode_rewards)

    collision_rate = (
        total_collisions / total_transmission_attempts
        if total_transmission_attempts > 0 else 0.0
    )
    successful_transmission_rate = (
        total_successful_transmissions / total_steps
        if total_steps > 0 else 0.0
    )
    idle_rate = total_idle_actions / total_steps if total_steps > 0 else 0.0
    channel_utilization = (
        total_transmission_attempts / total_steps if total_steps > 0 else 0.0
    )
    transmission_success_rate = (
        total_successful_transmissions / total_transmission_attempts
        if total_transmission_attempts > 0 else 0.0
    )

    channel_actions = action_counts[:env.num_channels]
    ch_total = channel_actions.sum()
    if ch_total > 0:
        probs = channel_actions / ch_total
        probs = probs[probs > 0]
        channel_entropy = float(-np.sum(probs * np.log(probs)))
        normalized_channel_entropy = (
            channel_entropy / np.log(env.num_channels) if env.num_channels > 1 else 0.0
        )
    else:
        channel_entropy = 0.0
        normalized_channel_entropy = 0.0


    # Uses normalized access efficiency because the simulator has no physical link-rate model.


    normalized_access_efficiency = transmission_success_rate

    return {
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'reward_ci95_low': ci_low,
        'reward_ci95_high': ci_high,
        'num_eval_episodes': int(num_episodes),
        'total_steps': int(total_steps),
        'total_transmission_attempts': int(total_transmission_attempts),
        'total_successful_transmissions': int(total_successful_transmissions),
        'total_collisions': int(total_collisions),
        'collision_rate': float(collision_rate),
        'successful_transmission_rate': float(successful_transmission_rate),
        'idle_rate': float(idle_rate),
        'channel_utilization': float(channel_utilization),
        'transmission_success_rate': float(transmission_success_rate),
        'throughput': float(successful_transmission_rate),
        'pu_interference_rate': float(collision_rate),
        'normalized_access_efficiency': float(normalized_access_efficiency),
        'channel_entropy': float(channel_entropy),
        'normalized_channel_entropy': float(normalized_channel_entropy),
    }


def run_evaluation(agent, environments, current_task_id, num_episodes=500):
    """
    Evaluates the frozen greedy policy on all tasks seen so far.

    Backward-compatible return signature:
      mean_rewards, collision_rates, transmission_metrics, extended_metrics
    """
    mean_rewards = []
    collision_rates = []
    transmission_metrics = []
    extended_metrics = []

    for task_id in range(current_task_id + 1):
        env = environments[task_id]
        m = evaluate_policy_on_env(agent, env, num_episodes=num_episodes)

        mean_rewards.append(m['reward_mean'])
        collision_rates.append(m['collision_rate'])

        transmission_metrics.append({
            'task': f'T{task_id + 1}',
            # Stores collision rate separately from successful transmission rate.

            'collision_rate': m['collision_rate'],
            'successful_transmission_rate': m['successful_transmission_rate'],
            'idle_rate': m['idle_rate'],
            'channel_utilization': m['channel_utilization'],
            'transmission_success_rate': m['transmission_success_rate'],
            'reward_std': m['reward_std'],
            'reward_ci95_low': m['reward_ci95_low'],
            'reward_ci95_high': m['reward_ci95_high'],
            'num_eval_episodes': m['num_eval_episodes'],
            'total_steps': m['total_steps'],
            'total_transmission_attempts': m['total_transmission_attempts'],
            'total_successful_transmissions': m['total_successful_transmissions'],
            'total_collisions': m['total_collisions'],
        })

        extended_metrics.append({
            'task': f'T{task_id + 1}',
            'throughput': m['throughput'],
            'pu_interference_rate': m['pu_interference_rate'],
            'normalized_access_efficiency': m['normalized_access_efficiency'],
            'channel_entropy': m['channel_entropy'],
            'normalized_channel_entropy': m['normalized_channel_entropy'],
        })

    return mean_rewards, collision_rates, transmission_metrics, extended_metrics


print("Core evaluation runner defined.")
