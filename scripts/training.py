"""Notebook definitions for training."""

from .drift import ChangePointDetector
from .agent import DQNAgent
from .ewc import EWC
from .memory import EpisodicMemory
from .baselines import GreedyMyopicBaseline
from .mec import MECServer
from .baselines import RandomBaseline
from .memory import ReplayBuffer
from .environments import Task1Env
from .environments import Task2Env
from .environments import Task3Env
from .environments import Task4Env
from .environments import Task5Env
from .environments import Task6Env
from .environments import Task7Env
from .continual_metrics import calculate_continual_metrics
import copy
from .utils import count_trainable_parameters
from .baselines import evaluate_baseline
from .evaluation import evaluate_policy_on_env
from .reporting import generate_metric_report
from .utils import load_rollback_checkpoint
from .adaptation import measure_adaptation_speed
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from .evaluation import run_evaluation
from .utils import save_checkpoint
from .utils import save_rollback_checkpoint
from .utils import set_global_seed
from .utils import setup_logger
import torch


def main():
    logger = setup_logger()
    logger.info("Initializing Continual RL Cognitive Radio Framework...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")


    EXPERIMENT_SEED = 42
    set_global_seed(EXPERIMENT_SEED)
    logger.info(f"Experiment seed: {EXPERIMENT_SEED}")


    envs = [Task1Env(), Task2Env(), Task3Env(), Task4Env(),
            Task5Env(), Task6Env(), Task7Env()]
    state_dim  = envs[0].state_dim
    output_dim = envs[0].action_space


    BATCH_SIZE       = 64
    GAMMA            = 0.97
    TARGET_UPDATE    = 1000
    ROLLBACK_INTERVAL = 100
    TASK_STEPS       = 10_000


    agent          = DQNAgent(state_dim, output_dim, device=device)
    active_buffer  = ReplayBuffer(capacity=10_000)
    episodic_memory = EpisodicMemory(capacity=2_000)
    ewc            = EWC(agent.policy_net)
    cpd            = ChangePointDetector(delta=0.002)
    mec            = MECServer(device=device)
    mec.start()


    global_step   = 0
    R_matrix      = []
    training_log  = []
    pre_task_scores = [None] * len(envs)
    task_boundary_steps = [i * TASK_STEPS for i in range(1, len(envs))]

    rollback_path = "checkpoints/rollback.pt"
    os.makedirs("checkpoints", exist_ok=True)

    try:
        for current_true_task_idx, env in enumerate(envs):
            logger.info(f"\n--- Environment → Task {current_true_task_idx + 1} ---")

            # Measures the current task before training to obtain its zero-shot score.

            if current_true_task_idx > 0:
                pre_eval = evaluate_policy_on_env(agent, env, num_episodes=50)
                pre_task_scores[current_true_task_idx] = pre_eval['reward_mean']
                logger.info(
                    f"Pre-task zero-shot T{current_true_task_idx + 1}: "
                    f"{pre_eval['reward_mean']:.3f} ± {pre_eval['reward_std']:.3f}"
                )

            state = env.reset()

            for step in range(TASK_STEPS):
                global_step += 1

                # Polls the MEC worker for completed Fisher estimates.
                fisher, theta_star = mec.check_results()
                if fisher is not None:
                    logger.info("MEC FIM received → injecting EWC penalty.")
                    ewc.update_task_penalty(fisher, theta_star)


                action = agent.get_action(state)


                next_state, reward, done, _ = env.step(action)
                active_buffer.push(state, action, reward, next_state, done)


                if len(active_buffer) > BATCH_SIZE:
                    num_episodic = int(BATCH_SIZE * 0.2)
                    num_active   = BATCH_SIZE - num_episodic

                    if len(episodic_memory) > num_episodic:
                        s_a, a_a, r_a, ns_a, d_a = active_buffer.sample(num_active)
                        s_e, a_e, r_e, ns_e, d_e = episodic_memory.sample(num_episodic)
                        states      = torch.cat([s_a,  s_e]).to(device)
                        actions_t   = torch.cat([a_a,  a_e]).to(device)
                        rewards_t   = torch.cat([r_a,  r_e]).to(device)
                        next_states = torch.cat([ns_a, ns_e]).to(device)
                        dones_t     = torch.cat([d_a,  d_e]).to(device)
                    else:
                        states, actions_t, rewards_t, next_states, dones_t = active_buffer.sample(BATCH_SIZE)
                        states, actions_t, rewards_t = states.to(device), actions_t.to(device), rewards_t.to(device)
                        next_states, dones_t         = next_states.to(device), dones_t.to(device)


                    q_values   = agent.policy_net(states)
                    q_expected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next    = agent.target_net(next_states).max(1)[0]
                        q_targets = rewards_t + GAMMA * q_next * (1 - dones_t)

                    td_errors_arr     = (q_expected - q_targets).detach().cpu().numpy()
                    current_td_error  = float(np.mean(np.abs(td_errors_arr)))

                    # Combines the Bellman error with the adaptive EWC penalty.
                    dqn_loss = nn.MSELoss()(q_expected, q_targets)
                    ewc_loss = ewc.compute_ewc_loss()
                    loss     = dqn_loss + ewc_loss

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=10.0)
                    agent.optimizer.step()

                    ewc.update_td_error(current_td_error)


                    training_log.append({
                        'global_step': global_step,
                        'task':        current_true_task_idx + 1,
                        'dqn_loss':    float(dqn_loss.item()),
                        'ewc_loss':    float(ewc_loss.item()),
                        'total_loss':  float(loss.item()),
                        'lambda_t':    ewc.get_adaptive_lambda(),
                        'td_error':    current_td_error,
                        'tau':         agent.tau,
                    })


                    boundary_detected, b_index = cpd.update(current_td_error, global_step=global_step)

                    if boundary_detected:
                        logger.info(f"ADWIN drift detected | estimated boundary: step {b_index}")

                        if load_rollback_checkpoint(agent.policy_net, rollback_path):
                            logger.info("Weight rollback applied.")
                            agent.update_target_network()

                        top_k = active_buffer.get_top_k_transitions(
                            k=286, policy_net=agent.policy_net,
                            target_net=agent.target_net, gamma=GAMMA, device=device
                        )
                        episodic_memory.add_transitions(top_k)
                        logger.info(f"Episodic memory: +{len(top_k)} transitions.")

                        frozen_net = copy.deepcopy(agent.policy_net).cpu()
                        mec.submit_task(list(active_buffer.buffer), frozen_net)
                        logger.info("FIM computation dispatched to MEC.")

                        active_buffer.flush()
                        cpd.reset(reset_step_counter=False)
                        agent.reset_temperature()
                        ewc.reset_td_window()

                # Copies policy weights to the target network at the configured interval.
                if global_step % TARGET_UPDATE == 0:
                    agent.update_target_network()

                # Saves the policy snapshot used by drift-triggered rollback.
                if global_step % ROLLBACK_INTERVAL == 0:
                    save_rollback_checkpoint(agent.policy_net, rollback_path)

                state = env.reset() if done else next_state


            logger.info(f"\n=== IEEE Evaluation Protocol: After Task {current_true_task_idx + 1} ===")
            mean_rewards, collision_rates, transmission_metrics, extended_metrics = run_evaluation(
                agent, envs, current_true_task_idx, num_episodes=100
            )

            padded = mean_rewards + [0.0] * (len(envs) - len(mean_rewards))
            R_matrix.append(padded)

            cl_metrics = calculate_continual_metrics(R_matrix)

            logger.info(f"Mean Rewards : {[round(r, 2) for r in mean_rewards]}")
            logger.info(f"Collision Rates: {[round(c, 3) for c in collision_rates]}")
            if cl_metrics:
                logger.info(f"CL Metrics : { {k: round(v, 4) for k, v in cl_metrics.items()} }")

            # Saves the task evaluation matrix with the current checkpoint.
            df = pd.DataFrame(R_matrix, columns=[f"T{i+1}" for i in range(len(envs))])
            save_checkpoint(
                task_id=current_true_task_idx + 1,
                policy_net=agent.policy_net,
                target_net=agent.target_net,
                fisher_dict=None,
                theta_star_dict=None,
                episodic_memory=episodic_memory,
                metrics_df=df
            )

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    finally:
        mec.stop()
        logger.info("MEC server shut down. Training complete.")

    # Persists per-step diagnostics for reproducibility and figures.
    if training_log:
        pd.DataFrame(training_log).to_csv("checkpoints/training_log.csv", index=False)


    logger.info("\n=== Generating Full IEEE Metric Report ===")

    adwin_stats = cpd.get_detection_stats(
        true_boundaries=task_boundary_steps,
        tolerance_steps=max(500, int(0.20 * TASK_STEPS))
    )
    logger.info(f"ADWIN Stats: {adwin_stats}")

    final_mean_rewards, final_collision_rates, final_tx_metrics, final_ext_metrics = run_evaluation(
        agent, envs, len(envs) - 1, num_episodes=200
    )

    # Evaluates baselines before computing forward transfer against the random reference.
    logger.info("\n=== Evaluating Baseline Agents ===")
    random_baseline = RandomBaseline(action_space=envs[0].action_space)
    greedy_baseline = GreedyMyopicBaseline(
        num_channels=envs[0].num_channels,
        history_length=envs[0].history_length
    )

    random_results = evaluate_baseline(random_baseline, envs, num_episodes=200)
    greedy_results = evaluate_baseline(greedy_baseline, envs, num_episodes=200)

    final_cl_metrics = calculate_continual_metrics(
        R_matrix,
        pre_task_scores=pre_task_scores,
        random_baseline_scores=random_results['mean_rewards'],
        single_task_oracle_scores=None
    )


    adaptation_steps = []
    for i, env in enumerate(envs):
        ref = R_matrix[i][i]
        adaptation_steps.append(
            measure_adaptation_speed(
                agent, env,
                within_task_reference_episode_reward=ref,
                threshold_fraction=0.95,
                max_steps=5000,
                eval_window=100
            )
        )

    report_df = generate_metric_report(
        R_matrix=R_matrix,
        cl_metrics=final_cl_metrics,
        transmission_metrics=final_tx_metrics,
        extended_metrics=final_ext_metrics,
        adwin_stats=adwin_stats,
        training_log=training_log,
        adaptation_steps=adaptation_steps,
        save_dir="checkpoints"
    )

    logger.info(f"Random   Mean Rewards: {[round(r, 2) for r in random_results['mean_rewards']]}")
    logger.info(f"Greedy   Mean Rewards: {[round(r, 2) for r in greedy_results['mean_rewards']]}")
    logger.info(f"Proposed Mean Rewards: {[round(r, 2) for r in final_mean_rewards]}")

    baseline_df = pd.DataFrame({
        'task': [f'T{i+1}' for i in range(len(envs))],

        'random_reward_mean': random_results['mean_rewards'],
        'random_reward_std': random_results['reward_stds'],
        'random_reward_ci95_low': random_results['reward_ci95_low'],
        'random_reward_ci95_high': random_results['reward_ci95_high'],
        'random_collision': random_results['collision_rates'],
        'random_throughput': random_results['throughputs'],

        'greedy_reward_mean': greedy_results['mean_rewards'],
        'greedy_reward_std': greedy_results['reward_stds'],
        'greedy_reward_ci95_low': greedy_results['reward_ci95_low'],
        'greedy_reward_ci95_high': greedy_results['reward_ci95_high'],
        'greedy_collision': greedy_results['collision_rates'],
        'greedy_throughput': greedy_results['throughputs'],

        'proposed_reward_mean': final_mean_rewards,
        'proposed_reward_std': [m['reward_std'] for m in final_tx_metrics],
        'proposed_reward_ci95_low': [m['reward_ci95_low'] for m in final_tx_metrics],
        'proposed_reward_ci95_high': [m['reward_ci95_high'] for m in final_tx_metrics],
        'proposed_collision': final_collision_rates,
        'proposed_throughput': [m['successful_transmission_rate'] for m in final_tx_metrics],
    })


    eps = 1e-12
    baseline_df['reward_gain_vs_greedy_pct'] = 100.0 * (
        baseline_df['proposed_reward_mean'] - baseline_df['greedy_reward_mean']
    ) / baseline_df['greedy_reward_mean'].abs().clip(lower=eps)

    baseline_df['collision_reduction_vs_greedy_pct'] = 100.0 * (
        baseline_df['greedy_collision'] - baseline_df['proposed_collision']
    ) / baseline_df['greedy_collision'].clip(lower=eps)

    baseline_df['throughput_gain_vs_greedy_pct'] = 100.0 * (
        baseline_df['proposed_throughput'] - baseline_df['greedy_throughput']
    ) / baseline_df['greedy_throughput'].clip(lower=eps)

    baseline_csv = "checkpoints/baseline_comparison.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    logger.info(f"Baseline comparison saved → {baseline_csv}")


    complexity_df = pd.DataFrame([{
        'seed': EXPERIMENT_SEED,
        'device': str(device),
        'trainable_parameters': count_trainable_parameters(agent.policy_net),
        'task_steps': TASK_STEPS,
        'num_tasks': len(envs),
        'evaluation_episodes_final': 200,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'target_update_steps': TARGET_UPDATE,
        'adwin_delta': cpd.delta,
    }])
    complexity_df.to_csv("checkpoints/experiment_metadata.csv", index=False)


    print("\n" + "=" * 92)
    print("PUBLICATION-READY RESULT SUMMARY")
    print("=" * 92)
    print(f"Final average performance (ACC_final): {final_cl_metrics.get('ACC_final', np.nan):.3f}")
    print(f"Diagonal learning score:              {final_cl_metrics.get('ACC_diagonal', np.nan):.3f}")
    print(f"BWT:                                  {final_cl_metrics.get('BWT', np.nan):.3f}")
    print(f"Mean normalized forgetting:           {final_cl_metrics.get('mean_forgetting_pct', np.nan):.3f}%")
    print(f"FWT vs random baseline:               {final_cl_metrics.get('FWT_mean', np.nan):.3f}")
    print(f"ADWIN detection rate:                 {adwin_stats.get('detection_rate', np.nan):.3f}")
    print(f"ADWIN false-positive rate:            {adwin_stats.get('false_positive_rate', np.nan):.3f}")
    print(f"ADWIN false-negative rate:            {adwin_stats.get('false_negative_rate', np.nan):.3f}")
    print(f"ADWIN mean detection delay:           {adwin_stats.get('mean_detection_delay', np.nan):.1f} steps")
    print("\nNOTE: true physical spectral efficiency (bit/s/Hz) is NOT claimed because this")
    print("environment has no bandwidth / payload-bit-rate / link-SINR model. The notebook")
    print("reports normalized access efficiency instead, avoiding a duplicate/mislabelled metric.")
    print("=" * 92)

    return report_df, baseline_df
