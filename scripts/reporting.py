"""Notebook definitions for reporting."""

import numpy as np
import os
import pandas as pd


def generate_metric_report(
    R_matrix: list,
    cl_metrics: dict,
    transmission_metrics: list,
    extended_metrics: list,
    adwin_stats: dict,
    training_log: list,
    adaptation_steps: list = None,
    save_dir: str = "checkpoints"
) -> pd.DataFrame:
    """
    Consolidates task-level and global metrics into publication-ready CSV output.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tasks = len(transmission_metrics)
    rows = []

    for t_idx in range(num_tasks):
        label = f'T{t_idx + 1}'
        tm = transmission_metrics[t_idx] if t_idx < len(transmission_metrics) else {}
        em = extended_metrics[t_idx] if t_idx < len(extended_metrics) else {}

        row = {
            'task': label,
            'R_self': R_matrix[t_idx][t_idx] if t_idx < len(R_matrix) else np.nan,
            'R_final': R_matrix[-1][t_idx] if R_matrix else np.nan,
            'reward_std': tm.get('reward_std', np.nan),
            'reward_ci95_low': tm.get('reward_ci95_low', np.nan),
            'reward_ci95_high': tm.get('reward_ci95_high', np.nan),
            'num_eval_episodes': tm.get('num_eval_episodes', np.nan),

            'forgetting_abs': cl_metrics.get(f'FR_{label}', np.nan),
            'forgetting_pct': cl_metrics.get(f'FRpct_{label}', np.nan),
            'fwt': cl_metrics.get(f'FWT_{label}', np.nan),
            'intransigence': cl_metrics.get(f'Intransigence_{label}', np.nan),

            # Reports the measured collision rate.
            'collision_rate': tm.get('collision_rate', np.nan),
            'successful_transmission_rate': tm.get('successful_transmission_rate', np.nan),
            'idle_rate': tm.get('idle_rate', np.nan),
            'channel_utilization': tm.get('channel_utilization', np.nan),
            'transmission_success_rate': tm.get('transmission_success_rate', np.nan),

            'throughput': em.get('throughput', np.nan),
            'pu_interference_rate': em.get('pu_interference_rate', np.nan),
            'normalized_access_efficiency': em.get('normalized_access_efficiency', np.nan),
            'channel_entropy_nats': em.get('channel_entropy', np.nan),
            'normalized_channel_entropy': em.get('normalized_channel_entropy', np.nan),

            'total_steps': tm.get('total_steps', np.nan),
            'total_transmission_attempts': tm.get('total_transmission_attempts', np.nan),
            'total_successful_transmissions': tm.get('total_successful_transmissions', np.nan),
            'total_collisions': tm.get('total_collisions', np.nan),
            'adaptation_steps_95pct': (
                adaptation_steps[t_idx] if adaptation_steps is not None and t_idx < len(adaptation_steps)
                else np.nan
            ),
        }
        rows.append(row)

    report_df = pd.DataFrame(rows)

    summary = {
        'task': 'SUMMARY',
        'R_final': cl_metrics.get('ACC_final', np.nan),
        'R_self': cl_metrics.get('ACC_diagonal', np.nan),
        'forgetting_abs': cl_metrics.get('mean_forgetting', np.nan),
        'forgetting_pct': cl_metrics.get('mean_forgetting_pct', np.nan),
        'fwt': cl_metrics.get('FWT_mean', np.nan),
        'intransigence': cl_metrics.get('Intransigence_mean', np.nan),
        'bwt': cl_metrics.get('BWT', np.nan),
        'bwt_pct': cl_metrics.get('BWT_pct', np.nan),
        'stability_score': cl_metrics.get('stability_score', np.nan),

        'adwin_detection_rate': adwin_stats.get('detection_rate', np.nan),
        'adwin_false_positive_rate': adwin_stats.get('false_positive_rate', np.nan),
        'adwin_false_negative_rate': adwin_stats.get('false_negative_rate', np.nan),
        'adwin_mean_delay': adwin_stats.get('mean_detection_delay', np.nan),
        'adwin_std_delay': adwin_stats.get('std_detection_delay', np.nan),
        'adwin_median_delay': adwin_stats.get('median_detection_delay', np.nan),
        'adwin_tp': adwin_stats.get('true_positive_count', np.nan),
        'adwin_fp': adwin_stats.get('false_positive_count', np.nan),
        'adwin_fn': adwin_stats.get('false_negative_count', np.nan),
    }

    if training_log:
        log_df = pd.DataFrame(training_log)
        for key in ['dqn_loss', 'ewc_loss', 'total_loss', 'lambda_t', 'td_error']:
            if key in log_df:
                summary[f'mean_{key}'] = float(log_df[key].mean())
                summary[f'std_{key}'] = float(log_df[key].std(ddof=1))

    report_df = pd.concat([report_df, pd.DataFrame([summary])], ignore_index=True)

    csv_path = os.path.join(save_dir, "ieee_full_metric_report.csv")
    report_df.to_csv(csv_path, index=False)


    notes = pd.DataFrame([
        ['throughput', 'successful transmissions / all environment steps', 'normalized per-step rate'],
        ['collision_rate', 'collisions / transmission attempts', 'also PU interference rate in this single-SU simulator'],
        ['normalized_access_efficiency', 'successful transmissions / transmission attempts', 'dimensionless proxy; NOT physical bit/s/Hz spectral efficiency'],
        ['channel_entropy_nats', 'Shannon entropy of channel-action distribution', 'nats'],
        ['normalized_channel_entropy', 'entropy / ln(number of channels)', '0..1'],
        ['ACC_final', 'mean final-row reward across all tasks', 'continual-learning final performance'],
        ['BWT', 'mean(final reward - reward after own-task learning)', 'negative indicates forgetting'],
    ], columns=['metric', 'definition', 'publication_note'])
    notes.to_csv(os.path.join(save_dir, "metric_definitions.csv"), index=False)

    print(f"IEEE metric report saved → {csv_path}")
    print(f"Metric definitions saved → {os.path.join(save_dir, 'metric_definitions.csv')}")
    return report_df


print("IEEE metric report generator defined.")
