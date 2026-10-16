"""Notebook definitions for charts."""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def apply_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    sns.set_style("whitegrid")


def generate_ieee_charts(
    rmatrix_csv: str = "checkpoints/metrics.csv",
    full_report_csv: str = "checkpoints/ieee_full_metric_report.csv",
    baseline_csv: str = "checkpoints/baseline_comparison.csv",
    training_log_csv: str = "checkpoints/training_log.csv",
    output_dir: str = "charts"
):
    """
    Publication figures using corrected metric definitions.
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)
    colors = sns.color_palette("husl", 7)

    def save(name):
        path = os.path.join(output_dir, name)
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


    if os.path.exists(rmatrix_csv):
        df = pd.read_csv(rmatrix_csv)
        tasks = [f"T{i+1}" for i in range(len(df.columns))]
        plt.figure(figsize=(7.2, 5.2))
        # Masks future tasks because their zero entries represent missing evaluations.
        arr = df.to_numpy(dtype=float)
        mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
        sns.heatmap(
            df, mask=mask, annot=True, fmt=".1f", cmap="YlGnBu",
            cbar_kws={'label': 'Mean episode reward'},
            xticklabels=tasks, yticklabels=tasks
        )
        plt.title(r'Continual-Learning Evaluation Matrix ($R_{k,j}$)')
        plt.xlabel(r'Evaluated on Task $j$')
        plt.ylabel(r'After Training Task $k$')
        plt.tight_layout()
        save("fig1_evaluation_matrix_heatmap.pdf")


    if os.path.exists(rmatrix_csv):
        df = pd.read_csv(rmatrix_csv)
        tasks = [f"T{i+1}" for i in range(len(df.columns))]
        plt.figure(figsize=(7.2, 4.4))
        for i, col in enumerate(df.columns):
            y = df[col].iloc[i:]
            x = range(i + 1, len(df) + 1)
            plt.plot(x, y, marker='o', linewidth=1.5, label=f'Eval {col}', color=colors[i])
        plt.title('Task Performance Evolution and Retention')
        plt.xlabel(r'Training Phase (after Task $k$)')
        plt.ylabel('Mean episode reward')
        plt.xticks(range(1, len(tasks) + 1), tasks)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        save("fig2_performance_evolution.pdf")


    if os.path.exists(full_report_csv):
        rdf = pd.read_csv(full_report_csv)
        summary = rdf[rdf['task'] == 'SUMMARY']
        if not summary.empty:
            s = summary.iloc[0]
            metric_vals = {
                'Final Avg.': s.get('R_final', np.nan),
                'Diagonal Avg.': s.get('R_self', np.nan),
                'BWT': s.get('bwt', np.nan),
                'FWT': s.get('fwt', np.nan),
            }
            metric_vals = {k: v for k, v in metric_vals.items() if np.isfinite(v)}
            if metric_vals:
                fig, ax = plt.subplots(figsize=(6.6, 4.0))
                bars = ax.bar(metric_vals.keys(), metric_vals.values())
                ax.axhline(0, linewidth=0.8, linestyle='--')
                ax.bar_label(bars, fmt='%.2f', padding=3)
                ax.set_title('Continual-Learning Reward Metrics')
                ax.set_ylabel('Reward units')
                plt.tight_layout()
                save("fig3_cl_summary_metrics.pdf")


    if os.path.exists(full_report_csv):
        rdf = pd.read_csv(full_report_csv)
        rows = rdf[rdf['task'] != 'SUMMARY']
        if 'forgetting_pct' in rows:
            valid = rows.dropna(subset=['forgetting_pct'])
            if not valid.empty:
                plt.figure(figsize=(7.2, 3.8))
                plt.bar(valid['task'], valid['forgetting_pct'])
                plt.axhline(0, linewidth=0.8, linestyle='--')
                plt.title('Per-Task Normalized Forgetting')
                plt.xlabel('Task')
                plt.ylabel('Forgetting (%)')
                plt.tight_layout()
                save("fig4_per_task_forgetting_rate.pdf")


    if os.path.exists(full_report_csv):
        rdf = pd.read_csv(full_report_csv)
        rows = rdf[rdf['task'] != 'SUMMARY']
        if {'throughput', 'normalized_access_efficiency'}.issubset(rows.columns):
            x = np.arange(len(rows))
            width = 0.36
            fig, ax = plt.subplots(figsize=(7.4, 4.0))
            ax.bar(x - width/2, rows['throughput'], width, label='Throughput (successes/step)')
            ax.bar(x + width/2, rows['normalized_access_efficiency'], width, label='Access efficiency (successes/attempt)')
            ax.set_xticks(x)
            ax.set_xticklabels(rows['task'])
            ax.set_ylim(0, 1.05)
            ax.set_title('CRN Throughput and Normalized Access Efficiency')
            ax.set_xlabel('Task')
            ax.set_ylabel('Normalized rate')
            ax.legend()
            plt.tight_layout()
            save("fig5_throughput_access_efficiency.pdf")


    if os.path.exists(full_report_csv):
        rdf = pd.read_csv(full_report_csv)
        rows = rdf[rdf['task'] != 'SUMMARY']
        if {'pu_interference_rate', 'normalized_channel_entropy'}.issubset(rows.columns):
            fig, ax1 = plt.subplots(figsize=(7.4, 4.0))
            x = np.arange(len(rows))
            width = 0.36
            ax1.bar(x - width/2, rows['pu_interference_rate'], width, label='PU interference rate')
            ax1.bar(x + width/2, rows['normalized_channel_entropy'], width, label='Normalized channel entropy')
            ax1.set_xticks(x)
            ax1.set_xticklabels(rows['task'])
            ax1.set_ylim(0, 1.05)
            ax1.set_title('CRN Coexistence and Exploration Metrics')
            ax1.set_xlabel('Task')
            ax1.set_ylabel('Normalized value')
            ax1.legend()
            plt.tight_layout()
            save("fig6_pu_interference_channel_entropy.pdf")


    if os.path.exists(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        x = np.arange(len(bdf))
        width = 0.27
        fig, ax = plt.subplots(figsize=(8.2, 4.5))

        def ci_half(lo, hi):
            return (np.asarray(hi) - np.asarray(lo)) / 2.0

        ax.bar(
            x - width, bdf['random_reward_mean'], width, label='Random',
            yerr=ci_half(bdf['random_reward_ci95_low'], bdf['random_reward_ci95_high']),
            capsize=2
        )
        ax.bar(
            x, bdf['greedy_reward_mean'], width, label='Greedy/Myopic',
            yerr=ci_half(bdf['greedy_reward_ci95_low'], bdf['greedy_reward_ci95_high']),
            capsize=2
        )
        ax.bar(
            x + width, bdf['proposed_reward_mean'], width, label='Proposed (EWC+ADWIN)',
            yerr=ci_half(bdf['proposed_reward_ci95_low'], bdf['proposed_reward_ci95_high']),
            capsize=2
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bdf['task'])
        ax.set_title('Baseline Comparison — Mean Episode Reward (95% CI)')
        ax.set_xlabel('Task')
        ax.set_ylabel('Mean episode reward')
        ax.legend()
        plt.tight_layout()
        save("fig7_baseline_reward_comparison.pdf")


    if os.path.exists(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        x = np.arange(len(bdf))
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(x, bdf['random_collision'], marker='o', label='Random collision')
        ax.plot(x, bdf['greedy_collision'], marker='o', label='Greedy collision')
        ax.plot(x, bdf['proposed_collision'], marker='o', label='Proposed collision')
        ax.plot(x, bdf['proposed_throughput'], marker='s', linestyle='--', label='Proposed throughput')
        ax.set_xticks(x)
        ax.set_xticklabels(bdf['task'])
        ax.set_ylim(0, 1.05)
        ax.set_title('Baseline Coexistence Comparison')
        ax.set_xlabel('Task')
        ax.set_ylabel('Rate')
        ax.legend(ncol=2)
        plt.tight_layout()
        save("fig8_baseline_collision_throughput.pdf")


    if os.path.exists(training_log_csv):
        log_df = pd.read_csv(training_log_csv)
        if {'dqn_loss', 'ewc_loss', 'global_step'}.issubset(log_df.columns):
            window = 200
            fig, ax = plt.subplots(figsize=(8.2, 4.0))
            ax.plot(log_df['global_step'], log_df['dqn_loss'].rolling(window).mean(), label='DQN loss')
            ax.plot(log_df['global_step'], log_df['ewc_loss'].rolling(window).mean(), label='EWC penalty')
            ax.set_title(f'Training Loss Decomposition ({window}-step rolling mean)')
            ax.set_xlabel('Global training step')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.tight_layout()
            save("fig9_loss_decomposition.pdf")


    if os.path.exists(training_log_csv):
        log_df = pd.read_csv(training_log_csv)
        if {'lambda_t', 'global_step'}.issubset(log_df.columns):
            plt.figure(figsize=(8.2, 3.5))
            plt.plot(log_df['global_step'], log_df['lambda_t'], linewidth=0.8)
            plt.title(r'Adaptive EWC Coefficient $\lambda_t$')
            plt.xlabel('Global training step')
            plt.ylabel(r'$\lambda_t$')
            plt.tight_layout()
            save("fig10_adaptive_lambda_trajectory.pdf")


    if os.path.exists(training_log_csv):
        log_df = pd.read_csv(training_log_csv)
        if {'td_error', 'task', 'global_step'}.issubset(log_df.columns):
            plt.figure(figsize=(8.2, 4.0))
            for t_id in sorted(log_df['task'].unique()):
                subset = log_df[log_df['task'] == t_id]
                plt.plot(
                    subset['global_step'],
                    subset['td_error'].rolling(50).mean(),
                    label=f'T{int(t_id)}', linewidth=0.9
                )
            plt.title('Rolling Mean TD Error (ADWIN Input)')
            plt.xlabel('Global training step')
            plt.ylabel('Mean absolute TD error')
            plt.legend(ncol=4)
            plt.tight_layout()
            save("fig11_td_error_per_task.pdf")

    print(f"\nAll IEEE figures saved to '{output_dir}/' as 300-DPI PDFs.")
