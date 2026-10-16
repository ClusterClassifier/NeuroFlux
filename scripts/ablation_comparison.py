"""Notebook definitions for ablation comparison."""

from IPython.display import display
import numpy as np
import os
import pandas as pd


def build_ablation_comparison(variant_dirs, output_csv="checkpoints/ablation_comparison.csv"):
    """
    variant_dirs example:
        {
            "Vanilla DQN": "runs/vanilla_seed42/checkpoints",
            "DQN+EWC": "runs/ewc_seed42/checkpoints",
            "DQN+ADWIN": "runs/adwin_seed42/checkpoints",
            "Proposed EWC+ADWIN": "checkpoints",
        }

    Every directory must contain ieee_full_metric_report.csv. Missing variants
    remain visibly missing rather than being inferred from the proposed run.
    """
    rows = []
    for variant, d in variant_dirs.items():
        p = os.path.join(d, "ieee_full_metric_report.csv")
        if not os.path.exists(p):
            print(f"Missing ablation run for {variant}: {p}")
            continue

        df = pd.read_csv(p)
        tasks = df[df['task'] != 'SUMMARY']
        summary = df[df['task'] == 'SUMMARY']
        if summary.empty:
            print(f"Missing SUMMARY row for {variant}: {p}")
            continue
        s = summary.iloc[0]

        rows.append({
            'variant': variant,
            'ACC_final': s.get('R_final', np.nan),
            'BWT': s.get('bwt', np.nan),
            'FWT': s.get('fwt', np.nan),
            'mean_forgetting_pct': s.get('forgetting_pct', np.nan),
            'mean_throughput': tasks['throughput'].mean() if 'throughput' in tasks else np.nan,
            'mean_collision_rate': tasks['collision_rate'].mean() if 'collision_rate' in tasks else np.nan,
            'mean_access_efficiency': (
                tasks['normalized_access_efficiency'].mean()
                if 'normalized_access_efficiency' in tasks else np.nan
            ),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out.to_csv(output_csv, index=False)
        print(f"Ablation comparison saved → {output_csv}")
        display(out)
    else:
        print("No ablation runs were available. No values were synthesized.")
    return out
