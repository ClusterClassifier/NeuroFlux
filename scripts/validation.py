"""Notebook definitions for validation."""

from .utils import mean_std_ci95
import numpy as np
import os
import pandas as pd


def validate_publication_outputs(
    report_csv="checkpoints/ieee_full_metric_report.csv",
    baseline_csv="checkpoints/baseline_comparison.csv",
    rmatrix_csv="checkpoints/metrics.csv"
):
    """Fails loudly on common publication-threatening reporting mistakes."""
    issues = []

    if not os.path.exists(report_csv):
        issues.append(f"Missing {report_csv}")
    else:
        rdf = pd.read_csv(report_csv)
        tasks = rdf[rdf['task'] != 'SUMMARY'].copy()

        # Checks for an accidental collision and throughput column substitution.
        if {'collision_rate', 'successful_transmission_rate'}.issubset(tasks.columns):
            if np.allclose(
                tasks['collision_rate'].to_numpy(float),
                tasks['successful_transmission_rate'].to_numpy(float),
                equal_nan=False
            ):
                issues.append("collision_rate equals successful_transmission_rate for all tasks.")

        # Checks that reported probabilities and rates lie within their valid range.
        rate_cols = [
            'collision_rate', 'successful_transmission_rate', 'idle_rate',
            'channel_utilization', 'transmission_success_rate', 'throughput',
            'pu_interference_rate', 'normalized_access_efficiency',
            'normalized_channel_entropy'
        ]
        for col in rate_cols:
            if col in tasks:
                vals = pd.to_numeric(tasks[col], errors='coerce').dropna()
                if ((vals < -1e-9) | (vals > 1 + 1e-9)).any():
                    issues.append(f"{col} contains values outside [0,1].")

        # Checks metric identities against their definitions.
        if {'collision_rate', 'pu_interference_rate'}.issubset(tasks.columns):
            if not np.allclose(tasks['collision_rate'], tasks['pu_interference_rate'], atol=1e-12):
                issues.append("PU interference rate is inconsistent with collision rate in this single-SU model.")

    if os.path.exists(rmatrix_csv):
        m = pd.read_csv(rmatrix_csv).to_numpy(float)
        if m.shape[0] == m.shape[1] and os.path.exists(report_csv):
            rdf = pd.read_csv(report_csv)
            summary = rdf[rdf['task'] == 'SUMMARY']
            if not summary.empty:
                expected_final = float(np.mean(m[-1, :]))
                reported_final = float(summary.iloc[0]['R_final'])
                if not np.isclose(expected_final, reported_final, atol=1e-6):
                    issues.append(
                        f"ACC_final mismatch: matrix={expected_final:.6f}, report={reported_final:.6f}"
                    )

    if issues:
        print("PUBLICATION VALIDATION: FAILED")
        for x in issues:
            print(" -", x)
        raise AssertionError("Publication-output validation failed.")
    else:
        print("PUBLICATION VALIDATION: PASSED")
        print("No duplicate collision/success bug, range violation, or ACC-final mismatch detected.")


def aggregate_independent_seed_runs(run_directories, output_csv="checkpoints/multiseed_summary.csv"):
    """
    Aggregates final task metrics from independently trained runs.

    Each directory must contain ieee_full_metric_report.csv produced by this
    notebook. This intentionally does not pretend that episode-level CIs are
    equivalent to independent-seed uncertainty.
    """
    records = []
    for run_dir in run_directories:
        p = os.path.join(run_dir, "ieee_full_metric_report.csv")
        if not os.path.exists(p):
            print(f"Skipping missing run: {p}")
            continue
        df = pd.read_csv(p)
        tasks = df[df['task'] != 'SUMMARY']
        summary = df[df['task'] == 'SUMMARY']
        if summary.empty:
            continue
        s = summary.iloc[0]
        records.append({
            'run_dir': run_dir,
            'ACC_final': s.get('R_final', np.nan),
            'BWT': s.get('bwt', np.nan),
            'FWT': s.get('fwt', np.nan),
            'mean_forgetting_pct': s.get('forgetting_pct', np.nan),
            'mean_throughput': tasks['throughput'].mean() if 'throughput' in tasks else np.nan,
            'mean_collision_rate': tasks['collision_rate'].mean() if 'collision_rate' in tasks else np.nan,
        })

    raw = pd.DataFrame(records)
    if raw.empty:
        print("No independent seed runs were found. Run the notebook under multiple seeds first.")
        return raw, pd.DataFrame()

    summary_rows = []
    for col in [c for c in raw.columns if c != 'run_dir']:
        vals = pd.to_numeric(raw[col], errors='coerce').dropna().to_numpy()
        mean, std, lo, hi = mean_std_ci95(vals)
        summary_rows.append({
            'metric': col,
            'n_seeds': len(vals),
            'mean': mean,
            'std_across_seeds': std,
            'ci95_low': lo,
            'ci95_high': hi
        })

    summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"Multi-seed summary saved → {output_csv}")
    return raw, summary
