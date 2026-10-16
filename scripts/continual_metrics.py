"""Notebook definitions for continual metrics."""

import numpy as np


def calculate_continual_metrics(
    R_matrix: list,
    pre_task_scores: list = None,
    random_baseline_scores: list = None,
    single_task_oracle_scores: list = None
) -> dict:
    """
    Publication-grade continual-learning metrics.

    R[k][j] = performance on task j after training through task k.

    Corrected definitions:
      Final Average Performance = mean_j R[K-1][j]
      Diagonal Learning Score   = mean_j R[j][j]
      Forgetting_j              = R[j][j] - R[K-1][j]
      BWT                       = mean_{j<K-1}(R[K-1][j] - R[j][j])
      FWT_j                     = pre_task_score[j] - random_baseline[j]

    FWT is reported as NaN when zero-shot pre-task evaluation was not measured;
    unevaluated future-task cells are never interpreted as zero transfer.
    Intransigence is reported only when a proper single-task oracle is supplied.
    """
    metrics = {}
    K = len(R_matrix)
    if K == 0:
        return metrics

    # Converts the observed task matrix entries to floating-point values.
    diag = [float(R_matrix[j][j]) for j in range(K)]
    final_row = [float(R_matrix[-1][j]) for j in range(K)]

    metrics['ACC_final'] = float(np.mean(final_row))
    metrics['ACC_diagonal'] = float(np.mean(diag))
    # Uses final average performance as ACC in downstream charts.
    metrics['ACC'] = metrics['ACC_final']


    fr_abs = []
    fr_pct = []
    for j in range(K - 1):
        f = diag[j] - final_row[j]
        metrics[f'FR_T{j+1}'] = float(f)
        denom = max(abs(diag[j]), 1e-12)
        fp = 100.0 * f / denom
        metrics[f'FRpct_T{j+1}'] = float(fp)
        fr_abs.append(f)
        fr_pct.append(fp)

    metrics['mean_forgetting'] = float(np.mean(fr_abs)) if fr_abs else 0.0
    metrics['mean_forgetting_pct'] = float(np.mean(fr_pct)) if fr_pct else 0.0


    if K > 1:
        bwt_vals = [final_row[j] - diag[j] for j in range(K - 1)]
        metrics['BWT'] = float(np.mean(bwt_vals))
        bwt_pct = [
            100.0 * (final_row[j] - diag[j]) / max(abs(diag[j]), 1e-12)
            for j in range(K - 1)
        ]
        metrics['BWT_pct'] = float(np.mean(bwt_pct))
    else:
        metrics['BWT'] = np.nan
        metrics['BWT_pct'] = np.nan

    # Computes forward transfer from measured pre-training task scores.
    fwt_values = []
    if pre_task_scores is not None and random_baseline_scores is not None:
        for j in range(1, min(K, len(pre_task_scores), len(random_baseline_scores))):
            pre = pre_task_scores[j]
            rnd = random_baseline_scores[j]
            if pre is not None and rnd is not None and np.isfinite(pre) and np.isfinite(rnd):
                fwt_j = float(pre - rnd)
                metrics[f'FWT_T{j+1}'] = fwt_j
                fwt_values.append(fwt_j)
        metrics['FWT_mean'] = float(np.mean(fwt_values)) if fwt_values else np.nan
    else:
        metrics['FWT_mean'] = np.nan

    # Computes intransigence only when an independent oracle is supplied.
    intransigence = []
    if single_task_oracle_scores is not None:
        for j in range(min(K, len(single_task_oracle_scores))):
            oracle = single_task_oracle_scores[j]
            if oracle is not None and np.isfinite(oracle):
                val = float(oracle - diag[j])
                metrics[f'Intransigence_T{j+1}'] = val
                intransigence.append(val)
        metrics['Intransigence_mean'] = float(np.mean(intransigence)) if intransigence else np.nan
    else:
        metrics['Intransigence_mean'] = np.nan


    metrics['stability_score'] = float(1.0 - max(0.0, metrics['mean_forgetting_pct']) / 100.0)

    return metrics


print("Continual learning metric suite defined.")
