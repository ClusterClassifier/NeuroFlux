"""Training-seed statistics and paired ablation comparisons."""

import json

from .io import _v1_atomic_json


def _v1_aggregate(root, plan, cfg):
    import numpy as np
    import pandas as pd
    from scipy.stats import t

    records, coverage = [], []
    for row in plan:
        run_dir = root / 'runs' / row['variant'] / ('seed_' + str(row['seed']))
        status_path = run_dir / 'status.json'
        state = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'not_started'}
        coverage.append(dict(**row, status=state['status']))
        if state['status'] != 'complete':
            continue
        report = pd.read_csv(run_dir / 'checkpoints' / 'ieee_full_metric_report.csv')
        summary = report.loc[report.task == 'SUMMARY'].iloc[0]
        tasks = report.loc[report.task != 'SUMMARY']
        baseline = pd.read_csv(run_dir / 'checkpoints' / 'baseline_comparison.csv')
        records.append(dict(**row, ACC_final=summary['R_final'], BWT=summary['bwt'],
                            mean_forgetting_pct=summary['forgetting_pct'],
                            mean_collision_rate=tasks['collision_rate'].mean(),
                            mean_throughput=tasks['throughput'].mean(),
                            final_policy_eval_reward=baseline['proposed_reward_mean'].mean(),
                            detector_false_discovery_fraction=summary.get('adwin_false_positive_rate', np.nan),
                            detector_mean_delay=summary.get('adwin_mean_delay', np.nan)))
    tables = root / 'tables'
    tables.mkdir(exist_ok=True)
    pd.DataFrame(coverage).to_csv(tables / 'coverage.csv', index=False)
    raw = pd.DataFrame(records)
    raw.to_csv(tables / 'per_seed.csv', index=False)
    if raw.empty:
        return raw
    metrics = [c for c in raw if c not in {'variant', 'seed'}]

    def statistics(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
        return dict(n=n, mean=mean, std=sd, ci95_low=mean-half, ci95_high=mean+half)

    summaries = [dict(variant=name, metric=metric, **statistics(group[metric]))
                 for name, group in raw.groupby('variant', sort=False) for metric in metrics]
    pd.DataFrame(summaries).to_csv(tables / 'multiseed_summary.csv', index=False)
    paired = []
    control = raw.loc[raw.variant == 'full_v1'].set_index('seed')
    for name, group in raw.loc[raw.variant != 'full_v1'].groupby('variant', sort=False):
        other = group.set_index('seed')
        common = control.index.intersection(other.index)
        for metric in metrics:
            stats = statistics(control.loc[common, metric].to_numpy() - other.loc[common, metric].to_numpy())
            paired.append(dict(variant=name, metric=metric, difference='full_v1_minus_variant', **stats))
    pd.DataFrame(paired).to_csv(tables / 'paired_comparisons.csv', index=False)
    _v1_atomic_json(tables / 'metric_notes.json', {
        'ACC_final': 'Mean final row of the original task matrix; after-task evaluation budget.',
        'final_policy_eval_reward': 'Mean of the separate original final-policy evaluation across tasks.',
        'BWT': 'Original final-row minus diagonal mean over prior tasks.',
        'mean_collision_rate': 'Unweighted task mean; collision / transmission attempts.',
        'confidence_intervals': 'Unadjusted Student-t intervals across independent training seeds; three seeds are exploratory.',
        'paired_comparisons': 'Full V1 minus each ablation, on shared completed seeds; no significance claim is generated.',
        'adaptation': 'Original frozen-policy threshold diagnostic remains in original reports; not aggregated as learning recovery time.',
        'execution': 'Original asynchronous MEC thread and original evaluation RNG behavior are retained.'
    })
    if cfg['make_plots']:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
            variants = raw.variant.unique().tolist()
            for ax, metric in zip(axes, ['ACC_final', 'BWT', 'mean_collision_rate']):
                values = [raw.loc[raw.variant == v, metric].to_numpy(float) for v in variants]
                means = [np.mean(x) for x in values]
                stds = [np.std(x, ddof=1) if len(x) > 1 else 0 for x in values]
                ax.bar(range(len(variants)), means, yerr=stds, capsize=3, color='#52759B', alpha=.65)
                for i, vals in enumerate(values):
                    ax.scatter(i + np.linspace(-.12, .12, len(vals)), vals, s=15, color='black', zorder=3)
                ax.set_xticks(range(len(variants)), variants, rotation=65, ha='right')
                ax.set_title(metric)
                ax.grid(axis='y', alpha=.2)
            fig.suptitle('Neurone V1 ablations: training-seed points and mean ± sample SD')
            fig.tight_layout()
            fig.savefig(root / 'ablation_summary.png', dpi=180, bbox_inches='tight')
            plt.close(fig)
        except ImportError:
            print('Plot skipped: install matplotlib to enable it.')
    return raw
