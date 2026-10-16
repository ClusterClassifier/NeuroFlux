# Results Status

The current paper reports four conditions evaluated on matched seeds `42`, `123`, and `2026`. Values are means ± sample standard deviations across independent training seeds; reward and BWT are points per 1,000-step episode.

| Condition | Final reward | BWT | Mean forgetting | Collision rate |
|---|---:|---:|---:|---:|
| Full Neuroflux | 913.76 ± 4.28 | −25.22 ± 22.10 | 2.47 ± 2.99% | 2.15 ± 1.67% |
| Fixed EWC | 927.26 ± 10.78 | −18.85 ± 15.35 | 2.22 ± 1.87% | 2.94 ± 1.14% |
| No EWC | 708.77 ± 2.13 | −245.98 ± 11.54 | 29.19 ± 1.51% | 12.29 ± 0.37% |
| No rollback | 915.41 ± 14.65 | −26.63 ± 19.21 | 2.94 ± 2.11% | 1.65 ± 1.24% |

Relative to full V1, no EWC loses `204.99` final-reward points on average over paired seeds and has more forgetting and collisions. Fixed EWC's paired reward difference is `+13.50`; no rollback's is `+1.65`, with mixed signs across seeds for rollback. These descriptive results support the EWC contrast in this synthetic protocol; they do not establish general superiority of coefficient adaptation or a reward benefit from rollback.

Numbers come from `neurone_v1_ablation_results.zip` and use the **after-task reward matrix** for final reward/BWT/forgetting. Other variant results require their own status and sample-size check before publication. No common-benchmark comparison with an external paper is reported here.
