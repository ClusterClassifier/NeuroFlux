# Metric Definitions

Let `R[k,j]` be mean 1,000-step episode reward on task `j` after training through task `k`, with seven tasks in total. The matrix uses the **100-episode after-task evaluations**; the separate 200-episode final-policy report is a different evaluation product.

| Metric | Definition | Unit or interpretation |
|---|---|---|
| Final average reward (`ACC_final`) | `mean_j R[7,j]` | Reward points per episode. |
| Backward transfer (`BWT`) | `mean_(j=1..6) [R[7,j] − R[j,j]]` | Points; negative means net earlier-task loss. |
| Mean normalized forgetting | `mean_(j=1..6) 100[R[j,j] − R[7,j]] / max(abs(R[j,j]), 1e−12)` | Percent; individual values may be negative. |
| Collision rate | `collisions / attempted transmissions` | Idle steps excluded. |
| Throughput proxy | `successful transmissions / all environment steps` | Successes per step, not bits/s or spectral efficiency. |
| Access-success proxy | `successful transmissions / attempted transmissions` | Dimensionless. |

The archive's detector “false-discovery fraction” is the share of alarms unmatched to known boundaries under its event-matching rule, not a generic per-step false-positive rate. Evaluation episodes measure a trained policy; **training seeds** are the independent replicates for condition comparisons. Do not treat the frozen-policy adaptation-speed diagnostic as learning recovery time.
