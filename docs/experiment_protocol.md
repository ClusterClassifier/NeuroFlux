# Experiment Protocol

## Sequential task design

One run trains T1–T7 for `10,000` interactions each (`70,000` total). Each task has eight channels and a task-specific base occupancy range; the generator also makes one channel relatively favorable.

| Task | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base range | .20–.40 | .60–.90 | .40–.60 | .05–.20 | .65–.85 | .80–.95 | .01–.10 |

Pre-task evaluation uses `50` episodes where required; after-task evaluations use `100` episodes per learned task and form the retention matrix; separate final-policy evaluation uses `200` episodes. Each evaluation episode has `1,000` steps and uses greedy action choice.

## Controlled comparisons

The reported matched comparison uses seeds `42`, `123`, and `2026` for `full_v1`, `fixed_ewc` (`λ=4`), `no_ewc` (no penalty), and `no_rollback` (no alarm-triggered policy restoration). Architecture, task order, training budget, and evaluation protocol are held fixed. Compare each intervention with full V1 **within seed** and summarize variation across independent seeds. The runner also defines replay interventions; consult the archive's coverage and per-run status before reporting them. This is a component ablation, not an external published-method benchmark.
