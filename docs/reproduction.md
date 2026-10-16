# Reproduction

Use the exact Neurone V1 notebook **with the appended ablation runner**, or the `run_v1_variant.py` source and `configuration.json` saved inside each archived run. Do not substitute scripts from a different NeuroFlux iteration. The archived top-level configuration records seeds `42`, `123`, `2026`, seven tasks, `10,000` steps per task, batch `64`, and evaluation counts of `50` pre-task, `100` after-task, and `200` final-policy episodes.

1. Create a Python environment with the dependency versions required by the selected notebook/source snapshot; record Python, PyTorch, NumPy, operating system, device, and determinism settings.
2. Run the notebook's implementation and ablation cells in their documented order, selecting the desired variant and seed. A full variant run has `70,000` training interactions plus evaluations; use a distinct output directory per run.
3. Retain each run's configuration, source worker, status, log, checkpoints, and metric CSVs. The archived runner configuration identifies source SHA-256 `71db701c7a31c84dffa794be0aba056db230464014757b928bd734427d3d12b5`.
4. Build summary tables from completed per-run reports and check `ACC_final` and BWT against the saved task reward matrices. Never mix matrix metrics with the separate final-policy evaluation.

The reference result archive is `neurone_v1_ablation_results.zip`. Publish a concrete repository revision and exact dependency lockfile before claiming independent bitwise reproduction. This document intentionally supplies no invented shell entry point for the notebook.
