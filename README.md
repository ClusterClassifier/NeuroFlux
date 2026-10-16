# NeuroFlux

**Continual deep Q-learning for sequential channel access in a synthetic cognitive-radio network.**

NeuroFlux studies how a single secondary user can learn channel-access decisions across changing primary-user occupancy regimes while retaining useful behavior from earlier regimes. The implemented system combines a deep Q-network (DQN), active and episodic replay, elastic weight consolidation (EWC), temporal-difference (TD) error change monitoring, and alarm-triggered policy rollback. It is a research prototype evaluated in a controlled simulator.

## At a glance

| Item               | Implementation                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| Environment        | One secondary user, eight synthetic channels, one idle action               |
| Observation        | Five recent SNR and occupancy values per channel; 80 inputs                 |
| Policy             | Feed-forward target-network DQN; 80 → 256 → 256 → 9                         |
| Continual learning | Persistent episodic replay and variance-adaptive EWC                        |
| Change response    | Custom ADWIN-style monitor on batch TD error; optional snapshot restoration |
| Core study         | Seven sequential occupancy regimes, T1–T7                                   |
| Main outcomes      | Final reward, backward transfer, forgetting, collision rate                 |

The regime names describe illustrative synthetic conditions. Occupancy is sampled from task-specific Bernoulli probabilities, with one relatively favorable channel per task. The agent receives recent occupancy history directly; this version does not simulate sensing errors, competing secondary users, or physical-layer bit rates.

## How it works

At each step, NeuroFlux observes recent channel conditions, samples a channel or idle action during training, receives a reward, and updates its DQN from replay. A clear-channel transmission earns `+1`, a collision earns `−1`, and idling earns `−0.05`. Evaluations use greedy actions.

The active replay buffer stores recent transitions. An episodic buffer retains selected high-absolute-TD-error transitions for later tasks. EWC penalizes changes to parameters estimated to matter to previous tasks; its global coefficient varies with recent TD-error variance. The detector monitors mean absolute batch TD error and may trigger restoration of a recent policy snapshot plus memory, importance-estimation, and exploration resets. Importance estimation runs in a local background worker; it is not a deployed edge-computing service.

The [system overview](docs/system_overview.md), [methods](docs/methods.md), and [metric definitions](docs/metric_definitions.md) give the compact technical specification.

## Install

Use Python 3.10 or newer. From the repository root (the directory containing `pyproject.toml`, `src/`, and `configs/`):

```bash
python -m venv .venv
python -m pip install -e .
```

Activate the virtual environment before installing or running the project. The editable install exposes the existing `neurone` command from `src/neurone/`; **NeuroFlux** is the system and publication name. The entry point also works as `python -m neurone` after installation. PyTorch installation may require a platform-specific wheel for the desired CPU or CUDA environment.

## Run a smoke check

The small preset checks that the pipeline works; its outputs are development artifacts, not research results.

```bash
python scripts/run_smoke.py --config configs/experiments/smoke.yaml --output results/smoke --hours 0.1
```

Run the existing unit checks separately when changing implementation or metrics:

```bash
python -m unittest discover -s tests -v
```

## Run experiments

The package's `final.yaml` preset launches a **separate comparison matrix** of `vanilla_dqn`, `dqn_ewc`, `dqn_adwin`, and `neurone_full` over its configured seeds. It is not the four-condition archived table below. The preset uses 10,000 interactions per task, but its evaluation counts are **10 pre-task, 10 after-task, and 30 final-policy episodes**, unlike the archived study.

```bash
python run_all.py --config configs/experiments/final.yaml --output results/final --hours 10
```

The component-ablation suite has its own protocol and configuration. Its full preset uses five configured seeds and includes additional component controls; it must be analyzed independently of the archived results.

```bash
python scripts/run_ablation_suite.py --config configs/experiments/ablation_overnight.yaml --output results/ablation --hours 9.5
```

Use a **new output directory when changing source or configuration**. Existing completed jobs may be skipped when rerunning an identical matrix; a saved inspection checkpoint does not imply exact continuation of an interrupted training job. Run budgets are caps, not guarantees that every job will finish. See [experiment protocol](docs/experiment_protocol.md) and [reproduction](docs/reproduction.md) before using outputs in a manuscript.

## Archived matched study

The reported study used the original notebook-backed ablation runner, **not** the `final.yaml` matrix above. It trained seven tasks for 10,000 interactions each and formed its continual-learning reward matrix from 100 after-task evaluation episodes per task. The reported component comparisons share training seeds `42`, `123`, and `2026`. Reward and backward transfer (BWT) are reward points per 1,000-step evaluation episode; values are mean ± sample standard deviation across training seeds.

| Condition      | Final average reward | BWT             | Mean forgetting | Collision rate |
| -------------- | --------------------:| ---------------:| ---------------:| --------------:|
| Full NeuroFlux | 913.76 ± 4.28        | −25.22 ± 22.10  | 2.47 ± 2.99%    | 2.15 ± 1.67%   |
| Fixed EWC      | 927.26 ± 10.78       | −18.85 ± 15.35  | 2.22 ± 1.87%    | 2.94 ± 1.14%   |
| No EWC         | 708.77 ± 2.13        | −245.98 ± 11.54 | 29.19 ± 1.51%   | 12.29 ± 0.37%  |
| No rollback    | 915.41 ± 14.65       | −26.63 ± 19.21  | 2.94 ± 2.11%    | 1.65 ± 1.24%   |

Relative to the full configuration, removing EWC lowered final reward by **204.99 points** on average across matched seeds, alongside more forgetting and collisions. The fixed-EWC and no-rollback contrasts do not establish a general advantage for adaptive coefficient scheduling or rollback. These are within-simulator component comparisons, not scores on a shared benchmark with external published methods. See [results status](docs/results_status.md) for the concise interpretation.

The archived run package contains the exact per-run worker source, configuration, logs, status, checkpoints, and metric reports. Its top-level configuration identifies source SHA-256 `71db701c7a31c84dffa794be0aba056db230464014757b928bd734427d3d12b5`. A repository revision should be linked to that source snapshot before claiming a fresh package run reproduces the table. Large checkpoints and logs belong in a versioned release or data archive rather than ordinary Git history.

## Repository map

```text
.
├── README.md                 Project entry point
├── CITATION.cff              Citation metadata
├── pyproject.toml            Python package metadata
├── configs/                  Task and experiment presets
├── docs/                     Methods, metrics, protocol, results, paper plan
├── legacy_original/          Preserved source snapshot
├── scripts/                  Smoke, run, ablation, aggregation, figure helpers
├── src/neurone/              Maintained Python package
├── tests/                    Fast implementation checks
└── run_all.py                Matrix runner
```

Generated `results/` directories are created by runs and should be handled according to the archive and Git ignore policy. The [paper blueprint](docs/paper_blueprint.md) outlines an IEEE-style manuscript; an editable manuscript draft is maintained separately.

## Scope and interpretation

- The detector responds to a learning-error statistic; it does not classify the type of traffic change.
- EWC importance uses squared Q-value gradients as a diagonal proxy, not a validated likelihood Fisher matrix.
- The reported throughput quantity counts successful transmissions per environment step; it is not bits per second or spectral efficiency.
- The simulator uses independent synthetic occupancy with supplied recent occupancy history. Real-radio, noisy-sensing, multi-user, and packet-level performance remain untested here.
- Evaluation episodes do not replace independent training seeds when describing uncertainty.

## Citation and license

Use [CITATION.cff](CITATION.cff) when citing the software. **Replace its author placeholders** and add the repository release/DOI after those exist. The [package metadata](pyproject.toml) also contains an author placeholder. Select a license and add a `LICENSE` file before a public release; no reuse license has been assumed here.
