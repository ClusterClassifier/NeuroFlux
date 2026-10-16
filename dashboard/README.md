# Neurone V1 Streamlit dashboard

The dashboard loads trained checkpoints automatically and starts inference when a
visitor clicks one of the seven scenario buttons. No checkpoint upload or model
selection is required from visitors.

## Start locally

Use Python 3.10 or newer. Extract this folder and run from inside it:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses CPU inference, so a GPU is not required.

## Configure GitHub once

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the
repository name and checkpoint folder. For Streamlit Community Cloud, enter these
same settings in the deployment's Secrets settings instead.

```toml
GITHUB_REPOSITORY = "YOUR_OWNER/YOUR_REPOSITORY"
GITHUB_REF = "main"
GITHUB_CHECKPOINT_PATH = "checkpoints"
GITHUB_POLICY_MODE = "final"
MAX_CHECKPOINT_MB = 64
```

No repository URL was present in the supplied code, so this package contains no
invented repository address. The repository setting accepts `owner/repository`,
a GitHub URL, or a GitHub SSH remote. When this setting is omitted, the app tries
the deployed checkout's `origin` remote. When neither a repository setting nor a
GitHub remote is available, it can use an existing local `checkpoints/` directory.
A configured GitHub failure is displayed explicitly; it does not silently load a
local model with a different provenance.

Environment variables with the same names override Streamlit secrets. An `.env`
file is not loaded automatically. Omit `GITHUB_REF` or set it to an empty string
to follow the repository's default branch. A commit SHA can be supplied to pin
an exact model revision.

For a private repository, configure `GITHUB_TOKEN` with repository Contents read
access. A token also increases API allowances for public repositories. Keep the
real secrets file out of Git. Tokens are not included in logs or exports.

### Checkpoint structure

Use the original training checkpoint names:

```text
checkpoints/task_1_policy.pt
checkpoints/task_2_policy.pt
...
checkpoints/task_7_policy.pt
```

Each checkpoint may contain `{"policy": state_dict, "target": state_dict}` or a
plain DQN policy state dictionary. The model's dimensions are inferred from its
`fc1`, `fc2`, and `fc3` weights. A missing target is cloned from the policy, and
this is disclosed in the dashboard. Loading uses `weights_only=True` without an
unsafe pickle fallback.

If your checkpoints are nested inside ablation outputs, set the exact run folder,
for example:

```toml
GITHUB_CHECKPOINT_PATH = "neurone_v1_ablation_results/runs/full_v1/seed_42/checkpoints"
```

The loader refuses ambiguous duplicate task checkpoints. It does not guess a
variant or training seed. Files must exist in the GitHub repository or its Git
LFS storage; release assets and GitHub Actions artifact ZIPs are not searched.

### Automatic model selection

| Mode | Behavior |
| --- | --- |
| `final` (default) | Uses the highest-numbered task checkpoint for all seven scenarios, allowing the same continual policy to be inspected across environments. |
| `matching` | Uses `task_1_policy.pt` for T1, `task_2_policy.pt` for T2, and so on. Changing checkpoint starts a new session. Automatic diurnal cycling is disabled in this mode. |

If the latest checkpoint is earlier than task 7, the UI identifies it as a partial
continual policy. `rollback.pt` is not selected automatically.

Repository discovery is cached, and the resolved commit stays fixed throughout
a session. **Refresh GitHub checkpoints** explicitly checks for updates and pauses
the simulation. Files are verified against the Git blob digest; Git LFS files are
also verified against their pointer's SHA-256 and size. A file larger than the
configured download limit is rejected. Regular Git blobs must also fit GitHub's
blob endpoint limit; use LFS for larger model files.

## Dashboard features

- One-click T1–T7 selection and automatic inference start.
- Pause, resume, single step, reset, and an automatic session step limit.
- Optional diurnal task cycling, with ADWIN history preserved across task shifts.
- Adjustable playback speed, detector sensitivity, and simulation seed.
- Inferred architecture, target-network origin, source commit, checkpoint SHA-256,
  and a link to the pinned GitHub file.
- Remote Fisher, reference-weight, and episodic-memory artifact availability,
  without downloading or unpickling those training artifacts.
- Decision-time channel occupancy, real Q-values, Bellman target, absolute TD error,
  detector window, and drift events.
- Rolling reward and TD-error charts, cumulative reward, and per-scenario rates.
- Retained inference latency and completed episode counts.
- Frozen export snapshots so logs, graph data, and metadata refer to the same step.

Clicking a different scenario preserves the session's detector and cumulative
telemetry when the checkpoint is unchanged. Reset clears the session and export
snapshot, reloads the original in-memory weights, and applies the selected seed.
Changing the seed setting alone does not alter the current session's environments.
ADWIN sensitivity changes reset only the detector. Scenario button selection during
automatic cycling immediately changes the active scenario and continues the cycle.

The original DQN architecture, CRN environment, and ADWIN implementation are
preserved. This remains fixed-policy inference, not online training: the dashboard
does not fabricate EWC updates, replay, rollback, or MEC computation. ADWIN events
are monitoring events, not verified detection accuracy measurements.

## Export logs and graphs

Click **Prepare export snapshot** after collecting at least one step. This pauses
the simulation and creates:

| File | Contents |
| --- | --- |
| `telemetry.csv` | Timestamped steps, scenario, action, outcome, reward, cumulative reward, Q-values, occupancy, Bellman target, TD error, drift flag, window, and latency. |
| `events.csv` | Timestamped structured event log. |
| `events.log` | Readable plain-text event log. |
| `scenario_summary.csv` | Per-scenario summary from the exported telemetry rows. |
| `metadata.json` | Session/checkpoint provenance, architecture, settings, cumulative counters, and retention information. |
| `graphs/dashboard.png` | Reward, TD-error/drift, cumulative reward, and action-frequency graphs. |
| `graphs/dashboard.svg` | The same graph panel in a scalable vector format. |

Download individual logs, telemetry, or PNG graphs, or download everything as a
ZIP. Downloads do not trigger simulation reruns. Prepare another snapshot to
capture later data; an earlier prepared snapshot remains fixed until replaced.

The visible event panel shows the latest 60 events. Export history retains up to
50,000 steps and 100,000 events per session. The step limit defaults to 10,000.
When telemetry retention truncates a longer session, metadata flags truncation;
per-scenario summaries and action counts describe the retained rows, while the
cumulative reward series retains its since-reset totals. Sessions are kept in
Streamlit session memory and are cleared on a server restart or session loss.

Throughput means successes / all steps. Collision rate means collisions /
transmission attempts. Access efficiency means successes / transmission attempts.
The simulated reward is not a physical bit-rate or spectral-efficiency measurement.
Inference latency measures the policy decision path on the server and excludes
browser rendering and the playback delay. Speed is a target playback rate; network
and Streamlit rerendering add overhead.

## Validation

```bash
python -m unittest -v test_dashboard
```

The included 16 tests cover discovery, immutable selection, ambiguity, download
hashes, Git LFS, size limits, token redirects, HTTP errors, exported rates and
files, scenario transitions, reset behavior, and step limits. HTTP responses and
tensor inference are mocked in these focused tests. The DQN, environment, and
ADWIN class definitions were also checked against the supplied source.

This environment did not have PyTorch or Streamlit installed, so a real-checkpoint
browser session and live GitHub access were not executed. No repository address
or real checkpoint was supplied for that integration check.

## Implementation references

- [GitHub Git trees API](https://docs.github.com/en/rest/git/trees)
- [GitHub Git blobs API](https://docs.github.com/en/rest/git/blobs)
- [Streamlit download button](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)
