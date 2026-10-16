"""Create consistent CSV, JSON, and publication-friendly graph snapshots."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd


def scenario_summary(records: list[dict]) -> pd.DataFrame:
    """Compute rates with explicit transmission-attempt denominators."""
    rows = []
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame()
    for scenario, group in frame.groupby("scenario", sort=False):
        steps = len(group)
        successes = int((group.outcome == "success").sum())
        collisions = int((group.outcome == "collision").sum())
        attempts = successes + collisions
        rows.append(dict(
            scenario=scenario, steps=steps, total_reward=float(group.reward.sum()),
            mean_reward=float(group.reward.mean()), throughput=successes / steps,
            collision_rate=collisions / attempts if attempts else 0.0,
            access_efficiency=successes / attempts if attempts else 0.0,
            idle_rate=float((group.outcome == "idle").mean()),
            mean_absolute_td_error=float(group.td_error.mean()),
            drift_events=int(group.drift.sum()), mean_inference_ms=float(group.inference_ms.mean()),
        ))
    return pd.DataFrame(rows)


def build_export(records: list[dict], events: list[dict], metadata: dict) -> dict[str, bytes]:
    """Build one frozen snapshot; graph data and CSV data use the same rows."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Run at least one simulation step before exporting.")
    metadata = dict(metadata, exported_rows=len(frame), first_exported_step=int(frame.step.min()), last_exported_step=int(frame.step.max()))
    event_frame = pd.DataFrame(events, columns=["timestamp", "step", "scenario", "category", "message"])
    flat = frame.copy()
    for column in ("q_values", "occupancy"):
        flat[column] = flat[column].map(json.dumps)
    files = {
        "telemetry.csv": flat.to_csv(index=False).encode(),
        "events.csv": event_frame.to_csv(index=False).encode(),
        "scenario_summary.csv": scenario_summary(records).to_csv(index=False).encode(),
        "metadata.json": json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False).encode(),
        "events.log": "\n".join(f"{e['timestamp']} [{e['step']:06d}] {e['scenario']} | {e['message']}" for e in events).encode(),
    }
    figure = Figure(figsize=(13, 10), constrained_layout=True)
    axes = figure.subplots(2, 2)
    steps = frame.step.to_numpy()
    axes[0, 0].plot(steps, frame.reward, alpha=.25, label="Reward")
    axes[0, 0].plot(steps, frame.reward.rolling(20, min_periods=1).mean(), label="20-step mean")
    axes[0, 0].set(title="Reward", xlabel="Simulation step", ylabel="Reward")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, frame.td_error, label="Absolute TD error")
    drift = frame.loc[frame.drift]
    axes[0, 1].scatter(drift.step, drift.td_error, color="#c73956", label="ADWIN event", zorder=3)
    axes[0, 1].set(title="TD-error drift signal", xlabel="Simulation step", ylabel="Absolute TD error")
    axes[0, 1].legend()
    axes[1, 0].plot(steps, frame.cumulative_reward, color="#268471")
    axes[1, 0].set(title="Cumulative reward", xlabel="Simulation step", ylabel="Reward since session reset")
    actions = frame.action.value_counts().sort_index()
    axes[1, 1].bar(actions.index.astype(str), actions.values, color="#52759b")
    axes[1, 1].set(title="Action distribution (zero-based; final action = idle)", xlabel="Action", ylabel="Count")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    figure.suptitle("Neurone V1 · frozen-policy simulation snapshot")
    for extension in ("png", "svg"):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=extension, dpi=200)
        files[f"graphs/dashboard.{extension}"] = buffer.getvalue()
    figure.clear()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, value in files.items():
            zipped.writestr(name, value)
    files["session_export.zip"] = archive.getvalue()
    return files
