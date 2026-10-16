"""Check GitHub resolution, checkpoint integrity, telemetry, and export behavior."""

import ast
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import time
import types
import unittest
from unittest.mock import patch
from urllib import request
import zipfile

import numpy as np
import pandas as pd

import github_checkpoints as github
from dashboard_exports import build_export, scenario_summary


class GitHubTests(unittest.TestCase):
    def test_repository_forms_and_invalid_input(self):
        for value in ["owner/repo", "https://github.com/owner/repo.git", "git@github.com:owner/repo.git"]:
            self.assertEqual(github.normalize_repository(value), "owner/repo")
        with self.assertRaises(github.CheckpointError):
            github.normalize_repository("https://untrusted.example/repo")
        with self.assertRaises(github.CheckpointError):
            github.normalize_directory("../checkpoints")

    def test_discovery_pins_commit_and_scopes_subtree(self):
        replies = [
            {"default_branch": "main"},
            {"sha": "commit", "commit": {"tree": {"sha": "root"}}},
            {"tree": [{"path": "checkpoints", "type": "tree", "sha": "folder"}]},
            {"tree": [
                {"path": "task_7_policy.pt", "type": "blob", "sha": "blob", "size": 10},
                {"path": "task_7_fisher.pkl", "type": "blob", "sha": "other", "size": 2},
            ]},
        ]
        with patch.object(github, "_json", side_effect=replies) as get:
            catalog = github.discover_repository("owner/repo", "", "checkpoints")
        self.assertEqual(catalog["commit"], "commit")
        self.assertEqual(catalog["checkpoints"][0]["path"], "checkpoints/task_7_policy.pt")
        self.assertIn("folder?recursive=1", get.call_args.args[0])

    def test_truncated_tree_fails_loudly(self):
        with patch.object(github, "_json", side_effect=[
            {"sha": "commit", "commit": {"tree": {"sha": "root"}}}, {"truncated": True}
        ]):
            with self.assertRaises(github.CheckpointError):
                github.discover_repository("owner/repo", "main", "")

    def test_missing_checkpoint_folder(self):
        with patch.object(github, "_json", side_effect=[
            {"sha": "commit", "commit": {"tree": {"sha": "root"}}}, {"tree": []}
        ]):
            with self.assertRaises(github.CheckpointError):
                github.discover_repository("owner/repo", "main", "missing")

    def test_final_matching_and_ambiguous_selection(self):
        catalog = {"checkpoints": [{"task": 1, "path": "one"}, {"task": 7, "path": "seven"}]}
        self.assertEqual(github.choose_checkpoint(catalog, 1)["task"], 7)
        self.assertEqual(github.choose_checkpoint(catalog, 1, "matching")["task"], 1)
        with self.assertRaises(github.CheckpointError):
            github.choose_checkpoint(catalog, 3, "matching")
        catalog["checkpoints"].append({"task": 7, "path": "different_run"})
        with self.assertRaises(github.CheckpointError):
            github.choose_checkpoint(catalog, 1)

    def fixture(self, data):
        selected = {"path": "checkpoints/task_7_policy.pt", "task": 7,
                    "sha": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
        catalog = {"repository": "owner/repo", "commit": "pinned", "entries": [
            {"path": "checkpoints/task_7_fisher.pkl"}
        ]}
        return catalog, selected

    def test_download_integrity_and_artifact_flags(self):
        data = b"checkpoint-fixture"
        catalog, selected = self.fixture(data)
        with patch.object(github, "_download", return_value=data) as get:
            result = github.fetch_checkpoint(catalog, selected)
        self.assertEqual(result["data"], data)
        self.assertTrue(result["artifacts"]["fisher"])
        self.assertFalse(result["artifacts"]["episodic"])
        self.assertIn("/git/blobs/" + selected["sha"], get.call_args.args[0])
        with patch.object(github, "_download", return_value=b"wrong"):
            with self.assertRaises(github.CheckpointError):
                github.fetch_checkpoint(catalog, selected)

    def test_lfs_download_and_digest(self):
        payload = b"large-file-fixture"
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n".encode()
        catalog, selected = self.fixture(pointer)
        with patch.object(github, "_download", side_effect=[pointer, payload]):
            self.assertEqual(github.fetch_checkpoint(catalog, selected)["data"], payload)
        with patch.object(github, "_download", side_effect=[pointer, b"bad"]):
            with self.assertRaises(github.CheckpointError):
                github.fetch_checkpoint(catalog, selected)

    def test_size_limit_without_content_length(self):
        response = io.BytesIO(b"x" * 20)
        response.headers = {}
        opener = types.SimpleNamespace(open=lambda *a, **k: response)
        with patch.object(github.request, "build_opener", return_value=opener):
            with self.assertRaises(github.CheckpointError):
                github._read_url("https://api.github.com/test", "", 10)

    def test_redirect_does_not_forward_token_to_another_host(self):
        handler = github._GitHubRedirect()
        req = request.Request("https://api.github.com/file", headers={"Authorization": "Bearer private"})
        redirected = handler.redirect_request(req, None, 302, "Found", {}, "https://objects.githubusercontent.com/file")
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaises(github.CheckpointError):
            handler.redirect_request(req, None, 302, "Found", {}, "https://untrusted.example/file")

    def test_access_errors_are_actionable(self):
        for code in [401, 403, 404, 429, 500]:
            with self.assertRaises(github.CheckpointError):
                github._http_error(code)


class ExportTests(unittest.TestCase):
    def records(self):
        return [dict(
            step=i + 1, scenario="T1", outcome=outcome, reward=reward,
            cumulative_reward=cumulative, td_error=.2, drift=(i == 1),
            inference_ms=1.0, action=i, q_values=[.1, .2, .3], occupancy=[0, 1],
        ) for i, (outcome, reward, cumulative) in enumerate([
            ("success", 1., 1.), ("collision", -1., 0.), ("idle", -.05, -.05)
        ])]

    def test_rates_have_correct_denominators(self):
        summary = scenario_summary(self.records()).iloc[0]
        self.assertAlmostEqual(summary.throughput, 1 / 3)
        self.assertAlmostEqual(summary.collision_rate, .5)
        self.assertAlmostEqual(summary.access_efficiency, .5)
        self.assertEqual(summary.drift_events, 1)

    def test_export_contains_consistent_graphs_and_logs(self):
        files = build_export(self.records(), [], {"session_steps": 3})
        self.assertTrue(files["graphs/dashboard.png"].startswith(b"\x89PNG"))
        self.assertIn(b"<svg", files["graphs/dashboard.svg"])
        self.assertEqual(len(pd.read_csv(io.BytesIO(files["telemetry.csv"]))), 3)
        self.assertEqual(json.loads(files["metadata.json"])["exported_rows"], 3)
        with zipfile.ZipFile(io.BytesIO(files["session_export.zip"])) as archive:
            self.assertIn("graphs/dashboard.svg", archive.namelist())
            self.assertEqual(archive.testzip(), None)

    def test_export_rejects_empty_run(self):
        with self.assertRaises(ValueError):
            build_export([], [], {})


class State(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class ArrayTensor:
    """Provide the tensor operations needed by the telemetry wiring test."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def float(self):
        return self

    def unsqueeze(self, axis):
        return ArrayTensor(np.expand_dims(self.array, axis))

    def squeeze(self, axis):
        return ArrayTensor(np.squeeze(self.array, axis))

    def __getitem__(self, index):
        return ArrayTensor(self.array[index])

    def item(self):
        return self.array.item()

    def max(self, dim):
        return types.SimpleNamespace(values=ArrayTensor(self.array.max(axis=dim)))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class RuntimeTests(unittest.TestCase):
    def namespace(self):
        from contextlib import nullcontext
        tree = ast.parse(Path(__file__).with_name("app.py").read_text())
        names = {"CRNEnvironment", "ADWIN", "append_log", "make_environment", "reset_runtime", "switch_task", "run_one_step"}
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        state = State(model_meta={"num_channels": 2, "history_length": 5},
                      simulation_seed=42, adwin_delta=.002, model_source_id="test",
                      simulation_mode="Automatic diurnal cycle", steps_per_task=1,
                      max_session_steps=10, running=True)
        namespace = dict(np=np, math=math, deque=deque, st=types.SimpleNamespace(session_state=state),
                         hashlib=hashlib, html=html, datetime=datetime, timezone=timezone, time=time,
                         MAX_HISTORY_ROWS=50_000, DEFAULT_ADWIN_DELTA=.002, EPISODE_LENGTH=1000,
                         GAMMA=.97)
        namespace["TASK_KEYS"] = ["T1", "T2"]
        namespace["TASK_META"] = {
            "T1": {"id": 1, "low": .2, "high": .4},
            "T2": {"id": 2, "low": .6, "high": .9},
        }
        namespace["torch"] = types.SimpleNamespace(
            from_numpy=ArrayTensor, no_grad=nullcontext,
            argmax=lambda x, dim: ArrayTensor(x.array.argmax(axis=dim)),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "dashboard_runtime", "exec"), namespace)
        state.policy_net = lambda x: ArrayTensor([[2., 1., 0.]])
        state.target_net = lambda x: ArrayTensor([[1., .5, 0.]])
        namespace["reset_runtime"]("T1")
        return namespace, state

    def test_auto_shift_keeps_decision_snapshot_and_detector(self):
        ns, state = self.namespace()
        detector = state.adwin
        ns["run_one_step"]()
        self.assertEqual(state.global_step, 1)
        self.assertEqual(state.active_task, "T2")
        self.assertEqual(state.last_decision_task, "T1")
        self.assertEqual(state.telemetry[0]["scenario"], "T1")
        self.assertIs(state.adwin, detector)
        self.assertEqual(len(state.telemetry[0]["q_values"]), 3)

    def test_reset_is_repeatable_and_clears_exports(self):
        ns, state = self.namespace()
        original = state.env.get_state().copy()
        ns["run_one_step"]()
        state.export_snapshot = {"old": True}
        ns["reset_runtime"]("T1")
        np.testing.assert_array_equal(original, state.env.get_state())
        self.assertIsNone(state.export_snapshot)
        self.assertEqual(len(state.telemetry), 0)
        self.assertEqual(state.global_step, 0)

    def test_step_limit_pauses_session(self):
        ns, state = self.namespace()
        state.max_session_steps = 1
        ns["run_one_step"]()
        self.assertFalse(state.running)


if __name__ == "__main__":
    unittest.main(verbosity=2)
