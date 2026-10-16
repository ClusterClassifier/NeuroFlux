"""Discover and download trained Neurone checkpoints from GitHub."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import PurePosixPath
from urllib.parse import quote, urlsplit
from urllib import error, request


class CheckpointError(RuntimeError):
    """Describe an actionable repository or checkpoint loading failure."""


def normalize_repository(value: str) -> str:
    """Accept owner/repo, a GitHub URL, or a GitHub SSH remote."""
    value = value.strip().rstrip("/")
    value = re.sub(r"^git@github\.com:", "", value)
    value = re.sub(r"^https?://github\.com/", "", value)
    value = re.sub(r"\.git$", "", value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise CheckpointError("Set GITHUB_REPOSITORY to owner/repository or its GitHub URL.")
    return value


def detect_repository() -> str:
    """Read the checkout remote without requiring the GitHub CLI."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True,
            text=True, timeout=3, check=True,
        )
        return normalize_repository(result.stdout)
    except (OSError, subprocess.SubprocessError, CheckpointError):
        return ""


def normalize_directory(value: str) -> str:
    """Keep repository-relative paths unambiguous."""
    value = value.strip().strip("/")
    if "\\" in value or any(p in {".", ".."} for p in value.split("/")):
        raise CheckpointError("GITHUB_CHECKPOINT_PATH must be a repository-relative folder.")
    return value


def _headers(token: str, raw: bool = False) -> dict:
    headers = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Neurone-Checkpoint-Dashboard",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_error(status):
    if status == 404:
        raise CheckpointError("GitHub repository, revision, or file was not found. For a private repository, configure GITHUB_TOKEN with read access.")
    if status in (401, 403, 429):
        raise CheckpointError("GitHub denied access or its API rate limit was reached. Check the deployment token and retry later.")
    raise CheckpointError(f"GitHub request failed (HTTP {status}).")


class _GitHubRedirect(request.HTTPRedirectHandler):
    """Keep credentials on their original host when GitHub redirects downloads."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = urlsplit(newurl)
        if destination.scheme != "https" or not (
            (destination.hostname or "") == "github.com"
            or (destination.hostname or "").endswith(".github.com")
            or (destination.hostname or "").endswith(".githubusercontent.com")
        ):
            raise CheckpointError("GitHub returned an unsupported download redirect.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and destination.netloc != urlsplit(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _read_url(url, token, maximum_bytes, raw=False):
    req = request.Request(url, headers=_headers(token, raw=raw))
    try:
        opener = request.build_opener(_GitHubRedirect())
        with opener.open(req, timeout=90) as response:
            if int(response.headers.get("Content-Length", "0")) > maximum_bytes:
                raise CheckpointError("GitHub file exceeds the configured download size limit.")
            chunks, size = [], 0
            while True:
                chunk = response.read(min(1024 * 1024, maximum_bytes - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise CheckpointError("GitHub download exceeded the configured size limit.")
                chunks.append(chunk)
            return b"".join(chunks)
    except error.HTTPError as exc:
        _http_error(exc.code)
    except (error.URLError, TimeoutError, OSError) as exc:
        raise CheckpointError("Could not reach GitHub. Check network access and repository configuration, then retry.") from exc


def _json(url: str, token: str) -> dict:
    try:
        return json.loads(_read_url(url, token, 32 * 1024 * 1024))
    except (ValueError, UnicodeError) as exc:
        raise CheckpointError("GitHub returned invalid repository metadata.") from exc


def discover_repository(repository: str, revision: str, directory: str, token: str = "") -> dict:
    """Resolve one revision and list only its configured checkpoint subtree."""
    repository = normalize_repository(repository)
    directory = normalize_directory(directory)
    base = f"https://api.github.com/repos/{repository}"
    if not revision:
        revision = _json(base, token)["default_branch"]
    commit = _json(f"{base}/commits/{quote(revision, safe='')}", token)
    commit_sha = commit["sha"]
    tree_sha = commit["commit"]["tree"]["sha"]
    for part in directory.split("/") if directory else []:
        tree = _json(f"{base}/git/trees/{tree_sha}", token)
        child = next((item for item in tree["tree"] if item["path"] == part and item["type"] == "tree"), None)
        if child is None:
            raise CheckpointError(f"Checkpoint folder '{directory}' was not found at the configured revision.")
        tree_sha = child["sha"]
    tree = _json(f"{base}/git/trees/{tree_sha}?recursive=1", token)
    if tree.get("truncated"):
        raise CheckpointError("Checkpoint listing is too large. Set GITHUB_CHECKPOINT_PATH to a more specific run folder.")
    entries = []
    for item in tree["tree"]:
        if item["type"] != "blob":
            continue
        path = f"{directory}/{item['path']}" if directory else item["path"]
        entries.append(dict(path=path, sha=item["sha"], size=item.get("size", 0)))
    checkpoints = []
    for item in entries:
        match = re.fullmatch(r"task_(\d+)_policy\.pt", PurePosixPath(item["path"]).name)
        if match:
            checkpoints.append(dict(item, task=int(match.group(1))))
    if not checkpoints:
        raise CheckpointError("No task_<number>_policy.pt checkpoints were found in the configured GitHub folder.")
    return dict(repository=repository, revision=revision, commit=commit_sha, entries=entries, checkpoints=checkpoints)


def choose_checkpoint(catalog: dict, task: int, mode: str = "final") -> dict:
    """Prefer the final policy unless the deployment explicitly maps tasks."""
    if mode not in {"final", "matching"}:
        raise CheckpointError("GITHUB_POLICY_MODE must be final or matching.")
    checkpoints = catalog["checkpoints"]
    wanted = task if mode == "matching" else max(item["task"] for item in checkpoints)
    selected = [item for item in checkpoints if item["task"] == wanted]
    if not selected:
        raise CheckpointError(f"No task_{wanted}_policy.pt checkpoint is available.")
    if len(selected) != 1:
        raise CheckpointError("Multiple runs contain the same task checkpoint. Narrow GITHUB_CHECKPOINT_PATH to one run's checkpoints folder.")
    return selected[0]


def _download(url: str, token: str, maximum_bytes: int) -> bytes:
    return _read_url(url, token, maximum_bytes, raw=True)


def fetch_checkpoint(catalog: dict, selected: dict, token: str = "", maximum_bytes: int = 64 * 1024 * 1024) -> dict:
    """Download immutable checkpoint bytes and verify their Git or LFS digest."""
    repository, commit, path = catalog["repository"], catalog["commit"], selected["path"]
    url = f"https://api.github.com/repos/{repository}/git/blobs/{selected['sha']}"
    data = _download(url, token, maximum_bytes)
    blob_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob_hash != selected["sha"]:
        raise CheckpointError("Downloaded checkpoint does not match the repository blob digest.")
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        oid = re.search(rb"oid sha256:([a-f0-9]{64})", data)
        size = re.search(rb"size (\d+)", data)
        if not oid or not size:
            raise CheckpointError("The Git LFS pointer is invalid.")
        if int(size.group(1)) > maximum_bytes:
            raise CheckpointError("Git LFS checkpoint exceeds the configured download limit.")
        lfs_url = f"https://media.githubusercontent.com/media/{repository}/{commit}/{quote(path, safe='/')}"
        data = _download(lfs_url, token, maximum_bytes)
        if len(data) != int(size.group(1)) or hashlib.sha256(data).hexdigest() != oid.group(1).decode():
            raise CheckpointError("Git LFS content could not be verified. Check repository LFS availability.")
    parent = str(PurePosixPath(path).parent)
    existing = {item["path"] for item in catalog["entries"]}
    task = selected["task"]
    artifacts = {
        name: f"{parent}/task_{task}_{suffix}.pkl" in existing
        for name, suffix in [("fisher", "fisher"), ("theta_star", "theta_star"), ("episodic", "episodic")]
    }
    return dict(
        data=data, repository=repository, commit=commit, path=path,
        task=task, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
        artifacts=artifacts,
    )
