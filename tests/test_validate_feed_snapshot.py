"""Tests for scripts/validate_feed_snapshot.py.

The invariant exists because the committed phishing snapshot held
189 domain strings and zero well-formed addresses while every
existing check counted entries and passed. These tests build
throwaway git repos with a committed baseline snapshot, then run
the real script against modified snapshots: junk substitution,
catastrophic shrink, and total loss must fail; a clean snapshot
and a snapshot with a single malformed entry must pass (with the
entry named as a warning).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "validate_feed_snapshot.py"


def _addr(i: int) -> str:
    return "0x" + format(i, "040x")


def _write(repo: Path, name: str, entries: list) -> None:
    snap = repo / "addresses"
    snap.mkdir(exist_ok=True)
    (snap / name).write_text(json.dumps(
        {"feed": name, "count": len(entries), "addresses": entries}))


def _make_repo(tmp_path: Path, committed: dict[str, list]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    for name, entries in committed.items():
        _write(repo, name, entries)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
    return repo


def _run(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "addresses/"],
        cwd=repo, capture_output=True, text=True)


def test_clean_snapshot_passes(tmp_path):
    entries = [_addr(i) for i in range(10)]
    repo = _make_repo(tmp_path, {"ofac.json": entries})
    _write(repo, "ofac.json", entries + [_addr(99)])
    r = _run(repo)
    assert r.returncode == 0, r.stderr


def test_domain_junk_fails(tmp_path):
    repo = _make_repo(tmp_path, {"phishing.json": [_addr(i) for i in range(50)]})
    _write(repo, "phishing.json", ["0x-scam.example", "0x-fake.top"] * 20)
    r = _run(repo)
    assert r.returncode == 1
    assert "not" in r.stderr and "well-formed" in r.stderr


def test_catastrophic_shrink_fails(tmp_path):
    repo = _make_repo(tmp_path, {"ofac.json": [_addr(i) for i in range(100)]})
    _write(repo, "ofac.json", [_addr(i) for i in range(40)])
    r = _run(repo)
    assert r.returncode == 1
    assert "shrank" in r.stderr


def test_small_shrink_passes(tmp_path):
    repo = _make_repo(tmp_path, {"ofac.json": [_addr(i) for i in range(100)]})
    _write(repo, "ofac.json", [_addr(i) for i in range(95)])
    r = _run(repo)
    assert r.returncode == 0, r.stderr


def test_single_malformed_warns_but_passes(tmp_path):
    entries = [_addr(i) for i in range(100)]
    repo = _make_repo(tmp_path, {"ofac.json": entries})
    entries[0] = "0xf6b5414f23a15c5fe41c37d7c8f7e4adfc30e0c"  # truncated
    _write(repo, "ofac.json", entries)
    r = _run(repo)
    assert r.returncode == 0, r.stderr
    assert "warning" in (r.stdout + r.stderr).lower()
    assert "0xf6b5414f" in (r.stdout + r.stderr)


def test_emptied_feed_fails(tmp_path):
    repo = _make_repo(tmp_path, {"ofac.json": [_addr(i) for i in range(50)]})
    _write(repo, "ofac.json", [])
    r = _run(repo)
    assert r.returncode == 1
