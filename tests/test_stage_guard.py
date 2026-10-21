"""Runtime tests for scripts/check_stage_paths.sh.

The guard exists because the feed-refresh workflow once staged a
gitignored path (data/feeds/) and failed 91 scheduled runs in a row.
These tests execute the real script inside a throwaway git repo and
prove both directions: it rejects ignored staging paths, including
the historical one, and it accepts the paths the workflow stages
today. Review feedback on PR #17 (2026-10-20) specifically required
a negative runtime test; the earlier in-workflow check could not see
ignored untracked files at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_stage_paths.sh"


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    (repo / ".gitignore").write_text("data/feeds/\n*.log\n")
    (repo / "data" / "feeds").mkdir(parents=True)
    (repo / "data" / "feeds" / "ofac_sdn.json").write_text("{}")
    (repo / "addresses").mkdir()
    (repo / "addresses" / "ofac.json").write_text("{}")
    return repo


def _run_guard(repo: Path, *paths: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *paths],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_guard_rejects_the_historical_ignored_path(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    result = _run_guard(repo, "data/feeds/")
    assert result.returncode == 1
    assert "data/feeds/" in result.stderr


def test_guard_rejects_ignored_path_mixed_with_valid_one(
    tmp_path: Path,
) -> None:
    """The exact April 2026 staging line: data/feeds/ addresses/."""
    repo = _make_repo(tmp_path)
    result = _run_guard(repo, "data/feeds/", "addresses/")
    assert result.returncode == 1


def test_guard_rejects_ignored_file_pattern(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    result = _run_guard(repo, "debug.log")
    assert result.returncode == 1


def test_guard_accepts_the_workflows_current_path(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    result = _run_guard(repo, "addresses/")
    assert result.returncode == 0, result.stderr


def test_guard_rejects_missing_argument(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    result = _run_guard(repo)
    assert result.returncode == 2
