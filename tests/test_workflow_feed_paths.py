"""Regression test for the Apr to Jul 2026 feed-refresh failure.

For 91 consecutive scheduled runs, refresh-addresses-daily.yml failed
because its commit step staged `data/feeds/`, a path that .gitignore
excludes (line 27). `git add` exits 1 when a pathspec matches only
ignored files, the Actions shell aborted the step, and nothing ever
committed. Nobody noticed for three months.

This test scans every workflow's `git add` lines and fails if a
staged literal path is a directory (or file) wholly covered by a
non-wildcard .gitignore entry, which is the shape of that bug.
Tracked files are exempt: ignore rules do not apply to them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ignore_dir_patterns() -> list[str]:
    """Non-wildcard .gitignore entries, normalised without trailing /."""
    patterns: list[str] = []
    for raw in (ROOT / ".gitignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if "*" in line:
            continue
        patterns.append(line.rstrip("/"))
    return patterns


def _tracked(path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def _covered_by_ignore(path: str, patterns: list[str]) -> bool:
    p = path.strip().strip('"').strip("'").rstrip("/")
    if not p or p.startswith("-") or p in {".", "./"}:
        return False
    return any(p == pat or p.startswith(pat + "/") for pat in patterns)


def _workflow_git_add_paths() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    wf_dir = ROOT / ".github" / "workflows"
    for wf in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        for line in wf.read_text().splitlines():
            match = re.search(r"git add\s+(.+)", line)
            if not match:
                continue
            for token in match.group(1).split():
                if token.startswith("-"):
                    continue
                found.append((wf.name, token))
    return found


def test_no_workflow_stages_a_gitignored_path() -> None:
    patterns = _ignore_dir_patterns()
    offenders = [
        (wf, token)
        for wf, token in _workflow_git_add_paths()
        if _covered_by_ignore(token, patterns) and not _tracked(token)
    ]
    assert not offenders, (
        "workflow(s) stage path(s) covered by .gitignore; "
        f"git add would fail the run: {offenders}"
    )


def test_refresh_workflow_stages_only_addresses() -> None:
    tokens = [
        token
        for wf, token in _workflow_git_add_paths()
        if wf == "refresh-addresses-daily.yml"
    ]
    assert tokens, "refresh workflow has no git add step; has it moved?"
    assert all(t.rstrip("/") == "addresses" for t in tokens), (
        f"refresh workflow stages unexpected paths: {tokens}; "
        "it must stage addresses/ only (data/feeds/ is gitignored)"
    )
