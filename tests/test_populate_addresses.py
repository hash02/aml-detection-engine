"""Tests for scripts/populate_addresses.py.

The populate step used to be an inline heredoc in the refresh
workflow: it silently skipped a feed whose cache file was missing
(fetch failed), kept the old snapshot, and still rewrote
MANIFEST.json with a fresh last_updated. Stale data, labeled fresh
(Claude review of PR #20, finding 2). These tests run the real
script against throwaway feeds/addresses directories: a missing
cache must fail before anything is written; an unchanged re-run
must produce byte-identical output; a changed feed must bump
last_updated and refresh only that feed's per-feed timestamp.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "populate_addresses.py"
WORKFLOW = ROOT / ".github" / "workflows" / "refresh-addresses-daily.yml"

# feed name -> snapshot filename, mirroring the script's MAPPING
FEEDS = {
    "ofac_sdn": "ofac.json",
    "phishing": "phishing.json",
    "mixers": "mixers.json",
    "bridges": "bridges.json",
    "lrt_offramps": "offramps.json",
}


def _addr(i: int) -> str:
    return "0x" + format(i, "040x")


def _write_cache(feeds_dir: Path, name: str, entries: list,
                 updated_at: float) -> None:
    feeds_dir.mkdir(parents=True, exist_ok=True)
    (feeds_dir / f"{name}.json").write_text(json.dumps(
        {"feed": name, "updated_at": updated_at,
         "count": len(entries), "addresses": entries}, indent=2))


def _write_all_caches(feeds_dir: Path, updated_at: float,
                      overrides: dict[str, list] | None = None) -> None:
    for i, name in enumerate(FEEDS):
        entries = [_addr(i * 100 + j) for j in range(5)]
        if overrides and name in overrides:
            entries = overrides[name]
        _write_cache(feeds_dir, name, entries, updated_at)


def _run(feeds_dir: Path, addr_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(feeds_dir), str(addr_dir)],
        capture_output=True, text=True)


def _manifest(addr_dir: Path) -> dict:
    return json.loads((addr_dir / "MANIFEST.json").read_text())


def test_first_run_writes_all_feeds_and_manifest(tmp_path):
    feeds_dir, addr_dir = tmp_path / "feeds", tmp_path / "addresses"
    _write_all_caches(feeds_dir, updated_at=1000.0)
    r = _run(feeds_dir, addr_dir)
    assert r.returncode == 0, r.stderr
    for filename in FEEDS.values():
        data = json.loads((addr_dir / filename).read_text())
        assert data["count"] == 5
    manifest = _manifest(addr_dir)
    assert manifest["feed_updated_at"] == {
        filename: 1000.0 for filename in FEEDS.values()}
    assert manifest["last_updated"] > 0
    assert set(manifest["checksums"]) == set(FEEDS.values())


def test_rerun_without_changes_is_byte_identical(tmp_path):
    feeds_dir, addr_dir = tmp_path / "feeds", tmp_path / "addresses"
    _write_all_caches(feeds_dir, updated_at=1000.0)
    assert _run(feeds_dir, addr_dir).returncode == 0
    manifest_before = (addr_dir / "MANIFEST.json").read_text()
    ofac_before = (addr_dir / "ofac.json").read_text()
    # Same content, fresher fetch timestamps: nothing may change,
    # or last_updated would claim stale data was refreshed.
    _write_all_caches(feeds_dir, updated_at=2000.0)
    r = _run(feeds_dir, addr_dir)
    assert r.returncode == 0, r.stderr
    assert (addr_dir / "MANIFEST.json").read_text() == manifest_before
    assert (addr_dir / "ofac.json").read_text() == ofac_before


def test_changed_feed_bumps_last_updated_and_only_its_timestamp(tmp_path):
    feeds_dir, addr_dir = tmp_path / "feeds", tmp_path / "addresses"
    _write_all_caches(feeds_dir, updated_at=1000.0)
    assert _run(feeds_dir, addr_dir).returncode == 0
    first = _manifest(addr_dir)
    ofac_before = (addr_dir / "ofac.json").read_text()
    grown = [_addr(100 + j) for j in range(6)]
    _write_all_caches(feeds_dir, updated_at=2000.0,
                      overrides={"phishing": grown})
    r = _run(feeds_dir, addr_dir)
    assert r.returncode == 0, r.stderr
    second = _manifest(addr_dir)
    assert second["last_updated"] > first["last_updated"]
    assert second["feed_updated_at"]["phishing.json"] == 2000.0
    assert second["feed_updated_at"]["ofac.json"] == 1000.0
    assert (addr_dir / "ofac.json").read_text() == ofac_before
    phishing = json.loads((addr_dir / "phishing.json").read_text())
    assert phishing["count"] == 6


def test_missing_cache_fails_closed_and_writes_nothing(tmp_path):
    feeds_dir, addr_dir = tmp_path / "feeds", tmp_path / "addresses"
    _write_all_caches(feeds_dir, updated_at=1000.0)
    assert _run(feeds_dir, addr_dir).returncode == 0
    manifest_before = (addr_dir / "MANIFEST.json").read_text()
    (feeds_dir / "phishing.json").unlink()
    r = _run(feeds_dir, addr_dir)
    assert r.returncode == 1
    assert "phishing" in r.stderr
    assert (addr_dir / "MANIFEST.json").read_text() == manifest_before


def test_missing_cache_on_first_run_creates_nothing(tmp_path):
    feeds_dir, addr_dir = tmp_path / "feeds", tmp_path / "addresses"
    _write_all_caches(feeds_dir, updated_at=1000.0)
    (feeds_dir / "ofac_sdn.json").unlink()
    r = _run(feeds_dir, addr_dir)
    assert r.returncode == 1
    assert "ofac_sdn" in r.stderr
    assert not addr_dir.exists() or not list(addr_dir.glob("*.json"))


def test_workflow_uses_the_populate_script():
    text = WORKFLOW.read_text()
    assert "scripts/populate_addresses.py" in text
    # The silent-skip line of the old inline heredoc must not return.
    assert "if not src.exists(): continue" not in text
