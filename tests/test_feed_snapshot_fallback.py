"""Snapshot fallback tests: get_feed() precedence cache > snapshot > baseline.

The committed addresses/ snapshot is the only copy of the feeds a
fresh deployment has. These tests pin the precedence rule and the
canary that the real committed snapshot loads through the runtime
code path (Claude's finding 1: before this fallback, no engine code
read addresses/*.json at all).
"""

from __future__ import annotations

import json
from pathlib import Path


def _addr(n: int) -> str:
    return "0x" + format(n, "040x")


def _fresh_feeds(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDS_OFFLINE", "1")
    import importlib

    import engine.feeds as feeds
    importlib.reload(feeds)
    feeds.FEEDS_DIR = tmp_path / "cache"
    feeds.FEEDS_DIR.mkdir(parents=True, exist_ok=True)
    feeds.SNAPSHOTS_DIR = tmp_path / "snapshots"
    feeds.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    feeds._MEM_CACHE.clear()
    return feeds


def _write(directory: Path, filename: str, addresses, feed="ofac_sdn"):
    blob = {
        "feed": feed,
        "updated_at": 1792500000.0,
        "count": len(addresses),
        "source": "test",
        "addresses": list(addresses),
    }
    (directory / filename).write_text(json.dumps(blob))


def test_cache_beats_snapshot(tmp_path, monkeypatch):
    feeds = _fresh_feeds(tmp_path, monkeypatch)
    cache_only, snap_only = _addr(1001), _addr(2002)
    _write(feeds.FEEDS_DIR, "ofac_sdn.json", [cache_only])
    _write(feeds.SNAPSHOTS_DIR, "ofac.json", [snap_only])
    got = feeds.get_feed("ofac_sdn")
    assert cache_only in got
    assert snap_only not in got  # snapshot not consulted when a cache exists


def test_snapshot_beats_baseline(tmp_path, monkeypatch):
    feeds = _fresh_feeds(tmp_path, monkeypatch)
    snap_only = _addr(3003)
    assert snap_only not in feeds.BASELINE_OFAC
    _write(feeds.SNAPSHOTS_DIR, "ofac.json", [snap_only])
    got = feeds.get_feed("ofac_sdn")
    assert snap_only in got
    # Baseline is unioned in, never displaced.
    assert feeds.BASELINE_OFAC <= got
    assert len(got) > len(feeds.BASELINE_OFAC)


def test_malformed_snapshot_entries_filtered(tmp_path, monkeypatch):
    feeds = _fresh_feeds(tmp_path, monkeypatch)
    good = _addr(4004)
    _write(feeds.SNAPSHOTS_DIR, "phishing.json",
           [good, "0x-auth.com", "0x-altseason.xyz", "not-an-address"],
           feed="phishing")
    got = feeds.get_feed("phishing")
    assert got == frozenset({good})


def test_unreadable_snapshot_falls_back_to_baseline(tmp_path, monkeypatch):
    feeds = _fresh_feeds(tmp_path, monkeypatch)
    (feeds.SNAPSHOTS_DIR / "ofac.json").write_text("{not json")
    assert feeds.get_feed("ofac_sdn") == feeds.BASELINE_OFAC


def test_chainalysis_has_no_snapshot_and_stays_empty(tmp_path, monkeypatch):
    feeds = _fresh_feeds(tmp_path, monkeypatch)
    assert feeds.FEEDS["chainalysis"].snapshot_path is None
    assert feeds.get_feed("chainalysis") == frozenset()


def test_canary_committed_snapshot_loads_via_get_feed(tmp_path, monkeypatch):
    """Canary: with an empty cache, the repo's real committed snapshot
    must come back through get_feed(). Expectations are computed from
    the committed files at run time, not hardcoded."""
    monkeypatch.setenv("FEEDS_OFFLINE", "1")
    import importlib

    import engine.feeds as feeds
    importlib.reload(feeds)
    feeds.FEEDS_DIR = tmp_path / "empty-cache"
    feeds.FEEDS_DIR.mkdir(parents=True, exist_ok=True)
    feeds._MEM_CACHE.clear()
    repo_addresses = Path(feeds.__file__).resolve().parent.parent / "addresses"
    assert feeds.SNAPSHOTS_DIR == repo_addresses

    ofac_snapshot = feeds._lower(
        json.loads((repo_addresses / "ofac.json").read_text())["addresses"])
    beyond_baseline = ofac_snapshot - feeds.BASELINE_OFAC
    assert beyond_baseline, "committed OFAC snapshot adds nothing over baseline"
    got_ofac = feeds.get_feed("ofac_sdn")
    # One well-known sanctioned address that exists only in the snapshot.
    assert sorted(beyond_baseline)[0] in got_ofac
    assert beyond_baseline <= got_ofac

    phish_snapshot = feeds._lower(
        json.loads((repo_addresses / "phishing.json").read_text())["addresses"])
    assert len(phish_snapshot) > 1000
    got_phish = feeds.get_feed("phishing")
    # Phishing baseline is empty, so every returned address is proof
    # the snapshot was loaded through the runtime path.
    assert got_phish == phish_snapshot
    assert sorted(phish_snapshot)[0] in got_phish
