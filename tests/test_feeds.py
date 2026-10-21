"""Feed loader tests — verify offline / baseline / caching invariants."""

from __future__ import annotations


def test_get_feed_returns_baseline_when_cache_missing(tmp_path, monkeypatch):
    # Point the feed dir at a fresh tmp so we start from an empty cache
    monkeypatch.setenv("FEEDS_OFFLINE", "1")
    import importlib

    import engine.feeds as feeds
    importlib.reload(feeds)
    feeds.FEEDS_DIR = tmp_path
    feeds._MEM_CACHE.clear()
    ofac = feeds.get_feed("ofac_sdn")
    assert len(ofac) >= 10
    assert "0x098b716b8aaf21512996dc57eb0615e2383e2f96" in ofac  # Ronin exploiter


def test_get_feed_unknown_feed_raises():
    import engine.feeds as feeds
    try:
        feeds.get_feed("does_not_exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_refresh_offline_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDS_OFFLINE", "1")
    import importlib

    import engine.feeds as feeds
    importlib.reload(feeds)
    feeds.FEEDS_DIR = tmp_path
    feeds._MEM_CACHE.clear()
    counts = feeds.refresh()
    # Even offline, every feed reports its baseline size
    for name in feeds.FEEDS:
        assert counts.get(name, -1) >= 0


def test_chainalysis_check_returns_none_without_rpc():
    import engine.feeds as feeds
    assert feeds.check_chainalysis("0x098b716b8aaf21512996dc57eb0615e2383e2f96", None) is None


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _mock_get(monkeypatch, payload):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResp(payload))


def test_scamsniffer_fetcher_keeps_only_wellformed(monkeypatch):
    import engine.feeds as feeds
    payload = ["0x" + format(i, "040x") for i in range(98)] + ["0x-scam.example", "junk"]
    _mock_get(monkeypatch, payload)
    out = feeds._fetch_scamsniffer_phish()
    assert len(out) == 98
    assert all(len(a) == 42 for a in out)


def test_scamsniffer_fetcher_rejects_domain_payload(monkeypatch):
    import engine.feeds as feeds
    _mock_get(monkeypatch, ["0x-alphafi.pages.dev", "0x-altseason.xyz"] * 30)
    try:
        feeds._fetch_scamsniffer_phish()
    except ValueError:
        return
    raise AssertionError("expected ValueError on a domain payload")


def test_scamsniffer_fetcher_rejects_nonlist_payload(monkeypatch):
    import engine.feeds as feeds
    _mock_get(monkeypatch, {"blacklist": ["0x" + "a" * 40]})
    try:
        feeds._fetch_scamsniffer_phish()
    except ValueError:
        return
    raise AssertionError("expected ValueError on a non-list payload")
