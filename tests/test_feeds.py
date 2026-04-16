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
