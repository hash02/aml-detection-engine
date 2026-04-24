"""Notifier tests — channel discovery + fail-soft dispatch."""

from __future__ import annotations


def test_configured_channels_empty_when_no_env(monkeypatch):
    for env in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "SLACK_WEBHOOK_URL", "GENERIC_WEBHOOK_URL"):
        monkeypatch.delenv(env, raising=False)
    from engine.notifier import configured_channels
    assert configured_channels() == []


def test_configured_channels_detects_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "123")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/slack")
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    from engine.notifier import configured_channels
    cc = configured_channels()
    assert "telegram" in cc
    assert "slack"    in cc
    assert "webhook"  not in cc


def test_notify_with_no_channels_logs_but_does_not_raise(monkeypatch, caplog):
    for env in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "SLACK_WEBHOOK_URL", "GENERIC_WEBHOOK_URL"):
        monkeypatch.delenv(env, raising=False)
    from engine.notifier import notify
    result = notify("hello world", level="INFO")
    assert result.ok is True
    assert result.sent == []
    # Every channel should be marked skipped
    assert set(result.skipped) == {"telegram", "slack", "webhook"}


def test_notify_dispatches_to_slack(monkeypatch):
    """Swap in a fake requests.post to verify dispatch without network."""
    import engine.notifier as notifier
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/slack")
    for env in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "GENERIC_WEBHOOK_URL"):
        monkeypatch.delenv(env, raising=False)

    calls: list[dict] = []

    class _Resp:
        def raise_for_status(self): pass

    class _FakeRequests:
        def post(self, url, json=None, timeout=None):  # noqa: A002
            calls.append({"url": url, "json": json})
            return _Resp()

    fake = _FakeRequests()
    monkeypatch.setitem(__import__("sys").modules, "requests", fake)

    res = notifier.notify("test", level="WARN")
    assert res.sent == ["slack"]
    assert calls[0]["url"] == "https://example.com/slack"
    assert "⚠️" in calls[0]["json"]["text"]


def test_notify_captures_channel_failure(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/slack")
    for env in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "GENERIC_WEBHOOK_URL"):
        monkeypatch.delenv(env, raising=False)

    class _BadRequests:
        def post(self, *_, **__):
            raise RuntimeError("network down")

    monkeypatch.setitem(__import__("sys").modules, "requests", _BadRequests())

    from engine.notifier import notify
    res = notify("test", level="ERROR")
    assert res.sent == []
    assert res.failed and res.failed[0][0] == "slack"
