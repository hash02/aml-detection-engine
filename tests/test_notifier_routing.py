"""Severity-routed notifier tests."""

from __future__ import annotations


class _FakeResponse:
    def raise_for_status(self):
        pass


class _FakeRequests:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.posts.append({"url": url, "json": json})
        return _FakeResponse()


def _install_fake(monkeypatch):
    fake = _FakeRequests()
    monkeypatch.setitem(__import__("sys").modules, "requests", fake)
    return fake


def test_routing_restricts_channels_per_level(monkeypatch):
    """WARN routes to slack; CRITICAL routes to telegram + slack."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL",  "https://example.com/slack")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID",   "123")
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    monkeypatch.setenv(
        "AML_NOTIFY_ROUTING",
        '{"WARN": ["slack"], "CRITICAL": ["slack", "telegram"]}',
    )
    fake = _install_fake(monkeypatch)

    from engine.notifier import notify
    r1 = notify("low-sev warning", level="WARN")
    assert r1.sent == ["slack"]
    assert "telegram" in r1.skipped

    fake.posts.clear()
    r2 = notify("critical!", level="CRITICAL")
    assert set(r2.sent) == {"slack", "telegram"}


def test_routing_wildcard_applies_when_level_missing(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/slack")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN",     raising=False)
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("AML_NOTIFY_ROUTING", '{"*": ["slack"]}')
    _install_fake(monkeypatch)

    from engine.notifier import notify
    r = notify("anything", level="INFO")
    assert r.sent == ["slack"]


def test_routing_unlisted_level_without_wildcard_is_suppressed(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/slack")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN",     raising=False)
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("AML_NOTIFY_ROUTING", '{"CRITICAL": ["slack"]}')
    _install_fake(monkeypatch)

    from engine.notifier import notify
    r = notify("info-level", level="INFO")
    # INFO is not listed, no wildcard → suppressed
    assert r.sent == []
    assert "slack" in r.skipped


def test_invalid_routing_env_falls_back_to_fanout(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/slack")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN",     raising=False)
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("AML_NOTIFY_ROUTING", "not-valid-json{{{")
    _install_fake(monkeypatch)

    from engine.notifier import notify
    r = notify("fallback behaviour", level="INFO")
    # Bad config should NOT suppress — fan out to every configured channel
    assert r.sent == ["slack"]
