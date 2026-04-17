"""
engine/notifier.py — Unified notification abstraction
======================================================

Why this exists: pipeline.py had Telegram logic hardcoded. Real ops
teams want to alert into Slack, a webhook, PagerDuty, OpsGenie, or
just stdout depending on severity + environment. This module gives
every caller one function — `notify(message, level)` — and handles
fan-out to whichever channels are configured.

Channels are discovered entirely from environment variables:
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID  → Telegram
  SLACK_WEBHOOK_URL                       → Slack incoming webhook
  GENERIC_WEBHOOK_URL                     → POST JSON to any URL
(nothing set)                            → stdout only

No channel is ever a hard dependency. When zero are configured,
`notify()` still logs the message at the `level` log level.

Severity map:
  DEBUG, INFO → fires on every channel
  WARN        → fires on every channel
  ERROR       → fires on every channel, plus prefixes with 🔴
  CRITICAL    → fires on every channel, prefixed 🚨 + forces sync send

Designed to be called from:
  - engine hot path (alert-driven notifications)
  - CI jobs (feed refresh failures, backtest regressions)
  - Streamlit user actions (admin-triggered manual alerts)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 8


@dataclass
class NotifyResult:
    sent: list[str]
    skipped: list[str]
    failed: list[tuple[str, str]]          # (channel, error)

    @property
    def ok(self) -> bool:
        return not self.failed


def _emoji_for(level: str) -> str:
    return {
        "DEBUG":    "·",
        "INFO":     "ℹ️",
        "WARN":     "⚠️",
        "WARNING":  "⚠️",
        "ERROR":    "🔴",
        "CRITICAL": "🚨",
    }.get(level.upper(), "•")


def _format(message: str, level: str) -> str:
    return f"{_emoji_for(level)} [{level.upper()}] {message}"


# ── Channel implementations ─────────────────────────────────────────

def _send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        raise RuntimeError("telegram: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID not set")
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram limit ~4096 chars
    body = text if len(text) <= 4000 else text[:3997] + "..."
    resp = requests.post(
        url,
        json={"chat_id": chat_id, "text": body, "parse_mode": "Markdown"},
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()


def _send_slack(text: str) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        raise RuntimeError("slack: SLACK_WEBHOOK_URL not set")
    import requests
    resp = requests.post(
        url,
        json={"text": text},
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()


def _send_webhook(text: str, payload: dict[str, Any] | None = None) -> None:
    url = os.environ.get("GENERIC_WEBHOOK_URL")
    if not url:
        raise RuntimeError("webhook: GENERIC_WEBHOOK_URL not set")
    import requests
    body = {"text": text, **(payload or {})}
    resp = requests.post(url, json=body, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()


_CHANNELS = (
    ("telegram", _send_telegram, "TELEGRAM_BOT_TOKEN"),
    ("slack",    _send_slack,    "SLACK_WEBHOOK_URL"),
    ("webhook",  _send_webhook,  "GENERIC_WEBHOOK_URL"),
)


# ── Public API ──────────────────────────────────────────────────────

def configured_channels() -> list[str]:
    """Channels whose required env var is set."""
    return [name for name, _, env in _CHANNELS if os.environ.get(env) or (
        name == "telegram" and os.environ.get("TELEGRAM_TOKEN")
    )]


def notify(
    message: str,
    level: str = "INFO",
    extra: dict[str, Any] | None = None,
) -> NotifyResult:
    """Dispatch `message` to every configured channel.

    Never raises — failures per channel are captured in the returned
    NotifyResult. Always logs at the requested level.
    """
    text = _format(message, level)
    log.log(getattr(logging, level.upper(), logging.INFO), text)

    sent: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for name, fn, env in _CHANNELS:
        has_cfg = bool(os.environ.get(env))
        if name == "telegram" and not has_cfg and os.environ.get("TELEGRAM_TOKEN"):
            has_cfg = True  # legacy env var
        if not has_cfg:
            skipped.append(name)
            continue
        try:
            if name == "webhook":
                fn(text, extra)  # type: ignore[call-arg]
            else:
                fn(text)
            sent.append(name)
        except Exception as e:  # noqa: BLE001
            failed.append((name, str(e)))

    return NotifyResult(sent=sent, skipped=skipped, failed=failed)


def notify_json(payload: dict[str, Any], level: str = "INFO") -> NotifyResult:
    """Convenience: dispatch a JSON-serialisable payload as its string."""
    return notify(json.dumps(payload, default=str)[:3800], level=level, extra=payload)
