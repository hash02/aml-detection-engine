"""Audit-log tests — verify append-only writes + idempotent replay."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def _alert_row(tx_id: str = "tx-1", amount: float = 9500.0) -> dict:
    return {
        "id":          tx_id,
        "sender_id":   "0xabc",
        "receiver_id": "0xdef",
        "amount":      amount,
        "timestamp":   datetime(2025, 4, 1, 12, 0, 0),
        "risk_score":  85,
        "risk_level":  "CRITICAL",
        "reasons":     "structuring;large_amount;",
        "alert":       True,
    }


def test_audit_log_writes_and_reads(tmp_path):
    from engine.audit import AuditLog

    log = AuditLog(tmp_path / "audit.db")
    rid = log.record_alert(_alert_row())
    assert rid is not None

    events = log.fetch(limit=10)
    assert len(events) == 1
    assert events[0]["tx_id"] == "tx-1"
    assert events[0]["risk_level"] == "CRITICAL"
    assert "structuring" in events[0]["rules_fired"]


def test_audit_log_deduplicates_identical_writes(tmp_path):
    """(tx_id, input_hash) is unique — replaying the same alert is a no-op."""
    from engine.audit import AuditLog

    log = AuditLog(tmp_path / "audit.db")
    row = _alert_row("tx-2")
    rid1 = log.record_alert(row)
    rid2 = log.record_alert(row)
    # Second write is INSERT OR IGNORE → lastrowid is 0 or the row's id,
    # but the row count stays at 1.
    assert rid1 is not None
    assert len(log.fetch()) == 1
    _ = rid2  # explicit: we don't care which value sqlite returned


def test_audit_log_record_batch_writes_only_alerts(tmp_path):
    from engine.audit import AuditLog

    log = AuditLog(tmp_path / "audit.db")
    df = pd.DataFrame([
        _alert_row("tx-A"),
        {**_alert_row("tx-B"), "alert": False},
        _alert_row("tx-C"),
    ])
    n = log.record_batch(df)
    assert n == 2  # only the two with alert=True
    tx_ids = {e["tx_id"] for e in log.fetch()}
    assert tx_ids == {"tx-A", "tx-C"}


def test_audit_log_review_event_links_to_alert(tmp_path):
    from engine.audit import AuditLog

    log = AuditLog(tmp_path / "audit.db")
    alert_id = log.record_alert(_alert_row("tx-X"))
    assert alert_id is not None
    ok = log.record_review(alert_id, "analyst1", "escalate", "manual review")
    assert ok is True


def test_audit_log_survives_malformed_row(tmp_path):
    """Best-effort writes must never raise into the caller."""
    from engine.audit import AuditLog

    log = AuditLog(tmp_path / "audit.db")
    # Missing keys, wrong types, weird payload — must not raise
    assert log.record_alert({"id": None, "amount": "nope"}) is None or log.record_alert({"id": None, "amount": "nope"}) >= 0
