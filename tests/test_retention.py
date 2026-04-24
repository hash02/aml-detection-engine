"""Retention + PII-redaction tests."""

from __future__ import annotations

import sqlite3
import time


def test_redact_pii_masks_common_patterns():
    from engine.retention import redact_pii

    text = "Flagged by jane.doe@example.com, ticket #12345. Phone: 415-555-1212."
    out = redact_pii(text)
    assert "jane.doe@example.com" not in out
    assert "[redacted-email]" in out
    assert "[redacted-ref]"   in out
    assert "[redacted-phone]" in out


def test_redact_pii_preserves_wallet_addresses():
    """Wallet addresses are public — redaction must not mask them."""
    from engine.retention import redact_pii
    text = "Subject 0x098b716b8aaf21512996dc57eb0615e2383e2f96 flagged"
    assert "0x098b716b8aaf21512996dc57eb0615e2383e2f96" in redact_pii(text)


def test_redact_dict_only_named_fields():
    from engine.retention import redact_dict
    d = {
        "tx_id":  "abc",
        "notes":  "contact alice@bank.com about this",
        "sender": "0xaaa",
    }
    out = redact_dict(d)
    assert "[redacted-email]" in out["notes"]
    assert out["tx_id"]  == "abc"
    assert out["sender"] == "0xaaa"


def test_purge_dry_run_does_not_delete(tmp_path):
    """Dry-run counts but doesn't mutate."""
    from engine.audit import AuditLog
    from engine.retention import purge

    audit_path = tmp_path / "audit.db"
    audit = AuditLog(audit_path)
    # Backdate a row by poking sqlite directly
    aid = audit.record_alert({
        "id": "old-tx", "sender_id": "0xa", "receiver_id": "0xb",
        "amount": 9_500, "timestamp": "2020-01-01",
        "risk_score": 90, "risk_level": "CRITICAL", "reasons": "structuring;",
    })
    assert aid
    ancient = time.time() - 400 * 86400
    with sqlite3.connect(str(audit_path)) as con:
        con.execute("UPDATE alert_events SET created_at = ? WHERE id = ?",
                    (ancient, aid))
        con.commit()

    result = purge(audit_db=audit_path, retention_days=365, dry_run=True)
    assert result.get("alert_events", 0) >= 1
    # Still there
    assert len(audit.fetch()) == 1


def test_purge_applies_retention(tmp_path):
    from engine.audit import AuditLog
    from engine.retention import purge

    audit_path = tmp_path / "audit.db"
    audit = AuditLog(audit_path)
    aid_old = audit.record_alert({
        "id": "old", "sender_id": "0xa", "receiver_id": "0xb",
        "amount": 1, "timestamp": "2020-01-01",
        "risk_score": 90, "risk_level": "CRITICAL", "reasons": "s;",
    })
    aid_new = audit.record_alert({
        "id": "new", "sender_id": "0xa", "receiver_id": "0xb",
        "amount": 1, "timestamp": "2025-04-01",
        "risk_score": 90, "risk_level": "CRITICAL", "reasons": "s;",
    })
    assert aid_old and aid_new
    ancient = time.time() - 400 * 86400
    with sqlite3.connect(str(audit_path)) as con:
        con.execute("UPDATE alert_events SET created_at = ? WHERE id = ?",
                    (ancient, aid_old))
        con.commit()

    result = purge(audit_db=audit_path, retention_days=365)
    assert result.get("alert_events", 0) == 1
    # Only the new row survives
    remaining = [e["tx_id"] for e in audit.fetch()]
    assert remaining == ["new"]


def test_forget_subject_removes_all_references(tmp_path):
    from engine.audit import AuditLog
    from engine.cases import CaseManager
    from engine.retention import forget_subject

    audit_path = tmp_path / "a.db"
    cases_path = tmp_path / "c.db"
    audit = AuditLog(audit_path)
    cases = CaseManager(cases_path)
    for subj in ("0xforget", "0xkeep"):
        aid = audit.record_alert({
            "id": f"tx-{subj}", "sender_id": subj, "receiver_id": "0xrx",
            "amount": 1, "timestamp": "2025-04-01",
            "risk_score": 90, "risk_level": "HIGH", "reasons": "s;",
        })
        cases.ingest_alert({
            "id": f"tx-{subj}", "sender_id": subj, "receiver_id": "0xrx",
            "amount": 1, "timestamp": "2025-04-01",
            "risk_score": 90, "risk_level": "HIGH", "reasons": "s;",
        })
        assert aid

    deleted = forget_subject("0xforget", audit_db=audit_path, cases_db=cases_path)
    assert any(v > 0 for v in deleted.values())

    # 0xkeep should still be in both stores
    assert any(e["sender_id"] == "0xkeep" for e in audit.fetch())
    assert any(c["subject_id"] == "0xkeep" for c in cases.triage_queue())


def test_purge_on_missing_db_is_noop():
    from engine.retention import purge
    assert purge(audit_db="/tmp/does-not-exist.db") == {}
