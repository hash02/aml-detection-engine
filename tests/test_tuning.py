"""Tuning tests — disposition feedback → threshold suggestions."""

from __future__ import annotations

import pytest


@pytest.fixture
def seeded_audit(tmp_path):
    """Build an AuditLog + review history mimicking a week of reviewer work."""
    from engine.audit import AuditLog
    audit = AuditLog(tmp_path / "audit.db")

    # 30 alerts on structuring — 25 dismissed = noisy rule
    for i in range(30):
        aid = audit.record_alert({
            "id":          f"struct-{i}",
            "sender_id":   f"0xaaa{i:03d}",
            "receiver_id": "0xdest",
            "amount":      9500,
            "timestamp":   "2025-04-01T12:00:00",
            "risk_score":  85,
            "risk_level":  "CRITICAL",
            "reasons":     "structuring;",
        })
        if aid and i < 25:
            audit.record_review(aid, "alice", "dismiss")
        elif aid:
            audit.record_review(aid, "alice", "escalate")

    # 15 alerts on OFAC — 12 escalated or SAR-filed = high-value rule
    for i in range(15):
        aid = audit.record_alert({
            "id":          f"ofac-{i}",
            "sender_id":   f"0xooo{i:03d}",
            "receiver_id": "0xdest",
            "amount":      50_000,
            "timestamp":   "2025-04-01T12:00:00",
            "risk_score":  150,
            "risk_level":  "CRITICAL",
            "reasons":     "OFAC_SDN_MATCH;",
        })
        if aid and i < 12:
            audit.record_review(aid, "alice", "sar_filed" if i % 2 else "escalate")
        elif aid:
            audit.record_review(aid, "alice", "dismiss")

    return tmp_path / "audit.db"


def test_suggest_flags_noisy_rule_as_raise_threshold(seeded_audit):
    from engine.tuning import suggest
    suggestions = suggest(audit_db=seeded_audit)
    by_rule = {s.rule: s for s in suggestions}
    assert "structuring" in by_rule
    assert by_rule["structuring"].suggestion == "raise_threshold"
    assert by_rule["structuring"].dismiss_rate > 0.6
    assert by_rule["structuring"].reviewed == 30


def test_suggest_flags_high_escalate_rule_as_lower_threshold(seeded_audit):
    from engine.tuning import suggest
    suggestions = suggest(audit_db=seeded_audit)
    by_rule = {s.rule: s for s in suggestions}
    assert "OFAC_SDN_MATCH" in by_rule
    assert by_rule["OFAC_SDN_MATCH"].suggestion == "lower_threshold"
    assert by_rule["OFAC_SDN_MATCH"].escalate_rate >= 0.70


def test_suggest_below_min_samples_is_skipped(tmp_path):
    """Rules with < min_samples reviewed alerts produce no suggestion."""
    from engine.audit import AuditLog
    from engine.tuning import suggest
    audit = AuditLog(tmp_path / "audit.db")
    for i in range(3):
        aid = audit.record_alert({
            "id": f"t-{i}", "sender_id": "0xa", "receiver_id": "0xb",
            "amount": 1, "timestamp": "2025-04-01T00:00:00",
            "risk_score": 99, "risk_level": "HIGH",
            "reasons": "phishing_hit;",
        })
        if aid:
            audit.record_review(aid, "a", "dismiss")
    assert suggest(audit_db=tmp_path / "audit.db") == []


def test_suggest_handles_missing_db():
    from engine.tuning import suggest
    assert suggest(audit_db="/tmp/does-not-exist.db") == []


def test_suggestion_stringifies():
    from engine.tuning import TuningSuggestion
    s = TuningSuggestion(
        rule="foo", reviewed=20, dismiss_rate=0.7, escalate_rate=0.1,
        suggestion="raise_threshold", confidence=0.4,
        rationale="14 of 20 dismissed",
    )
    out = str(s)
    assert "foo" in out and "n=20" in out and "raise_threshold" in out
