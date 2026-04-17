"""SAR-SF export tests — verify FinCEN-compatible JSON shape."""

from __future__ import annotations

import json
from datetime import datetime


def _alert(reasons: str = "structuring;large_amount;") -> dict:
    return {
        "id":          "tx-sar-1",
        "sender_id":   "0xaaa",
        "receiver_id": "0xbbb",
        "amount":      9876.50,
        "timestamp":   datetime(2025, 4, 1, 12, 0, 0),
        "risk_score":  90,
        "risk_level":  "CRITICAL",
        "reasons":     reasons,
    }


def test_sar_sf_shape():
    from engine.sar_sf import build_sar_sf_report

    r = build_sar_sf_report(_alert())
    assert r["form_type"] == "SAR-SF"
    assert len(r["control_number"]) == 14 and r["control_number"].isdigit()
    assert r["suspicious_activity"]["amount_involved_usd"] == 9876.50
    assert r["suspicious_activity"]["risk_level"] == "CRITICAL"
    # Subjects include both sender and receiver
    roles = {s["role"] for s in r["subjects"]}
    assert roles == {"subject", "counterparty"}
    # Serialisable as JSON
    assert isinstance(json.dumps(r, default=str), str)


def test_sar_sf_activity_code_mapping():
    from engine.sar_sf import build_sar_sf_report

    # Structuring + layering → 906 + 809
    r = build_sar_sf_report(_alert("structuring;layering_cycle;"))
    codes = r["suspicious_activity"]["activity_codes"]
    assert "906" in codes
    assert "809" in codes


def test_sar_sf_unknown_rule_falls_back_to_other():
    from engine.sar_sf import build_sar_sf_report

    # No known tag in reasons — code list should just be []
    r = build_sar_sf_report(_alert("profile_UNKNOWN;"))
    assert r["suspicious_activity"]["activity_codes"] == []


def test_sar_sf_ofac_maps_to_sanctions():
    from engine.sar_sf import build_sar_sf_report

    r = build_sar_sf_report(_alert("OFAC_SDN_MATCH;large_amount;"))
    assert "501" in r["suspicious_activity"]["activity_codes"]


def test_sar_sf_narrative_override():
    from engine.sar_sf import build_sar_sf_report

    narrative = {"summary": "Ronin Bridge exploit", "signals": ["a"], "rec_actions": ["b"]}
    r = build_sar_sf_report(_alert(), narrative=narrative)
    assert r["narrative"]["summary"] == "Ronin Bridge exploit"
    assert r["narrative"]["signals"] == ["a"]
    assert r["narrative"]["recommended_actions"] == ["b"]
