"""Case-management tests — ingest, transitions, triage queue."""

from __future__ import annotations

import pandas as pd
import pytest


def _alert(tx_id: str, sender: str, score: int = 85) -> dict:
    return {
        "id":          tx_id,
        "sender_id":   sender,
        "receiver_id": "0xdest",
        "amount":      9_500,
        "timestamp":   "2025-04-01T12:00:00",
        "risk_score":  score,
        "risk_level":  "CRITICAL",
        "reasons":     "structuring;",
        "alert":       True,
    }


def test_ingest_opens_new_case(tmp_path):
    from engine.cases import CaseManager
    cm = CaseManager(tmp_path / "c.db")
    case_id = cm.ingest_alert(_alert("tx1", "0xaaa"))
    assert case_id is not None
    case = cm.case(case_id)
    assert case["subject_id"] == "0xaaa"
    assert case["status"] == "OPEN"
    assert case["alert_count"] == 1
    assert case["alerts"][0]["tx_id"] == "tx1"


def test_ingest_merges_into_existing_open_case(tmp_path):
    from engine.cases import CaseManager
    cm = CaseManager(tmp_path / "c.db")
    c1 = cm.ingest_alert(_alert("tx1", "0xaaa", score=70))
    c2 = cm.ingest_alert(_alert("tx2", "0xaaa", score=95))
    assert c1 == c2
    case = cm.case(c1)
    assert case["alert_count"] == 2
    # Priority rolls up to max
    assert case["priority"] == 95


def test_ingest_batch_handles_mixed_alerts(tmp_path):
    from engine.cases import CaseManager
    cm = CaseManager(tmp_path / "c.db")
    df = pd.DataFrame([
        _alert("tx-a", "0xaaa"),
        {**_alert("tx-b", "0xbbb"), "alert": False},
        _alert("tx-c", "0xaaa"),
    ])
    n = cm.ingest_batch(df)
    assert n == 2
    assert cm.stats()["open"] == 1  # both 0xaaa alerts merged
    # 0xbbb has alert=False → not ingested
    queue = cm.triage_queue()
    subjects = {c["subject_id"] for c in queue}
    assert subjects == {"0xaaa"}


def test_valid_transition_in_review_to_escalated(tmp_path):
    from engine.cases import CaseManager
    cm = CaseManager(tmp_path / "c.db")
    cid = cm.ingest_alert(_alert("tx1", "0xaaa"))
    assert cm.transition(cid, "IN_REVIEW", actor="alice")
    assert cm.transition(cid, "ESCALATED", actor="alice", notes="dpos exposure")
    case = cm.case(cid)
    assert case["status"] == "ESCALATED"
    assert len(case["events"]) == 2


def test_invalid_transition_raises(tmp_path):
    from engine.cases import CaseManager, InvalidTransition
    cm = CaseManager(tmp_path / "c.db")
    cid = cm.ingest_alert(_alert("tx1", "0xaaa"))
    with pytest.raises(InvalidTransition):
        cm.transition(cid, "CLOSED", actor="alice")


def test_triage_queue_sorted_by_priority(tmp_path):
    from engine.cases import CaseManager
    cm = CaseManager(tmp_path / "c.db")
    cm.ingest_alert(_alert("tx-low",  "0xsubj1", score=40))
    cm.ingest_alert(_alert("tx-high", "0xsubj2", score=95))
    cm.ingest_alert(_alert("tx-mid",  "0xsubj3", score=65))
    queue = cm.triage_queue()
    assert [c["priority"] for c in queue] == [95, 65, 40]


def test_case_assignment(tmp_path):
    from engine.cases import CaseManager
    cm = CaseManager(tmp_path / "c.db")
    cid = cm.ingest_alert(_alert("tx1", "0xaaa"))
    assert cm.assign(cid, "reviewer_42")
    assert cm.case(cid)["assignee"] == "reviewer_42"
