"""Phase-6 API tests — /score explain, /cases, /drift."""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    os.environ.setdefault("FEEDS_OFFLINE", "1")
    os.environ.pop("AML_API_TOKEN", None)
    import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def test_score_with_explain_returns_breakdown(client):
    """Requesting `explain=True` attaches per-rule breakdown to every result."""
    ronin = "0x098b716b8aaf21512996dc57eb0615e2383e2f96"
    req = {
        "transactions": [
            {
                "id": "tx-ex", "sender_id": ronin,
                "receiver_id": "0x0000000000000000000000000000000000000001",
                "amount": 500_000, "timestamp": "2025-04-01T12:00:00",
                "country": "KP",
            },
        ],
        "explain": True,
    }
    r = client.post("/score", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["breakdown"] is not None
    rules = {p["rule"] for p in body["results"][0]["breakdown"]}
    # Large amount + OFAC should both contribute
    assert "large_amount" in rules
    assert "OFAC_SDN_MATCH" in rules


def test_score_without_explain_omits_breakdown(client):
    req = {
        "transactions": [
            {
                "id": "tx-noex", "sender_id": "0xaaa", "receiver_id": "0xbbb",
                "amount": 100, "timestamp": "2025-04-01T12:00:00",
            },
        ],
    }
    r = client.post("/score", json=req)
    assert r.status_code == 200
    assert r.json()["results"][0].get("breakdown") is None


def test_cases_endpoint_returns_queue(client, tmp_path, monkeypatch):
    """Freshly-started case db returns an empty queue but valid shape."""
    monkeypatch.setenv("AML_CASES_DB", str(tmp_path / "cases.db"))
    # Re-import api to pick up new env var for _CASES
    import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    r = c.get("/cases")
    assert r.status_code == 200
    body = r.json()
    assert "queue" in body and "stats" in body
    assert isinstance(body["queue"], list)


def test_drift_endpoint_returns_no_alerts_on_empty_db(client, tmp_path, monkeypatch):
    """With no history, drift detection returns zero alerts cleanly."""
    monkeypatch.setenv("AML_AUDIT_DB", str(tmp_path / "audit.db"))
    import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    r = c.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["alert_count"] == 0
    assert body["alerts"] == []
