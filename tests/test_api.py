"""FastAPI sidecar tests — exercised via TestClient (no network)."""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(scope="module")
def client():
    fastapi = pytest.importorskip("fastapi")
    starlette = pytest.importorskip("starlette")
    assert fastapi and starlette  # silence unused-import
    os.environ.setdefault("FEEDS_OFFLINE", "1")
    os.environ.pop("AML_API_TOKEN", None)  # open mode for these tests

    # Force-reimport so env is picked up
    import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_feeds_status(client):
    r = client.get("/feeds")
    assert r.status_code == 200
    body = r.json()
    assert body.get("available") is True
    names = {f["name"] for f in body["feeds"]}
    assert "ofac_sdn" in names


def test_score_endpoint_flags_ofac_sender(client):
    """Ronin exploiter address is in the OFAC baseline — should flag."""
    ronin = "0x098b716b8aaf21512996dc57eb0615e2383e2f96"
    req = {
        "transactions": [
            {
                "id": "tx-1",
                "sender_id": ronin,
                "receiver_id": "0x0000000000000000000000000000000000000001",
                "amount": 1_000_000,
                "timestamp": "2025-04-01T12:00:00",
                "country": "KP",
                "sender_profile": "PERSONAL_LIKE",
            },
        ],
    }
    r = client.post("/score", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["flagged"] == 1
    assert body["results"][0]["alert"] is True
    assert "OFAC" in body["results"][0]["reasons"]


def test_score_empty_batch_returns_empty(client):
    r = client.post("/score", json={"transactions": []})
    assert r.status_code == 200
    body = r.json()
    assert body == {"count": 0, "flagged": 0, "aggregate_rate": 0.0, "results": []}


def test_audit_endpoint_requires_token_when_configured(monkeypatch):
    """With AML_API_TOKEN set, /audit requires matching Bearer token."""
    fastapi = pytest.importorskip("fastapi")
    assert fastapi
    monkeypatch.setenv("AML_API_TOKEN", "s3cret")
    monkeypatch.setenv("FEEDS_OFFLINE", "1")

    import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)

    assert c.get("/audit").status_code == 401
    assert c.get("/audit", headers={"Authorization": "Bearer wrong"}).status_code == 403
    ok = c.get("/audit", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
