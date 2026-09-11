import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api
from engine.audit import AuditLog


@pytest.fixture
def audit_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_API_TOKEN", "test-only-token")
    audit = AuditLog(tmp_path / "audit.db")
    monkeypatch.setattr(api, "_AUDIT", audit)
    def scored(df, cfg):
        return df.assign(risk_score=90, risk_level="HIGH", alert=True, reasons="test")
    monkeypatch.setattr("scripts.backtest.run_full_pipeline", scored)
    return TestClient(api.app), audit


def request_body(write=True):
    return {"write_audit": write, "transactions": [{
        "id": "test-1", "sender_id": "sender", "receiver_id": "receiver",
        "amount": 10, "timestamp": "2026-01-01T00:00:00",
    }]}


@pytest.mark.parametrize("header,status", [(None, 401), ("Bearer wrong", 403)])
def test_audit_write_requires_token(audit_client, header, status):
    client, audit = audit_client
    headers = {"Authorization": header} if header else {}
    response = client.post("/score", json=request_body(), headers=headers)
    assert response.status_code == status
    assert audit.fetch() == []


def test_missing_token_configuration_blocks_protected_routes(audit_client, monkeypatch):
    client, audit = audit_client
    monkeypatch.delenv("AML_API_TOKEN")
    assert client.post("/score", json=request_body()).status_code == 503
    assert client.get("/audit").status_code == 503
    assert audit.fetch() == []


def test_public_scoring_does_not_write(audit_client):
    client, audit = audit_client
    response = client.post("/score", json=request_body(False))
    assert response.status_code == 200
    assert audit.fetch() == []


def test_authenticated_audit_write_is_persisted_and_replay_safe(audit_client):
    client, audit = audit_client
    for _ in range(2):
        response = client.post("/score", json=request_body(),
                               headers={"Authorization": "Bearer test-only-token"})
        assert response.status_code == 200
    assert len(audit.fetch()) == 1


def test_underlying_database_failure_cannot_report_success(audit_client, monkeypatch):
    client, audit = audit_client
    def unavailable():
        raise sqlite3.OperationalError("synthetic database failure")
    monkeypatch.setattr(audit, "_connect", unavailable)
    response = client.post("/score", json=request_body(),
                           headers={"Authorization": "Bearer test-only-token"})
    assert response.status_code == 503
    assert "synthetic database failure" not in response.text


def test_batch_exception_cannot_report_success(audit_client, monkeypatch):
    client, audit = audit_client
    def unavailable(df):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(audit, "record_batch", unavailable)
    response = client.post("/score", json=request_body(),
                           headers={"Authorization": "Bearer test-only-token"})
    assert response.status_code == 503
