"""
engine/audit.py — Append-only audit log for compliance review
==============================================================

Every alert fire and every analyst action is recorded to a local SQLite
database. In production this would be shipped to an immutable store
(CloudTrail / S3 Object Lock / append-only Postgres) — for this demo,
SQLite is a good-enough approximation that proves the pattern.

Design constraints:
  - Writes never block the engine hot path (best-effort, swallow errors)
  - Schema is strictly additive — new columns only, never drops
  - Every write is timestamped with microsecond UTC
  - Integrity: (id, input_hash) unique — prevents duplicate logging if
    the same batch is replayed

Public API:
  AuditLog(path)                  — opens/creates the db at `path`
  log.record_alert(row, cfg)      — write a single alert row
  log.record_batch(df, cfg)       — write every alert in a DataFrame
  log.record_review(id, reviewer, disposition, notes)
                                  — analyst action on a prior alert
  log.fetch(limit=50)             — most recent N events (for UI tail)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   REAL    NOT NULL,
  tx_id        TEXT,
  sender_id    TEXT,
  receiver_id  TEXT,
  amount       REAL,
  risk_score   INTEGER,
  risk_level   TEXT,
  rules_fired  TEXT,
  input_hash   TEXT    NOT NULL,
  UNIQUE(tx_id, input_hash)
);
CREATE INDEX IF NOT EXISTS alert_events_ts ON alert_events(created_at DESC);

CREATE TABLE IF NOT EXISTS review_events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at    REAL    NOT NULL,
  alert_id      INTEGER NOT NULL,
  reviewer      TEXT    NOT NULL,
  disposition   TEXT    NOT NULL,     -- 'escalate' | 'dismiss' | 'sar_filed'
  notes         TEXT,
  FOREIGN KEY(alert_id) REFERENCES alert_events(id)
);
"""


class AuditLog:
    """Thread-safe append-only audit log over SQLite."""

    def __init__(self, path: str | Path = "data/audit.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as con:
            con.executescript(SCHEMA)
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    @staticmethod
    def _input_hash(row: dict[str, Any]) -> str:
        """Stable hash of the (tx_id, sender, receiver, amount, ts) tuple."""
        key = f"{row.get('id', '')}|{row.get('sender_id', '')}|{row.get('receiver_id', '')}|{row.get('amount', '')}|{row.get('timestamp', '')}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # ── Writes (best-effort) ────────────────────────────────────────────
    def record_alert(self, row: dict[str, Any]) -> int | None:
        try:
            with self._lock, closing(self._connect()) as con:
                cur = con.execute(
                    """INSERT OR IGNORE INTO alert_events
                       (created_at, tx_id, sender_id, receiver_id, amount,
                        risk_score, risk_level, rules_fired, input_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        time.time(),
                        str(row.get("id", ""))[:128],
                        str(row.get("sender_id", ""))[:128],
                        str(row.get("receiver_id", ""))[:128],
                        float(row.get("amount", 0) or 0),
                        int(row.get("risk_score", 0) or 0),
                        str(row.get("risk_level", "")),
                        str(row.get("reasons", ""))[:1024],
                        self._input_hash(row),
                    ),
                )
                con.commit()
                return cur.lastrowid
        except Exception as e:  # noqa: BLE001 — audit log must never crash the caller
            log.warning("audit: record_alert failed: %s", e)
            return None

    def record_batch(self, df) -> int:
        """Write every `alert=True` row in a DataFrame. Returns count written."""
        if df is None or df.empty or "alert" not in df.columns:
            return 0
        alerts = df[df["alert"] == True]  # noqa: E712
        n = 0
        for _, row in alerts.iterrows():
            if self.record_alert(row.to_dict()) is not None:
                n += 1
        return n

    def record_review(
        self,
        alert_id: int,
        reviewer: str,
        disposition: str,
        notes: str = "",
    ) -> bool:
        try:
            with self._lock, closing(self._connect()) as con:
                con.execute(
                    """INSERT INTO review_events
                       (created_at, alert_id, reviewer, disposition, notes)
                       VALUES (?, ?, ?, ?, ?)""",
                    (time.time(), alert_id, reviewer[:64], disposition[:32], notes[:1024]),
                )
                con.commit()
                return True
        except Exception as e:  # noqa: BLE001
            log.warning("audit: record_review failed: %s", e)
            return False

    # ── Reads ───────────────────────────────────────────────────────────
    def fetch(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as con:
                cur = con.execute(
                    """SELECT id, created_at, tx_id, sender_id, receiver_id,
                              amount, risk_score, risk_level, rules_fired
                         FROM alert_events
                         ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                )
                cols = [c[0] for c in cur.description]
                return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
        except Exception as e:  # noqa: BLE001
            log.warning("audit: fetch failed: %s", e)
            return []

    def dump_json(self, limit: int = 1000) -> str:
        return json.dumps(self.fetch(limit), indent=2, default=str)
