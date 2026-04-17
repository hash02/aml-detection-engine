"""
engine/cases.py — Case management for grouped alerts
======================================================

A "case" is an investigation — one or more alerts tied to the same
subject (usually a sender wallet), worked as a unit by reviewers.
Without cases, every tx alert is an island; reviewers re-learn the
context every time. With cases, the analyst opens one record,
sees every alert in chronological order, and advances the case
through its lifecycle.

Case lifecycle (strict FSM — enforced by transition()):
  OPEN → IN_REVIEW → ESCALATED → CLOSED
  OPEN → IN_REVIEW → DISMISSED  → CLOSED
  OPEN → IN_REVIEW → SAR_FILED  → CLOSED
  any  → REOPENED (admin only)

Schema (SQLite, sibling to the audit db):
  cases         — one row per investigation
  case_alerts   — many-to-one: every alert belongs to one case
  case_events   — append-only history of state transitions

Cases are derived deterministically: the first OPEN case per subject
within a rolling `case_window_days` gets subsequent alerts attached.
Reviewers can also merge / split manually via the UI (later PR).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id    TEXT    NOT NULL,           -- usually the sender address
  opened_at     REAL    NOT NULL,
  updated_at    REAL    NOT NULL,
  status        TEXT    NOT NULL DEFAULT 'OPEN',
  priority      INTEGER NOT NULL DEFAULT 0, -- rolled-up max alert score
  alert_count   INTEGER NOT NULL DEFAULT 0,
  assignee      TEXT    DEFAULT NULL,
  notes         TEXT    DEFAULT ''
);
CREATE INDEX IF NOT EXISTS cases_subject ON cases(subject_id, status);
CREATE INDEX IF NOT EXISTS cases_priority ON cases(priority DESC, updated_at DESC);

CREATE TABLE IF NOT EXISTS case_alerts (
  case_id  INTEGER NOT NULL,
  tx_id    TEXT    NOT NULL,
  added_at REAL    NOT NULL,
  PRIMARY KEY (case_id, tx_id),
  FOREIGN KEY (case_id) REFERENCES cases(id)
);

CREATE TABLE IF NOT EXISTS case_events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  case_id    INTEGER NOT NULL,
  created_at REAL    NOT NULL,
  actor      TEXT    NOT NULL,
  from_status TEXT,
  to_status  TEXT    NOT NULL,
  notes      TEXT
);
"""

# Valid transitions — anything not here raises InvalidTransition.
_TRANSITIONS = {
    "OPEN":       {"IN_REVIEW", "DISMISSED"},
    "IN_REVIEW":  {"ESCALATED", "DISMISSED", "SAR_FILED", "OPEN"},
    "ESCALATED":  {"CLOSED", "SAR_FILED"},
    "DISMISSED":  {"CLOSED", "OPEN"},    # admin can reopen
    "SAR_FILED":  {"CLOSED"},
    "CLOSED":     {"OPEN"},              # admin re-open
}


class InvalidTransition(ValueError):
    """Raised when a caller tries an illegal status transition."""


class CaseManager:
    """Thread-safe case store backed by SQLite."""

    def __init__(self, path: str | Path = "data/cases.db", window_days: int = 14):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window_seconds = window_days * 86400
        self._lock = threading.Lock()
        with closing(self._connect()) as con:
            con.executescript(SCHEMA)
            con.commit()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    # ── Ingest ──────────────────────────────────────────────────────────
    def ingest_alert(self, row: dict[str, Any]) -> int | None:
        """Attach this alert to the right case (or open a new one).

        Returns the case id. Returns None only on I/O failure — callers
        can treat None as "case layer unavailable" and continue.
        """
        subject = str(row.get("sender_id") or "").strip()
        tx_id   = str(row.get("id") or "").strip()
        if not subject or not tx_id:
            return None
        score = int(row.get("risk_score") or 0)
        now   = time.time()
        try:
            with self._lock, closing(self._connect()) as con:
                # Find an open case for this subject within the window
                cur = con.execute(
                    """SELECT id, priority FROM cases
                       WHERE subject_id = ? AND status IN ('OPEN', 'IN_REVIEW')
                         AND opened_at >= ?
                       ORDER BY opened_at DESC LIMIT 1""",
                    (subject, now - self.window_seconds),
                )
                existing = cur.fetchone()
                if existing:
                    case_id, old_priority = existing
                    con.execute(
                        """UPDATE cases
                             SET updated_at  = ?,
                                 priority    = MAX(priority, ?),
                                 alert_count = alert_count + 1
                           WHERE id = ?""",
                        (now, score, case_id),
                    )
                else:
                    cur = con.execute(
                        """INSERT INTO cases
                           (subject_id, opened_at, updated_at, status,
                            priority, alert_count)
                           VALUES (?, ?, ?, 'OPEN', ?, 1)""",
                        (subject, now, now, score),
                    )
                    case_id = cur.lastrowid
                con.execute(
                    """INSERT OR IGNORE INTO case_alerts (case_id, tx_id, added_at)
                       VALUES (?, ?, ?)""",
                    (case_id, tx_id, now),
                )
                con.commit()
                return int(case_id) if case_id else None
        except Exception as e:  # noqa: BLE001
            log.warning("cases: ingest_alert failed: %s", e)
            return None

    def ingest_batch(self, df) -> int:
        """Attach every alert=True row in a DataFrame to a case."""
        if df is None or df.empty or "alert" not in df.columns:
            return 0
        alerts = df[df["alert"] == True]  # noqa: E712
        n = 0
        for _, r in alerts.iterrows():
            if self.ingest_alert(r.to_dict()) is not None:
                n += 1
        return n

    # ── Transitions ────────────────────────────────────────────────────
    def transition(
        self,
        case_id: int,
        to_status: str,
        actor: str,
        notes: str = "",
    ) -> bool:
        """Move a case to a new status. Raises InvalidTransition on illegal move."""
        to_status = to_status.upper()
        with self._lock, closing(self._connect()) as con:
            cur = con.execute("SELECT status FROM cases WHERE id = ?", (case_id,))
            r = cur.fetchone()
            if not r:
                return False
            from_status = r[0]
            if to_status not in _TRANSITIONS.get(from_status, set()):
                raise InvalidTransition(
                    f"cannot move case {case_id} from {from_status} to {to_status}; "
                    f"allowed: {sorted(_TRANSITIONS.get(from_status, set()))}"
                )
            now = time.time()
            con.execute(
                "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?",
                (to_status, now, case_id),
            )
            con.execute(
                """INSERT INTO case_events
                   (case_id, created_at, actor, from_status, to_status, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (case_id, now, actor[:64], from_status, to_status, notes[:1024]),
            )
            con.commit()
            return True

    def assign(self, case_id: int, assignee: str) -> bool:
        try:
            with self._lock, closing(self._connect()) as con:
                con.execute(
                    "UPDATE cases SET assignee = ?, updated_at = ? WHERE id = ?",
                    (assignee[:64], time.time(), case_id),
                )
                con.commit()
                return True
        except Exception as e:  # noqa: BLE001
            log.warning("cases: assign failed: %s", e)
            return False

    # ── Queries ────────────────────────────────────────────────────────
    def triage_queue(self, limit: int = 50) -> list[dict[str, Any]]:
        """Open + in-review cases, priority-sorted (highest first)."""
        with closing(self._connect()) as con:
            cur = con.execute(
                """SELECT id, subject_id, opened_at, updated_at, status,
                          priority, alert_count, assignee
                     FROM cases
                    WHERE status IN ('OPEN', 'IN_REVIEW')
                    ORDER BY priority DESC, updated_at DESC
                    LIMIT ?""",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

    def case(self, case_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as con:
            cur = con.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [c[0] for c in cur.description]
            out = dict(zip(cols, row, strict=False))
            cur = con.execute(
                "SELECT tx_id, added_at FROM case_alerts WHERE case_id = ? ORDER BY added_at",
                (case_id,),
            )
            out["alerts"] = [{"tx_id": r[0], "added_at": r[1]} for r in cur.fetchall()]
            cur = con.execute(
                """SELECT created_at, actor, from_status, to_status, notes
                     FROM case_events WHERE case_id = ? ORDER BY created_at""",
                (case_id,),
            )
            out["events"] = [
                {"created_at": r[0], "actor": r[1], "from": r[2],
                 "to": r[3], "notes": r[4]}
                for r in cur.fetchall()
            ]
            return out

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as con:
            out: dict[str, int] = {}
            for status in ("OPEN", "IN_REVIEW", "ESCALATED",
                           "DISMISSED", "SAR_FILED", "CLOSED"):
                cur = con.execute("SELECT COUNT(*) FROM cases WHERE status = ?", (status,))
                out[status.lower()] = int(cur.fetchone()[0])
            return out
