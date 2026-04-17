"""
engine/retention.py — Data retention + PII redaction
======================================================

Two responsibilities:

1. **Retention purge** — delete alert_events / review_events / case
   rows older than the configured retention window. Run it on a cron
   (or via `scripts/purge.py`) to keep the audit footprint bounded.

2. **PII redaction** — every wallet address is public, but notes and
   free-text disposition fields written by reviewers may contain PII
   (names, emails, ticket numbers). redact_pii() applies conservative
   regex masking for export to external systems.

Design principle: redaction is applied at *export time*, never in-place.
The audit store keeps full fidelity; only outbound copies are scrubbed.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

log = logging.getLogger(__name__)


# Default retention: 2 years. FinCEN requires SAR records to be kept
# for 5 years, but we're not the SAR itself — we're the upstream
# detection store. Operators pick their own window per jurisdiction.
DEFAULT_RETENTION_DAYS = 730


# ── PII redaction ──────────────────────────────────────────────────
# Conservative: mask email, phone, and "ticket:<NNN>" style references.
# Intentionally does NOT mask wallet addresses — those are public.
_EMAIL_RE  = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE  = re.compile(r"\b\+?\d[\d\-\s]{7,}\d\b")
_SSN_RE    = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_TICKET_RE = re.compile(r"(?i)\b(ticket|case|ref|jira)[:\s#-]*\d+")


def redact_pii(text: str) -> str:
    """Return text with conservative PII masks applied."""
    if not text:
        return text
    out = _EMAIL_RE.sub("[redacted-email]",   text)
    out = _PHONE_RE.sub("[redacted-phone]",   out)
    out = _SSN_RE.sub("[redacted-ssn]",       out)
    out = _TICKET_RE.sub("[redacted-ref]",    out)
    return out


def redact_dict(d: dict, fields: tuple[str, ...] = ("notes", "rules_fired")) -> dict:
    """Return a copy of `d` with the named string fields redacted."""
    out = dict(d)
    for f in fields:
        if f in out and isinstance(out[f], str):
            out[f] = redact_pii(out[f])
    return out


# ── Retention purge ────────────────────────────────────────────────

def purge(
    audit_db: str | Path = "data/audit.db",
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
) -> dict[str, int]:
    """Delete rows older than `retention_days` from the audit store.

    Returns a dict of (table → rows_deleted). On dry_run=True, returns
    (table → rows_that_would_be_deleted) without mutating anything.
    """
    cutoff = time.time() - (retention_days * 86400)
    result: dict[str, int] = {}
    path = Path(audit_db)
    if not path.exists():
        return result
    try:
        # isolation_level=None → autocommit; required because VACUUM
        # can't run inside an implicit transaction. Without this, the
        # DELETE appears to succeed but the SELECT-COUNT is reported
        # first and the VACUUM raises, silently voiding the change set.
        with closing(sqlite3.connect(str(path), isolation_level=None)) as con:
            for table, ts_col in (
                ("review_events", "created_at"),
                ("alert_events",  "created_at"),
                ("rule_fires",    None),  # day-indexed, handled below
            ):
                if ts_col is None:
                    # rule_fires uses ISO date strings, not epochs
                    import datetime as _dt
                    cutoff_day = (_dt.date.today() - _dt.timedelta(days=retention_days)).isoformat()
                    cur = con.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE day < ?",
                        (cutoff_day,),
                    )
                    n = int(cur.fetchone()[0])
                    if not dry_run and n:
                        con.execute(f"DELETE FROM {table} WHERE day < ?", (cutoff_day,))
                    result[table] = n
                    continue
                cur = con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {ts_col} < ?",
                    (cutoff,),
                )
                n = int(cur.fetchone()[0])
                if not dry_run and n:
                    con.execute(f"DELETE FROM {table} WHERE {ts_col} < ?", (cutoff,))
                result[table] = n
            if not dry_run:
                # Autocommit mode means every DELETE already persisted;
                # VACUUM is optional but fine here since we're outside
                # any transaction.
                try:
                    con.execute("VACUUM")
                except sqlite3.OperationalError:
                    pass
    except sqlite3.OperationalError as e:
        # Table doesn't exist yet (fresh install) — safe to report 0
        log.debug("retention: %s", e)
    return result


def forget_subject(
    subject_id: str,
    audit_db: str | Path = "data/audit.db",
    cases_db:  str | Path = "data/cases.db",
) -> dict[str, int]:
    """Right-to-be-forgotten / GDPR deletion for a specific wallet.

    Wallet addresses on a public chain aren't strictly PII in most
    jurisdictions, but some regulators treat them as such. This purges
    every record that mentions the address.
    """
    deleted: dict[str, int] = {}
    subject = str(subject_id).strip().lower()
    if not subject:
        return deleted

    for path, tables in (
        (audit_db, [("alert_events", "sender_id"), ("alert_events", "receiver_id")]),
        (cases_db, [("cases", "subject_id")]),
    ):
        p = Path(path)
        if not p.exists():
            continue
        try:
            with closing(sqlite3.connect(str(p))) as con:
                for table, col in tables:
                    cur = con.execute(
                        f"DELETE FROM {table} WHERE lower({col}) = ?",
                        (subject,),
                    )
                    deleted[f"{p.name}:{table}.{col}"] = cur.rowcount
                con.commit()
        except sqlite3.OperationalError as e:
            log.debug("forget_subject on %s: %s", p, e)
    return deleted
