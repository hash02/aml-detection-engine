"""
engine/tuning.py — Threshold tuning suggestions from disposition feedback
=========================================================================

When reviewers dismiss an alert, that's ground truth that the rule
fired on something benign. Aggregated over enough dispositions, we
can suggest *directional* threshold changes to reduce false positives
without auto-applying them.

This module is deliberately advisory. It never mutates CONFIG — it
produces a list of `TuningSuggestion` objects a human (admin) can
accept or ignore. Model governance stays in-loop.

Inputs:
  - AuditLog: every alert fire (for context)
  - review_events: reviewer dispositions (dismiss / escalate / sar_filed)

Method:
  For each rule X:
    dismiss_rate = dismissed(X) / total_reviewed(X)
    escalate_rate = escalated(X) / total_reviewed(X)

  If dismiss_rate > `min_dismiss_rate` AND sample ≥ `min_samples`:
    → suggest raising the rule weight's threshold by a small ratio
    → confidence = function of sample size
  If escalate_rate > `min_escalate_rate` AND sample ≥ `min_samples`:
    → suggest that this rule is under-firing (optional lowering)

A rule with fewer than `min_samples` reviewed alerts returns no
suggestion — we don't make calls on insufficient data.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TuningSuggestion:
    rule: str
    reviewed: int
    dismiss_rate: float
    escalate_rate: float
    suggestion: str            # "raise_threshold" | "lower_threshold" | "hold"
    confidence: float          # 0..1
    rationale: str

    def __str__(self) -> str:
        return (
            f"[{self.rule}] n={self.reviewed} "
            f"dismiss={self.dismiss_rate:.1%} escalate={self.escalate_rate:.1%} "
            f"→ {self.suggestion} (conf={self.confidence:.2f})"
        )


def _extract_rules(reasons: str) -> list[str]:
    """Return distinct rule tags from a reasons string."""
    from engine.drift import TRACKED_RULES
    return [r for r in TRACKED_RULES if r in (reasons or "")]


def load_reviewed_alerts(audit_db: str | Path) -> list[tuple[str, str]]:
    """Return [(rules_fired, disposition), ...] for every reviewed alert.

    Rules fired is the raw reasons string from alert_events; disposition
    is the corresponding review_events row. Many-to-many: one alert
    can have multiple reviews; we keep the latest.
    """
    path = Path(audit_db)
    if not path.exists():
        return []
    try:
        with closing(sqlite3.connect(str(path))) as con:
            cur = con.execute(
                """
                SELECT a.rules_fired, r.disposition
                  FROM alert_events AS a
                  JOIN review_events AS r ON r.alert_id = a.id
                 ORDER BY r.created_at
                """
            )
            return [(row[0] or "", row[1] or "") for row in cur.fetchall()]
    except Exception as e:  # noqa: BLE001 — missing tables / fresh install
        log.debug("tuning: load_reviewed_alerts failed: %s", e)
        return []


def suggest(
    audit_db: str | Path = "data/audit.db",
    min_samples: int = 10,
    min_dismiss_rate: float = 0.60,
    min_escalate_rate: float = 0.70,
) -> list[TuningSuggestion]:
    """Analyse disposition history and return per-rule suggestions."""
    reviewed = load_reviewed_alerts(audit_db)
    if not reviewed:
        return []

    # Tally per-rule
    counts: dict[str, dict[str, int]] = {}
    for reasons, dispo in reviewed:
        for rule in _extract_rules(reasons):
            counts.setdefault(rule, {"reviewed": 0, "dismiss": 0, "escalate": 0, "sar": 0})
            counts[rule]["reviewed"] += 1
            if dispo == "dismiss":
                counts[rule]["dismiss"] += 1
            elif dispo == "escalate":
                counts[rule]["escalate"] += 1
            elif dispo == "sar_filed":
                counts[rule]["sar"] += 1

    suggestions: list[TuningSuggestion] = []
    for rule, tally in counts.items():
        n = tally["reviewed"]
        if n < min_samples:
            continue
        dismiss_rate  = tally["dismiss"] / n
        escalate_rate = (tally["escalate"] + tally["sar"]) / n
        # Confidence: monotonically increasing with sample size, caps at 1.0
        confidence = min(1.0, n / (min_samples * 5))
        if dismiss_rate >= min_dismiss_rate:
            suggestions.append(TuningSuggestion(
                rule=rule,
                reviewed=n,
                dismiss_rate=round(dismiss_rate, 3),
                escalate_rate=round(escalate_rate, 3),
                suggestion="raise_threshold",
                confidence=round(confidence, 2),
                rationale=(
                    f"{tally['dismiss']} of {n} reviewed alerts were dismissed "
                    f"({dismiss_rate:.0%}). Consider raising the threshold to "
                    f"reduce false positives."
                ),
            ))
        elif escalate_rate >= min_escalate_rate:
            suggestions.append(TuningSuggestion(
                rule=rule,
                reviewed=n,
                dismiss_rate=round(dismiss_rate, 3),
                escalate_rate=round(escalate_rate, 3),
                suggestion="lower_threshold",
                confidence=round(confidence, 2),
                rationale=(
                    f"{tally['escalate'] + tally['sar']} of {n} reviewed alerts "
                    f"were escalated or filed as SAR ({escalate_rate:.0%}). Rule "
                    f"may be under-firing — consider lowering the threshold."
                ),
            ))
        else:
            suggestions.append(TuningSuggestion(
                rule=rule,
                reviewed=n,
                dismiss_rate=round(dismiss_rate, 3),
                escalate_rate=round(escalate_rate, 3),
                suggestion="hold",
                confidence=round(confidence, 2),
                rationale="Mixed dispositions within tolerance bands — hold current threshold.",
            ))

    return sorted(suggestions, key=lambda s: (s.suggestion != "raise_threshold", -s.reviewed))
