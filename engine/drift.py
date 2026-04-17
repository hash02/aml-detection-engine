"""
engine/drift.py — Rule fire-rate drift detection
==================================================

A rule that fires 5x more often today than the 30-day baseline is a
signal that EITHER the world changed (new exploit, new feed entry)
OR the rule is broken (threshold regression, bad input).

This module records daily per-rule fire counts and flags statistically
significant deviations. It's a 1-file, dependency-free approach that
a real deployment would later swap for Prometheus alert rules.

Public API:
  record_daily(df, cfg, today=...)     → persists today's rule fire counts
  detect_drift(window_days=30, sigma=3.0) → list[DriftAlert]

Storage: same SQLite file as audit (by default), new table `rule_fires`.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS rule_fires (
  day        TEXT    NOT NULL,   -- ISO date
  rule       TEXT    NOT NULL,
  fires      INTEGER NOT NULL,
  total_rows INTEGER NOT NULL,
  PRIMARY KEY (day, rule)
);
CREATE INDEX IF NOT EXISTS rule_fires_rule_day ON rule_fires(rule, day);
"""


# Reasons tags we track. Mirrors the ALL_RULE_TAGS list in the backtest
# harness (one-off duplication is cheaper than importing across layers).
TRACKED_RULES = (
    "large_amount", "velocity_many_tx", "structuring", "fan_in",
    "foreign_country", "layering_cycle",
    "mixer_touch", "mixer_withdraw", "bridge_hop", "peel_chain",
    "novel_wallet_dump", "concentrated_inflow",
    "OFAC_SDN_MATCH", "flash_loan_burst", "coordinated_burst",
    "dormant_activation",
    "wash_cycle", "smurfing", "exit_rush", "rapid_succession",
    "high_risk_jurisdiction", "exchange_avoidance", "layering_deep",
    "phishing_hit", "sub_threshold_tranching",
    "machine_cadence", "sybil_fan_in",
    "drainer_signature", "address_poisoning",
)


@dataclass
class DriftAlert:
    rule: str
    today_fires: int
    baseline_mean: float
    baseline_std: float
    z_score: float
    baseline_days: int

    def __str__(self) -> str:
        return (
            f"DriftAlert[{self.rule}] fires={self.today_fires} "
            f"baseline={self.baseline_mean:.1f}±{self.baseline_std:.1f} "
            f"z={self.z_score:+.1f} over {self.baseline_days}d"
        )


class DriftMonitor:
    """Lightweight daily rule-fire tracker with z-score drift detection."""

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
        return con

    # ── Writes ───────────────────────────────────────────────────────
    def record_daily(self, df, today: date | None = None) -> int:
        """Persist today's per-rule fire counts from a scored DataFrame."""
        if df is None or df.empty or "reasons" not in df.columns:
            return 0
        today = today or date.today()
        day   = today.isoformat()
        total = len(df)

        counts: dict[str, int] = {r: 0 for r in TRACKED_RULES}
        for reasons in df["reasons"].fillna(""):
            for rule in TRACKED_RULES:
                if rule in reasons:
                    counts[rule] += 1

        try:
            with self._lock, closing(self._connect()) as con:
                for rule, fires in counts.items():
                    con.execute(
                        """INSERT INTO rule_fires (day, rule, fires, total_rows)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(day, rule) DO UPDATE SET
                               fires = excluded.fires,
                               total_rows = excluded.total_rows""",
                        (day, rule, fires, total),
                    )
                con.commit()
            return sum(1 for v in counts.values() if v > 0)
        except Exception as e:  # noqa: BLE001
            log.warning("drift: record_daily failed: %s", e)
            return 0

    # ── Drift check ──────────────────────────────────────────────────
    def detect_drift(
        self,
        window_days: int = 30,
        sigma: float = 3.0,
        today: date | None = None,
    ) -> list[DriftAlert]:
        """Return drift alerts for any rule whose today fires are > sigma σ
        from its baseline mean. Needs at least 7 historical days to fire."""
        today = today or date.today()
        start = today - timedelta(days=window_days)
        alerts: list[DriftAlert] = []

        with closing(self._connect()) as con:
            for rule in TRACKED_RULES:
                cur = con.execute(
                    """SELECT day, fires FROM rule_fires
                        WHERE rule = ? AND day >= ? AND day <= ?
                        ORDER BY day""",
                    (rule, start.isoformat(), today.isoformat()),
                )
                rows = cur.fetchall()
                if not rows:
                    continue
                series = [r[1] for r in rows]
                days   = [r[0] for r in rows]
                if today.isoformat() not in days or len(series) < 8:
                    continue   # not enough baseline history
                today_idx = days.index(today.isoformat())
                today_fires = series[today_idx]
                baseline = series[:today_idx] + series[today_idx + 1:]
                if not baseline:
                    continue
                mean = statistics.mean(baseline)
                std  = statistics.pstdev(baseline) if len(baseline) > 1 else 0.0
                if std == 0:
                    # Constant baseline — σ is degenerate. Flag when
                    # today's count differs materially from the flat line:
                    #   mean == 0 and today > 0      → new rule hitting
                    #   mean  > 0 and abs change ≥ max(2×mean, 5)  → spike
                    threshold_delta = max(2 * mean, 5) if mean > 0 else 1
                    if abs(today_fires - mean) >= threshold_delta:
                        alerts.append(DriftAlert(
                            rule=rule, today_fires=today_fires,
                            baseline_mean=round(mean, 2), baseline_std=0.0,
                            z_score=float("inf"),
                            baseline_days=len(baseline),
                        ))
                    continue
                z = (today_fires - mean) / std
                if abs(z) >= sigma:
                    alerts.append(DriftAlert(
                        rule=rule, today_fires=today_fires,
                        baseline_mean=round(mean, 2),
                        baseline_std=round(std, 2),
                        z_score=round(z, 2),
                        baseline_days=len(baseline),
                    ))
        return sorted(alerts, key=lambda a: abs(a.z_score if math.isfinite(a.z_score) else 1e9),
                      reverse=True)
