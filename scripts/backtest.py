"""
scripts/backtest.py — Replay historical cases, measure rule performance
========================================================================

Runs the engine against every CSV in a directory (default `data/cases/`)
and reports per-rule fire counts, per-case detection rate, and
aggregate precision proxies. Output is JSON; intended to be diffed
between engine revisions to catch detection regressions in CI.

The expected CSV schema matches the engine input (`sender_id`,
`receiver_id`, `amount`, `country`, `timestamp`, `sender_profile`, …).
A sidecar file `<case>.expected.json` can optionally declare the ground
truth — {"true_positives": ["tx_id_1", ...]} — so precision and recall
can be computed; without it the backtest just reports fire counts.

Usage:
  python scripts/backtest.py                            # uses data/cases/
  python scripts/backtest.py --dir data/synthetic/      # custom dir
  python scripts/backtest.py --out backtest_report.json # custom output

Exit code: non-zero if detection rate drops below `--min-rate` (default
disabled — set it in CI to wire up a hard gate).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "engine"))

os.environ.setdefault("FEEDS_OFFLINE", "1")

from engine_v11_blockchain import (  # noqa: E402
    CONFIG,
    compute_features,
    detect_bridge_hops,
    detect_concentrated_inflow,
    detect_coordinated_burst,
    detect_dormant_activation,
    detect_exchange_avoidance,
    detect_exit_rush,
    detect_flash_loan_burst,
    detect_high_risk_country,
    detect_layering,
    detect_layering_deep,
    detect_machine_cadence,
    detect_mixer_touch,
    detect_novel_wallet_dump,
    detect_ofac_hit,
    detect_peel_chain,
    detect_phish_hit,
    detect_rapid_succession,
    detect_smurfing,
    detect_sub_threshold_tranching,
    detect_sybil_fan_in,
    detect_wash_cycle,
    risk_level,
    score_transactions,
)

ALL_RULE_TAGS = [
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
]


def run_full_pipeline(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = compute_features(df, cfg)
    df, _ = detect_layering(df, cfg)
    for fn in (
        detect_mixer_touch, detect_bridge_hops, detect_peel_chain,
        detect_novel_wallet_dump, detect_concentrated_inflow, detect_ofac_hit,
        detect_flash_loan_burst, detect_coordinated_burst, detect_dormant_activation,
        detect_wash_cycle, detect_smurfing, detect_exit_rush,
        detect_rapid_succession, detect_high_risk_country,
        detect_exchange_avoidance, detect_layering_deep, detect_phish_hit,
        detect_sub_threshold_tranching, detect_machine_cadence, detect_sybil_fan_in,
    ):
        df = fn(df, cfg)
    df = score_transactions(df, cfg)
    df["risk_level"] = df["risk_score"].apply(risk_level)
    return df


def count_rule_fires(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for reasons in df["reasons"].fillna(""):
        for tag in ALL_RULE_TAGS:
            if tag in reasons:
                counts[tag] += 1
    return dict(counts)


def backtest_case(csv_path: Path, cfg: dict) -> dict:
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Normalise optional columns
    for col in ("is_known_mixer", "is_bridge"):
        if col not in df.columns:
            df[col] = False
    if "sender_profile" not in df.columns:
        df["sender_profile"] = "PERSONAL_LIKE"
    for col in ("sender_tx_count", "sender_active_days", "account_age_days"):
        if col not in df.columns:
            df[col] = 0
    scored = run_full_pipeline(df, cfg)
    elapsed = time.perf_counter() - t0

    n_total   = len(scored)
    n_flagged = int(scored["alert"].sum())
    detection = (n_flagged / n_total * 100) if n_total else 0.0

    expected_path = csv_path.with_suffix(".expected.json")
    precision: float | None = None
    recall:    float | None = None
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        tps = set(expected.get("true_positives", []))
        if tps and "id" in scored.columns:
            flagged_ids = set(scored.loc[scored["alert"] == True, "id"].astype(str))  # noqa: E712
            tp = len(flagged_ids & tps)
            fp = len(flagged_ids - tps)
            fn = len(tps - flagged_ids)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "case":            csv_path.name,
        "rows":            n_total,
        "flagged":         n_flagged,
        "detection_rate":  round(detection, 2),
        "precision":       round(precision, 3) if precision is not None else None,
        "recall":          round(recall, 3)    if recall    is not None else None,
        "rule_fires":      count_rule_fires(scored),
        "elapsed_s":       round(elapsed, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/cases", help="Directory of case CSVs")
    ap.add_argument("--out", default="backtest_report.json")
    ap.add_argument("--min-rate", type=float, default=None,
                    help="Fail if aggregate detection rate drops below this")
    args = ap.parse_args()

    case_dir = Path(args.dir)
    if not case_dir.exists():
        # Fall back to the bundled sample so `backtest` always has work to do
        case_dir = Path("data")

    csvs = sorted(p for p in case_dir.glob("*.csv") if not p.name.startswith("."))
    if not csvs:
        print(f"[BACKTEST] No CSVs found under {case_dir}")
        return 2

    results: list[dict] = []
    for csv in csvs:
        print(f"[BACKTEST] {csv.name} …")
        try:
            results.append(backtest_case(csv, CONFIG))
        except Exception as e:  # noqa: BLE001
            print(f"[BACKTEST]   failed: {e}")
            results.append({"case": csv.name, "error": str(e)})

    total_rows    = sum(r.get("rows", 0) for r in results)
    total_flagged = sum(r.get("flagged", 0) for r in results)
    aggregate     = (total_flagged / total_rows * 100) if total_rows else 0.0

    report = {
        "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count":     len(results),
        "total_rows":     total_rows,
        "total_flagged":  total_flagged,
        "aggregate_rate": round(aggregate, 2),
        "cases":          results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[BACKTEST] wrote {args.out}")
    print(f"[BACKTEST] aggregate detection: {aggregate:.2f}% across {total_rows} rows")

    if args.min_rate is not None and aggregate < args.min_rate:
        print(f"[BACKTEST] FAIL: {aggregate:.2f}% < {args.min_rate}% threshold")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
