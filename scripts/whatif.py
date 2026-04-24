"""
scripts/whatif.py — Counterfactual threshold replay
=====================================================

"What would happen if I raised the structuring threshold to 6 txns?"

Takes a base dataset, applies config overrides, and reports:
  - rows that would newly flag (FP/TP churn)
  - rows that would no longer flag
  - per-rule fire-count delta

Nothing is persisted. Use this before opening a PR that changes a
rule weight or threshold.

Usage:
  python scripts/whatif.py --input data/sample_transactions.csv \
    --set alert_threshold=60 \
    --set w_structuring=20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "engine"))

os.environ.setdefault("FEEDS_OFFLINE", "1")


def parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set expected key=value, got {p!r}")
        k, _, v = p.partition("=")
        k = k.strip()
        v = v.strip()
        # Try int, then float, fall back to str
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def score(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from scripts.backtest import run_full_pipeline
    return run_full_pipeline(df.copy(), cfg)


def fires_by_rule(df: pd.DataFrame) -> dict[str, int]:
    from scripts.backtest import ALL_RULE_TAGS
    counts: dict[str, int] = {}
    for tag in ALL_RULE_TAGS:
        counts[tag] = int(df["reasons"].astype(str).str.contains(tag, regex=False).sum())
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sample_transactions.csv")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="CONFIG override, e.g. --set alert_threshold=60")
    ap.add_argument("--out", default="whatif_report.json")
    args = ap.parse_args()

    from engine_v11_blockchain import CONFIG

    from engine.schema import normalise

    raw = pd.read_csv(args.input, parse_dates=["timestamp"])
    raw = normalise(raw)
    base_cfg  = dict(CONFIG)
    new_cfg   = dict(CONFIG) | parse_overrides(args.overrides)

    base_scored = score(raw, base_cfg)
    new_scored  = score(raw, new_cfg)

    base_alerted = set(base_scored.loc[base_scored["alert"], "id"].astype(str))
    new_alerted  = set(new_scored.loc[new_scored["alert"],  "id"].astype(str))

    added   = sorted(new_alerted - base_alerted)
    removed = sorted(base_alerted - new_alerted)
    stable  = sorted(base_alerted & new_alerted)

    base_fires = fires_by_rule(base_scored)
    new_fires  = fires_by_rule(new_scored)
    rule_delta = {
        r: {"before": base_fires[r], "after": new_fires[r],
            "delta": new_fires[r] - base_fires[r]}
        for r in base_fires if base_fires[r] != new_fires[r]
    }

    report = {
        "input": args.input,
        "overrides": parse_overrides(args.overrides),
        "rows": len(raw),
        "before": {"alerts": len(base_alerted)},
        "after":  {"alerts": len(new_alerted)},
        "added":   {"count": len(added),   "ids": added[:50]},
        "removed": {"count": len(removed), "ids": removed[:50]},
        "stable":  {"count": len(stable)},
        "rule_delta": rule_delta,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[WHATIF] before={len(base_alerted)}  after={len(new_alerted)}  "
          f"added={len(added)}  removed={len(removed)}")
    print(f"[WHATIF] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
