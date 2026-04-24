"""
scripts/benchmark.py — Latency + throughput benchmarks
========================================================

Measures end-to-end pipeline timing at a few transaction-count
levels. Useful for portfolio numbers ("engine sustains N tx/sec on
commodity hardware") and for detecting perf regressions in CI.

Usage:
  python scripts/benchmark.py                     # default sizes
  python scripts/benchmark.py --sizes 100 1000 10000
  python scripts/benchmark.py --out bench.json

Output is a JSON report with per-size latency (mean, p50, p95) plus
rule-fire counts so a perf regression can be correlated with a
detection-rate drop.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "engine"))

os.environ.setdefault("FEEDS_OFFLINE", "1")

random.seed(42)


def synthetic_frame(n: int) -> pd.DataFrame:
    """Generate a deterministic synthetic transaction frame of size n."""
    t0 = datetime(2025, 4, 1, 12, 0, 0)
    rows = []
    senders = [f"0x{i:040x}" for i in range(max(5, n // 10))]
    for i in range(n):
        sender = random.choice(senders)
        receiver = random.choice(senders)
        if receiver == sender:
            receiver = f"0x{(i + 7):040x}"
        amount = random.choice([150, 4_500, 9_500, 50_000, 500_000])
        rows.append({
            "id":          f"bench-{i}",
            "sender_id":   sender,
            "receiver_id": receiver,
            "amount":      amount,
            "country":     random.choice(["US", "CA", "KP", "IR", "UNKNOWN"]),
            "timestamp":   t0 + timedelta(seconds=i * 37),
            "sender_profile": "PERSONAL_LIKE",
            "is_known_mixer": False,
            "is_bridge":      False,
            "sender_tx_count": random.randint(1, 500),
            "sender_active_days": random.randint(1, 1000),
            "account_age_days":   random.randint(1, 1500),
        })
    return pd.DataFrame(rows)


def run_once(df: pd.DataFrame) -> tuple[float, int]:
    """Run the full 28-rule pipeline once; return (elapsed_s, flagged)."""
    from engine_v11_blockchain import CONFIG

    from scripts.backtest import run_full_pipeline
    t0 = time.perf_counter()
    scored = run_full_pipeline(df, CONFIG)
    elapsed = time.perf_counter() - t0
    return elapsed, int(scored["alert"].sum())


def bench_size(n: int, repeats: int = 3) -> dict:
    df = synthetic_frame(n)
    samples: list[float] = []
    flagged_last = 0
    for _ in range(repeats):
        elapsed, flagged_last = run_once(df)
        samples.append(elapsed)
    samples.sort()
    return {
        "rows":      n,
        "repeats":   repeats,
        "mean_s":    round(statistics.mean(samples), 4),
        "p50_s":     round(samples[len(samples) // 2], 4),
        "p95_s":     round(samples[int(len(samples) * 0.95)] if len(samples) > 1 else samples[-1], 4),
        "min_s":     round(min(samples), 4),
        "max_s":     round(max(samples), 4),
        "flagged":   flagged_last,
        "tx_per_s":  round(n / statistics.mean(samples), 1) if samples else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="bench_report.json")
    args = ap.parse_args()

    print(f"[BENCH] sizes={args.sizes} repeats={args.repeats}")
    results = []
    for n in args.sizes:
        print(f"[BENCH] size={n} …")
        results.append(bench_size(n, repeats=args.repeats))
        r = results[-1]
        print(f"         mean={r['mean_s']}s p95={r['p95_s']}s "
              f"throughput={r['tx_per_s']} tx/s  flagged={r['flagged']}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_rules": 28,
        "results":      results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[BENCH] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
