"""
scripts/purge.py — Retention purge CLI
========================================

Cron-runnable wrapper for engine.retention.purge(). Intended to run
nightly in production to bound the audit footprint.

Usage:
  python scripts/purge.py                         # 2-year retention, apply
  python scripts/purge.py --days 365              # 1 year
  python scripts/purge.py --dry-run               # count only, no delete
  python scripts/purge.py --db /path/to/audit.db  # custom db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   default="data/audit.db")
    ap.add_argument("--days", type=int, default=730, help="retention window")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from engine.retention import purge
    result = purge(audit_db=args.db, retention_days=args.days, dry_run=args.dry_run)
    verb = "would delete" if args.dry_run else "deleted"
    print(f"[PURGE] {verb} (days={args.days}):")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
