"""
scripts/refresh_feeds.py — Pull latest threat-intel feeds into cache
=====================================================================

Run this on a cron (hourly / daily) in production. Safe to run locally
or in CI — never fails the process even if every feed is unreachable.

Exit codes:
  0  at least one feed refreshed or all baselines available
  2  all feeds failed AND no cached copy exists (hard error)

Env:
  FEEDS_OFFLINE=1   skip all network calls (useful for CI smoke tests)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make the engine importable when this file is invoked from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from feeds import FEEDS, feed_age_hours, refresh  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("refresh_feeds")

    log.info("Refreshing %d feeds…", len(FEEDS))
    counts = refresh()
    report: list[dict] = []
    for name, n in counts.items():
        age = feed_age_hours(name)
        report.append({
            "feed": name,
            "count": n,
            "age_hours": round(age, 2) if age is not None else None,
        })
    log.info("Summary:\n%s", json.dumps(report, indent=2))

    total_addresses = sum(counts.values())
    if total_addresses == 0:
        log.error("All feeds empty — aborting")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
