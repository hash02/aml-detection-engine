#!/usr/bin/env python3
"""populate_addresses.py: build the committed addresses/ snapshot
from the refreshed feed cache (data/feeds/) plus MANIFEST.json.

Why this is a script and why it fails closed: this step used to be
an inline heredoc in refresh-addresses-daily.yml. If a feed's fetch
failed, refresh wrote no cache file, and the heredoc silently
skipped that feed (`if not src.exists(): continue`), so the old
snapshot stayed in place while MANIFEST.json was rewritten with a
fresh last_updated. Stale data, labeled fresh (Claude review of
PR #20, finding 2). Every other workflow step is already a script
under scripts/; this was the last inline one.

Behavior:

- Every mapped feed must have a cache file. On a fresh checkout
  the refresh step materializes all of them: fetcher feeds write
  on success, and baseline-only feeds are written precisely so
  this step picks them up (see engine/feeds.py refresh()). A
  missing cache file therefore means the refresh did not do its
  job: fail (exit 1) before writing anything, so a partial
  snapshot can never be committed under a fresh timestamp.
- A feed file is rewritten only when its content (everything
  except the embedded updated_at) differs from the committed
  snapshot. A pure re-fetch timestamp never dirties a file.
- MANIFEST.json records a per-feed updated_at taken from each
  feed's committed snapshot file, i.e. when that feed's content
  last changed. The top-level last_updated is bumped only when
  at least one feed file changed; otherwise the previous value
  is carried over, so an unchanged run produces no diff at all.

Usage: populate_addresses.py [feeds_dir] [addresses_dir]
(defaults: data/feeds addresses). Exit 0 ok, 1 missing cache,
2 usage error.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

MAPPING = {
    "ofac_sdn":     ("ofac.json",     "https://raw.githubusercontent.com/0xB10C/ofac-sanctioned-digital-currency-addresses"),
    "phishing":     ("phishing.json", "https://github.com/scamsniffer/scam-database"),
    "mixers":       ("mixers.json",   "engine/feeds.py BASELINE_MIXERS"),
    "bridges":      ("bridges.json",  "engine/feeds.py BASELINE_BRIDGES"),
    "lrt_offramps": ("offramps.json", "engine/feeds.py BASELINE_LRT_OFFRAMPS"),
}

SCHEMA = {
    "ofac.json": "OFAC SDN ethereum addresses",
    "phishing.json": "ScamSniffer scam-database aggregate (GPL-3.0)",
    "exploits.json": "Analyst-curated exploit wallet list",
    "mixers.json": "Tornado Cash pools + routers",
    "bridges.json": "Cross-chain bridge routers",
    "offramps.json": "No-KYC offramps + drainer infrastructure",
}


def _content_key(blob: dict) -> dict:
    """A feed snapshot minus its timestamp: what 'changed' means."""
    return {k: v for k, v in blob.items() if k != "updated_at"}


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def main(argv: list[str]) -> int:
    if len(argv) not in (0, 2):
        print(__doc__.splitlines()[0], file=sys.stderr)
        print("usage: populate_addresses.py [feeds_dir] [addresses_dir]",
              file=sys.stderr)
        return 2
    feeds_dir = Path(argv[0]) if argv else Path("data/feeds")
    addr_dir = Path(argv[1]) if argv else Path("addresses")

    missing = [name for name in MAPPING
               if not (feeds_dir / f"{name}.json").exists()]
    if missing:
        for name in missing:
            print(f"::error::cache file missing for feed {name!r} "
                  f"({feeds_dir}/{name}.json): the refresh step did "
                  f"not produce it, so the previous snapshot would "
                  f"go stale under a fresh manifest timestamp. "
                  f"Failing instead.", file=sys.stderr)
        return 1

    addr_dir.mkdir(parents=True, exist_ok=True)
    changed: list[str] = []
    feed_updated_at: dict[str, float] = {}
    for feed_name, (addr_name, source) in MAPPING.items():
        data = json.loads((feeds_dir / f"{feed_name}.json").read_text())
        out = {
            "feed": feed_name,
            "updated_at": data["updated_at"],
            "count": data["count"],
            "source": source,
            "addresses": data["addresses"],
        }
        dest = addr_dir / addr_name
        existing = _read_json(dest) if dest.exists() else None
        if existing is not None and _content_key(existing) == _content_key(out):
            if existing.get("updated_at") is not None:
                feed_updated_at[addr_name] = existing["updated_at"]
            print(f"{addr_name}: unchanged ({out['count']} entries)")
            continue
        dest.write_text(json.dumps(out, indent=2))
        changed.append(addr_name)
        feed_updated_at[addr_name] = out["updated_at"]
        print(f"{addr_name}: updated ({out['count']} entries)")

    manifest_path = addr_dir / "MANIFEST.json"
    prev = _read_json(manifest_path) if manifest_path.exists() else None
    if changed:
        last_updated = time.time()
        last_updated_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elif prev is not None and "last_updated" in prev:
        last_updated = prev["last_updated"]
        last_updated_iso = prev.get(
            "last_updated_iso",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_updated)))
    else:
        last_updated = time.time()
        last_updated_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    checksums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                 for f in sorted(addr_dir.glob("*.json"))
                 if f.name != "MANIFEST.json"}
    manifest = {
        "version": "0.2.0",
        "last_updated": last_updated,
        "last_updated_iso": last_updated_iso,
        "source": "hash02/aml-detection-engine",
        "schema": SCHEMA,
        "feed_updated_at": feed_updated_at,
        "checksums": checksums,
        "consumers": ["hash02/aml-roaster"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"MANIFEST.json: {len(changed)} feed file(s) changed "
          f"({', '.join(changed) if changed else 'none'}); "
          f"last_updated {'bumped' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
