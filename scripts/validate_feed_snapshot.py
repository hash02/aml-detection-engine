#!/usr/bin/env python3
"""validate_feed_snapshot.py: data-quality invariant for refreshed feeds.

Why this exists: the phishing feed's upstream stopped publishing
addresses, its fetcher silently fell back to a domain list, and
the committed "phishing addresses" snapshot turned out to contain
189 domain strings and zero well-formed Ethereum addresses. Every
check that existed counted entries; none validated them. A count
is not evidence that data is what it claims to be.

Run after the populate step, before staging. Fails the job (exit
1) when any feed file in the snapshot:

- contains entries but more than 5 percent of them are not
  well-formed 0x addresses (substitution or format catastrophe),
- is empty while its committed version is not (total loss), or
- shrank by more than 50 percent versus its committed version
  (upstream deletion or silent fallback).

Individual malformed entries at or below the 5 percent threshold
are printed as warnings, not failures: list hygiene (for example
one truncated address) gets its own cleanup and must not block
every refresh. Committed counts are read with `git show
HEAD:<path>`; a file with no committed version skips the shrink
check. Usage: validate_feed_snapshot.py <addresses_dir>.
Exit 0 pass, 1 invariant violated, 2 usage error.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
MALFORMED_FAIL_SHARE = 0.05
SHRINK_FAIL_SHARE = 0.50


def entries_of(path: Path) -> list:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return list(data.get("addresses", []))
    if isinstance(data, list):
        return list(data)
    return []


def committed_count(repo: Path, rel: str) -> int | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{rel}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return len(data.get("addresses", []))
    if isinstance(data, list):
        return len(data)
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.splitlines()[0], file=sys.stderr)
        print("usage: validate_feed_snapshot.py <addresses_dir>", file=sys.stderr)
        return 2
    snap_dir = Path(argv[0])
    if not snap_dir.is_dir():
        print(f"::error::snapshot dir not found: {snap_dir}", file=sys.stderr)
        return 2
    repo = snap_dir.resolve().parent
    failures: list[str] = []
    for path in sorted(snap_dir.glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        entries = entries_of(path)
        malformed = [e for e in entries
                     if not (isinstance(e, str) and ADDRESS_RE.match(e))]
        rel = f"{snap_dir.name}/{path.name}"
        base = committed_count(repo, rel)
        if malformed:
            share = len(malformed) / len(entries) if entries else 1.0
            if share > MALFORMED_FAIL_SHARE:
                failures.append(
                    f"{rel}: {len(malformed)}/{len(entries)} entries are not "
                    f"well-formed addresses ({share:.0%} malformed)")
            else:
                print(f"::warning::{rel}: {len(malformed)} malformed "
                      f"entr{'y' if len(malformed) == 1 else 'ies'} "
                      f"(at or below threshold): {malformed}")
        if base is not None:
            if base > 0 and len(entries) == 0:
                failures.append(f"{rel}: emptied (committed {base}, now 0)")
            elif base > 0 and len(entries) < base * (1 - SHRINK_FAIL_SHARE):
                failures.append(
                    f"{rel}: shrank {base} -> {len(entries)} "
                    f"(more than {SHRINK_FAIL_SHARE:.0%})")
        print(f"{rel}: {len(entries)} entries "
              f"({len(entries) - len(malformed)} well-formed, committed: "
              f"{base if base is not None else 'n/a'})")
    if failures:
        for f in failures:
            print(f"::error::{f}", file=sys.stderr)
        return 1
    print("Feed snapshot satisfies the data-quality invariant.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
