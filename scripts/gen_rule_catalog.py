"""
scripts/gen_rule_catalog.py — Auto-generate rule catalogue from docstrings
==========================================================================

docs/RULES.md is hand-maintained and lists all 28 scoring rules.
This script only dumps detect_* docstrings when --out is given.
It does not write docs/RULES.md.

Usage:
  python scripts/gen_rule_catalog.py --out /tmp/detectors.md
  python scripts/gen_rule_catalog.py --out /tmp/detectors.md --check
"""

from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "engine" / "engine_v11_blockchain.py"


def extract_detectors(source: str) -> list[tuple[str, str]]:
    """Return [(function_name, docstring), ...] for every detect_* function."""
    tree = ast.parse(source)
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("detect_"):
            doc = ast.get_docstring(node) or "(no docstring)"
            out.append((node.name, doc))
    return sorted(out, key=lambda t: t[0])


def format_entry(name: str, doc: str) -> str:
    """Pretty-print one catalogue entry in Markdown."""
    title = name.removeprefix("detect_").replace("_", " ").title()
    # Trim indentation + collapse blank-line-separated blocks
    body = textwrap.dedent(doc).strip()
    return f"### `{name}` — {title}\n\n```text\n{body}\n```\n"


def build_catalog(source: str) -> str:
    entries = extract_detectors(source)
    header = (
        "# AML Rule Catalogue\n\n"
        "> Auto-generated from docstrings in `engine/engine_v11_blockchain.py`.\n"
        "> Do not edit by hand — run `python scripts/gen_rule_catalog.py` to refresh.\n\n"
        f"**{len(entries)} detectors** are wired into the engine.\n\n"
        "---\n\n"
    )
    sections = [format_entry(n, d) for n, d in entries]
    return header + "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="Write a detect_* docstring dump to this path")
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if --out does not match a fresh dump")
    args = ap.parse_args()

    if args.out is None:
        print(
            "docs/RULES.md is hand-maintained (28 scoring rules). "
            "Pass --out PATH to write a detect_* docstring dump.",
            file=sys.stderr,
        )
        return 2

    source = ENGINE_PATH.read_text()
    new = build_catalog(source)

    if args.check:
        if not args.out.exists():
            print(f"[gen_rule_catalog] {args.out} missing — run without --check", file=sys.stderr)
            return 2
        existing = args.out.read_text()
        if existing.strip() != new.strip():
            print(
                f"[gen_rule_catalog] {args.out} is stale. "
                f"Run `python scripts/gen_rule_catalog.py` to refresh.",
                file=sys.stderr,
            )
            return 1
        print(f"[gen_rule_catalog] {args.out} up to date ✓")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(new)
    print(f"[gen_rule_catalog] wrote {args.out} ({len(new):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
