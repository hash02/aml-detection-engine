"""
scripts/gen_rule_catalog.py — Auto-generate rule catalogue from docstrings
==========================================================================

Every detector in engine/engine_v11_blockchain.py starts with a
structured docstring that explains what the rule does, what pattern
it catches, and the config knobs that drive it. This script scans
those docstrings and emits `docs/RULES.md` — a human-readable catalogue
that compliance analysts can consult without opening Python.

Keeps documentation in sync with code by construction: the catalogue
is generated from the source of truth. A CI step (`make rules`) can
diff the committed catalogue against a freshly-generated one to block
PRs that forget to refresh it.

Usage:
  python scripts/gen_rule_catalog.py            # writes docs/RULES.md
  python scripts/gen_rule_catalog.py --out X    # custom path
  python scripts/gen_rule_catalog.py --check    # exit != 0 if stale
"""

from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "engine" / "engine_v11_blockchain.py"
DEFAULT_OUT = REPO_ROOT / "docs" / "RULES.md"


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
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output path for the generated catalogue")
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if the committed catalogue is stale")
    args = ap.parse_args()

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
