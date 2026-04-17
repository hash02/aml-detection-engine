"""Rule catalogue generator tests — extraction + staleness check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_extract_detectors_finds_every_rule():
    from scripts.gen_rule_catalog import extract_detectors
    source = (REPO / "engine" / "engine_v11_blockchain.py").read_text()
    names = {n for n, _ in extract_detectors(source)}
    # Spot-check a representative sample across v6 → v13
    for expected in (
        "detect_layering",
        "detect_ofac_hit",
        "detect_mixer_touch",
        "detect_sub_threshold_tranching",
        "detect_sybil_fan_in",
        "detect_drainer_signature",
        "detect_address_poisoning",
    ):
        assert expected in names, f"missing detector: {expected}"


def test_build_catalog_produces_markdown(tmp_path):
    from scripts.gen_rule_catalog import build_catalog
    source = (REPO / "engine" / "engine_v11_blockchain.py").read_text()
    md = build_catalog(source)
    assert md.startswith("# AML Rule Catalogue")
    assert "detectors" in md
    assert "`detect_ofac_hit`" in md


def test_catalog_check_mode_passes_on_fresh(tmp_path):
    """--check mode after a regeneration returns 0."""
    out = tmp_path / "RULES.md"
    gen  = REPO / "scripts" / "gen_rule_catalog.py"
    r1 = subprocess.run(
        [sys.executable, str(gen), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(
        [sys.executable, str(gen), "--out", str(out), "--check"],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr


def test_catalog_check_mode_fails_when_stale(tmp_path):
    out = tmp_path / "RULES.md"
    out.write_text("# Stale content\n")
    gen = REPO / "scripts" / "gen_rule_catalog.py"
    r = subprocess.run(
        [sys.executable, str(gen), "--out", str(out), "--check"],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "stale" in r.stderr.lower()
