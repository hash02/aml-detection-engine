"""Backtest harness smoke test — full pipeline against bundled sample data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_backtest_runs_against_sample_data(tmp_path):
    """`python scripts/backtest.py --dir data --out …` emits a valid report."""
    repo = Path(__file__).resolve().parent.parent
    out = tmp_path / "report.json"
    env = {**os.environ, "FEEDS_OFFLINE": "1", "PYTHONPATH": str(repo)}
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "backtest.py"),
         "--dir", str(repo / "data"),
         "--out", str(out)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert r.returncode == 0, f"backtest failed: stderr={r.stderr[-500:]}"
    assert out.exists()
    report = json.loads(out.read_text())
    assert "aggregate_rate" in report
    assert report["case_count"] >= 1
    # Each case reports either a detection rate or an error
    for case in report["cases"]:
        assert "case" in case
        assert "rule_fires" in case or "error" in case
