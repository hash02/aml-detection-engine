"""What-if replay tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_whatif_runs_on_sample_data(tmp_path):
    out = tmp_path / "report.json"
    env = {**os.environ, "FEEDS_OFFLINE": "1", "PYTHONPATH": str(REPO)}
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "whatif.py"),
         "--input", str(REPO / "data" / "sample_transactions.csv"),
         "--set",  "alert_threshold=80",
         "--out",  str(out)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert r.returncode == 0, f"whatif failed: {r.stderr[-500:]}"
    assert out.exists()
    report = json.loads(out.read_text())
    assert "before" in report and "after" in report
    assert report["overrides"] == {"alert_threshold": 80}
    # Raising the threshold strictly reduces or keeps the same alert count
    assert report["after"]["alerts"] <= report["before"]["alerts"]


def test_whatif_rule_delta_captures_changes(tmp_path):
    out = tmp_path / "report.json"
    env = {**os.environ, "FEEDS_OFFLINE": "1", "PYTHONPATH": str(REPO)}
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "whatif.py"),
         "--input", str(REPO / "data" / "sample_transactions.csv"),
         "--set",  "alert_threshold=999",   # impossibly high → nothing fires
         "--out",  str(out)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(out.read_text())
    # Threshold is the gate, not a rule weight → rule fire counts stay the same
    # but alert counts drop to zero.
    assert report["after"]["alerts"] == 0
