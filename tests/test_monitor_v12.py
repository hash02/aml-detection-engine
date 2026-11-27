"""Daily monitor must run the four v12 detectors the Streamlit app already calls.

Drainer signature and address poisoning stay unwired: the monitor fetcher
has no token fields for them.
"""

from __future__ import annotations

import sys
import types
from datetime import timedelta

import pandas as pd

# blockscout_fetcher replaces sys.stdout at import. The monitor imports it
# for live fetches; these tests only call run_engine_on_df, so stub it
# before that import or pytest's capture stream is closed.
if "scripts.blockscout_fetcher" not in sys.modules:
    _blockscout = types.ModuleType("scripts.blockscout_fetcher")
    _blockscout.fetch_and_build = None
    sys.modules["scripts.blockscout_fetcher"] = _blockscout

OFFRAMP = "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f"


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_monitor_phish_hit_fires_on_offramp(row, t0):
    from scripts.monitor import run_engine_on_df

    df = _frame([
        row("0x00000000000000000000000000000000000000aa", OFFRAMP, 5_000, t0),
    ])
    out = run_engine_on_df(df)
    assert bool(out.loc[0, "phish_flag"]) is True
    assert out.loc[0, "phish_source"] == "no_kyc_offramp_receiver"
    assert "phishing_hit" in out.loc[0, "reasons"]


def test_monitor_sub_threshold_tranching_fires(row, t0):
    from scripts.monitor import run_engine_on_df

    sender = "0x00000000000000000000000000000000000000d1"
    rx = "0x00000000000000000000000000000000000000d2"
    df = _frame([
        row(sender, rx, 9_100, t0),
        row(sender, rx, 9_500, t0 + timedelta(hours=2)),
        row(sender, rx, 8_800, t0 + timedelta(hours=4)),
        row(sender, rx, 200, t0 + timedelta(hours=5)),
    ])
    out = run_engine_on_df(df)
    assert int(out["tranching_flag"].sum()) >= 3
    assert bool(out.loc[out["amount"] == 200, "tranching_flag"].iloc[0]) is False


def test_monitor_machine_cadence_fires(row, t0):
    from scripts.monitor import run_engine_on_df

    sender = "0x" + "a1" * 20
    rx = "0x" + "b1" * 20
    df = _frame([
        row(sender, rx, 200, t0 + timedelta(seconds=30 * i))
        for i in range(6)
    ])
    out = run_engine_on_df(df)
    assert bool(out["cadence_flag"].all()) is True
    assert out["reasons"].str.contains("machine_cadence").all()


def test_monitor_sybil_fan_in_fires(row, t0):
    from scripts.monitor import run_engine_on_df

    rx = "0x" + "c1" * 20
    df = _frame([
        row(f"0x{i:040x}", rx, 1_000.0, t0 + timedelta(seconds=5 * i))
        for i in range(1, 7)
    ])
    out = run_engine_on_df(df)
    assert int(out["sybil_flag"].sum()) == 6
    assert out["reasons"].str.contains("sybil_fan_in").all()


def test_monitor_leaves_v13_rules_unwired(row, t0):
    from scripts.monitor import run_engine_on_df

    df = _frame([
        row("0x00000000000000000000000000000000000000aa",
            "0x00000000000000000000000000000000000000bb", 50, t0),
    ])
    out = run_engine_on_df(df)
    assert "drainer_flag" not in out.columns
    assert "poison_flag" not in out.columns
    assert "phish_flag" in out.columns
    assert "tranching_flag" in out.columns
    assert "cadence_flag" in out.columns
    assert "sybil_flag" in out.columns
    assert bool(out.loc[0, "phish_flag"]) is False
    assert bool(out.loc[0, "tranching_flag"]) is False
    assert bool(out.loc[0, "cadence_flag"]) is False
    assert bool(out.loc[0, "sybil_flag"]) is False
