"""Mixer-touch selection: is_known_mixer, MX_DARK, and the unused lookback.

Scores below are produced by scripts.monitor.run_engine_on_df, the same
path the daily monitor uses. Amounts stay under the personal large-amount
threshold so the mixer term is the only point source.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

MIXER = "0x12d66f87a04a9e220c9d35925b72aca3ca8c78e2"
ALICE = "0x" + "aa" * 20
CHARLIE = "0x" + "cc" * 20
DAVE = "0x" + "dd" * 20


def _flags(df):
    from engine_v11_blockchain import CONFIG, detect_mixer_touch

    return detect_mixer_touch(df, CONFIG)


def test_known_mixer_flag_sets_deposit_and_withdraw(row, t0, build_df):
    """A truthy is_known_mixer row is a mixer row even when country is not MX_DARK.

    Withdraw overwrites the mixer row itself, because its receiver is a
    withdrawer. The depositor's later send to anyone else stays
    mixer_deposit. A later payment to the withdrawer is mixer_withdraw.
    """
    df = build_df([
        row(ALICE, MIXER, 2_500, t0, country="US", is_known_mixer=True),
        row(ALICE, CHARLIE, 2_500, t0 + timedelta(hours=1), country="US"),
        row(DAVE, MIXER, 2_500, t0 + timedelta(hours=2), country="US"),
    ])
    out = _flags(df)

    assert bool(out.loc[0, "mixer_flag"]) is True
    assert out.loc[0, "mixer_type"] == "mixer_withdraw"
    assert out.loc[1, "mixer_type"] == "mixer_deposit"
    assert bool(out.loc[1, "mixer_flag"]) is True
    assert out.loc[2, "mixer_type"] == "mixer_withdraw"
    assert bool(out.loc[2, "mixer_flag"]) is True


def test_known_mixer_string_and_missing_values(row, t0, build_df):
    truthy = build_df([
        row(ALICE, MIXER, 100, t0, country="US", is_known_mixer=True),
    ])
    truthy["is_known_mixer"] = "true"
    assert _flags(truthy).loc[0, "mixer_type"] == "mixer_withdraw"

    for raw in ("TRUE", "1", "yes", "1.0"):
        frame = build_df([row(ALICE, MIXER, 100, t0, country="US")])
        frame["is_known_mixer"] = raw
        assert _flags(frame).loc[0, "mixer_flag"] == True  # noqa: E712

    for raw in ("false", "0", "", "no", "nan", float("nan"), None):
        frame = build_df([row(ALICE, MIXER, 100, t0, country="CA")])
        frame["is_known_mixer"] = raw
        out = _flags(frame)
        assert bool(out.loc[0, "mixer_flag"]) is False
        assert out.loc[0, "mixer_type"] == ""


def test_missing_is_known_mixer_column_does_not_flag_ordinary_row(row, t0, build_df):
    df = build_df([row(ALICE, CHARLIE, 100, t0, country="US")])
    df = df.drop(columns=["is_known_mixer"])
    out = _flags(df)
    assert "is_known_mixer" not in df.columns
    assert bool(out.loc[0, "mixer_flag"]) is False


def test_mx_dark_behavior_unchanged_without_known_mixer_flag(row, t0, build_df):
    """Country MX_DARK still selects the row, and withdraw still overwrites."""
    df = build_df([
        row(ALICE, MIXER, 2_500, t0, country="MX_DARK", is_known_mixer=False),
        row(ALICE, CHARLIE, 2_500, t0 + timedelta(hours=1), country="CA"),
        row(DAVE, MIXER, 2_500, t0 + timedelta(hours=2), country="CA"),
    ])
    out = _flags(df)
    assert out.loc[0, "mixer_type"] == "mixer_withdraw"
    assert out.loc[1, "mixer_type"] == "mixer_deposit"
    assert out.loc[2, "mixer_type"] == "mixer_withdraw"

    bare = build_df([row(ALICE, MIXER, 2_500, t0, country="MX_DARK")])
    bare = bare.drop(columns=["is_known_mixer"])
    assert _flags(bare).loc[0, "mixer_type"] == "mixer_withdraw"


def test_mixer_window_hours_does_not_drop_older_rows(row, t0, build_df):
    from engine_v11_blockchain import CONFIG

    assert CONFIG["mixer_window_hours"] == 24
    df = build_df([
        row(ALICE, MIXER, 100, t0, country="US", is_known_mixer=True),
        row(ALICE, CHARLIE, 100, t0 + timedelta(hours=48), country="US"),
    ])
    out = _flags(df)
    assert bool(out.loc[0, "mixer_flag"]) is True
    assert out.loc[1, "mixer_type"] == "mixer_deposit"


def test_known_mixer_row_scores_withdraw_not_high_risk(row, t0, build_df):
    """US + is_known_mixer is not a high-risk country. Withdraw is 55 * 1.3."""
    from scripts.monitor import run_engine_on_df

    df = build_df([
        row(ALICE, MIXER, 2_500, t0, country="US", is_known_mixer=True,
            profile="PERSONAL_LIKE"),
    ])
    out = run_engine_on_df(df, threshold=0.4)
    assert out.loc[0, "mixer_type"] == "mixer_withdraw"
    assert bool(out.loc[0, "high_risk_country_flag"]) is False
    assert int(out.loc[0, "risk_score"]) == 72
    assert "mixer_withdraw" in out.loc[0, "reasons"]
    assert "high_risk_jurisdiction" not in out.loc[0, "reasons"]


def test_mx_dark_row_score_unchanged(row, t0, build_df):
    from scripts.monitor import run_engine_on_df

    df = build_df([
        row(ALICE, MIXER, 2_500, t0, country="MX_DARK", is_known_mixer=False,
            profile="PERSONAL_LIKE"),
    ])
    out = run_engine_on_df(df, threshold=0.4)
    assert out.loc[0, "mixer_type"] == "mixer_withdraw"
    assert bool(out.loc[0, "high_risk_country_flag"]) is False
    assert int(out.loc[0, "risk_score"]) == 72
    assert "foreign_context_only" in out.loc[0, "reasons"]


def test_nan_country_with_false_flag_is_not_a_mixer_row(row, t0, build_df):
    df = build_df([row(ALICE, CHARLIE, 100, t0, country="US")])
    df.loc[0, "country"] = pd.NA
    df.loc[0, "is_known_mixer"] = float("nan")
    out = _flags(df)
    assert bool(out.loc[0, "mixer_flag"]) is False
