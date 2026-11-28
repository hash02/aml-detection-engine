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
    """Direction follows which side is the mixer address.

    Alice → mixer is placement (mixer_deposit), including her later send
    to someone else. Mixer → Bob is an exit (mixer_withdraw). A later
    payment to Bob stays mixer_withdraw. Paying the mixer again is still
    a deposit: the mixer address is not a withdrawal recipient.
    """
    bob = "0x" + "b0" * 20
    df = build_df([
        row(ALICE, MIXER, 2_500, t0, country="US", is_known_mixer=True),
        row(ALICE, CHARLIE, 2_500, t0 + timedelta(hours=1), country="US"),
        row(MIXER, bob, 2_500, t0 + timedelta(hours=2), country="US", is_known_mixer=True),
        row(DAVE, bob, 2_500, t0 + timedelta(hours=3), country="US"),
        row(DAVE, MIXER, 2_500, t0 + timedelta(hours=4), country="US", is_known_mixer=True),
    ])
    out = _flags(df)

    assert out.loc[0, "mixer_type"] == "mixer_deposit"
    assert out.loc[1, "mixer_type"] == "mixer_deposit"
    assert out.loc[2, "mixer_type"] == "mixer_withdraw"
    assert out.loc[3, "mixer_type"] == "mixer_withdraw"
    assert out.loc[4, "mixer_type"] == "mixer_deposit"
    assert bool(out["mixer_flag"].all()) is True


def test_known_mixer_string_and_missing_values(row, t0, build_df):
    """Flag coercion uses two ordinary addresses so the mixer roster cannot decide the row."""
    for raw in ("true", "TRUE", "1", "yes", "1.0"):
        frame = build_df([row(ALICE, CHARLIE, 100, t0, country="US")])
        frame["is_known_mixer"] = raw
        out = _flags(frame)
        assert bool(out.loc[0, "mixer_flag"]) is True
        assert out.loc[0, "mixer_type"] == "mixer_deposit"

    for raw in ("false", "0", "", "no", "nan", float("nan"), None):
        frame = build_df([row(ALICE, CHARLIE, 100, t0, country="CA")])
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


def test_mx_dark_into_known_mixer_is_deposit(row, t0, build_df):
    """Receiver address wins when it is a known mixer, even if country is MX_DARK.

    Country is inferred from the sender, so MX_DARK alone means the sender
    is the mixer. Here the receiver is the mixer contract, so the row is
    placement. The mixer address does not become a withdrawal recipient.
    """
    df = build_df([
        row(ALICE, MIXER, 2_500, t0, country="MX_DARK", is_known_mixer=False),
        row(ALICE, CHARLIE, 2_500, t0 + timedelta(hours=1), country="CA"),
        row(DAVE, MIXER, 2_500, t0 + timedelta(hours=2), country="CA"),
    ])
    out = _flags(df)
    assert out.loc[0, "mixer_type"] == "mixer_deposit"
    assert out.loc[1, "mixer_type"] == "mixer_deposit"
    assert out.loc[2, "mixer_type"] == "mixer_deposit"

    bare = build_df([row(ALICE, MIXER, 2_500, t0, country="MX_DARK")])
    bare = bare.drop(columns=["is_known_mixer"])
    assert _flags(bare).loc[0, "mixer_type"] == "mixer_deposit"


def test_mx_dark_country_without_known_address_means_sender_is_mixer(row, t0, build_df):
    """MX_DARK with neither side in the mixer roster: the sender is the mixer."""
    mixer_side = "0x" + "11" * 20
    customer = "0x" + "22" * 20
    other = "0x" + "33" * 20
    df = build_df([
        row(mixer_side, customer, 2_500, t0, country="MX_DARK", is_known_mixer=False),
        row(other, customer, 100, t0 + timedelta(hours=1), country="CA"),
        row(mixer_side, other, 100, t0 + timedelta(hours=2), country="CA"),
    ])
    out = _flags(df)
    assert out.loc[0, "mixer_type"] == "mixer_withdraw"
    assert out.loc[1, "mixer_type"] == "mixer_withdraw"
    # The inferred mixer side is not a depositor and not a withdrawal recipient.
    assert out.loc[2, "mixer_type"] == ""


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


def test_deposit_scores_55_and_withdrawal_scores_72(row, t0, build_df):
    """PERSONAL_LIKE multiplier is 1.0. Country CA is home, so foreign adds 0.

    $2,500 is under the $10,000 large-amount line. One row per frame, so
    velocity, structuring, and fan-in stay off. Deposit is w_mixer_touch
    (55). Withdrawal is 55 * 1.3 = 71.5, rounded to 72.
    """
    from scripts.monitor import run_engine_on_df

    deposit = build_df([
        row(ALICE, MIXER, 2_500, t0, country="CA", is_known_mixer=True,
            profile="PERSONAL_LIKE"),
    ])
    out_dep = run_engine_on_df(deposit, threshold=0.4)
    assert out_dep.loc[0, "mixer_type"] == "mixer_deposit"
    assert int(out_dep.loc[0, "risk_score"]) == 55
    assert "mixer_touch" in out_dep.loc[0, "reasons"]
    assert "mixer_withdraw" not in out_dep.loc[0, "reasons"]
    assert "foreign_country" not in out_dep.loc[0, "reasons"]

    withdraw = build_df([
        row(MIXER, ALICE, 2_500, t0, country="CA", is_known_mixer=True,
            profile="PERSONAL_LIKE"),
    ])
    out_wd = run_engine_on_df(withdraw, threshold=0.4)
    assert out_wd.loc[0, "mixer_type"] == "mixer_withdraw"
    assert int(out_wd.loc[0, "risk_score"]) == 72
    assert "mixer_withdraw" in out_wd.loc[0, "reasons"]
    assert "foreign_country" not in out_wd.loc[0, "reasons"]


def test_nan_country_with_false_flag_is_not_a_mixer_row(row, t0, build_df):
    df = build_df([row(ALICE, CHARLIE, 100, t0, country="US")])
    df.loc[0, "country"] = pd.NA
    df.loc[0, "is_known_mixer"] = float("nan")
    out = _flags(df)
    assert bool(out.loc[0, "mixer_flag"]) is False


def test_flag_only_does_not_contaminate_either_wallet(row, t0, build_df):
    """Unknown side: the row is a deposit, and neither wallet is stored."""
    other = "0x" + "44" * 20
    df = build_df([
        row(ALICE, CHARLIE, 100, t0, country="CA", is_known_mixer=True),
        row(ALICE, other, 100, t0 + timedelta(hours=1), country="CA"),
        row(other, CHARLIE, 100, t0 + timedelta(hours=2), country="CA"),
    ])
    out = _flags(df)
    assert out.loc[0, "mixer_type"] == "mixer_deposit"
    assert out.loc[1, "mixer_type"] == ""
    assert out.loc[2, "mixer_type"] == ""


def test_three_case_direction_scores(row, t0, build_df):
    """The three $2,500 shapes from the November write-up, after direction is fixed.

    The withdrawal and the later payment share one frame. The deposit is its
    own frame. Scoring them together makes a round trip, and layering then
    adds points that are not the mixer term.

    Mixer → customer uses profile HIGH_RISK (multiplier 1.0) and country
    MX_DARK. The later payment is country US. The deposit is country UNKNOWN
    and PERSONAL_LIKE. No v6 co-signal fires, so foreign context adds 0.
    """
    from etherscan_fetcher import classify_profile, infer_country
    from scripts.monitor import run_engine_on_df

    exit_df = build_df([
        row(
            MIXER, ALICE, 2_500, t0,
            country=infer_country(MIXER),
            is_known_mixer=True,
            profile=classify_profile(MIXER, False),
        ),
        row(
            DAVE, ALICE, 2_500, t0 + timedelta(minutes=30),
            country="US",
            profile="PERSONAL_LIKE",
        ),
    ])
    exit_out = run_engine_on_df(exit_df, threshold=0.4)
    withdrawal = exit_out.loc[0]
    assert withdrawal["mixer_type"] == "mixer_withdraw"
    assert int(withdrawal["risk_score"]) == 72
    assert withdrawal["risk_level"] == "HIGH"
    later = exit_out.loc[1]
    assert later["mixer_type"] == "mixer_withdraw"
    assert int(later["risk_score"]) == 72
    assert later["risk_level"] == "HIGH"

    deposit_df = build_df([
        row(
            ALICE, MIXER, 2_500, t0,
            country="UNKNOWN",
            is_known_mixer=True,
            profile="PERSONAL_LIKE",
        ),
    ])
    deposit = run_engine_on_df(deposit_df, threshold=0.4).loc[0]
    assert deposit["mixer_type"] == "mixer_deposit"
    assert int(deposit["risk_score"]) == 55
    assert deposit["risk_level"] == "MEDIUM"
    assert "mixer_withdraw" not in deposit["reasons"]
    assert "mixer_touch" in deposit["reasons"]


def test_infer_country_known_mixer_is_mx_dark():
    """Both fetchers share etherscan_fetcher.infer_country. Blockscout has no copy."""
    import etherscan_fetcher
    from etherscan_fetcher import KNOWN_BRIDGES, KNOWN_MIXERS, infer_country
    from scripts.blockscout_fetcher import infer_country as blockscout_infer

    assert blockscout_infer is etherscan_fetcher.infer_country
    assert MIXER in KNOWN_MIXERS
    assert infer_country(MIXER) == "MX_DARK"
    assert infer_country(MIXER.upper()) == "MX_DARK"
    for addr in KNOWN_MIXERS:
        assert infer_country(addr) == "MX_DARK"
    assert infer_country("0x" + "ab" * 20) == "UNKNOWN"
    bridge = "0x3666f603cc164936c1b87e207f36beba4ac5f18d"
    assert bridge in KNOWN_BRIDGES
    assert infer_country(bridge) == "BRIDGE"


def test_fetched_mixer_sender_and_receiver_scores(row, t0, build_df):
    """Country comes from infer_country(sender), so a mixer sender is MX_DARK.

    That row is mixer_withdraw because the sender address is the mixer.
    A later payment to the customer who received the funds stays
    mixer_withdraw. $2,500 stays under the personal large-amount line.
    MX_DARK is foreign context only (0 points) and is not high-risk.
    The only points are 55 * 1.3, rounded to 72.
    """
    from etherscan_fetcher import classify_profile, infer_country
    from scripts.monitor import run_engine_on_df

    country = infer_country(MIXER)
    assert country == "MX_DARK"
    df = build_df([
        row(
            MIXER, ALICE, 2_500, t0,
            country=country,
            is_known_mixer=True,
            profile=classify_profile(MIXER, False),
        ),
        row(
            DAVE, ALICE, 2_500, t0 + timedelta(minutes=30),
            country="US",
            profile="PERSONAL_LIKE",
        ),
    ])
    out = run_engine_on_df(df, threshold=0.4)

    sender = out.loc[0]
    assert sender["mixer_type"] == "mixer_withdraw"
    assert bool(sender["high_risk_country_flag"]) is False
    assert bool(sender["ofac_flag"]) is False
    assert int(sender["risk_score"]) == 72
    assert sender["risk_level"] == "HIGH"
    assert "foreign_context_only" in sender["reasons"]
    assert "high_risk_jurisdiction" not in sender["reasons"]

    receiver_row = out.loc[1]
    assert receiver_row["mixer_type"] == "mixer_withdraw"
    assert int(receiver_row["risk_score"]) == 72
    assert receiver_row["risk_level"] == "HIGH"
    assert "high_risk_jurisdiction" not in receiver_row["reasons"]
