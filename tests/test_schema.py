"""Schema validator tests — required/optional columns + error messages."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest


def _base_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sender_id":   "0xaaa",
            "receiver_id": "0xbbb",
            "amount":      100.0,
            "timestamp":   datetime(2025, 4, 1),
        },
    ])


def test_validate_accepts_minimal_df():
    from engine.schema import validate_dataframe
    issues = validate_dataframe(_base_df())
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []


def test_validate_flags_missing_required_column():
    from engine.schema import validate_dataframe
    df = _base_df().drop(columns=["amount"])
    issues = validate_dataframe(df)
    codes  = {i.code for i in issues}
    assert "E_MISSING_COLUMN" in codes


def test_validate_flags_non_numeric_amount():
    from engine.schema import validate_dataframe
    # Build with amount column as object so we can legally hold a string
    df = pd.DataFrame([{
        "sender_id": "0xaaa", "receiver_id": "0xbbb",
        "amount": "not a number", "timestamp": datetime(2025, 4, 1),
    }])
    issues = validate_dataframe(df)
    assert any(i.code == "E_AMOUNT_NOT_NUMERIC" for i in issues)


def test_validate_flags_unparseable_timestamp():
    from engine.schema import validate_dataframe
    df = pd.DataFrame([{
        "sender_id": "0xaaa", "receiver_id": "0xbbb",
        "amount": 100.0, "timestamp": "not a date",
    }])
    issues = validate_dataframe(df)
    assert any(i.code == "E_TIMESTAMP_UNPARSEABLE" for i in issues)


def test_validate_flags_null_addresses():
    from engine.schema import validate_dataframe
    df = _base_df()
    df.loc[0, "sender_id"] = None
    issues = validate_dataframe(df)
    assert any(i.code == "E_NULL_ADDRESS" for i in issues)


def test_require_valid_raises_on_errors():
    from engine.schema import SchemaError, require_valid
    bad = _base_df().drop(columns=["sender_id"])
    with pytest.raises(SchemaError) as exc:
        require_valid(bad)
    assert "E_MISSING_COLUMN" in str(exc.value)


def test_normalise_fills_optional_columns():
    from engine.schema import normalise
    out = normalise(_base_df())
    assert "sender_profile" in out.columns
    assert out.loc[0, "sender_profile"] == "PERSONAL_LIKE"
    assert "is_known_mixer" in out.columns
    assert out.loc[0, "is_known_mixer"] is False or out.loc[0, "is_known_mixer"] == False  # noqa: E712
    # v13 multi-asset columns
    assert "asset_type" in out.columns
    assert out.loc[0, "asset_type"] == "native"


def test_validate_warning_for_empty_df():
    from engine.schema import validate_dataframe
    issues = validate_dataframe(pd.DataFrame())
    assert any(i.code == "W_EMPTY" for i in issues)


def test_validate_csv_path_missing_file(tmp_path):
    from engine.schema import validate_csv_path
    issues = validate_csv_path(tmp_path / "does_not_exist.csv")
    assert issues and issues[0].code == "E_FILE_NOT_FOUND"
