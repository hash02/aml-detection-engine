"""
engine/schema.py — Input validation for engine CSVs
=====================================================

The engine expects a specific schema. Until this module existed, a
missing column produced a KeyError three stack frames deep in
compute_features. Analysts can't debug that.

Public API:
  validate_dataframe(df)          → list[ValidationIssue]
  validate_csv_path(path)         → list[ValidationIssue]
  require_valid(df)               → raises SchemaError if anything's wrong

Philosophy: strict required columns, lenient defaults for optional
ones (auto-populate rather than raise). Errors are *messages for
humans* — they say which rows, which values, how to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"sender_id", "receiver_id", "amount", "timestamp"}
OPTIONAL_COLUMNS = {
    "country":             "UNKNOWN",
    "sender_profile":      "PERSONAL_LIKE",
    "is_known_mixer":      False,
    "is_bridge":           False,
    "sender_tx_count":     0,
    "sender_active_days":  0,
    "account_age_days":    0,
    "id":                  None,      # auto-generated if absent
}
# v13 multi-asset columns — entirely optional, defaults produce legacy behaviour
V13_DEFAULTS = {
    "asset_type":           "native",
    "token_contract":       "",
    "token_symbol":         "",
    "token_amount_decimal": 0.0,
}


class SchemaError(ValueError):
    """Raised by require_valid() when one or more issues are fatal."""


@dataclass
class ValidationIssue:
    severity: str     # "error" | "warning"
    code:     str
    message:  str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


def validate_dataframe(df: pd.DataFrame) -> list[ValidationIssue]:
    """Return a list of issues. Empty list means the frame is good."""
    issues: list[ValidationIssue] = []
    if df is None:
        issues.append(ValidationIssue("error", "E_NULL",
                                      "DataFrame is None"))
        return issues
    if len(df) == 0:
        issues.append(ValidationIssue("warning", "W_EMPTY",
                                      "DataFrame is empty — the engine will "
                                      "return zero alerts without raising."))
        return issues

    # Required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    for col in sorted(missing):
        issues.append(ValidationIssue(
            "error", "E_MISSING_COLUMN",
            f"Required column {col!r} is missing. The engine needs "
            f"{sorted(REQUIRED_COLUMNS)} at minimum."))

    # Spot-check types
    if "amount" in df.columns:
        try:
            amounts = pd.to_numeric(df["amount"], errors="coerce")
            bad = amounts.isna().sum()
            if bad:
                issues.append(ValidationIssue(
                    "error", "E_AMOUNT_NOT_NUMERIC",
                    f"{bad} row(s) have non-numeric `amount`. "
                    f"Sample: {df.loc[amounts.isna(), 'amount'].head(3).tolist()}"))
            if (amounts < 0).any():
                n = int((amounts < 0).sum())
                issues.append(ValidationIssue(
                    "warning", "W_NEGATIVE_AMOUNT",
                    f"{n} row(s) have negative `amount` — allowed, but "
                    f"unusual for AML workloads."))
        except Exception as e:  # noqa: BLE001
            issues.append(ValidationIssue("error", "E_AMOUNT_CHECK_FAILED", str(e)))

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        bad = ts.isna().sum()
        if bad:
            issues.append(ValidationIssue(
                "error", "E_TIMESTAMP_UNPARSEABLE",
                f"{bad} row(s) have an unparseable `timestamp`. "
                f"Sample: {df.loc[ts.isna(), 'timestamp'].head(3).tolist()}"))

    # Duplicate IDs (if id is present)
    if "id" in df.columns:
        dup = int(df["id"].duplicated().sum())
        if dup:
            issues.append(ValidationIssue(
                "warning", "W_DUPLICATE_ID",
                f"{dup} duplicate id value(s). Audit log dedupes, but "
                f"per-rule counts may be inflated."))

    # Null senders / receivers
    for col in ("sender_id", "receiver_id"):
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                issues.append(ValidationIssue(
                    "error", "E_NULL_ADDRESS",
                    f"{n} row(s) have a null {col}."))

    return issues


def validate_csv_path(path: str | Path) -> list[ValidationIssue]:
    p = Path(path)
    if not p.exists():
        return [ValidationIssue("error", "E_FILE_NOT_FOUND", f"No such file: {p}")]
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        return [ValidationIssue("error", "E_CSV_READ_FAILED", f"{type(e).__name__}: {e}")]
    return validate_dataframe(df)


def require_valid(df: pd.DataFrame) -> None:
    """Raise SchemaError if any error-severity issues exist."""
    issues = validate_dataframe(df)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise SchemaError(
            "Schema validation failed:\n" + "\n".join(f"  {i}" for i in errors)
        )


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional + v13 columns with safe defaults. Never raises.

    Callers that want a clean, engine-ready frame without having to
    remember the full list of optional columns just call this.
    """
    df = df.copy()
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            if col == "id":
                df[col] = (
                    df["sender_id"].astype(str).str[-6:] + "-"
                    + df["receiver_id"].astype(str).str[-6:] + "-"
                    + df.index.astype(str)
                )
            else:
                df[col] = default
    for col, default in V13_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df
