"""
Pytest fixtures — share a minimal "base" transaction frame that every
rule test can mutate, so individual tests stay under 20 lines.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Keep feed fetches offline during tests — we rely on bundled baselines.
os.environ.setdefault("FEEDS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "engine"))


def _row(
    sender: str,
    receiver: str,
    amount: float,
    ts: datetime,
    profile: str = "PERSONAL_LIKE",
    country: str = "US",
    is_known_mixer: bool = False,
    is_bridge: bool = False,
    sender_tx_count: int = 50,
    sender_active_days: int = 365,
    account_age_days: int = 365,
) -> dict:
    return {
        "id": f"tx_{sender[-4:]}_{receiver[-4:]}_{int(ts.timestamp())}",
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": amount,
        "country": country,
        "timestamp": ts,
        "sender_profile": profile,
        "is_known_mixer": is_known_mixer,
        "is_bridge": is_bridge,
        "sender_tx_count": sender_tx_count,
        "sender_active_days": sender_active_days,
        "account_age_days": account_age_days,
    }


@pytest.fixture
def row():
    return _row


@pytest.fixture
def t0() -> datetime:
    return datetime(2025, 4, 1, 12, 0, 0)


@pytest.fixture
def build_df():
    """Factory: given a list of dicts, return a sorted DataFrame."""
    def _build(rows):
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    return _build


@pytest.fixture
def cfg():
    """Engine CONFIG dict — import lazily to keep module-import time low."""
    from engine_v11_blockchain import CONFIG
    return {**CONFIG}
