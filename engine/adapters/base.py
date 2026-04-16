"""
engine/adapters/base.py — Abstract adapter + canonical schema
==============================================================

Every chain-specific adapter subclasses `Adapter` and:
  - `fetch_transactions(address, limit)` → list[CanonicalTx]
  - `to_dataframe(txs)` → DataFrame in the engine's schema

The canonical schema is defined by `CanonicalTx`. Adding a field here
means every adapter must produce it; adding an optional v13 field (see
the `asset_type` / token columns) without breaking older engine calls
is done by giving it a sensible default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

# v13 multi-asset tag values — keep this list closed so downstream rules
# can switch on it without a dictionary of magic strings.
ASSET_NATIVE = "native"     # ETH, TRX, SOL, BTC — the chain's base asset
ASSET_ERC20  = "erc20"      # Any fungible token
ASSET_ERC721 = "erc721"     # NFT; most rules ignore this
ASSET_UTXO   = "utxo"       # Bitcoin-style output


@dataclass
class CanonicalTx:
    """One transaction in engine-canonical form."""

    # Core fields (required by every detector)
    id:          str
    sender_id:   str
    receiver_id: str
    amount:      float                  # USD-denominated, decimal
    timestamp:   pd.Timestamp | str
    country:     str = "UNKNOWN"
    sender_profile: str = "PERSONAL_LIKE"

    # Context flags (optional — default False when unknown)
    is_known_mixer: bool = False
    is_bridge:      bool = False

    # Sender-centric history (optional — 0 is a safe default)
    sender_tx_count:    int = 0
    sender_active_days: int = 0
    account_age_days:   int = 0

    # v13 multi-asset columns (NEW in Phase 3)
    asset_type:         str  = ASSET_NATIVE
    token_contract:     str  = ""       # empty for native; ERC-20 address
    token_symbol:       str  = ""       # e.g. "USDC"
    token_amount_decimal: float = 0.0   # raw token quantity (post-decimals)

    # Arbitrary chain-specific extras — not used by rules, preserved for audit
    extra: dict[str, Any] = field(default_factory=dict)


class Adapter(ABC):
    """Abstract base class for chain ingestion adapters."""

    chain: str = "unknown"

    @abstractmethod
    def fetch_transactions(self, address: str, limit: int = 200) -> list[CanonicalTx]:
        """Pull transactions for an address, normalised to CanonicalTx."""
        raise NotImplementedError

    def to_dataframe(self, txs: list[CanonicalTx]) -> pd.DataFrame:
        """Convert a list of CanonicalTx into the engine's DataFrame schema."""
        if not txs:
            return pd.DataFrame(columns=[f.name for f in _dc_fields(CanonicalTx)])
        rows = [asdict(t) for t in txs]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Drop the `extra` dict column — engine doesn't consume it
        if "extra" in df.columns:
            df = df.drop(columns=["extra"])
        return df


def _dc_fields(cls):
    # Small wrapper so importers don't need dataclasses import
    from dataclasses import fields
    return fields(cls)
