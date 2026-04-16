"""
engine/adapters — Cross-chain ingestion adapters
==================================================

Each adapter is responsible for turning raw chain-native data into the
engine's canonical schema:

    id, sender_id, receiver_id, amount, country, timestamp,
    sender_profile, is_known_mixer, is_bridge,
    sender_tx_count, sender_active_days, account_age_days,
    # v13 multi-asset columns (NEW)
    asset_type, token_contract, token_symbol, token_amount_decimal

Adapter discovery:
    from engine.adapters import get_adapter
    eth = get_adapter("ethereum")
    df  = eth.fetch_transactions(address, limit=200)

Adding a chain is a matter of dropping a new module in here and
registering it in `_REGISTRY` at the bottom of this file. Nothing else
in the engine needs to know which chain data came from.
"""

from __future__ import annotations

from .base import Adapter, CanonicalTx  # noqa: F401
from .ethereum import EthereumAdapter

_REGISTRY: dict[str, type[Adapter]] = {
    "ethereum": EthereumAdapter,
    # Registered but not fully implemented — see stub modules for TODOs:
    #   "tron":    TronAdapter,
    #   "solana":  SolanaAdapter,
    #   "bitcoin": BitcoinAdapter,
}


def get_adapter(name: str) -> Adapter:
    """Return an instantiated adapter for the named chain.

    Raises KeyError if the chain isn't registered.
    """
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise KeyError(
            f"no adapter for {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return cls()


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)
