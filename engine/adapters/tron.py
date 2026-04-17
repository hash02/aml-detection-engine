"""
engine/adapters/tron.py — Tron adapter (stub)
==============================================

Tron is the dominant stablecoin-laundering rail (USDT-TRC20). A full
adapter would use the Tron gRPC or TronGrid JSON-RPC to pull the
`transferContract` + `triggerSmartContract` streams, then normalise
them the same way EthereumAdapter does.

Left as a stub so the registry pattern + tests exercise it, but it
raises `NotImplementedError` rather than silently returning empty —
making it explicit that Tron ingestion still needs wiring.
"""

from __future__ import annotations

from .base import Adapter, CanonicalTx


class TronAdapter(Adapter):
    chain = "tron"

    def fetch_transactions(self, address: str, limit: int = 200) -> list[CanonicalTx]:
        raise NotImplementedError(
            "Tron adapter is a stub. Pending: TronGrid API client + TRC20 "
            "log decoding. See engine/adapters/ethereum.py for the reference "
            "shape."
        )
