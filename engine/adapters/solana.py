"""
engine/adapters/solana.py — Solana adapter (stub)
==================================================

Solana's tx model (program invocations, multiple transfers per txn,
SPL tokens) is chunkier than EVM. A full adapter would paginate
`getSignaturesForAddress` → `getTransaction` and split each compound
transaction into per-transfer canonical rows.

Left as a stub so the registry exposes it but honestly flags that the
hard part (account-level vs instruction-level modelling) is unfinished.
"""

from __future__ import annotations

from .base import Adapter, CanonicalTx


class SolanaAdapter(Adapter):
    chain = "solana"

    def fetch_transactions(self, address: str, limit: int = 200) -> list[CanonicalTx]:
        raise NotImplementedError(
            "Solana adapter is a stub. Pending: JSON-RPC client + SPL Token "
            "instruction decoding. Pagination via getSignaturesForAddress."
        )
