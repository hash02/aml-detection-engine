"""
engine/adapters/ethereum.py — ETH mainnet adapter
==================================================

Wraps the existing etherscan_fetcher + a new tokentx (ERC-20) fetcher.
Produces canonical rows for both native ETH transfers and ERC-20
transfers, so the engine can see multi-asset movement in a single DF.

Downstream impact: rules that are asset-agnostic (fan_in, velocity,
sybil_fan_in, …) naturally work on both. Rules that care about dollar
values (large_amount, sub_threshold_tranching) use the `amount` column,
which we populate with USD-equivalent regardless of asset type.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import ASSET_ERC20, ASSET_NATIVE, Adapter, CanonicalTx

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"


def _session():
    # Local import so the adapter module stays importable in test
    # environments that don't have etherscan_fetcher available.
    try:
        from etherscan_fetcher import SESSION
        return SESSION
    except ImportError:
        import requests
        return requests.Session()


def _eth_price_usd() -> float:
    try:
        from etherscan_fetcher import get_eth_price_usd
        return float(get_eth_price_usd())
    except Exception:  # noqa: BLE001
        return 2500.0


class EthereumAdapter(Adapter):
    chain = "ethereum"

    # ── public API ─────────────────────────────────────────────────────
    def fetch_transactions(self, address: str, limit: int = 200) -> list[CanonicalTx]:
        """Fetch native ETH + ERC-20 transactions for `address`.

        If `ETHERSCAN_API_KEY` is set, uses Etherscan V2; otherwise we
        fall back to the existing Blockchair path in etherscan_fetcher.
        Token transfers require the Etherscan key — without it, we only
        return native transfers.
        """
        key = os.environ.get("ETHERSCAN_API_KEY", "")
        native = self._fetch_native(address, limit, key)
        tokens = self._fetch_tokens(address, limit, key) if key else []
        return native + tokens

    # ── native ETH ─────────────────────────────────────────────────────
    def _fetch_native(self, address: str, limit: int, key: str) -> list[CanonicalTx]:
        try:
            from etherscan_fetcher import fetch_transactions
        except ImportError:
            return []
        raw = fetch_transactions(address, api_key=key, limit=limit)
        price = _eth_price_usd()
        out: list[CanonicalTx] = []
        for tx in raw:
            try:
                out.append(self._native_to_canonical(tx, price))
            except (KeyError, ValueError, TypeError):
                continue
        return out

    @staticmethod
    def _native_to_canonical(tx: dict[str, Any], eth_price: float) -> CanonicalTx:
        sender = str(tx.get("from") or tx.get("sender") or "").lower()
        recv   = str(tx.get("to")   or tx.get("recipient") or "").lower()
        wei    = int(tx.get("value", 0) or 0)
        eth    = wei / 1e18
        amount = eth * eth_price
        ts     = tx.get("timeStamp") or tx.get("time") or ""
        if isinstance(ts, str) and ts.isdigit():
            ts_iso = int(ts)
        else:
            ts_iso = ts
        return CanonicalTx(
            id          = str(tx.get("hash", f"eth-{sender[:10]}-{recv[:10]}-{time.time_ns()}")),
            sender_id   = sender,
            receiver_id = recv,
            amount      = amount,
            timestamp   = ts_iso,
            asset_type  = ASSET_NATIVE,
            token_symbol = "ETH",
            token_amount_decimal = eth,
        )

    # ── ERC-20 tokentx ─────────────────────────────────────────────────
    def _fetch_tokens(self, address: str, limit: int, key: str) -> list[CanonicalTx]:
        """Etherscan V2 `tokentx` endpoint — ERC-20 transfers in/out."""
        if not key:
            return []
        session = _session()
        try:
            resp = session.get(
                ETHERSCAN_V2_URL,
                params={
                    "chainid": 1,
                    "module":  "account",
                    "action":  "tokentx",
                    "address": address,
                    "page":    1,
                    "offset":  min(limit, 10000),
                    "sort":    "asc",
                    "apikey":  key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                return []
        except Exception:  # noqa: BLE001 — token fetch is best-effort
            return []

        eth_price = _eth_price_usd()
        # Rough stablecoin set so we can price USDC/USDT/DAI at parity
        STABLES = {"USDC", "USDT", "DAI", "FRAX", "LUSD", "TUSD", "USDP"}

        out: list[CanonicalTx] = []
        for tx in data.get("result", []):
            try:
                decimals = int(tx.get("tokenDecimal", 18))
                raw_amt  = int(tx.get("value", 0))
                token_amt = raw_amt / (10 ** decimals) if decimals >= 0 else float(raw_amt)
                symbol   = (tx.get("tokenSymbol") or "").upper()
                # Best-effort USD amount — stable @ 1, WETH @ spot, else 0
                if symbol in STABLES:
                    usd = token_amt
                elif symbol in {"WETH", "STETH", "RETH"}:
                    usd = token_amt * eth_price
                else:
                    usd = 0.0   # Unknown token; downstream rules still get count/velocity
                out.append(CanonicalTx(
                    id          = str(tx.get("hash", "")),
                    sender_id   = str(tx.get("from", "")).lower(),
                    receiver_id = str(tx.get("to", "")).lower(),
                    amount      = float(usd),
                    timestamp   = tx.get("timeStamp", ""),
                    asset_type  = ASSET_ERC20,
                    token_contract       = str(tx.get("contractAddress", "")).lower(),
                    token_symbol         = symbol,
                    token_amount_decimal = token_amt,
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return out
