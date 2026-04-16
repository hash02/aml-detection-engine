"""Adapter registry + canonical schema tests."""

from __future__ import annotations

import pytest


def test_adapter_registry_lists_ethereum():
    from engine.adapters import list_adapters
    assert "ethereum" in list_adapters()


def test_get_adapter_ethereum_returns_adapter():
    from engine.adapters import get_adapter
    from engine.adapters.base import Adapter
    eth = get_adapter("ethereum")
    assert isinstance(eth, Adapter)
    assert eth.chain == "ethereum"


def test_get_adapter_unknown_raises():
    from engine.adapters import get_adapter
    with pytest.raises(KeyError):
        get_adapter("cardano")


def test_canonical_tx_to_dataframe_preserves_v13_columns():
    from engine.adapters.base import ASSET_ERC20, CanonicalTx
    from engine.adapters.ethereum import EthereumAdapter

    txs = [
        CanonicalTx(
            id="tx1", sender_id="0xa", receiver_id="0xb", amount=100.0,
            timestamp="2025-04-01T12:00:00",
            asset_type=ASSET_ERC20, token_symbol="USDC",
            token_contract="0xusdc", token_amount_decimal=100.0,
        ),
    ]
    df = EthereumAdapter().to_dataframe(txs)
    assert "asset_type" in df.columns
    assert "token_contract" in df.columns
    assert df.iloc[0]["asset_type"] == ASSET_ERC20
    assert df.iloc[0]["token_symbol"] == "USDC"
    assert "extra" not in df.columns  # extras get stripped before handing to engine


def test_canonical_tx_empty_list_returns_empty_df():
    from engine.adapters.ethereum import EthereumAdapter

    df = EthereumAdapter().to_dataframe([])
    assert df.empty
    assert "asset_type" in df.columns


def test_tron_and_solana_stubs_raise_not_implemented():
    """Stubs are explicit — they don't pretend to work."""
    import pytest

    from engine.adapters.solana import SolanaAdapter
    from engine.adapters.tron import TronAdapter

    with pytest.raises(NotImplementedError):
        TronAdapter().fetch_transactions("TX")
    with pytest.raises(NotImplementedError):
        SolanaAdapter().fetch_transactions("SO")
