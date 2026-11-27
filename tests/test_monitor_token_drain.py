"""Offline checks for token rows and the monitor's drainer wiring.

No network. Token payloads are fixtures. Scoring goes through
scripts.monitor.run_engine_on_df, the same function the daily job uses.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta

import pandas as pd

VICTIM = "0x" + "11" * 20
DRAINER = "0x" + "22" * 20
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
OTHER = "0x" + "33" * 20


def _native(tx_hash: str, sender: str, receiver: str, ts: int, wei: int = 10**18) -> dict:
    return {
        "hash": tx_hash,
        "from": sender,
        "to": receiver,
        "value": str(wei),
        "timeStamp": str(ts),
        "isError": "0",
    }


def _tokentx(tx_hash: str, sender: str, receiver: str, contract: str, symbol: str, ts: int) -> dict:
    return {
        "hash": tx_hash,
        "from": sender,
        "to": receiver,
        "contractAddress": contract,
        "value": "10000000000",  # 10,000 units at 6 decimals
        "tokenDecimal": "6",
        "tokenSymbol": symbol,
        "timeStamp": str(ts),
    }


def test_build_token_rows_keeps_two_assets_that_share_a_hash():
    from etherscan_fetcher import build_token_rows

    ts = 1_710_000_000
    rows = build_token_rows(
        [
            _tokentx("0xabc", VICTIM, DRAINER, USDC, "USDC", ts),
            _tokentx("0xabc", VICTIM, DRAINER, USDT, "USDT", ts + 10),
        ],
        eth_price=2000.0,
    )
    assert len(rows) == 2
    assert {r["token_contract"] for r in rows} == {USDC, USDT}
    assert {r["asset_type"] for r in rows} == {"erc20"}
    assert rows[0]["id"] != rows[1]["id"]
    assert rows[0]["amount"] == 10000.0


def test_native_rows_carry_asset_type_so_the_drainer_rule_can_run():
    from etherscan_fetcher import build_transaction_rows

    rows = build_transaction_rows(
        [_native("0xnat", VICTIM, OTHER, 1_710_000_000)],
        eth_price=2000.0,
        source_address=VICTIM,
    )
    assert len(rows) == 1
    assert rows[0]["asset_type"] == "native"
    assert rows[0]["token_contract"] == ""


def test_fetch_live_data_merges_tokentx_without_network(monkeypatch, tmp_path):
    import etherscan_fetcher as fetcher

    ts = 1_710_000_000
    monkeypatch.setattr(fetcher, "get_eth_price_usd", lambda: 2000.0)
    monkeypatch.setattr(
        fetcher,
        "fetch_transactions",
        lambda *args, **kwargs: [_native("0xnat", VICTIM, OTHER, ts)],
    )
    monkeypatch.setattr(
        fetcher,
        "fetch_etherscan_tokentx",
        lambda *args, **kwargs: [
            _tokentx("0xdrain", VICTIM, DRAINER, USDC, "USDC", ts),
            _tokentx("0xdrain", VICTIM, DRAINER, USDT, "USDT", ts + 20),
        ],
    )
    df = fetcher.fetch_live_data(
        addresses=[VICTIM],
        api_key="not-a-real-key",
        limit=10,
        expand_hops=False,
        output_path=str(tmp_path / "out.csv"),
    )
    assert set(df["asset_type"]) == {"native", "erc20"}
    token_rows = df[df["asset_type"] == "erc20"]
    assert set(token_rows["token_contract"]) == {USDC, USDT}
    assert len(token_rows) == 2


def test_tokentx_is_not_called_without_an_api_key(monkeypatch, tmp_path):
    import etherscan_fetcher as fetcher

    called = {"n": 0}

    def _boom(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("tokentx should not run without a key")

    monkeypatch.setattr(fetcher, "get_eth_price_usd", lambda: 2000.0)
    monkeypatch.setattr(
        fetcher,
        "fetch_transactions",
        lambda *args, **kwargs: [_native("0xnat", VICTIM, OTHER, 1_710_000_000)],
    )
    monkeypatch.setattr(fetcher, "fetch_etherscan_tokentx", _boom)
    df = fetcher.fetch_live_data(
        addresses=[VICTIM],
        api_key="",
        limit=10,
        expand_hops=False,
        output_path=str(tmp_path / "out.csv"),
    )
    assert called["n"] == 0
    assert list(df["asset_type"]) == ["native"]


def test_blockscout_token_rows_keep_distinct_assets():
    from scripts.blockscout_fetcher import build_rows_from_blockscout_tokens

    ts = "2024-06-01T12:00:00.000000Z"
    rows = build_rows_from_blockscout_tokens([
        {
            "from": {"hash": VICTIM},
            "to": {"hash": DRAINER},
            "tx_hash": "0xsame",
            "log_index": 1,
            "timestamp": ts,
            "token": {
                "address": USDC, "symbol": "USDC", "decimals": "6",
                "type": "ERC-20", "exchange_rate": "1.0",
            },
            "total": {"decimals": "6", "value": "5000000"},
        },
        {
            "from": {"hash": VICTIM},
            "to": {"hash": DRAINER},
            "tx_hash": "0xsame",
            "log_index": 2,
            "timestamp": ts,
            "token": {
                "address": USDT, "symbol": "USDT", "decimals": "6",
                "type": "ERC-20", "exchange_rate": "1",
            },
            "total": {"decimals": "6", "value": "5000000"},
        },
        {
            "from": {"hash": VICTIM},
            "to": {"hash": DRAINER},
            "tx_hash": "0xnft",
            "log_index": 3,
            "timestamp": ts,
            "token": {"address": "0xnft", "symbol": "NFT", "decimals": "0", "type": "ERC-721"},
            "total": {"decimals": "0", "value": "1"},
        },
    ])
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {f"0xsame:{USDC}:1", f"0xsame:{USDT}:2"}
    assert rows[0]["amount"] == 5.0


def _score(rows: list[dict]) -> pd.DataFrame:
    from scripts.monitor import run_engine_on_df

    df = pd.DataFrame(rows)
    for col, default in (
        ("is_known_mixer", False),
        ("is_bridge", False),
        ("sender_tx_count", 40),
        ("sender_active_days", 400),
        ("sender_profile", "PERSONAL_LIKE"),
        ("country", "US"),
        ("label", "SYNTH"),
    ):
        if col not in df.columns:
            df[col] = default
    return run_engine_on_df(df, threshold=0.7)


def _erc20(sender, receiver, contract, symbol, ts: datetime, amount=10_000.0) -> dict:
    return {
        "id": f"{int(ts.timestamp())}:{contract}",
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": amount,
        "timestamp": ts,
        "asset_type": "erc20",
        "token_contract": contract,
        "token_symbol": symbol,
        "token_amount_decimal": amount,
    }


def test_monitor_drainer_fires_on_two_tokens_and_stays_quiet_on_normal_rows():
    from scripts.monitor import run_engine_on_df

    source = inspect.getsource(run_engine_on_df)
    assert "df = detect_drainer_signature(df, cfg)" in source
    assert "detect_address_poisoning(" not in source

    t0 = datetime(2024, 6, 1, 12, 0, 0)
    drain = _score([
        _erc20(VICTIM, DRAINER, USDC, "USDC", t0),
        _erc20(VICTIM, DRAINER, USDT, "USDT", t0 + timedelta(seconds=30)),
    ])
    assert int(drain["drainer_flag"].sum()) == 2
    assert drain["reasons"].astype(str).str.contains("drainer_signature").all()

    same_token = _score([
        _erc20(VICTIM, DRAINER, USDC, "USDC", t0),
        _erc20(VICTIM, DRAINER, USDC, "USDC", t0 + timedelta(seconds=30)),
    ])
    assert int(same_token["drainer_flag"].sum()) == 0

    native = _score([
        {
            "id": "n1", "sender_id": VICTIM, "receiver_id": DRAINER,
            "amount": 12000.0, "timestamp": t0, "asset_type": "native",
            "token_contract": "", "token_symbol": "ETH", "token_amount_decimal": 4.0,
        },
        {
            "id": "n2", "sender_id": VICTIM, "receiver_id": DRAINER,
            "amount": 8000.0, "timestamp": t0 + timedelta(seconds=40),
            "asset_type": "native", "token_contract": "", "token_symbol": "ETH",
            "token_amount_decimal": 3.0,
        },
    ])
    assert int(native["drainer_flag"].sum()) == 0

    too_slow = _score([
        _erc20(VICTIM, DRAINER, USDC, "USDC", t0),
        _erc20(VICTIM, DRAINER, USDT, "USDT", t0 + timedelta(minutes=10)),
    ])
    assert int(too_slow["drainer_flag"].sum()) == 0
