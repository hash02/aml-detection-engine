"""
engine/feeds.py — Live threat-intelligence feed loader
=======================================================

Replaces the hard-coded OFAC_SDN_ADDRESSES / KNOWN_MIXERS sets in the
engine with live, refreshable feeds. Every lookup is O(1) against a
pre-built set, so the hot path in detect_* functions is unchanged.

Feeds supported:
  - ofac_sdn         → OFAC SDN virtual-currency addendum (US Treasury)
  - phishing         → ScamSniffer scam-database address blacklist (GitHub, GPL-3.0)
  - chainalysis      → Chainalysis Sanctions Oracle (on-chain, ETH mainnet)
  - mixers           → Tornado-style privacy pool contracts
  - bridges          → Cross-chain bridge routers
  - exchanges        → Centralised exchange hot wallets
  - lrt_offramps     → LRT / no-KYC off-ramp clusters (analyst-maintained)

All feeds persist to data/feeds/<feed>.json with a last_updated timestamp.
Callers use `get_feed(name)` → frozenset[str] of lowercased 0x addresses.

Offline / air-gapped: the loader never raises on network failure; it
returns the last cached copy and logs the staleness. CI sets
`FEEDS_OFFLINE=1` to skip all network calls.

Source precedence in `get_feed()` (highest first):
  1. Local cache, data/feeds/<feed>.json: the freshest copy, written
     by `refresh()` on this machine and operator-controlled.
  2. Committed snapshot, addresses/<file>.json: the repo-committed
     copy of the feeds, refreshed by the daily workflow. This is the
     fallback that keeps a fresh deployment (no cache yet) from
     running on baselines alone.
  3. Hardcoded baseline below: the never-empty floor bundled in code.
The baseline is unioned into whichever tier wins, so a feed never
returns fewer than its bundled entries.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

FEEDS_DIR = Path(__file__).resolve().parent.parent / "data" / "feeds"
FEEDS_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "addresses"

# Committed snapshot filename per feed key. Mirrors the populate
# MAPPING in .github/workflows/refresh-addresses-daily.yml, which
# writes addresses/<file> from the cache of the same feed key; keep
# the two in sync. "chainalysis" has no snapshot: its fetcher
# returns empty by design and lookups go through check_chainalysis()
# per address, so there is nothing for a snapshot to carry.
SNAPSHOT_FILES: dict[str, str] = {
    "ofac_sdn": "ofac.json",
    "phishing": "phishing.json",
    "mixers": "mixers.json",
    "bridges": "bridges.json",
    "lrt_offramps": "offramps.json",
}

OFFLINE = os.environ.get("FEEDS_OFFLINE") == "1"
DEFAULT_TIMEOUT = 15
STALE_AFTER_HOURS = 24


# ── Bundled baseline (never empty, ships in the repo) ────────────────────────
# Matches the previously inlined OFAC_SDN_ADDRESSES in engine_v11_blockchain.py
# so behaviour is identical when the network is unreachable.
BASELINE_OFAC = frozenset({
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
    "0x9ad122c22b14202b4490edaf288fdb3c7cb3ff5d",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
    "0x5efda50f22d34f262c29268506c5fa42cb56a1ce",
    "0x8589427373d6d84e98730d7795d8f6f8731fda16",
    "0xd96f2b1c14db8458374d9aca76e26c3950113464",
    "0x4736dcf1b7a3d580672a2389a73823eb9ea0c5b6",
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96",
    "0x172370d5cd63279efa6d502dab29171933a610af",
    "0x9c9e10e1f65d3ffd0b61f3b6d6c3b5de60c1e31f",
    "0xf2bd9aa5ff88de44d0d8ab0f85bfe4d6e89eb04e",
    "0x974caa59e49682cda0ad2bbe82983419a2ecc400",
    "0x96221423681a6d52e184d440a8efcebb105c7242",
    "0xbdd077f651ebe7f7b3ce16fe5f2b025be2969516",
})

BASELINE_MIXERS = frozenset({
    "0x12d66f87a04a9e220c9d35925b72aca3ca8c78e2",
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
    "0x722122df12d4e14e13ac3b6895a86e84145b6967",
    "0x2717c5e28cf931547b621a5dddb772ab6a35b701",
    "0x7f367cc41522ce07553e823bf3be79a889debe1b",
})

BASELINE_BRIDGES = frozenset({
    "0x3666f603cc164936c1b87e207f36beba4ac5f18d",
    "0x3e4a3a4796d16c0cd582c382691998f7c06420b6",
    "0xb8901acb165ed027e32754e0ffe830802919727f",
    "0x4d9079bb4165aeb4084c526a32695dcfd2f77381",
    "0x8731d54e9d02c286767d56ac03e8037c07e01e98",
    "0x66a71dcef29a0ffbdbe3c6a460a3b5bc225cd675",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",
    "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf",
    "0x2796317b0ff8538f253012862c06787adfb8ceb6",
})

# No-KYC / LRT off-ramps and known drainer infrastructure (analyst-curated).
# These are NOT sanctioned — the engine weights them lower than OFAC.
BASELINE_LRT_OFFRAMPS = frozenset({
    # FixedFloat (no-KYC swap) — widely used in 2024-2025 DPRK laundering
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",
    # ChangeNOW hot wallet cluster (no-KYC)
    "0x077d360f11d220e4d5d831430c81c26c9be7c4a4",
    # Inferno Drainer relayer (2024 documented)
    "0x0000db5c8b030ae20308ac975898e09741e70000",
})


@dataclass
class Feed:
    name: str
    baseline: frozenset[str]
    fetch: Callable[[], frozenset[str]] | None = None

    @property
    def cache_path(self) -> Path:
        return FEEDS_DIR / f"{self.name}.json"

    @property
    def snapshot_path(self) -> Path | None:
        filename = SNAPSHOT_FILES.get(self.name)
        return SNAPSHOTS_DIR / filename if filename else None


# ── Fetchers (all return lowercased 0x addresses) ────────────────────────────

_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _lower(addrs) -> frozenset[str]:
    # Strict format check, deliberately: a prefix-only check is
    # what let domain strings from a phishing domain list pass as
    # addresses (a string like "0x-auth.com" starts with 0x). Only
    # well-formed addresses survive, here and at cache load.
    return frozenset(a.lower() for a in addrs
                     if isinstance(a, str) and _ADDR_RE.match(a))


def _fetch_scamsniffer_phish() -> frozenset[str]:
    """Phishing and scam addresses from ScamSniffer's database.

    Source: github.com/scamsniffer/scam-database,
    blacklist/address.json. Data is GPL-3.0; attribution and the
    license are recorded in addresses/README.md and the snapshot
    manifest. The publisher discloses that the open data lags its
    internal database by 7 days; that delay is accepted and
    recorded with the feed.

    Replaces the MetaMask eth-phishing-detect source, which no
    longer publishes addresses (its config is a domain detector).
    The old fetcher's fallback silently filtered that domain list
    by the "0x" prefix and shipped domain strings as addresses.

    Fails loudly: a non-list payload, zero well-formed addresses,
    or a mostly malformed payload raises, so the caller keeps the
    previous cache and the snapshot validator gates any commit.
    """
    import requests
    url = "https://raw.githubusercontent.com/scamsniffer/scam-database/main/blacklist/address.json"
    r = requests.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("scamsniffer payload is not a list")
    addrs = _lower(data)
    if not addrs:
        raise ValueError("scamsniffer payload yielded no valid addresses")
    malformed = len(data) - len(addrs)
    if malformed > 0.05 * len(data):
        raise ValueError(
            f"scamsniffer payload mostly malformed: {malformed}/{len(data)}")
    return addrs


def _fetch_ofac_sdn() -> frozenset[str]:
    """US Treasury OFAC SDN virtual currency addendum.

    The canonical feed is an XML delta; we mirror the flat
    community-maintained extract that most analytics vendors consume.
    """
    import requests
    url = "https://raw.githubusercontent.com/0xB10C/ofac-sanctioned-digital-currency-addresses/lists/sanctioned_addresses_ETH.json"
    r = requests.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return _lower(r.json())


def _fetch_chainalysis_oracle() -> frozenset[str]:
    """Chainalysis Sanctions Oracle is an on-chain contract with a
    per-address `isSanctioned(address)` view. We don't materialise the
    full set (that'd require an archive node); instead the engine calls
    `check_chainalysis(addr)` live for any address that passes the
    local OFAC check, as belt-and-braces. This fetcher returns empty
    so the cache exists but lookups go through the dedicated helper.
    """
    return frozenset()


FEEDS: dict[str, Feed] = {
    "ofac_sdn":       Feed("ofac_sdn",       BASELINE_OFAC,         _fetch_ofac_sdn),
    "phishing":       Feed("phishing",       frozenset(),           _fetch_scamsniffer_phish),
    "chainalysis":    Feed("chainalysis",    frozenset(),           _fetch_chainalysis_oracle),
    "mixers":         Feed("mixers",         BASELINE_MIXERS,       None),
    "bridges":        Feed("bridges",        BASELINE_BRIDGES,      None),
    "lrt_offramps":   Feed("lrt_offramps",   BASELINE_LRT_OFFRAMPS, None),
}


# ── Cache I/O ────────────────────────────────────────────────────────────────

def _load_cache(feed: Feed) -> tuple[frozenset[str], float]:
    if not feed.cache_path.exists():
        return feed.baseline, 0.0
    try:
        blob = json.loads(feed.cache_path.read_text())
        return _lower(blob.get("addresses", [])), float(blob.get("updated_at", 0.0))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        log.warning("feed %s cache unreadable: %s", feed.name, e)
        return feed.baseline, 0.0


def _load_snapshot(feed: Feed) -> frozenset[str] | None:
    """Addresses from the committed snapshot, or None if there is no
    usable snapshot for this feed (no snapshot file mapped, the file
    is absent, or it is unreadable). Entries pass through the same
    strict well-formedness filter as cache loads, so a malformed
    snapshot entry can never reach the engine."""
    path = feed.snapshot_path
    if path is None or not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
        raw = blob if isinstance(blob, list) else blob.get("addresses", [])
        return _lower(raw)
    except (json.JSONDecodeError, ValueError, OSError, AttributeError) as e:
        log.warning("feed %s snapshot unreadable: %s", feed.name, e)
        return None


def _save_cache(feed: Feed, addrs: frozenset[str]) -> None:
    blob = {
        "feed": feed.name,
        "updated_at": time.time(),
        "count": len(addrs),
        "addresses": sorted(addrs),
    }
    feed.cache_path.write_text(json.dumps(blob, indent=2))


# ── Public API ───────────────────────────────────────────────────────────────

_MEM_CACHE: dict[str, frozenset[str]] = {}


def get_feed(name: str) -> frozenset[str]:
    """Return the current set of addresses for a feed. Never raises
    for a known feed.

    Precedence (see module docstring): if a local cache file exists
    it wins and the snapshot is not consulted; with no cache, the
    committed snapshot is used; with neither, the hardcoded
    baseline. The baseline is unioned into the winning tier.
    """
    if name in _MEM_CACHE:
        return _MEM_CACHE[name]
    feed = FEEDS.get(name)
    if feed is None:
        raise KeyError(f"unknown feed: {name}")
    if feed.cache_path.exists():
        cached, _ = _load_cache(feed)
        combined = feed.baseline | cached
    else:
        snapshot = _load_snapshot(feed)
        combined = feed.baseline | snapshot if snapshot is not None else feed.baseline
    _MEM_CACHE[name] = combined
    return combined


def refresh(name: str | None = None) -> dict[str, int]:
    """Refresh one feed (or all if name is None). Returns {name: count}.

    Baseline-only feeds (no fetcher) are reported with their baseline size.
    """
    targets = [FEEDS[name]] if name else list(FEEDS.values())
    results: dict[str, int] = {}
    for feed in targets:
        if feed.fetch is None:
            # Baseline-only feeds still need a cache file so workflow step 5 / MANIFEST
            # picks them up. Write the baseline if no cache exists OR the cache is stale.
            if not feed.cache_path.exists():
                _save_cache(feed, feed.baseline)
            results[feed.name] = len(feed.baseline)
            continue
        if OFFLINE:
            log.info("FEEDS_OFFLINE=1, skipping %s", feed.name)
            results[feed.name] = len(get_feed(feed.name))
            continue
        try:
            fetched = feed.fetch()
            combined = feed.baseline | fetched
            _save_cache(feed, combined)
            _MEM_CACHE[feed.name] = combined
            results[feed.name] = len(combined)
            log.info("feed %s refreshed: %d addresses", feed.name, len(combined))
        except Exception as e:  # noqa: BLE001 — network fetch, keep old cache
            log.warning("feed %s refresh failed (%s), keeping cached copy", feed.name, e)
            results[feed.name] = len(get_feed(feed.name))
    return results


def feed_age_hours(name: str) -> float | None:
    """Hours since the feed was last refreshed. None if never."""
    feed = FEEDS.get(name)
    if feed is None or not feed.cache_path.exists():
        return None
    _, updated = _load_cache(feed)
    if updated == 0.0:
        return None
    return (time.time() - updated) / 3600.0


def is_stale(name: str) -> bool:
    age = feed_age_hours(name)
    return age is None or age > STALE_AFTER_HOURS


# ── Chainalysis Sanctions Oracle (on-chain belt-and-braces) ──────────────────
# Address of the sanctions oracle on Ethereum mainnet. Public, documented
# at https://go.chainalysis.com/chainalysis-oracle-docs.html
CHAINALYSIS_ORACLE = "0x40c57923924b5c5c5455c48d93317139addac8fb"


def check_chainalysis(address: str, rpc_url: str | None = None) -> bool | None:
    """Query the Chainalysis Sanctions Oracle for a single address.

    Returns True/False, or None if the RPC is unreachable. The oracle
    is read-only and free; the only cost is an eth_call to mainnet.
    """
    if OFFLINE or not rpc_url:
        return None
    import requests
    addr = address.lower().replace("0x", "").rjust(64, "0")
    # selector for isSanctioned(address) = 0xdf592f7d
    data = "0xdf592f7d" + addr
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": CHAINALYSIS_ORACLE, "data": data}, "latest"],
    }
    try:
        r = requests.post(rpc_url, json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        result = r.json().get("result", "0x0")
        return int(result, 16) == 1
    except Exception as e:  # noqa: BLE001
        log.debug("chainalysis oracle unreachable: %s", e)
        return None
