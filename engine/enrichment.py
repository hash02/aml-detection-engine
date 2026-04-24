"""
engine/enrichment.py — Subject attribution lookups
====================================================

When an analyst sees a flagged wallet, the first question is always
"what do we know about this address?" This module gives every alert
a stable, cached enrichment record:

  - label      — human-readable, e.g. "Binance hot wallet 3"
  - category   — "exchange" | "bridge" | "mixer" | "lrt_offramp" |
                 "sanctioned" | "phishing" | "unknown"
  - source     — where the label came from ("local" | "ofac" |
                 "provider:arkham" | ...)
  - confidence — 0..1

The lookup pipeline:
  1. Local `data/labels.json` (analyst-curated, committed to repo)
  2. Bundled feed sets from engine.feeds (OFAC, mixers, bridges, ...)
  3. Optional external provider via `EnrichmentProvider` protocol —
     callers inject one (e.g. Arkham, Etherscan labels) if they have
     API credentials. Zero providers = the first two layers are still
     useful.

Results are cached in-memory with a TTL so a batch of 500 alerts for
the same wallet doesn't blow up the provider's rate limit.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "labels.json"
DEFAULT_TTL_SECONDS = 3600


@dataclass(frozen=True)
class Enrichment:
    address: str
    label: str
    category: str
    source: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "address": self.address, "label": self.label,
            "category": self.category, "source": self.source,
            "confidence": self.confidence,
        }


UNKNOWN = Enrichment(address="", label="unknown", category="unknown",
                     source="none", confidence=0.0)


class EnrichmentProvider(Protocol):
    """Pluggable upstream — wire in Arkham, Etherscan labels, etc."""

    def lookup(self, address: str) -> Enrichment | None:  # pragma: no cover
        ...


def _load_local_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("enrichment: can't read %s: %s", path, e)
        return {}
    return {addr.lower(): entry for addr, entry in raw.items()}


class EnrichmentResolver:
    """Pipeline of resolution sources with TTL caching."""

    def __init__(
        self,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        provider: EnrichmentProvider | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self._labels = _load_local_labels(Path(labels_path))
        self._provider = provider
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[Enrichment, float]] = {}

    def lookup(self, address: str) -> Enrichment:
        """Return the best-effort enrichment for `address`. Never raises."""
        if not address:
            return UNKNOWN
        key = address.lower().strip()
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - hit[1] < self._ttl:
            return hit[0]

        resolved = self._resolve(key)
        self._cache[key] = (resolved, now)
        return resolved

    def lookup_batch(self, addresses) -> dict[str, Enrichment]:
        return {addr: self.lookup(addr) for addr in addresses}

    def _resolve(self, key: str) -> Enrichment:
        # 1. Local analyst-curated labels
        local = self._labels.get(key)
        if local:
            return Enrichment(
                address=key,
                label=str(local.get("label", "labeled")),
                category=str(local.get("category", "labeled")),
                source="local",
                confidence=float(local.get("confidence", 0.9)),
            )

        # 2. Feed-backed categories — OFAC > phish > mixer > bridge > off-ramp
        try:
            from engine.feeds import get_feed
        except ImportError:
            get_feed = None  # type: ignore[assignment]
        if get_feed is not None:
            if key in get_feed("ofac_sdn"):
                return Enrichment(key, "OFAC SDN listed", "sanctioned",
                                  "ofac", 1.0)
            if key in get_feed("metamask_phish"):
                return Enrichment(key, "Phishing / drainer", "phishing",
                                  "metamask", 0.85)
            if key in get_feed("mixers"):
                return Enrichment(key, "Privacy mixer contract", "mixer",
                                  "mixers_feed", 0.9)
            if key in get_feed("bridges"):
                return Enrichment(key, "Cross-chain bridge", "bridge",
                                  "bridges_feed", 0.85)
            if key in get_feed("lrt_offramps"):
                return Enrichment(key, "No-KYC off-ramp cluster", "lrt_offramp",
                                  "analyst_list", 0.8)

        # 3. External provider
        if self._provider is not None:
            try:
                out = self._provider.lookup(key)
                if out is not None:
                    return out
            except Exception as e:  # noqa: BLE001
                log.debug("enrichment provider failed: %s", e)

        return Enrichment(address=key, label="unknown", category="unknown",
                          source="none", confidence=0.0)

    # ── Admin operations ───────────────────────────────────────────
    def invalidate(self, address: str | None = None) -> None:
        """Drop cached entries. `None` clears the whole cache."""
        if address is None:
            self._cache.clear()
        else:
            self._cache.pop(address.lower(), None)

    def stats(self) -> dict:
        return {
            "local_labels":    len(self._labels),
            "cache_size":      len(self._cache),
            "has_provider":    self._provider is not None,
        }


# ── Module-level singleton for convenience ──────────────────────────
_default: EnrichmentResolver | None = None


def default_resolver() -> EnrichmentResolver:
    global _default
    if _default is None:
        _default = EnrichmentResolver()
    return _default
