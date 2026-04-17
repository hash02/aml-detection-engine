"""Enrichment resolver tests — local → feed → provider pipeline + TTL."""

from __future__ import annotations

import json


def test_lookup_returns_unknown_for_missing_address(tmp_path):
    from engine.enrichment import EnrichmentResolver
    r = EnrichmentResolver(labels_path=tmp_path / "empty.json")
    out = r.lookup("0xdeadbeef")
    assert out.category == "unknown"
    assert out.source == "none"
    assert out.confidence == 0.0


def test_lookup_hits_local_labels(tmp_path):
    from engine.enrichment import EnrichmentResolver
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "0xaaa": {"label": "Test Exchange", "category": "exchange",
                   "confidence": 0.9}
    }))
    r = EnrichmentResolver(labels_path=labels)
    out = r.lookup("0xAAA")   # case-insensitive
    assert out.label == "Test Exchange"
    assert out.category == "exchange"
    assert out.source == "local"
    assert out.confidence == 0.9


def test_lookup_flags_ofac_via_feed(tmp_path):
    """Ronin exploiter is in the OFAC baseline — resolver should catch it."""
    from engine.enrichment import EnrichmentResolver
    r = EnrichmentResolver(labels_path=tmp_path / "empty.json")
    ronin = "0x098b716b8aaf21512996dc57eb0615e2383e2f96"
    out = r.lookup(ronin)
    assert out.category == "sanctioned"
    assert out.source == "ofac"
    assert out.confidence == 1.0


def test_lookup_flags_mixer_via_feed(tmp_path):
    from engine.enrichment import EnrichmentResolver
    r = EnrichmentResolver(labels_path=tmp_path / "empty.json")
    # Tornado Cash 1 ETH pool — in BASELINE_MIXERS
    mixer = "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936"
    out = r.lookup(mixer)
    assert out.category == "mixer"


def test_lookup_uses_provider_when_unknown(tmp_path):
    from engine.enrichment import Enrichment, EnrichmentResolver

    class _Prov:
        def lookup(self, addr):
            return Enrichment(address=addr, label="Arkham: relayer",
                              category="relay", source="provider:arkham",
                              confidence=0.7)

    r = EnrichmentResolver(labels_path=tmp_path / "empty.json", provider=_Prov())
    out = r.lookup("0xfeedbeef")
    assert out.source == "provider:arkham"
    assert out.category == "relay"


def test_lookup_caches_results(tmp_path):
    """Second lookup of the same address hits the cache, not the provider."""
    from engine.enrichment import Enrichment, EnrichmentResolver

    calls = {"n": 0}

    class _Counter:
        def lookup(self, addr):
            calls["n"] += 1
            return Enrichment(addr, "x", "x", "provider:x", 0.5)

    r = EnrichmentResolver(labels_path=tmp_path / "empty.json", provider=_Counter())
    r.lookup("0xaaa")
    r.lookup("0xaaa")
    r.lookup("0xaaa")
    assert calls["n"] == 1


def test_invalidate_clears_cache(tmp_path):
    from engine.enrichment import Enrichment, EnrichmentResolver

    calls = {"n": 0}

    class _C:
        def lookup(self, a):
            calls["n"] += 1
            return Enrichment(a, "x", "x", "provider:x", 0.5)

    r = EnrichmentResolver(labels_path=tmp_path / "empty.json", provider=_C())
    r.lookup("0xaaa")
    r.invalidate("0xaaa")
    r.lookup("0xaaa")
    assert calls["n"] == 2


def test_bundled_labels_file_is_parseable():
    """The committed data/labels.json must be valid JSON with reasonable keys."""
    from engine.enrichment import DEFAULT_LABELS_PATH
    raw = json.loads(DEFAULT_LABELS_PATH.read_text())
    assert isinstance(raw, dict)
    for addr, entry in raw.items():
        assert addr.startswith("0x")
        assert "label" in entry and "category" in entry
