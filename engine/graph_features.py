"""
engine/graph_features.py — Network topology features
=====================================================

Computes per-sender graph metrics from the transaction window:
  - out_degree     (unique receivers the sender has sent to)
  - in_degree      (unique senders that sent to the address)
  - passthrough    (min(in_degree, out_degree) — hot relay indicator)
  - unique_ratio   (out_degree / total_outgoing; 1 = fan-out, ~0 = rebroadcast)
  - betweenness    (approximate; full nx betweenness is O(V*E) so we
                    use sampled betweenness when V > 500)

These features slot into the ML anomaly Layer 2 as extra columns when
available, strengthening the anomaly score for relay wallets and
structurally unusual addresses without changing any Layer-1 rule.

Dependency: `networkx`. Gracefully no-ops when networkx isn't installed.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

FEATURE_COLUMNS = (
    "graph_out_degree",
    "graph_in_degree",
    "graph_passthrough",
    "graph_unique_ratio",
    "graph_betweenness",
)


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Append graph-topology columns to df. Never raises; no-ops gracefully."""
    df = df.copy()
    if df.empty or "sender_id" not in df.columns or "receiver_id" not in df.columns:
        return _zero_fill(df)

    try:
        import networkx as nx
    except ImportError:
        log.info("networkx unavailable — graph features disabled")
        return _zero_fill(df)

    G: Any = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(str(row["sender_id"]), str(row["receiver_id"]))

    out_deg: dict[str, int] = dict(G.out_degree())
    in_deg: dict[str, int]  = dict(G.in_degree())

    # Betweenness — full-graph O(V·E); sample when the graph is large.
    k_samples: int | None = None
    if G.number_of_nodes() > 500:
        k_samples = min(200, G.number_of_nodes())
    try:
        between = nx.betweenness_centrality(G, k=k_samples, normalized=True)
    except Exception:  # noqa: BLE001
        between = {n: 0.0 for n in G.nodes()}

    # Per-row features keyed on sender_id
    def _unique_ratio(sender: str) -> float:
        sender_rows = df[df["sender_id"] == sender]
        if sender_rows.empty:
            return 0.0
        unique = sender_rows["receiver_id"].nunique()
        return unique / max(len(sender_rows), 1)

    df["graph_out_degree"]   = df["sender_id"].map(lambda s: float(out_deg.get(str(s), 0)))
    df["graph_in_degree"]    = df["sender_id"].map(lambda s: float(in_deg.get(str(s), 0)))
    df["graph_passthrough"]  = df.apply(
        lambda r: float(min(out_deg.get(str(r["sender_id"]), 0),
                             in_deg.get(str(r["sender_id"]), 0))),
        axis=1,
    )
    df["graph_unique_ratio"] = df["sender_id"].map(_unique_ratio).astype(float)
    df["graph_betweenness"]  = df["sender_id"].map(lambda s: float(between.get(str(s), 0.0)))

    return df


def metadata() -> dict[str, Any]:
    """Pinned metadata for audit / model-card consumption."""
    return {
        "features": list(FEATURE_COLUMNS),
        "betweenness_sampling_threshold": 500,
        "betweenness_sample_size": 200,
    }
