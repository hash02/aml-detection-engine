"""
engine/observability.py — Sentry + Prometheus + structured logging
==================================================================

Everything in this module is env-gated. If the relevant library isn't
installed or the env var isn't set, the exported helpers are no-ops —
so the engine still runs in minimal environments (local dev, CI, Docker
without observability sidecars) without conditional imports everywhere.

Env vars:
  SENTRY_DSN              — if set + sentry_sdk installed, init Sentry
  PROMETHEUS_PORT         — if set + prometheus_client installed, start
                            an HTTP server on that port exposing metrics
  LOG_LEVEL               — one of DEBUG / INFO / WARNING / ERROR
                            (default INFO)

Public API:
  init_observability()                 — call once at process start
  log                                  — stdlib logger, preconfigured
  rules_fired_total(rule: str)         — increment a per-rule counter
  txs_processed_total.inc(n)           — increment tx-processed counter
  engine_latency_seconds.observe(s)    — record a pipeline latency sample
  capture_exception(exc)               — send to Sentry if wired
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

log = logging.getLogger("nexus")


# ── Sentry ──────────────────────────────────────────────────────────────────
_sentry_ready = False

def _init_sentry() -> None:
    global _sentry_ready
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk  # type: ignore[import-not-found]
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            environment=os.environ.get("APP_ENV", "dev"),
            release=os.environ.get("APP_RELEASE", "dev"),
        )
        _sentry_ready = True
        log.info("sentry initialised")
    except ImportError:
        log.debug("sentry_sdk not installed; skipping")


def capture_exception(exc: BaseException) -> None:
    if not _sentry_ready:
        return
    try:
        import sentry_sdk  # type: ignore[import-not-found]
        sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001
        pass


# ── Prometheus ──────────────────────────────────────────────────────────────
# Counters and histograms are singletons; if prometheus_client is missing
# we substitute a null object so .inc() / .observe() are still callable.
class _NullMetric:
    def inc(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def labels(self, *_, **__): return self
    def set(self, *_, **__): pass


rules_fired_total_metric = _NullMetric()
txs_processed_total      = _NullMetric()
engine_latency_seconds   = _NullMetric()
alerts_raised_total      = _NullMetric()


def _init_prometheus() -> None:
    global rules_fired_total_metric, txs_processed_total, engine_latency_seconds
    global alerts_raised_total
    port_str = os.environ.get("PROMETHEUS_PORT")
    if not port_str:
        return
    try:
        from prometheus_client import Counter, Histogram, start_http_server  # type: ignore[import-not-found]
    except ImportError:
        log.debug("prometheus_client not installed; skipping")
        return
    rules_fired_total_metric = Counter(
        "aml_rules_fired_total",
        "Total rule firings by rule name",
        labelnames=("rule",),
    )
    txs_processed_total = Counter(
        "aml_txs_processed_total",
        "Total transactions processed by the engine",
    )
    alerts_raised_total = Counter(
        "aml_alerts_raised_total",
        "Total alerts raised (risk_score >= threshold) by risk level",
        labelnames=("risk_level",),
    )
    engine_latency_seconds = Histogram(
        "aml_engine_latency_seconds",
        "End-to-end pipeline latency in seconds",
        buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60),
    )
    start_http_server(int(port_str))
    log.info("prometheus exporter on :%s", port_str)


def rules_fired_total(rule: str, n: int = 1) -> None:
    rules_fired_total_metric.labels(rule=rule).inc(n)


@contextmanager
def timed(metric_name: str = "pipeline"):
    """Context manager: times the block, observes to engine_latency_seconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        engine_latency_seconds.observe(time.perf_counter() - t0)


# ── Init hook ───────────────────────────────────────────────────────────────
_initialised = False

def init_observability() -> None:
    """Idempotent — safe to call from Streamlit AND from the engine CLI."""
    global _initialised
    if _initialised:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    _init_sentry()
    _init_prometheus()
    _initialised = True
