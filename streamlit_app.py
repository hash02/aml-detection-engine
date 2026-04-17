"""
NEXUS-RISK — Live AML Detection Demo
=====================================
Bionic Banker · bionicbanker.tech

A live, browser-based interface to the NEXUS-RISK AML engine v11.
Upload a transaction CSV or run on the built-in sample data to see
22 blockchain AML detection rules fire in real time.

Run locally:
    pip install streamlit pandas
    streamlit run streamlit_app.py
"""

import glob
import io
import json
import os
import sys
import time

import pandas as pd
import streamlit as st

# ── Import live fetcher ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    from etherscan_fetcher import DEFAULT_SEEDS, fetch_live_data
    LIVE_FEED_AVAILABLE = True
except ImportError:
    LIVE_FEED_AVAILABLE = False

# ── Observability + auth + audit (all env-gated / fail-open) ────────────────
try:
    from engine.observability import (
        alerts_raised_total,
        engine_latency_seconds,
        init_observability,
        rules_fired_total,
        txs_processed_total,
    )
    init_observability()
except Exception:  # noqa: BLE001 — observability must never crash the app
    def init_observability(): pass
    def rules_fired_total(_rule, _n=1): pass
    class _Null:
        def inc(self, *_, **__): pass
        def observe(self, *_, **__): pass
        def labels(self, *_, **__): return self
    alerts_raised_total    = _Null()
    engine_latency_seconds = _Null()
    txs_processed_total    = _Null()

try:
    from engine.audit import AuditLog
    from engine.auth import current_user, require_auth
    from engine.sar_sf import build_sar_sf_report
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# ── Import the engine (all pure functions, no file I/O needed) ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "engine"))

from engine_v11_blockchain import (
    compute_features,
    detect_layering,
    detect_mixer_touch,
    detect_bridge_hops,
    detect_peel_chain,
    detect_ofac_hit,
    detect_flash_loan_burst,
    detect_coordinated_burst,
    detect_novel_wallet_dump,
    detect_concentrated_inflow,
    detect_dormant_activation,
    detect_wash_cycle,
    detect_smurfing,
    detect_exit_rush,
    detect_rapid_succession,
    detect_high_risk_country,
    detect_layering_deep,
    detect_exchange_avoidance,
    detect_phish_hit,
    detect_sub_threshold_tranching,
    detect_machine_cadence,
    detect_sybil_fan_in,
    detect_drainer_signature,
    detect_address_poisoning,
    score_transactions,
    risk_level,
    risk_emoji,
    CONFIG,
)

try:
    from feeds import FEEDS, feed_age_hours
    FEED_STATUS_AVAILABLE = True
except ImportError:
    FEED_STATUS_AVAILABLE = False

# ── Auth gate + audit log (both fail-open to the public demo) ───────────────
if AUTH_AVAILABLE:
    USER = require_auth()
    AUDIT = AuditLog(os.environ.get("AML_AUDIT_DB", "data/audit.db"))
else:
    USER = None
    AUDIT = None

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS-RISK · AML Engine Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark theme matching bionicbanker.tech ───────────────────────
st.markdown("""
<style>
  /* Background */
  .stApp { background-color: #08080e; }
  .stApp > header { background-color: transparent !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0e0e1c;
    border-right: 1px solid rgba(91,115,248,0.14);
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background-color: #0e0e1c;
    border: 1px solid rgba(91,115,248,0.14);
    border-radius: 12px;
    padding: 1rem 1.25rem;
  }

  /* Metric label */
  div[data-testid="metric-container"] label {
    color: #8a8aa0 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }

  /* Metric value */
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ededf5 !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
  }

  /* Expander */
  div[data-testid="stExpander"] {
    background-color: #0e0e1c;
    border: 1px solid rgba(91,115,248,0.14);
    border-radius: 12px;
  }

  /* Dataframe */
  .stDataFrame { border-radius: 10px; }

  /* Headings */
  h1, h2, h3 { color: #ededf5 !important; }

  /* Info/warning/success boxes */
  div[data-testid="stAlert"] { border-radius: 10px; }

  /* Caption text */
  .stCaption { color: #505068 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #5b73f8, #8b6cf7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* File uploader */
  div[data-testid="stFileUploader"] {
    background: #0e0e1c;
    border: 1px dashed rgba(91,115,248,0.3);
    border-radius: 10px;
    padding: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Helper: run the full detection pipeline ──────────────────────────────────
def run_engine(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the complete 26-rule AML detection pipeline on a DataFrame."""
    df = compute_features(df, cfg)
    df, _ = detect_layering(df, cfg)
    df = detect_mixer_touch(df, cfg)
    df = detect_bridge_hops(df, cfg)
    df = detect_peel_chain(df, cfg)
    df = detect_novel_wallet_dump(df, cfg)
    df = detect_concentrated_inflow(df, cfg)
    df = detect_ofac_hit(df, cfg)
    df = detect_flash_loan_burst(df, cfg)
    df = detect_coordinated_burst(df, cfg)
    df = detect_dormant_activation(df, cfg)
    df = detect_wash_cycle(df, cfg)
    df = detect_smurfing(df, cfg)
    df = detect_exit_rush(df, cfg)
    df = detect_rapid_succession(df, cfg)
    df = detect_high_risk_country(df, cfg)
    df = detect_exchange_avoidance(df, cfg)
    df = detect_layering_deep(df, cfg)
    # v12 detectors
    df = detect_phish_hit(df, cfg)
    df = detect_sub_threshold_tranching(df, cfg)
    df = detect_machine_cadence(df, cfg)
    df = detect_sybil_fan_in(df, cfg)
    # v13 detectors
    df = detect_drainer_signature(df, cfg)
    df = detect_address_poisoning(df, cfg)
    df = score_transactions(df, cfg)
    df["risk_level"] = df["risk_score"].apply(risk_level)
    df["risk_emoji"] = df["risk_score"].apply(risk_emoji)

    # Layer 2 — unsupervised anomaly score (Isolation Forest).
    try:
        from engine.ml_anomaly import fit_predict as ml_fit
        df = ml_fit(df)
    except Exception:  # noqa: BLE001 — ML layer is best-effort
        df["ml_anomaly_score"] = 0.0
        df["ml_anomaly_flag"]  = False

    # Alert suppression — dedupe same-sender/same-rule bursts into one
    # representative alert per time bucket. Pass-through if disabled.
    try:
        from engine.suppression import apply_suppression
        df = apply_suppression(df, cfg)
    except Exception:  # noqa: BLE001
        df["suppressed"]   = False
        df["supp_repr_id"] = ""

    return df


def format_reasons(reasons_str: str) -> list[str]:
    """Parse the reasons string into clean signal names."""
    SIGNAL_LABELS = {
        "large_amount":                    "💰 Large Transaction",
        "velocity_many_tx":                "⚡ High Velocity",
        "structuring":                     "🔢 Structuring / Amount Splitting",
        "fan_in":                          "🕸️ Fan-In (Many Senders → 1)",
        "foreign_country":                 "🌍 Foreign Jurisdiction",
        "layering_cycle":                  "🔄 Layering Cycle Detected",
        "mixer_touch":                     "🌪️ Mixer Contact",
        "mixer_withdraw":                  "🚨 Mixer Withdrawal",
        "bridge_hop":                      "🌉 Multi-Bridge Hop",
        "peel_chain":                      "🍌 Peel Chain (Linear Layering)",
        "novel_wallet_dump":               "🆕 Novel Wallet Dump",
        "concentrated_inflow":             "📥 Concentrated Inflow",
        "OFAC_SDN_MATCH":                  "🚫 OFAC SDN Sanction Hit",
        "flash_loan_burst":                "⚡ Flash Loan Burst",
        "coordinated_burst":               "🤝 Coordinated Multi-Sender Burst",
        "dormant_activation":              "💤 Dormant Wallet Activated",
        "wash_cycle":                      "♻️ Wash Cycle (A→B→A)",
        "smurfing":                        "🔵 Smurfing (Threshold Avoidance)",
        "exit_rush":                       "🏃 Exit Rush (Novel Wallet → Bridge)",
        "rapid_succession":                "🔫 Rapid Fan-Out",
        "high_risk_jurisdiction":          "⚠️ High-Risk Jurisdiction (FATF)",
        "high_risk_jurisdiction_amplified":"🔴 High-Risk Jurisdiction (Amplified)",
        "exchange_avoidance":              "🚪 Exchange Avoidance Routing",
        "layering_deep":                   "🕳️ Deep Layering Chain (5+ hops)",
        "phishing_hit":                    "🎣 Phishing / Drainer Address",
        "sub_threshold_tranching":         "🪙 Sub-Threshold Tranching ($8k–$10k)",
        "machine_cadence":                 "🤖 Machine Cadence (bot timing)",
        "sybil_fan_in":                    "👥 Sybil Fan-In (coordinated senders)",
        "drainer_signature":               "🪓 Drainer Signature (multi-asset drain)",
        "address_poisoning":               "☠️ Address Poisoning (lookalike dust)",
    }
    signals = []
    for part in reasons_str.split(";"):
        part = part.strip()
        if not part or part.startswith("profile_") or part.startswith("foreign_context"):
            continue
        # Match by prefix (some signals have dynamic suffixes like bridge_hop_3)
        matched = False
        for key, label in SIGNAL_LABELS.items():
            if part.startswith(key):
                if label not in signals:
                    signals.append(label)
                matched = True
                break
        if not matched and part:
            signals.append(f"⚠️ {part}")
    return signals


def risk_color(level: str) -> str:
    return {
        "CRITICAL": "#ef4444",
        "HIGH":     "#f97316",
        "MEDIUM":   "#eab308",
        "LOW":      "#22c55e",
    }.get(level, "#8a8aa0")


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
      <span style='font-size: 1.1rem; font-weight: 700; color: #ededf5;'>
        NEXUS<span style='color: #5b73f8;'>-RISK</span>
      </span><br>
      <span style='font-size: 0.75rem; color: #505068; letter-spacing: 0.06em; text-transform: uppercase;'>
        AML Engine v11 · 22 Rules
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Data Source**")
    st.markdown("""
    <div style='font-size:0.75rem; color:#505068; line-height:1.55; margin-bottom:0.6rem;'>
      ⚠️ <strong style='color:#8a8aa0;'>Sample data uses known sanctioned wallets</strong>
      (Lazarus Group, Wormhole exploiter). Expect 90–100% CRITICAL — that's correct.
      Paste your own address in Live mode to see how the scoring varies.
    </div>
    """, unsafe_allow_html=True)
    data_mode = st.radio(
        "Choose input",
        [
            "📋 Use Sample Data (Wormhole, Ronin, Lazarus)",
            "🔗 Fetch Live Blockchain Data",
            "📤 Upload My CSV",
            "📡 Live Monitoring",
        ],
        label_visibility="collapsed",
    )

    uploaded_file = None
    live_api_key = ""
    live_address = ""

    if data_mode == "📤 Upload My CSV":
        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=["csv"],
            help="Required columns: sender_id, receiver_id, amount, country, timestamp, sender_profile, sender_tx_count, sender_avg_amount, sender_active_days",
        )
        st.caption("Required columns: sender_id, receiver_id, amount, country, timestamp, sender_profile, sender_tx_count, sender_avg_amount, sender_active_days")

    elif data_mode == "🔗 Fetch Live Blockchain Data":
        st.markdown("""
        <div style='font-size:0.78rem; color:#8a8aa0; margin-bottom:0.75rem; line-height:1.6;'>
          Pulls real Ethereum transactions and runs them through the engine live.
          Uses publicly documented high-risk wallet clusters by default.
        </div>
        """, unsafe_allow_html=True)
        live_address = st.text_input(
            "Ethereum address (optional)",
            placeholder="0x... or leave blank for default seeds",
            help="Leave blank to use default seed addresses (Ronin hacker, publicly documented exploit wallets)"
        )
        live_api_key = st.text_input(
            "Etherscan API key (optional but recommended)",
            type="password",
            placeholder="Get free key at etherscan.io/register",
            help="Free key = 5 req/sec. Without = 1 req/5sec (slower but works)"
        )
        st.caption("⚠️ Fetching live data may take 30–60 seconds depending on rate limits.")

    st.divider()

    st.markdown("**Engine Settings**")
    alert_threshold = st.slider(
        "Alert Threshold (score ≥)",
        min_value=20, max_value=100, value=40, step=5,
        help="Transactions scoring at or above this value trigger an alert. Default: 40"
    )

    st.divider()
    st.markdown("""
    <div style='font-size: 0.78rem; color: #505068; line-height: 1.6;'>
      Built by <strong style='color: #8a8aa0;'>HASH</strong><br>
      Computer Engineer · Financial Advisor<br>
      <a href='https://bionicbanker.tech' style='color: #5b73f8;'>bionicbanker.tech</a><br>
      <a href='https://github.com/hash02/aml-detection-engine' style='color: #5b73f8;'>GitHub →</a>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
  <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #5b73f8; margin-bottom: 0.5rem;'>
    NEXUS-RISK · Live Demo
  </div>
  <h1 style='font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; color: #ededf5;'>
    Blockchain AML Detection Engine
  </h1>
  <p style='color: #8a8aa0; margin-top: 0.5rem; font-size: 0.95rem; line-height: 1.7;'>
    22 detection rules. Real blockchain transaction patterns. The same engine that caught Tornado Cash,
    Ronin, Lazarus Group, Wormhole, and Nomad — running live in your browser.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
sample_data_path = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")
raw_df = None
data_loaded = False
data_label = ""

if data_mode == "📡 Live Monitoring":
    pass  # Handled in the monitoring section below

elif data_mode == "📋 Use Sample Data (Wormhole, Ronin, Lazarus)":
    try:
        raw_df = pd.read_csv(sample_data_path)
        data_loaded = True
        data_label = "sample dataset (Wormhole · Ronin · Lazarus Group)"
    except FileNotFoundError:
        st.error("Sample data file not found. Please upload your own CSV.")

elif data_mode == "🔗 Fetch Live Blockchain Data":
    if not LIVE_FEED_AVAILABLE:
        st.error("Live feed module not found. Make sure etherscan_fetcher.py is in the root folder.")
    else:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            fetch_clicked = st.button("🔗 Fetch Live Data Now", use_container_width=True)
        with col_info:
            st.caption("Fetches real Ethereum transactions from Etherscan and runs the engine on them.")

        if fetch_clicked:
            addresses = [live_address.strip()] if live_address.strip() else DEFAULT_SEEDS
            with st.spinner(f"Fetching {len(addresses)} address(es) from Etherscan... (30–60 sec)"):
                try:
                    live_df = fetch_live_data(
                        addresses=addresses,
                        api_key=live_api_key.strip(),
                        limit=300,
                        expand_hops=True,
                        output_path=os.path.join(os.path.dirname(__file__), "data", "live_transactions.csv")
                    )
                    if not live_df.empty:
                        raw_df = live_df
                        data_loaded = True
                        n_addr = len(addresses)
                        addr_preview = addresses[0][:10] + "..." if addresses else "default seeds"
                        data_label = f"live Ethereum data · {len(raw_df)} transactions · {n_addr} address(es)"
                        st.success(f"✅ Fetched {len(raw_df)} live transactions. Running engine...")
                    else:
                        st.error("No transactions returned. Check your address or try with an API key.")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
        else:
            st.info("👆 Click 'Fetch Live Data Now' to pull real Ethereum transactions.")

elif data_mode == "📤 Upload My CSV":
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            data_loaded = True
            data_label = f"uploaded file: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# ── Show schema guide if no data ──────────────────────────────────────────────
if data_mode == "📡 Live Monitoring":
    pass  # Skip — monitoring page handles its own rendering
elif not data_loaded:
    st.info("👈 Select a data source in the sidebar to run the engine.")

    st.markdown("### Expected CSV Schema")
    schema = pd.DataFrame({
        "Column": ["sender_id", "receiver_id", "amount", "country", "timestamp",
                   "sender_profile", "sender_tx_count", "sender_avg_amount", "sender_active_days"],
        "Type": ["string", "string", "float", "string (2-letter ISO)", "datetime string",
                 "NEW / PERSONAL_LIKE / BUSINESS_LIKE / CONTRACT", "int", "float", "float"],
        "Example": ["0x3ee18B2...", "0x629e7D...", "384000000.0", "US",
                    "2022-02-02 18:24:00", "CONTRACT", "1", "384000000.0", "0.0"],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.stop()

# ── Context banner for sample data ───────────────────────────────────────────
if data_mode == "📋 Use Sample Data (Wormhole, Ronin, Lazarus)":
    st.markdown("""
    <div style='background:#1a0f0f; border:1px solid rgba(239,68,68,0.25); border-radius:10px;
                padding:0.85rem 1.1rem; margin-bottom:1.25rem; font-size:0.85rem; line-height:1.6;'>
      <strong style='color:#ef4444;'>Why is everything CRITICAL?</strong>
      <span style='color:#a8a8b8;'> — This dataset contains real transactions from
      <strong style='color:#ededf5;'>OFAC-sanctioned wallets</strong>: the Ronin Bridge exploiter
      (Lazarus Group, North Korea) and the Wormhole attacker. Every transaction touching a sanctioned
      address scores CRITICAL by design. That's the engine working correctly, not a bug.
      To see varied scoring, switch to <strong style='color:#5b73f8;'>🔗 Fetch Live Blockchain Data</strong>
      and paste any Ethereum address.</span>
    </div>
    """, unsafe_allow_html=True)

# ── Run engine ────────────────────────────────────────────────────────────────
cfg = {**CONFIG, "alert_threshold": alert_threshold}

# Short-circuit on empty input — avoids a downstream KeyError on `risk_level`
# when the engine has nothing to score.
if raw_df is None or raw_df.empty:
    st.warning("No transactions to analyse. Upload a CSV or pick a different data source.")
    st.stop()

# Schema validation — fast-fail with human-readable messages before we
# commit the cost of running the full pipeline.
try:
    from engine.schema import normalise, validate_dataframe
    issues = validate_dataframe(raw_df)
    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    if errors:
        st.error("Input CSV failed schema validation:")
        for i in errors:
            st.code(str(i))
        st.stop()
    if warnings:
        with st.expander("⚠️ Schema validation — warnings", expanded=False):
            for i in warnings:
                st.caption(str(i))
    raw_df = normalise(raw_df)
except ImportError:
    pass

_pipeline_start = time.perf_counter()
with st.spinner("Running 28-rule detection pipeline..."):
    try:
        # Ensure timestamp is parsed
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

        # Normalise required columns that may be missing in user uploads
        for col in ["is_known_mixer", "is_bridge"]:
            if col not in raw_df.columns:
                raw_df[col] = False

        result_df = run_engine(raw_df.copy(), cfg)
        engine_ok = True
    except Exception as e:
        st.error(f"Engine error: {e}")
        import traceback
        st.code(traceback.format_exc())
        try:
            from engine.observability import capture_exception
            capture_exception(e)
        except Exception:  # noqa: BLE001
            pass
        engine_ok = False

engine_latency_seconds.observe(time.perf_counter() - _pipeline_start)

if not engine_ok:
    st.stop()

# Defensive: engine can return an empty frame if every row fails validation.
if result_df.empty:
    st.warning("Engine ran but produced no scored transactions.")
    st.stop()

# ── Summary Metrics ───────────────────────────────────────────────────────────
flagged    = result_df[result_df["alert"] == True]
total      = len(result_df)
n_flagged  = len(flagged)
n_critical = len(flagged[flagged["risk_level"] == "CRITICAL"])
n_high     = len(flagged[flagged["risk_level"] == "HIGH"])
n_medium   = len(flagged[flagged["risk_level"] == "MEDIUM"])

# Metrics + audit (both best-effort, never block the UI) ──────────────────
txs_processed_total.inc(total)
for lvl, n in (("CRITICAL", n_critical), ("HIGH", n_high), ("MEDIUM", n_medium)):
    if n:
        alerts_raised_total.labels(risk_level=lvl).inc(n)
if AUDIT is not None:
    try:
        AUDIT.record_batch(result_df)
    except Exception:  # noqa: BLE001
        pass
detect_rate = f"{(n_flagged / total * 100):.1f}%" if total > 0 else "—"

st.markdown(f"<p style='color: #505068; font-size: 0.82rem; margin-bottom: 1rem;'>Engine run complete · {data_label}</p>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Transactions", total)
with col2: st.metric("Flagged (Alerts)", n_flagged, delta=f"{detect_rate} of total", delta_color="off")
with col3: st.metric("🔴 Critical", n_critical)
with col4: st.metric("🟠 High", n_high)
with col5: st.metric("🟡 Medium", n_medium)

st.divider()

# ── Risk distribution chart ───────────────────────────────────────────────────
if n_flagged > 0:
    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        st.markdown("#### Risk Distribution")
        dist_data = flagged["risk_level"].value_counts().reindex(
            ["CRITICAL", "HIGH", "MEDIUM"], fill_value=0
        ).reset_index()
        dist_data.columns = ["Risk Level", "Count"]
        dist_data["Color"] = dist_data["Risk Level"].map({
            "CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308"
        })

        # Manual bar chart using st.markdown (avoids matplotlib dependency)
        max_count = dist_data["Count"].max() or 1
        for _, row in dist_data.iterrows():
            bar_pct = int((row["Count"] / max_count) * 100)
            st.markdown(f"""
            <div style='margin-bottom: 0.75rem;'>
              <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                <span style='font-size: 0.8rem; color: {row["Color"]}; font-weight: 600;'>{row["Risk Level"]}</span>
                <span style='font-size: 0.8rem; color: #8a8aa0;'>{row["Count"]}</span>
              </div>
              <div style='background: #141428; border-radius: 4px; height: 8px; overflow: hidden;'>
                <div style='background: {row["Color"]}; width: {bar_pct}%; height: 100%; border-radius: 4px; opacity: 0.85;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Top triggered rules
        st.markdown("#### Most Fired Rules")
        all_reasons = ";".join(flagged["reasons"].fillna("").tolist())
        rule_counts = {}
        for part in all_reasons.split(";"):
            part = part.strip()
            if not part or part.startswith("profile_") or part.startswith("foreign_context"):
                continue
            # Group by prefix
            prefix = part.split("_")[0] + "_" + part.split("_")[1] if "_" in part else part
            rule_counts[prefix] = rule_counts.get(prefix, 0) + 1

        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for rule, count in top_rules:
            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; font-size: 0.8rem; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);'>
              <span style='color: #8a8aa0;'>{rule}</span>
              <span style='color: #5b73f8; font-weight: 600;'>{count}×</span>
            </div>
            """, unsafe_allow_html=True)

    with col_table:
        st.markdown("#### Flagged Transactions")
        display_df = flagged[[
            "sender_id", "receiver_id", "amount", "country", "risk_score", "risk_level"
        ]].copy()
        display_df["sender_id"]   = display_df["sender_id"].str[:18] + "..."
        display_df["receiver_id"] = display_df["receiver_id"].str[:18] + "..."
        display_df["amount"]      = display_df["amount"].apply(lambda x: f"${x:,.0f}")
        display_df.columns = ["Sender", "Receiver", "Amount", "Country", "Score", "Risk"]
        display_df = display_df.sort_values("Score", ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Per-transaction drill-down ──────────────────────────────────────────
    st.markdown("#### Transaction Drill-Down")
    st.caption("Expand any flagged transaction to see which rules fired and why.")

    sorted_flagged = flagged.sort_values("risk_score", ascending=False)

    for i, (idx, row) in enumerate(sorted_flagged.iterrows()):
        level  = row["risk_level"]
        score  = int(row["risk_score"])
        emoji  = row["risk_emoji"]
        sender = str(row["sender_id"])
        amount = row["amount"]
        signals = format_reasons(str(row.get("reasons", "")))

        label = f"{emoji} **{level}** · Score: {score} · ${amount:,.0f} · {sender[:20]}..."

        with st.expander(label):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style='font-size: 0.82rem; line-height: 2; color: #8a8aa0;'>
                  <strong style='color: #ededf5;'>Sender</strong><br>
                  <code style='color: #5b73f8; font-size: 0.78rem;'>{sender}</code><br>
                  <strong style='color: #ededf5;'>Receiver</strong><br>
                  <code style='color: #5b73f8; font-size: 0.78rem;'>{row["receiver_id"]}</code><br>
                  <strong style='color: #ededf5;'>Amount</strong> &nbsp; ${amount:,.2f}<br>
                  <strong style='color: #ededf5;'>Country</strong> &nbsp; {row["country"]}<br>
                  <strong style='color: #ededf5;'>Timestamp</strong> &nbsp; {row["timestamp"]}
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div style='margin-bottom: 0.5rem;'>
                  <span style='font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #5b73f8;'>Risk Score</span><br>
                  <span style='font-size: 2.5rem; font-weight: 800; color: {risk_color(level)}; letter-spacing: -0.03em;'>{score}</span>
                  <span style='font-size: 1rem; color: #8a8aa0;'> / ∞</span><br>
                  <span style='font-size: 0.85rem; font-weight: 700; color: {risk_color(level)};'>{level}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Rules Fired:**")
                if signals:
                    for sig in signals:
                        st.markdown(f"<div style='font-size: 0.82rem; padding: 0.2rem 0; color: #ededf5;'>{sig}</div>", unsafe_allow_html=True)
                else:
                    st.caption("No specific signals decoded.")

            # Per-rule contribution breakdown — shows how each rule
            # contributed to the final risk score. Purely additive;
            # identical numbers to score_row(), just disaggregated.
            try:
                from engine.explain import score_breakdown
                breakdown = score_breakdown(row.to_dict(), cfg)
                if breakdown:
                    st.markdown("**Score breakdown:**")
                    bd_df = pd.DataFrame(breakdown)
                    st.dataframe(bd_df, use_container_width=True, hide_index=True)
            except Exception:  # noqa: BLE001
                pass

            # SAR-SF export button — gated by RBAC.can("download_sar").
            # Any authenticated analyst qualifies; anonymous users see a
            # disabled placeholder so the capability is still discoverable.
            if AUTH_AVAILABLE and USER is not None and USER.can("download_sar"):
                try:
                    sar_json = json.dumps(
                        build_sar_sf_report(row.to_dict(), {"summary": label}),
                        indent=2, default=str,
                    )
                    st.download_button(
                        "⬇️ Download SAR-SF JSON",
                        sar_json,
                        file_name=f"sar_sf_{row.get('id', 'alert')}.json",
                        mime="application/json",
                        key=f"sar_dl_{idx}",
                    )
                except Exception:  # noqa: BLE001
                    pass

            # Reviewer disposition — gated on can("file_disposition").
            # Analysts see a read-only note; reviewers + admins see the buttons.
            if AUTH_AVAILABLE and USER is not None and AUDIT is not None:
                if USER.can("file_disposition"):
                    st.markdown("**Disposition**")
                    d1, d2, d3, d4 = st.columns([1, 1, 1, 2])
                    alert_id_hint = int(idx)  # local dataframe index as a proxy
                    notes = d4.text_input("Notes", key=f"notes_{idx}",
                                          label_visibility="collapsed",
                                          placeholder="Optional reviewer notes")
                    def _dispo(action: str):
                        AUDIT.record_review(
                            alert_id=alert_id_hint,
                            reviewer=USER.username,
                            disposition=action,
                            notes=notes or "",
                        )
                        st.toast(f"Recorded: {action} by {USER.username}")
                    if d1.button("Escalate", key=f"esc_{idx}"):
                        _dispo("escalate")
                    if d2.button("Dismiss",  key=f"dis_{idx}"):
                        _dispo("dismiss")
                    if d3.button("SAR filed", key=f"sar_{idx}"):
                        _dispo("sar_filed")
                else:
                    st.caption("🔒 Disposition actions require reviewer role.")

        if i >= 19:  # Cap at 20 expanded rows for performance
            st.caption(f"... and {len(sorted_flagged) - 20} more flagged transactions.")
            break

else:
    st.success(f"✅ No transactions met the alert threshold (score ≥ {alert_threshold}). Try lowering the threshold in the sidebar.")

# ── Live-ops panel: feed freshness + audit tail ──────────────────────────────
with st.expander("🛰️ Live-Ops — Feed Status + Audit Tail"):
    lop_a, lop_b = st.columns(2)
    with lop_a:
        st.markdown("**Threat-intel feeds**")
        if FEED_STATUS_AVAILABLE:
            rows = []
            for name in FEEDS:
                age = feed_age_hours(name)
                status = "baseline only" if age is None else f"{age:.1f}h ago"
                rows.append({"feed": name, "last refresh": status})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Feeds module unavailable in this build.")
    with lop_b:
        st.markdown("**Recent audit events**")
        if AUDIT is not None:
            events = AUDIT.fetch(limit=10)
            if events:
                audit_df = pd.DataFrame(events)[["created_at", "risk_level", "tx_id", "amount"]]
                audit_df["created_at"] = pd.to_datetime(audit_df["created_at"], unit="s")
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No audit events yet — run the engine to populate.")
        else:
            st.caption("Audit log disabled (auth module unavailable).")

# ── Raw data toggle ────────────────────────────────────────────────────────────
with st.expander("🔬 View Raw Scored Data (all transactions)"):
    raw_display = result_df[["sender_id", "receiver_id", "amount", "country",
                              "risk_score", "risk_level", "alert", "reasons"]].copy()
    raw_display["sender_id"]   = raw_display["sender_id"].str[:20] + "..."
    raw_display["receiver_id"] = raw_display["receiver_id"].str[:20] + "..."
    raw_display["amount"]      = raw_display["amount"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(raw_display, use_container_width=True, hide_index=True)

    # Download button
    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Full Results CSV",
        data=csv_out,
        file_name="nexus_risk_results.csv",
        mime="text/csv",
    )

# ── MONITORING PAGE ─────────────────────────────────────────────────────────
if data_mode == "📡 Live Monitoring":
    st.markdown("### Live Monitoring Dashboard")
    st.markdown("<p style='color: #8a8aa0; font-size: 0.88rem;'>Track watched addresses, view recent alerts, and manage your watchlist.</p>", unsafe_allow_html=True)

    # Load watchlist
    watchlist_path = os.path.join(os.path.dirname(__file__), "config", "watchlist.json")
    watchlist = {}
    if os.path.exists(watchlist_path):
        with open(watchlist_path, "r", encoding="utf-8") as wf:
            watchlist = json.load(wf)

    # Watchlist status
    st.markdown("#### Watched Addresses")
    wl_addresses = watchlist.get("addresses", [])
    last_checked_wl = watchlist.get("last_checked", {})

    if wl_addresses:
        wl_data = []
        for entry in wl_addresses:
            wl_data.append({
                "Label": entry.get("label", "Unknown"),
                "Address": entry["address"][:20] + "...",
                "Chain": entry.get("chain", 1),
                "Source": entry.get("source", "etherscan"),
                "Last Checked": last_checked_wl.get(entry["address"], "Never"),
            })
        st.dataframe(pd.DataFrame(wl_data), use_container_width=True, hide_index=True)
    else:
        st.info("No addresses in watchlist. Add addresses to config/watchlist.json")

    st.divider()

    # Add address form
    st.markdown("#### Add Address to Watchlist")
    col_addr, col_label, col_chain = st.columns([3, 2, 1])
    with col_addr:
        new_addr = st.text_input("Address", placeholder="0x...", key="mon_addr")
    with col_label:
        new_label = st.text_input("Label", placeholder="e.g., Suspicious Wallet", key="mon_label")
    with col_chain:
        new_chain = st.selectbox("Chain", [1, 137, 10, 8453, 42161], key="mon_chain")

    if st.button("Add to Watchlist", key="mon_add"):
        if new_addr and new_addr.startswith("0x"):
            new_entry = {
                "address": new_addr.strip(),
                "label": new_label.strip() or new_addr[:10],
                "chain": new_chain,
                "source": "blockscout" if new_chain != 1 else "etherscan",
            }
            watchlist.setdefault("addresses", []).append(new_entry)
            with open(watchlist_path, "w", encoding="utf-8") as wf:
                json.dump(watchlist, wf, indent=2)
            st.success(f"Added {new_label or new_addr[:10]} to watchlist")
            st.rerun()
        else:
            st.warning("Enter a valid 0x address")

    st.divider()

    # Recent alerts
    st.markdown("#### Recent Alerts")
    monitoring_dir = os.path.join(os.path.dirname(__file__), "data", "monitoring")
    if os.path.exists(monitoring_dir):
        alert_files = sorted(glob.glob(os.path.join(monitoring_dir, "alert-*.md")), reverse=True)
        if alert_files:
            latest = alert_files[0]
            with open(latest, "r", encoding="utf-8") as af:
                alert_content = af.read()
            st.markdown(f"**Latest report:** `{os.path.basename(latest)}`")

            with st.expander("View Full Alert Report", expanded=True):
                st.markdown(alert_content)

            if len(alert_files) > 1:
                st.markdown("**Previous reports:**")
                for af_path in alert_files[1:5]:
                    with st.expander(os.path.basename(af_path)):
                        with open(af_path, "r", encoding="utf-8") as af:
                            st.markdown(af.read())
        else:
            st.info("No alert reports yet. Run `python scripts/monitor.py` to generate the first one.")
    else:
        st.info("Monitoring directory not found. Run the monitor script first.")

    # Transaction data viewer
    if os.path.exists(monitoring_dir):
        csv_files = sorted(glob.glob(os.path.join(monitoring_dir, "*.csv")), reverse=True)
        if csv_files:
            st.divider()
            st.markdown("#### Transaction Data")
            selected_csv = st.selectbox("Select dataset", [os.path.basename(f) for f in csv_files[:10]])
            if selected_csv:
                csv_path = os.path.join(monitoring_dir, selected_csv)
                try:
                    mon_df = pd.read_csv(csv_path)
                    display_cols = [c for c in ["sender_id", "receiver_id", "amount", "country", "timestamp"] if c in mon_df.columns]
                    st.dataframe(mon_df[display_cols].head(50), use_container_width=True, hide_index=True)
                    st.caption(f"{len(mon_df)} total transactions in this dataset")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05);
     text-align: center; font-size: 0.78rem; color: #505068;'>
  NEXUS-RISK AML Engine v11 · Built by
  <a href='https://bionicbanker.tech' style='color: #5b73f8;'>HASH @ Bionic Banker</a> ·
  <a href='https://github.com/hash02/aml-detection-engine' style='color: #5b73f8;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
