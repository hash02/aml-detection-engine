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

import sys
import os
import io
import pandas as pd
import streamlit as st

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
    score_transactions,
    risk_level,
    risk_emoji,
    CONFIG,
)

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
    """Run the complete 22-rule AML detection pipeline on a DataFrame."""
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
    df = score_transactions(df, cfg)
    df["risk_level"] = df["risk_score"].apply(risk_level)
    df["risk_emoji"] = df["risk_score"].apply(risk_emoji)
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
    data_mode = st.radio(
        "Choose input",
        ["📋 Use Sample Data (Wormhole, Ronin, Lazarus)", "📤 Upload My CSV"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    if data_mode == "📤 Upload My CSV":
        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=["csv"],
            help="Required columns: sender_id, receiver_id, amount, country, timestamp, sender_profile, sender_tx_count, sender_avg_amount, sender_active_days",
        )
        st.caption("Required columns: sender_id, receiver_id, amount, country, timestamp, sender_profile, sender_tx_count, sender_avg_amount, sender_active_days")

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

if data_mode == "📋 Use Sample Data (Wormhole, Ronin, Lazarus)":
    try:
        raw_df = pd.read_csv(sample_data_path)
        data_loaded = True
        data_label = "sample dataset (Wormhole · Ronin · Lazarus Group)"
    except FileNotFoundError:
        st.error("Sample data file not found. Please upload your own CSV.")
        data_loaded = False
        raw_df = None
else:
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            data_loaded = True
            data_label = f"uploaded file: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            data_loaded = False
            raw_df = None
    else:
        data_loaded = False
        raw_df = None

# ── Show schema guide if no data ──────────────────────────────────────────────
if not data_loaded:
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

# ── Run engine ────────────────────────────────────────────────────────────────
cfg = {**CONFIG, "alert_threshold": alert_threshold}

with st.spinner("Running 22-rule detection pipeline..."):
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
        engine_ok = False

if not engine_ok:
    st.stop()

# ── Summary Metrics ───────────────────────────────────────────────────────────
flagged    = result_df[result_df["alert"] == True]
total      = len(result_df)
n_flagged  = len(flagged)
n_critical = len(flagged[flagged["risk_level"] == "CRITICAL"])
n_high     = len(flagged[flagged["risk_level"] == "HIGH"])
n_medium   = len(flagged[flagged["risk_level"] == "MEDIUM"])
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

        if i >= 19:  # Cap at 20 expanded rows for performance
            st.caption(f"... and {len(sorted_flagged) - 20} more flagged transactions.")
            break

else:
    st.success(f"✅ No transactions met the alert threshold (score ≥ {alert_threshold}). Try lowering the threshold in the sidebar.")

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

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05);
     text-align: center; font-size: 0.78rem; color: #505068;'>
  NEXUS-RISK AML Engine v11 · Built by
  <a href='https://bionicbanker.tech' style='color: #5b73f8;'>HASH @ Bionic Banker</a> ·
  <a href='https://github.com/hash02/aml-detection-engine' style='color: #5b73f8;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
