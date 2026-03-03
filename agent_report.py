"""
NEXUS-AGENT Report Generator — Visual Investigation Reports
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates colorful HTML reports from agent investigation results.

Styles:
  --style roast      🔥 Fun, sarcastic, emoji-heavy (HASH's default)
  --style pro        📋 Clean professional for compliance teams
  --style dark       🌙 Dark theme cyberpunk investigator
  --style minimal    ⚡ Stripped down, just the facts

Usage:
  python agent_report.py                          # uses latest results
  python agent_report.py --input results.json     # specific file
  python agent_report.py --style dark             # change style
"""

import json, os, sys, re
from datetime import datetime

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STYLE TEMPLATES — swap these to change the whole vibe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STYLES = {
    "roast": {
        "name": "🔥 Roast Mode",
        "bg": "#0a0a0f",
        "card_bg": "#14141f",
        "text": "#e0e0e0",
        "accent": "#ff6b35",
        "accent2": "#ffd166",
        "critical": "#ff2d55",
        "high": "#ff9500",
        "medium": "#ffcc00",
        "low": "#34c759",
        "header_gradient": "linear-gradient(135deg, #ff6b35 0%, #ff2d55 100%)",
        "font": "'Segoe UI', system-ui, -apple-system, sans-serif",
        "border_radius": "16px",
        "verdict_emoji": {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "👀", "LOW": "✅"},
        "header_emoji": "🔥",
        "report_title": "NEXUS-AGENT Investigation Report",
        "subtitle": "Roasting suspicious transactions since 2026",
    },
    "pro": {
        "name": "📋 Professional",
        "bg": "#ffffff",
        "card_bg": "#f8f9fa",
        "text": "#1a1a2e",
        "accent": "#2563eb",
        "accent2": "#7c3aed",
        "critical": "#dc2626",
        "high": "#ea580c",
        "medium": "#d97706",
        "low": "#16a34a",
        "header_gradient": "linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%)",
        "font": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "border_radius": "8px",
        "verdict_emoji": {"CRITICAL": "●", "HIGH": "●", "MEDIUM": "●", "LOW": "●"},
        "header_emoji": "📋",
        "report_title": "AML Investigation Summary",
        "subtitle": "NEXUS-RISK Automated Investigation Report",
    },
    "dark": {
        "name": "🌙 Cyber Dark",
        "bg": "#08080e",
        "card_bg": "#111122",
        "text": "#c8c8d0",
        "accent": "#5b73f8",
        "accent2": "#8b6cf7",
        "critical": "#ff3366",
        "high": "#ff8800",
        "medium": "#ffaa00",
        "low": "#00ff88",
        "header_gradient": "linear-gradient(135deg, #5b73f8 0%, #8b6cf7 100%)",
        "font": "'JetBrains Mono', 'Fira Code', monospace",
        "border_radius": "12px",
        "verdict_emoji": {"CRITICAL": "⛔", "HIGH": "🔶", "MEDIUM": "🔷", "LOW": "🟢"},
        "header_emoji": "🌙",
        "report_title": "NEXUS-AGENT // INVESTIGATION LOG",
        "subtitle": "[ CLASSIFIED — AUTOMATED THREAT ANALYSIS ]",
    },
    "minimal": {
        "name": "⚡ Minimal",
        "bg": "#fafafa",
        "card_bg": "#ffffff",
        "text": "#333333",
        "accent": "#000000",
        "accent2": "#666666",
        "critical": "#e53e3e",
        "high": "#dd6b20",
        "medium": "#d69e2e",
        "low": "#38a169",
        "header_gradient": "linear-gradient(135deg, #333 0%, #666 100%)",
        "font": "'Georgia', serif",
        "border_radius": "4px",
        "verdict_emoji": {"CRITICAL": "!", "HIGH": "!", "MEDIUM": "~", "LOW": "✓"},
        "header_emoji": "⚡",
        "report_title": "Investigation Report",
        "subtitle": "",
    },
}


def parse_report_sections(report_text):
    """Extract roast, SAR, verdict, and action from agent report text."""
    sections = {"roast": "", "sar": "", "verdict": "", "verdict_level": "UNKNOWN", "action": ""}

    if not report_text:
        return sections

    # Extract roast
    roast_match = re.search(r"ROAST:\s*[\"']?(.*?)(?=[\"']?\s*(?:📋|SAR NARRATIVE:|$))", report_text, re.DOTALL)
    if roast_match:
        sections["roast"] = roast_match.group(1).strip().strip('"').strip("'")

    # Extract SAR
    sar_match = re.search(r"SAR NARRATIVE:\s*(.*?)(?=⚡|RISK VERDICT:|$)", report_text, re.DOTALL)
    if sar_match:
        sections["sar"] = sar_match.group(1).strip()

    # Extract verdict
    verdict_match = re.search(r"RISK VERDICT:\s*(CRITICAL|HIGH|MEDIUM|LOW)\s*[—–-]\s*(.*?)(?=🎯|RECOMMENDED|$)", report_text, re.DOTALL)
    if verdict_match:
        sections["verdict_level"] = verdict_match.group(1).strip()
        sections["verdict"] = verdict_match.group(2).strip()

    # Extract action
    action_match = re.search(r"(?:RECOMMENDED ACTION|ACTION):\s*(.*?)$", report_text, re.DOTALL)
    if action_match:
        sections["action"] = action_match.group(1).strip()

    return sections


def generate_html(results_data, style_name="roast"):
    """Generate a full HTML report from investigation results."""
    s = STYLES.get(style_name, STYLES["roast"])
    results = results_data.get("results", [])
    timestamp = results_data.get("timestamp", datetime.now().isoformat())
    model = results_data.get("model", "unknown")

    # Build investigation cards
    cards_html = ""
    for i, r in enumerate(results):
        tx_id = r.get("tx_id", f"TXN-{i}")
        loops = r.get("loops", "?")
        report = r.get("report", "")
        error = r.get("error", "")

        sections = parse_report_sections(report)
        level = sections["verdict_level"]
        level_color = s.get(level.lower(), s["accent"])

        emoji = s["verdict_emoji"].get(level, "❓")

        if error:
            cards_html += f"""
            <div class="card error-card">
                <div class="card-header">
                    <span class="tx-id">{tx_id}</span>
                    <span class="badge" style="background: {s['critical']}">❌ ERROR</span>
                </div>
                <p class="error-msg">{error}</p>
            </div>"""
            continue

        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <div class="tx-info">
                    <span class="tx-id">{tx_id}</span>
                    <span class="loops">🔄 {loops} loops</span>
                </div>
                <span class="badge" style="background: {level_color}">{emoji} {level}</span>
            </div>

            {"<div class='roast-section'><div class='section-label'>🔥 THE ROAST</div><p class='roast-text'>" + sections['roast'] + "</p></div>" if sections['roast'] else ""}

            <div class="sar-section">
                <div class="section-label">📋 SAR NARRATIVE</div>
                <p class="sar-text">{sections['sar']}</p>
            </div>

            <div class="verdict-section" style="border-left: 4px solid {level_color}">
                <div class="section-label">⚡ VERDICT</div>
                <p class="verdict-text"><strong>{level}</strong> — {sections['verdict']}</p>
            </div>

            {"<div class='action-section'><div class='section-label'>🎯 RECOMMENDED ACTION</div><p class='action-text'>" + sections['action'] + "</p></div>" if sections['action'] else ""}
        </div>"""

    # Stats
    total = len(results)
    completed = sum(1 for r in results if r.get("report"))
    critical = sum(1 for r in results if "CRITICAL" in str(r.get("report", "")))
    avg_loops = sum(r.get("loops", 0) for r in results) / max(total, 1)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{s['report_title']}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Bricolage+Grotesque:wght@700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: {s['bg']};
    color: {s['text']};
    font-family: {s['font']};
    line-height: 1.6;
    padding: 2rem;
  }}

  .container {{ max-width: 900px; margin: 0 auto; }}

  .header {{
    background: {s['header_gradient']};
    color: white;
    padding: 2.5rem 2rem;
    border-radius: {s['border_radius']};
    margin-bottom: 2rem;
    text-align: center;
  }}

  .header h1 {{
    font-family: 'Bricolage Grotesque', {s['font']};
    font-size: 2rem;
    margin-bottom: 0.5rem;
  }}

  .header .subtitle {{
    opacity: 0.85;
    font-size: 0.95rem;
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }}

  .stat-card {{
    background: {s['card_bg']};
    border-radius: {s['border_radius']};
    padding: 1.2rem;
    text-align: center;
    border: 1px solid {'#222233' if s['bg'].startswith('#0') else '#e5e7eb'};
  }}

  .stat-number {{
    font-size: 2rem;
    font-weight: 700;
    color: {s['accent']};
  }}

  .stat-label {{
    font-size: 0.8rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .card {{
    background: {s['card_bg']};
    border-radius: {s['border_radius']};
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid {'#222233' if s['bg'].startswith('#0') else '#e5e7eb'};
    transition: transform 0.2s;
  }}

  .card:hover {{ transform: translateY(-2px); }}

  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid {'#222233' if s['bg'].startswith('#0') else '#e5e7eb'};
  }}

  .tx-id {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 1.1rem;
    color: {s['accent']};
  }}

  .loops {{
    font-size: 0.85rem;
    opacity: 0.6;
    margin-left: 1rem;
  }}

  .badge {{
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    color: white;
  }}

  .section-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.6;
    margin-bottom: 0.4rem;
    font-weight: 600;
  }}

  .roast-section {{
    background: {'#1a1020' if s['bg'].startswith('#0') else '#fff7ed'};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border-left: 4px solid {s['accent']};
  }}

  .roast-text {{
    font-size: 1.05rem;
    font-style: italic;
    color: {s['accent']};
    line-height: 1.5;
  }}

  .sar-section {{ margin-bottom: 1rem; }}
  .sar-text {{ font-size: 0.95rem; line-height: 1.6; }}

  .verdict-section {{
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    background: {'#0f0f1a' if s['bg'].startswith('#0') else '#f9fafb'};
    border-radius: 8px;
  }}

  .action-section {{
    background: {'#0a1520' if s['bg'].startswith('#0') else '#eff6ff'};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 4px solid {s['accent2']};
  }}

  .footer {{
    text-align: center;
    margin-top: 2rem;
    padding: 1.5rem;
    opacity: 0.5;
    font-size: 0.85rem;
  }}

  .footer a {{ color: {s['accent']}; text-decoration: none; }}

  .error-card {{ border-left: 4px solid {s['critical']}; opacity: 0.7; }}
  .error-msg {{ color: {s['critical']}; font-style: italic; }}

  .meta-bar {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    opacity: 0.5;
    margin-bottom: 2rem;
    padding: 0 0.5rem;
  }}

  @media (max-width: 600px) {{
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
    body {{ padding: 1rem; }}
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>{s['header_emoji']} {s['report_title']}</h1>
    <div class="subtitle">{s['subtitle']}</div>
  </div>

  <div class="meta-bar">
    <span>Generated: {timestamp[:19].replace('T', ' ')}</span>
    <span>Model: {model}</span>
    <span>Style: {s['name']}</span>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-number">{total}</div>
      <div class="stat-label">Investigated</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">{completed}</div>
      <div class="stat-label">Completed</div>
    </div>
    <div class="stat-card">
      <div class="stat-number" style="color: {s['critical']}">{critical}</div>
      <div class="stat-label">Critical</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">{avg_loops:.1f}</div>
      <div class="stat-label">Avg Loops</div>
    </div>
  </div>

  {cards_html}

  <div class="footer">
    Built by <a href="https://bionicbanker.tech">Bionic Banker</a> — NEXUS-AGENT v1<br>
    Powered by {model} | {style_name} theme | $0.00 API cost
  </div>

</div>
</body>
</html>"""

    return html


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NEXUS-AGENT Report Generator")
    parser.add_argument("--input", type=str, default=os.path.join(WORKSPACE, "agent_v1_results.json"),
                        help="Path to agent results JSON")
    parser.add_argument("--style", type=str, default="roast", choices=list(STYLES.keys()),
                        help="Report style: roast, pro, dark, minimal")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: agent_report_{style}.html)")
    parser.add_argument("--all-styles", action="store_true",
                        help="Generate reports in ALL styles")
    args = parser.parse_args()

    # Load results
    if not os.path.exists(args.input):
        print(f"❌ Results file not found: {args.input}")
        print("   Run agent_v1.py --all first to generate results.")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    print(f"📊 Loaded {len(data.get('results', []))} investigation results")

    if args.all_styles:
        # Generate all styles
        for style_name in STYLES:
            html = generate_html(data, style_name)
            out_path = args.output or os.path.join(WORKSPACE, f"agent_report_{style_name}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  {STYLES[style_name]['name']} → {out_path}")
    else:
        html = generate_html(data, args.style)
        out_path = args.output or os.path.join(WORKSPACE, f"agent_report_{args.style}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n✅ Report generated: {out_path}")
        print(f"   Style: {STYLES[args.style]['name']}")
        print(f"   Open in browser to view")


if __name__ == "__main__":
    main()
