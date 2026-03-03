"""
NEXUS-RISK Agent v1 — The Roasting Investigator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your AML engine scores transactions. This agent INVESTIGATES them.

What it does:
  1. Takes a flagged transaction (or batch of alerts from the engine)
  2. Loops: examines the transaction, decides what tool to use next
  3. Builds an investigation report with evidence chain
  4. Roasts the transaction — sarcastic commentary that makes the pattern memorable
  5. Writes a draft SAR narrative (human-reviewable)

Architecture (the ReAct loop):
  GOAL → THINK → ACT → OBSERVE → THINK → ... → REPORT

This is your AML engine wrapped in a reasoning loop.
The engine is now a TOOL the agent calls, not the whole system.

v1 capabilities:
  - Tool 1: score_transaction — calls your AML engine rules
  - Tool 2: check_history — looks at sender/receiver past behavior
  - Tool 3: trace_flow — follows money through hops
  - Tool 4: roast — generates sarcastic analysis of the pattern
  - Tool 5: write_sar — drafts investigation narrative

Built by HASH (Bionic Banker) — learning agents by building them.
"""

import json, os, time, re
from datetime import datetime, timedelta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Load API keys
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
ANTHROPIC_KEY = None
GROQ_KEY = None
GEMINI_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith("ANTHROPIC_API_KEY="):
                ANTHROPIC_KEY = line_s.split("=", 1)[1].strip()
            if line_s.startswith("GROQ_API_KEY="):
                GROQ_KEY = line_s.split("=", 1)[1].strip()
            import re as _re
            # Groq key might be in format: groq"gsk_..."
            _gm = _re.search(r'groq["\s]*(gsk_[^"]+)"?', line_s, _re.IGNORECASE)
            if _gm and not GROQ_KEY:
                GROQ_KEY = _gm.group(1)
            _m = _re.search(r'"(AIza[^"]+)"', line_s)
            if _m:
                GEMINI_KEY = _m.group(1)

# Agent config
MAX_LOOPS = 8          # safety: agent stops after 8 reasoning steps
VERBOSE = True         # print agent's thinking in real-time

# Model priority: FREE first (Groq/Gemini), paid last (Anthropic)
# Groq = Llama 3.3 70B = FREE, no credit card, 0 cost
# Gemini = FREE tier (daily quota limits)
# Anthropic = PAID per token (avoid unless needed)
if GROQ_KEY:
    PROVIDER = "groq"
    MODEL = "llama-3.3-70b-versatile"
elif GEMINI_KEY:
    PROVIDER = "gemini"
    MODEL = "gemini-2.5-flash"
elif ANTHROPIC_KEY:
    PROVIDER = "anthropic"
    MODEL = "claude-haiku-4-5-20251001"
else:
    PROVIDER = "simulation"
    MODEL = "none"

# Override: force a specific provider via command line (--provider groq/gemini/anthropic)
# See argparse at bottom


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATED DATA (v1 — no live database yet)
# In v2 this connects to real transaction feeds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_TRANSACTIONS = [
    {
        "id": "TXN-2024-00147",
        "sender": "0x7a3B...f291",
        "receiver": "0x9cD4...8e33",
        "amount": 847000,
        "currency": "USDT",
        "timestamp": "2024-03-15T02:14:33Z",
        "chain": "Ethereum",
        "sender_profile": {
            "wallet_age_days": 3,
            "total_tx_count": 2,
            "known_labels": [],
            "country_exposure": ["KP"],  # North Korea
        },
        "receiver_profile": {
            "wallet_age_days": 891,
            "total_tx_count": 4200,
            "known_labels": ["exchange_deposit", "binance_hot"],
            "country_exposure": ["KY"],
        },
        "flags_from_engine": ["novel_wallet_dump", "high_risk_country", "exit_rush"],
        "risk_score": 185,
        "risk_level": "CRITICAL",
    },
    {
        "id": "TXN-2024-00392",
        "sender": "0x1eA2...c901",
        "receiver": "0x3fB7...d445",
        "amount": 9800,
        "currency": "ETH",
        "timestamp": "2024-03-15T14:22:01Z",
        "chain": "Ethereum",
        "sender_profile": {
            "wallet_age_days": 730,
            "total_tx_count": 12,
            "known_labels": ["tornado_user"],
            "country_exposure": [],
        },
        "receiver_profile": {
            "wallet_age_days": 2,
            "total_tx_count": 1,
            "known_labels": [],
            "country_exposure": [],
        },
        "flags_from_engine": ["mixer_touch", "peel_chain_linear", "exchange_avoidance"],
        "risk_score": 140,
        "risk_level": "CRITICAL",
    },
    {
        "id": "TXN-2024-00510",
        "sender": "0x4dC8...a223",
        "receiver": "0x8eF1...b667",
        "amount": 4950,
        "currency": "USDC",
        "timestamp": "2024-03-16T08:45:12Z",
        "chain": "Polygon",
        "sender_profile": {
            "wallet_age_days": 45,
            "total_tx_count": 87,
            "known_labels": [],
            "country_exposure": ["CA"],
        },
        "receiver_profile": {
            "wallet_age_days": 45,
            "total_tx_count": 87,
            "known_labels": [],
            "country_exposure": ["CA"],
        },
        "related_transactions": [
            {"amount": 4900, "time_gap_min": 3, "receiver": "0x8eF1...b667"},
            {"amount": 4850, "time_gap_min": 7, "receiver": "0x8eF1...b667"},
            {"amount": 4975, "time_gap_min": 2, "receiver": "0x8eF1...b667"},
            {"amount": 4800, "time_gap_min": 5, "receiver": "0x8eF1...b667"},
        ],
        "flags_from_engine": ["structuring", "smurfing", "velocity"],
        "risk_score": 115,
        "risk_level": "HIGH",
    },
    {
        "id": "TXN-2024-00788",
        "sender": "0xDeAd...0001",
        "receiver": "0xBeeF...0002",
        "amount": 2300000,
        "currency": "ETH",
        "timestamp": "2024-03-17T03:00:05Z",
        "chain": "Ethereum",
        "sender_profile": {
            "wallet_age_days": 2190,  # 6 years old
            "total_tx_count": 3,      # but barely used
            "last_active_days_ago": 1825,  # dormant 5 years
            "known_labels": ["bitfinex_hack_2016"],
            "country_exposure": [],
        },
        "receiver_profile": {
            "wallet_age_days": 1,
            "total_tx_count": 0,
            "known_labels": [],
            "country_exposure": [],
        },
        "flags_from_engine": ["dormant_activation", "novel_wallet_dump", "layering_deep", "ofac_hit"],
        "risk_score": 395,
        "risk_level": "CRITICAL",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOLS — what the agent can use
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tool_score_transaction(tx_id):
    """Look up a transaction's risk score and flags from the AML engine."""
    tx_id = tx_id.strip().strip('"').strip("'")  # clean quotes from LLM output
    for tx in SAMPLE_TRANSACTIONS:
        if tx["id"] == tx_id:
            return json.dumps({
                "id": tx["id"],
                "amount": tx["amount"],
                "currency": tx["currency"],
                "risk_score": tx["risk_score"],
                "risk_level": tx["risk_level"],
                "flags": tx["flags_from_engine"],
                "sender": tx["sender"],
                "receiver": tx["receiver"],
                "chain": tx["chain"],
            }, indent=2)
    return json.dumps({"error": f"Transaction {tx_id} not found"})


def tool_check_history(wallet_address):
    """Check a wallet's profile and history."""
    wallet_address = wallet_address.strip().strip('"').strip("'")
    for tx in SAMPLE_TRANSACTIONS:
        if wallet_address in tx["sender"]:
            profile = tx["sender_profile"]
            return json.dumps({
                "wallet": wallet_address,
                "role": "sender",
                "age_days": profile["wallet_age_days"],
                "total_transactions": profile["total_tx_count"],
                "labels": profile.get("known_labels", []),
                "country_exposure": profile.get("country_exposure", []),
                "last_active_days_ago": profile.get("last_active_days_ago", "recent"),
            }, indent=2)
        if wallet_address in tx["receiver"]:
            profile = tx["receiver_profile"]
            return json.dumps({
                "wallet": wallet_address,
                "role": "receiver",
                "age_days": profile["wallet_age_days"],
                "total_transactions": profile["total_tx_count"],
                "labels": profile.get("known_labels", []),
                "country_exposure": profile.get("country_exposure", []),
            }, indent=2)
    return json.dumps({"error": f"No history found for {wallet_address}"})


def tool_trace_flow(tx_id):
    """Trace the flow of funds — related transactions, hops, patterns."""
    tx_id = tx_id.strip().strip('"').strip("'")
    for tx in SAMPLE_TRANSACTIONS:
        if tx["id"] == tx_id:
            related = tx.get("related_transactions", [])
            return json.dumps({
                "origin_tx": tx_id,
                "sender": tx["sender"],
                "receiver": tx["receiver"],
                "amount": tx["amount"],
                "related_flows": related if related else "No related flows detected in current window.",
                "hop_count": len(related),
                "total_related_volume": sum(r.get("amount", 0) for r in related) if related else 0,
            }, indent=2)
    return json.dumps({"error": f"Transaction {tx_id} not found"})


def tool_check_sanctions(wallet_address):
    """Check if wallet appears on OFAC/sanctions lists."""
    wallet_address = wallet_address.strip().strip('"').strip("'")
    # Simulated — in production this hits real OFAC API
    sanctioned = {
        "0xDeAd...0001": {"list": "OFAC SDN", "reason": "BitFinex 2016 hack proceeds", "added": "2022-02-08"},
    }
    for addr, info in sanctioned.items():
        if addr in wallet_address or wallet_address in addr:
            return json.dumps({"match": True, "wallet": wallet_address, **info})
    return json.dumps({"match": False, "wallet": wallet_address})


# Tool registry — the agent sees this menu
TOOLS = {
    "score_transaction": {
        "fn": tool_score_transaction,
        "description": "Get risk score and flags for a transaction ID",
        "params": "tx_id (string)",
    },
    "check_history": {
        "fn": tool_check_history,
        "description": "Check a wallet's profile, age, labels, and transaction history",
        "params": "wallet_address (string — partial match ok, e.g. '0x7a3B')",
    },
    "trace_flow": {
        "fn": tool_trace_flow,
        "description": "Trace related fund flows and hop patterns for a transaction",
        "params": "tx_id (string)",
    },
    "check_sanctions": {
        "fn": tool_check_sanctions,
        "description": "Check if a wallet is on OFAC or other sanctions lists",
        "params": "wallet_address (string)",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM CALLER — supports Groq (FREE), Gemini (FREE), Anthropic (PAID)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def call_llm(system_prompt, messages, max_tokens=800):
    """Universal LLM caller. Uses free providers first."""
    for attempt in range(3):
        try:
            if PROVIDER == "groq":
                from openai import OpenAI
                client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
                groq_msgs = [{"role": "system", "content": system_prompt}] + messages
                r = client.chat.completions.create(
                    model=MODEL, max_tokens=max_tokens, messages=groq_msgs
                )
                return r.choices[0].message.content.strip()

            elif PROVIDER == "gemini":
                from google import genai
                client = genai.Client(api_key=GEMINI_KEY)
                # Flatten messages into a single prompt for Gemini
                convo = system_prompt + "\n\n"
                for m in messages:
                    role = "User" if m["role"] == "user" else "Assistant"
                    convo += f"{role}: {m['content']}\n\n"
                convo += "Assistant:"
                r = client.models.generate_content(model=MODEL, contents=convo)
                return r.text.strip()

            elif PROVIDER == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
                r = client.messages.create(
                    model=MODEL, max_tokens=max_tokens,
                    system=system_prompt, messages=messages,
                )
                return r.content[0].text.strip()

        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate" in err or "quota" in err:
                wait = (attempt + 1) * 10
                print(f"      [{PROVIDER} rate limited, waiting {wait}s...]")
                time.sleep(wait)
            else:
                return f"LLM_ERROR ({PROVIDER}): {e}"
    return f"LLM_ERROR: {PROVIDER} exhausted retries"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE AGENT LOOP (this is the core — everything above is setup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """You are NEXUS-AGENT, an autonomous AML investigation agent built by HASH at Bionic Banker.

Your job: investigate flagged transactions, build an evidence chain, and produce two outputs:
1. A ROAST — sarcastic, memorable commentary that makes the suspicious pattern obvious
2. A SAR NARRATIVE — professional investigation summary suitable for compliance review

You have access to these tools:
{tools}

HOW TO WORK:
1. Start by scoring the transaction to understand what the engine flagged
2. Check the sender and receiver wallet histories
3. Trace fund flows if layering/structuring is suspected
4. Check sanctions lists if high-risk indicators present
5. Once you have enough evidence, produce your report

RESPOND IN THIS FORMAT (every turn):
THINK: [your reasoning about what you know and what you need next]
ACTION: [tool_name] [parameter]
--- OR when you have enough evidence ---
THINK: [final reasoning]
REPORT:
🔥 ROAST: [sarcastic 2-3 sentence roast of the transaction pattern — make it memorable, funny, cutting]
📋 SAR NARRATIVE: [professional 3-5 sentence investigation summary with evidence chain]
⚡ RISK VERDICT: [CRITICAL/HIGH/MEDIUM/LOW] — [one-line justification]
🎯 RECOMMENDED ACTION: [what should happen next — freeze, escalate, monitor, etc.]

RULES:
- Use tools to gather evidence. Don't guess — investigate.
- Pay attention to WHICH wallet addresses appear in tool results. Use the EXACT addresses from the data, not from other transactions.
- The ROAST must be genuinely funny, cutting, and reference SPECIFIC numbers and patterns from the evidence. Not generic jokes — make fun of the actual absurdity. Examples of good roasts:
  * "This wallet sat dormant for 5 years and suddenly moved $2.3M. That's not waking up — that's a heist alarm going off in slow motion."
  * "Four transactions, all conveniently $4,950. Nothing says 'I'm not structuring' like a receipt book that looks like it was printed by a money laundering tutorial."
  * "A 3-day-old wallet with $847K and North Korean exposure. That's less of a red flag and more of a red billboard on fire."
- The SAR NARRATIVE must be professional, evidence-based, and reference specific wallet addresses, amounts, and flags.
- Always check sanctions before closing a CRITICAL case.
- Maximum {max_loops} tool calls per investigation (be efficient).
"""


def build_tool_descriptions():
    """Format tool descriptions for the system prompt."""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"  - {name}({info['params']}): {info['description']}")
    return "\n".join(lines)


def parse_agent_response(text):
    """Parse the agent's response to extract THINK, ACTION, or REPORT."""
    result = {"think": "", "action": None, "action_param": None, "report": None}

    # Extract THINK
    think_match = re.search(r"THINK:\s*(.+?)(?=ACTION:|REPORT:|$)", text, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    # Check for ACTION — handles both "tool_name param" and "tool_name(param)"
    action_match = re.search(r"ACTION:\s*(\w+)\s*\(?\s*([^)\n]+?)\s*\)?\s*$", text, re.MULTILINE)
    if action_match:
        result["action"] = action_match.group(1).strip()
        result["action_param"] = action_match.group(2).strip()

    # Check for REPORT (means agent is done)
    if "REPORT:" in text or "ROAST:" in text:
        report_match = re.search(r"(?:REPORT:|🔥\s*ROAST:)(.*)", text, re.DOTALL)
        if report_match:
            result["report"] = text[text.index("ROAST:"):] if "ROAST:" in text else report_match.group(1).strip()

    return result


def run_agent(tx_id, use_api=True):
    """
    Run the investigation agent on a transaction.

    If use_api=True, uses Claude API for reasoning.
    If use_api=False, runs in simulation mode (prints what would happen).
    """
    print(f"\n{'━'*65}")
    print(f"🕵️ NEXUS-AGENT v1 — Investigating {tx_id}")
    print(f"{'━'*65}")

    # Build system prompt
    system = SYSTEM_PROMPT.format(
        tools=build_tool_descriptions(),
        max_loops=MAX_LOOPS,
    )

    # Conversation history (the agent's memory within this investigation)
    messages = []

    # Initial user message: the investigation goal
    user_msg = f"Investigate transaction {tx_id}. Build the evidence chain, then deliver your ROAST and SAR NARRATIVE."
    messages.append({"role": "user", "content": user_msg})

    if VERBOSE:
        print(f"\n📌 Goal: Investigate {tx_id}")

    for loop in range(MAX_LOOPS):
        if VERBOSE:
            print(f"\n{'─'*40} Loop {loop+1}/{MAX_LOOPS} {'─'*10}")

        if use_api and PROVIDER != "simulation":
            agent_text = call_llm(system, messages)
        else:
            # Simulation mode — no API call
            agent_text = simulate_agent_step(tx_id, loop, messages)

        if VERBOSE:
            print(f"🤖 Agent:\n{agent_text}")

        # Parse response
        parsed = parse_agent_response(agent_text)

        if parsed["think"] and VERBOSE:
            print(f"\n💭 Thinking: {parsed['think'][:150]}...")

        # Check if agent produced final report
        if parsed["report"]:
            print(f"\n{'━'*65}")
            print(f"📊 INVESTIGATION COMPLETE — {tx_id}")
            print(f"{'━'*65}")
            print(agent_text[agent_text.index("ROAST:"):] if "ROAST:" in agent_text else parsed["report"])
            print(f"{'━'*65}")
            print(f"🔄 Loops used: {loop+1}/{MAX_LOOPS}")

            return {
                "tx_id": tx_id,
                "loops": loop + 1,
                "report": parsed["report"],
                "full_conversation": messages,
            }

        # Execute tool action
        if parsed["action"]:
            tool_name = parsed["action"]
            tool_param = parsed["action_param"]

            if tool_name in TOOLS:
                if VERBOSE:
                    print(f"\n🔧 Tool: {tool_name}({tool_param})")

                # Call the tool
                tool_result = TOOLS[tool_name]["fn"](tool_param)

                if VERBOSE:
                    print(f"📄 Result: {tool_result[:200]}...")

                # Add agent's response and tool result to conversation
                messages.append({"role": "assistant", "content": agent_text})
                messages.append({"role": "user", "content": f"TOOL RESULT ({tool_name}):\n{tool_result}"})
            else:
                messages.append({"role": "assistant", "content": agent_text})
                messages.append({"role": "user", "content": f"ERROR: Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"})
        else:
            # Agent didn't call a tool or produce a report — nudge it
            messages.append({"role": "assistant", "content": agent_text})
            messages.append({"role": "user", "content": "Continue investigating. Use a tool or produce your final REPORT with ROAST and SAR NARRATIVE."})

    # Hit max loops without report
    print(f"\n⚠️ Agent hit max loops ({MAX_LOOPS}) without producing a report.")
    return {"tx_id": tx_id, "loops": MAX_LOOPS, "report": None, "error": "max_loops_exceeded"}


def simulate_agent_step(tx_id, loop, messages):
    """Simulation mode — shows what the agent WOULD do without API calls."""
    # Simple state machine to demonstrate the loop
    steps = [
        f"THINK: Starting investigation of {tx_id}. First, I need the risk score and flags.\nACTION: score_transaction {tx_id}",
        f"THINK: Got the risk score. Now checking the sender's wallet history.\nACTION: check_history 0x7a3B",
        f"THINK: Sender looks suspicious. Let me check the receiver too.\nACTION: check_history 0x9cD4",
        f"THINK: Need to trace the fund flow to see the full pattern.\nACTION: trace_flow {tx_id}",
        f"THINK: High-risk country detected. Better check sanctions.\nACTION: check_sanctions 0x7a3B",
        (
            f"THINK: I have enough evidence. The pattern is clear — a brand new wallet with NK exposure "
            f"dumping funds to an exchange. Classic exploit-and-exit.\n"
            f"REPORT:\n"
            f"🔥 ROAST: This wallet is 3 days old, has exactly 2 transactions, and somehow has $847K in USDT "
            f"with North Korean exposure. That's not an investment portfolio — that's a getaway car with the engine "
            f"still running. The only thing faster than this exit was the wallet's entire lifespan.\n\n"
            f"📋 SAR NARRATIVE: Transaction {tx_id} involves a novel wallet (3 days old, 2 total transactions) "
            f"sending $847,000 USDT to a known Binance hot wallet. Sender wallet shows DPRK country exposure. "
            f"Engine flagged novel_wallet_dump, high_risk_country, and exit_rush. Pattern consistent with "
            f"exploit proceeds rapid liquidation via centralized exchange.\n\n"
            f"⚡ RISK VERDICT: CRITICAL — Novel wallet + DPRK exposure + immediate exchange exit = textbook exploit cashout\n\n"
            f"🎯 RECOMMENDED ACTION: Immediate freeze request to Binance. File SAR. Escalate to FINTRAC and law enforcement."
        ),
    ]
    return steps[min(loop, len(steps) - 1)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH MODE — investigate all flagged transactions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def investigate_all(use_api=True):
    """Run the agent on all sample transactions."""
    print(f"\n{'='*65}")
    print("NEXUS-AGENT v1 — Batch Investigation")
    print(f"{'='*65}")
    print(f"Transactions to investigate: {len(SAMPLE_TRANSACTIONS)}")
    print(f"Mode: {'API (Claude)' if use_api and ANTHROPIC_KEY else 'Simulation'}")
    print(f"Model: {MODEL}")
    print(f"Max loops per investigation: {MAX_LOOPS}")

    results = []
    for tx in SAMPLE_TRANSACTIONS:
        result = run_agent(tx["id"], use_api=use_api)
        results.append(result)
        time.sleep(1)  # breathing room between investigations

    # Summary
    print(f"\n{'='*65}")
    print("BATCH SUMMARY")
    print(f"{'='*65}")
    completed = sum(1 for r in results if r.get("report"))
    total_loops = sum(r.get("loops", 0) for r in results)
    print(f"  Investigated: {len(results)}")
    print(f"  Completed:    {completed}/{len(results)}")
    print(f"  Total loops:  {total_loops}")
    print(f"  Avg loops:    {total_loops/len(results):.1f}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_v1_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "results": [{
                "tx_id": r["tx_id"],
                "loops": r.get("loops"),
                "report": r.get("report"),
                "error": r.get("error"),
            } for r in results]
        }, f, indent=2)
    print(f"\n💾 Results saved: {out_path}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEXUS-AGENT v1 — AML Investigation Agent")
    parser.add_argument("--tx", type=str, help="Investigate a specific transaction ID")
    parser.add_argument("--all", action="store_true", help="Investigate all sample transactions")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode (no API calls)")
    parser.add_argument("--provider", type=str, choices=["groq", "gemini", "anthropic"],
                        help="Force a specific provider (default: cheapest available)")
    args = parser.parse_args()

    # Override provider if specified
    if args.provider:
        PROVIDER = args.provider
        if args.provider == "groq" and GROQ_KEY:
            MODEL = "llama-3.3-70b-versatile"
        elif args.provider == "gemini" and GEMINI_KEY:
            MODEL = "gemini-2.5-flash"
        elif args.provider == "anthropic" and ANTHROPIC_KEY:
            MODEL = "claude-haiku-4-5-20251001"
        else:
            print(f"⚠️  No API key found for {args.provider} — check .env")

    use_api = not args.sim

    if PROVIDER == "simulation" and use_api:
        print("⚠️  No API keys found in .env — running in simulation mode")
        print("   Add GROQ_API_KEY=gsk_... to .env (FREE, no credit card)")
        print("   Get it at: https://console.groq.com/keys")
        use_api = False

    # Show cost info
    cost_info = {
        "groq": "FREE (Llama 3.3 70B)",
        "gemini": "FREE tier (daily quota)",
        "anthropic": "⚠️ PAID ($0.01-0.05 per investigation)",
        "simulation": "FREE (no API)",
    }
    print(f"\n💰 Provider: {PROVIDER} — {cost_info.get(PROVIDER, '?')}")
    print(f"   Model: {MODEL}")

    if args.tx:
        run_agent(args.tx, use_api=use_api)
    elif args.all:
        investigate_all(use_api=use_api)
    else:
        # Default: investigate the most suspicious one
        print("No --tx or --all specified. Running on most suspicious transaction.")
        most_sus = max(SAMPLE_TRANSACTIONS, key=lambda t: t["risk_score"])
        run_agent(most_sus["id"], use_api=use_api)
