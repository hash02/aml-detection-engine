"""
NEXUS-AGENT v2 — The Investigator With Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 → v2 upgrades:
  1. REAL DATA INPUT — reads CSV/JSON, not hardcoded samples
  2. MEMORY — remembers wallets across investigations (knowledge graph)
  3. PERSONALITY — configurable investigation voice (roast, detective, auditor)
  4. CHAIN TOOLS — new tools for deeper investigation

Architecture:
  DATA (csv/json) → AGENT LOOP → MEMORY UPDATE → REPORT → HTML

Built by HASH (Bionic Banker) — learning agents by building them.
"""

import json, os, time, re, csv
from datetime import datetime
from collections import defaultdict

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(WORKSPACE, ".env")

# Load API keys
GROQ_KEY = None
GEMINI_KEY = None
ANTHROPIC_KEY = None

if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith("ANTHROPIC_API_KEY="):
                ANTHROPIC_KEY = line_s.split("=", 1)[1].strip()
            if line_s.startswith("GROQ_API_KEY="):
                GROQ_KEY = line_s.split("=", 1)[1].strip()
            gm = re.search(r'groq["\s]*(gsk_[^"]+)"?', line_s, re.IGNORECASE)
            if gm and not GROQ_KEY:
                GROQ_KEY = gm.group(1)
            m = re.search(r'"(AIza[^"]+)"', line_s)
            if m:
                GEMINI_KEY = m.group(1)

# Provider priority: FREE first
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

MAX_LOOPS = 10
VERBOSE = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEMORY — the agent's brain across investigations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentMemory:
    """Persistent knowledge graph — wallets, patterns, connections."""

    def __init__(self, memory_path=None):
        self.path = memory_path or os.path.join(WORKSPACE, "agent_memory.json")
        self.wallets = {}          # wallet_addr → {profile, investigations, risk_history}
        self.patterns = []         # detected cross-case patterns
        self.investigations = []   # investigation summaries
        self.connections = defaultdict(set)  # wallet → set of connected wallets
        self.load()

    def load(self):
        """Load memory from disk."""
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.wallets = data.get("wallets", {})
            self.patterns = data.get("patterns", [])
            self.investigations = data.get("investigations", [])
            # Rebuild connections from sets
            for k, v in data.get("connections", {}).items():
                self.connections[k] = set(v)
            print(f"🧠 Memory loaded: {len(self.wallets)} wallets, {len(self.investigations)} investigations")
        else:
            print("🧠 Fresh memory — no prior investigations")

    def save(self):
        """Save memory to disk."""
        data = {
            "wallets": self.wallets,
            "patterns": self.patterns,
            "investigations": self.investigations,
            "connections": {k: list(v) for k, v in self.connections.items()},
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def remember_wallet(self, address, info):
        """Store or update wallet info."""
        addr = address.strip().strip('"').strip("'")
        if addr not in self.wallets:
            self.wallets[addr] = {
                "first_seen": datetime.now().isoformat(),
                "investigations": [],
                "risk_scores": [],
                "labels": [],
                "notes": [],
            }
        w = self.wallets[addr]
        if info.get("risk_score"):
            w["risk_scores"].append(info["risk_score"])
        if info.get("labels"):
            w["labels"] = list(set(w["labels"] + info["labels"]))
        if info.get("note"):
            w["notes"].append(info["note"])
        w["last_seen"] = datetime.now().isoformat()

    def remember_connection(self, addr1, addr2):
        """Record that two wallets are connected (appeared in same tx)."""
        a1 = addr1.strip().strip('"').strip("'")
        a2 = addr2.strip().strip('"').strip("'")
        self.connections[a1].add(a2)
        self.connections[a2].add(a1)

    def remember_investigation(self, tx_id, summary):
        """Store investigation summary."""
        self.investigations.append({
            "tx_id": tx_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        })

    def recall_wallet(self, address):
        """What do we know about this wallet from past investigations?"""
        addr = address.strip().strip('"').strip("'")
        # Check partial matches too
        for known_addr, data in self.wallets.items():
            if addr in known_addr or known_addr in addr:
                connections = list(self.connections.get(known_addr, set()))
                return {
                    "known": True,
                    "address": known_addr,
                    "times_seen": len(data.get("investigations", [])),
                    "risk_history": data.get("risk_scores", []),
                    "labels": data.get("labels", []),
                    "notes": data.get("notes", []),
                    "connections": connections[:10],
                    "first_seen": data.get("first_seen"),
                    "last_seen": data.get("last_seen"),
                }
        return {"known": False, "address": addr}

    def detect_patterns(self):
        """Scan memory for cross-investigation patterns."""
        patterns = []

        # Pattern: wallets appearing in multiple investigations
        for addr, data in self.wallets.items():
            if len(data.get("risk_scores", [])) >= 2:
                patterns.append({
                    "type": "repeat_offender",
                    "wallet": addr,
                    "appearances": len(data["risk_scores"]),
                    "avg_risk": sum(data["risk_scores"]) / len(data["risk_scores"]),
                })

        # Pattern: clusters of connected wallets
        for addr, conns in self.connections.items():
            if len(conns) >= 3:
                patterns.append({
                    "type": "hub_wallet",
                    "wallet": addr,
                    "connections": len(conns),
                })

        self.patterns = patterns
        return patterns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADER — reads real transaction data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_transactions(source):
    """Load transactions from CSV, JSON, or use built-in samples."""
    if source == "sample":
        return get_sample_transactions()

    if not os.path.exists(source):
        print(f"❌ File not found: {source}")
        return []

    ext = os.path.splitext(source)[1].lower()

    if ext == ".json":
        with open(source) as f:
            data = json.load(f)
        # Handle both list and dict-with-list formats
        if isinstance(data, list):
            return data
        if "transactions" in data:
            return data["transactions"]
        if "results" in data:
            return data["results"]
        return [data]

    elif ext == ".csv":
        txns = []
        with open(source, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx = {
                    "id": row.get("id", row.get("tx_id", row.get("hash", f"TXN-{len(txns)}"))),
                    "sender": row.get("sender", row.get("sender_id", row.get("from", "unknown"))),
                    "receiver": row.get("receiver", row.get("receiver_id", row.get("to", "unknown"))),
                    "amount": float(row.get("amount", 0)),
                    "currency": row.get("currency", row.get("token", "USD")),
                    "timestamp": row.get("timestamp", row.get("date", "")),
                    "chain": row.get("chain", row.get("network", "unknown")),
                    "risk_score": float(row.get("risk_score", row.get("score", 0))),
                    "risk_level": row.get("risk_level", row.get("level", "UNKNOWN")),
                    "flags": [],
                }
                # Parse flags from various possible column formats
                flags_raw = row.get("flags", row.get("rules_triggered", ""))
                if flags_raw:
                    tx["flags"] = [f.strip() for f in flags_raw.split(",") if f.strip()]
                txns.append(tx)
        print(f"📂 Loaded {len(txns)} transactions from CSV")
        return txns

    else:
        print(f"❌ Unsupported format: {ext}. Use .json or .csv")
        return []


def get_sample_transactions():
    """Built-in sample data for testing."""
    return [
        {
            "id": "TXN-2024-00147",
            "sender": "0x7a3B...f291",
            "receiver": "0x9cD4...8e33",
            "amount": 847000,
            "currency": "USDT",
            "timestamp": "2024-03-15T02:14:33Z",
            "chain": "Ethereum",
            "risk_score": 185,
            "risk_level": "CRITICAL",
            "flags": ["novel_wallet_dump", "high_risk_country", "exit_rush"],
            "sender_profile": {"wallet_age_days": 3, "total_tx_count": 2, "known_labels": [], "country_exposure": ["KP"]},
            "receiver_profile": {"wallet_age_days": 891, "total_tx_count": 4200, "known_labels": ["exchange_deposit", "binance_hot"], "country_exposure": ["KY"]},
        },
        {
            "id": "TXN-2024-00392",
            "sender": "0x1eA2...c901",
            "receiver": "0x3fB7...d445",
            "amount": 9800,
            "currency": "ETH",
            "timestamp": "2024-03-15T14:22:01Z",
            "chain": "Ethereum",
            "risk_score": 140,
            "risk_level": "CRITICAL",
            "flags": ["mixer_touch", "peel_chain_linear", "exchange_avoidance"],
            "sender_profile": {"wallet_age_days": 730, "total_tx_count": 12, "known_labels": ["tornado_user"], "country_exposure": []},
            "receiver_profile": {"wallet_age_days": 2, "total_tx_count": 1, "known_labels": [], "country_exposure": []},
        },
        {
            "id": "TXN-2024-00510",
            "sender": "0x4dC8...a223",
            "receiver": "0x8eF1...b667",
            "amount": 4950,
            "currency": "USDC",
            "timestamp": "2024-03-16T08:45:12Z",
            "chain": "Polygon",
            "risk_score": 115,
            "risk_level": "HIGH",
            "flags": ["structuring", "smurfing", "velocity"],
            "sender_profile": {"wallet_age_days": 45, "total_tx_count": 87, "known_labels": [], "country_exposure": ["CA"]},
            "receiver_profile": {"wallet_age_days": 45, "total_tx_count": 87, "known_labels": [], "country_exposure": ["CA"]},
            "related_transactions": [
                {"amount": 4900, "time_gap_min": 3, "receiver": "0x8eF1...b667"},
                {"amount": 4850, "time_gap_min": 7, "receiver": "0x8eF1...b667"},
                {"amount": 4975, "time_gap_min": 2, "receiver": "0x8eF1...b667"},
            ],
        },
        {
            "id": "TXN-2024-00788",
            "sender": "0xDeAd...0001",
            "receiver": "0xBeeF...0002",
            "amount": 2300000,
            "currency": "ETH",
            "timestamp": "2024-03-17T03:00:05Z",
            "chain": "Ethereum",
            "risk_score": 395,
            "risk_level": "CRITICAL",
            "flags": ["dormant_activation", "novel_wallet_dump", "layering_deep", "ofac_hit"],
            "sender_profile": {"wallet_age_days": 2190, "total_tx_count": 3, "last_active_days_ago": 1825, "known_labels": ["bitfinex_hack_2016"], "country_exposure": []},
            "receiver_profile": {"wallet_age_days": 1, "total_tx_count": 0, "known_labels": [], "country_exposure": []},
        },
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PERSONALITY SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERSONALITIES = {
    "roast": {
        "name": "🔥 Roast Mode",
        "instructions": """Your voice: sarcastic, cutting, funny. You're a comedian who happens to be an AML investigator.
Make fun of the transaction's absurdity. Reference specific numbers. The joke should be technically accurate.
Examples:
- "This wallet sat dormant for 5 years then moved $2.3M. That's not waking up — that's a heist alarm in slow motion."
- "Four transactions all at $4,950? Nothing says 'I'm not structuring' like a receipt book from a money laundering tutorial."
Your SAR narrative stays professional. The roast is separate — it's the hook that makes people actually READ the report.""",
    },
    "detective": {
        "name": "🕵️ Noir Detective",
        "instructions": """Your voice: hardboiled detective narrating a case. Think Raymond Chandler meets blockchain forensics.
Describe the investigation like you're writing a crime novel — "The wallet hadn't moved in five years. When it finally did, it wasn't subtle."
Be dramatic but accurate. Every detail you mention must come from the evidence.
Your SAR narrative stays professional. The detective narration is the color commentary.""",
    },
    "auditor": {
        "name": "📊 Strict Auditor",
        "instructions": """Your voice: precise, methodical, by-the-book. You cite regulation numbers. You don't joke.
Structure findings as: Finding → Evidence → Risk Assessment → Regulatory Reference.
Reference FINTRAC/PCMLTFA for Canadian context, FinCEN/BSA for US.
Everything professional. No roasts. No personality. Pure compliance.""",
    },
    "explain": {
        "name": "🎓 Plain English",
        "instructions": """Your voice: clear, simple, educational. Explain the suspicious pattern like you're teaching someone who's never seen a SAR before.
Use analogies. "Imagine someone opens a brand new bank account, deposits $847K, then immediately wires it overseas — in 3 days."
No jargon unless you define it first. Make the pattern obvious to a non-expert.
This mode is for training, demos, and non-technical stakeholders.""",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOLS (v2 — expanded with memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Transaction store — populated by load_transactions()
TX_STORE = {}
MEMORY = None  # initialized in main


def tool_score_transaction(tx_id):
    """Get risk score and flags for a transaction."""
    tx_id = tx_id.strip().strip('"').strip("'")
    tx = TX_STORE.get(tx_id)
    if not tx:
        return json.dumps({"error": f"Transaction {tx_id} not found"})
    return json.dumps({
        "id": tx["id"],
        "amount": tx.get("amount"),
        "currency": tx.get("currency"),
        "risk_score": tx.get("risk_score"),
        "risk_level": tx.get("risk_level"),
        "flags": tx.get("flags", []),
        "sender": tx.get("sender"),
        "receiver": tx.get("receiver"),
        "chain": tx.get("chain"),
        "timestamp": tx.get("timestamp"),
    }, indent=2)


def tool_check_history(wallet_address):
    """Check wallet profile from transaction data + memory."""
    wallet_address = wallet_address.strip().strip('"').strip("'")

    # Check transaction store first
    for tx in TX_STORE.values():
        if wallet_address in str(tx.get("sender", "")):
            profile = tx.get("sender_profile", {})
            result = {
                "wallet": wallet_address,
                "role": "sender",
                "age_days": profile.get("wallet_age_days", "unknown"),
                "total_transactions": profile.get("total_tx_count", "unknown"),
                "labels": profile.get("known_labels", []),
                "country_exposure": profile.get("country_exposure", []),
                "last_active_days_ago": profile.get("last_active_days_ago", "unknown"),
            }
            # Enrich with memory
            if MEMORY:
                mem = MEMORY.recall_wallet(wallet_address)
                if mem.get("known"):
                    result["memory"] = {
                        "times_investigated": mem["times_seen"],
                        "risk_history": mem["risk_history"],
                        "past_labels": mem["labels"],
                        "known_connections": mem["connections"][:5],
                    }
            return json.dumps(result, indent=2)

        if wallet_address in str(tx.get("receiver", "")):
            profile = tx.get("receiver_profile", {})
            result = {
                "wallet": wallet_address,
                "role": "receiver",
                "age_days": profile.get("wallet_age_days", "unknown"),
                "total_transactions": profile.get("total_tx_count", "unknown"),
                "labels": profile.get("known_labels", []),
                "country_exposure": profile.get("country_exposure", []),
            }
            if MEMORY:
                mem = MEMORY.recall_wallet(wallet_address)
                if mem.get("known"):
                    result["memory"] = {
                        "times_investigated": mem["times_seen"],
                        "risk_history": mem["risk_history"],
                        "past_labels": mem["labels"],
                        "known_connections": mem["connections"][:5],
                    }
            return json.dumps(result, indent=2)

    # Fall back to memory only
    if MEMORY:
        mem = MEMORY.recall_wallet(wallet_address)
        if mem.get("known"):
            return json.dumps({"wallet": wallet_address, "source": "memory_only", **mem}, indent=2)

    return json.dumps({"error": f"No history for {wallet_address}"})


def tool_trace_flow(tx_id):
    """Trace fund flows for a transaction."""
    tx_id = tx_id.strip().strip('"').strip("'")
    tx = TX_STORE.get(tx_id)
    if not tx:
        return json.dumps({"error": f"Transaction {tx_id} not found"})

    related = tx.get("related_transactions", [])
    return json.dumps({
        "origin_tx": tx_id,
        "sender": tx.get("sender"),
        "receiver": tx.get("receiver"),
        "amount": tx.get("amount"),
        "related_flows": related if related else "No related flows detected.",
        "hop_count": len(related),
        "total_related_volume": sum(r.get("amount", 0) for r in related) if related else 0,
    }, indent=2)


def tool_check_sanctions(wallet_address):
    """Check OFAC/sanctions lists."""
    wallet_address = wallet_address.strip().strip('"').strip("'")
    # Simulated sanctions list (v3 will hit real OFAC API)
    sanctioned = {
        "0xDeAd...0001": {"list": "OFAC SDN", "reason": "BitFinex 2016 hack proceeds", "added": "2022-02-08"},
    }
    for addr, info in sanctioned.items():
        if addr in wallet_address or wallet_address in addr:
            return json.dumps({"match": True, "wallet": wallet_address, **info})
    return json.dumps({"match": False, "wallet": wallet_address})


def tool_recall_memory(query):
    """Search agent's memory for past investigations, known wallets, patterns."""
    query = query.strip().strip('"').strip("'").lower()
    if not MEMORY:
        return json.dumps({"error": "No memory initialized"})

    results = {"query": query, "matches": []}

    # Search wallets
    for addr, data in MEMORY.wallets.items():
        if query in addr.lower() or any(query in l.lower() for l in data.get("labels", [])):
            results["matches"].append({"type": "wallet", "address": addr, "data": data})

    # Search investigations
    for inv in MEMORY.investigations:
        if query in inv.get("tx_id", "").lower() or query in inv.get("summary", "").lower():
            results["matches"].append({"type": "investigation", **inv})

    # Search patterns
    for pat in MEMORY.patterns:
        if query in str(pat).lower():
            results["matches"].append({"type": "pattern", **pat})

    results["total_matches"] = len(results["matches"])
    return json.dumps(results, indent=2, default=str)


TOOLS = {
    "score_transaction": {
        "fn": tool_score_transaction,
        "description": "Get risk score and flags for a transaction ID",
        "params": "tx_id (string)",
    },
    "check_history": {
        "fn": tool_check_history,
        "description": "Check a wallet's profile, age, labels, transaction history, and MEMORY of past investigations",
        "params": "wallet_address (string)",
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
    "recall_memory": {
        "fn": tool_recall_memory,
        "description": "Search your memory for past investigations, known wallets, or detected patterns. Use this to check if you've seen a wallet before.",
        "params": "query (string — wallet address, label, or keyword)",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM CALLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_provider(provider, model, system_prompt, messages, max_tokens):
    """Call a specific provider. Returns (response_text, None) or (None, error_string)."""
    if provider == "groq" and GROQ_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1", timeout=30.0)
        groq_msgs = [{"role": "system", "content": system_prompt}] + messages
        r = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=groq_msgs)
        return r.choices[0].message.content.strip(), None
    elif provider == "gemini" and GEMINI_KEY:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        convo = system_prompt + "\n\n"
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            convo += f"{role}: {m['content']}\n\n"
        convo += "Assistant:"
        r = client.models.generate_content(model=model, contents=convo)
        return r.text.strip(), None
    elif provider == "anthropic" and ANTHROPIC_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        r = client.messages.create(model=model, max_tokens=max_tokens, system=system_prompt, messages=messages)
        return r.content[0].text.strip(), None
    return None, f"No key for {provider}"


# Provider fallback chain: free first, paid last
PROVIDER_CHAIN = [
    ("groq", "llama-3.3-70b-versatile"),
    ("gemini", "gemini-2.5-flash"),
    ("anthropic", "claude-haiku-4-5-20251001"),
]


def call_llm(system_prompt, messages, max_tokens=800):
    """Universal LLM caller with automatic fallback chain. Free providers first."""
    # Try primary provider first with retries
    for attempt in range(2):
        try:
            text, err = _call_provider(PROVIDER, MODEL, system_prompt, messages, max_tokens)
            if text:
                return text
        except Exception as e:
            err_s = str(e).lower()
            if "429" in str(e) or "rate" in err_s or "quota" in err_s:
                if attempt == 0:
                    print(f"      [{PROVIDER} rate limited, waiting 10s...]")
                    time.sleep(10)
                    continue
                else:
                    break  # fall through to chain
            else:
                return f"LLM_ERROR ({PROVIDER}): {e}"

    # Primary exhausted — try fallback chain
    print(f"      ⚡ {PROVIDER} exhausted — trying fallback chain...")
    for fb_provider, fb_model in PROVIDER_CHAIN:
        if fb_provider == PROVIDER:
            continue  # skip the one that just failed
        try:
            text, err = _call_provider(fb_provider, fb_model, system_prompt, messages, max_tokens)
            if text:
                cost = {"groq": "FREE", "gemini": "FREE", "anthropic": "⚠️ PAID"}.get(fb_provider, "?")
                print(f"      ✅ Fallback to {fb_provider} ({cost})")
                return text
        except Exception as e:
            err_s = str(e).lower()
            if "429" in str(e) or "rate" in err_s or "quota" in err_s:
                print(f"      [{fb_provider} also rate limited, trying next...]")
                continue
            else:
                print(f"      [{fb_provider} error: {str(e)[:80]}]")
                continue

    return "LLM_ERROR: All providers exhausted"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPT BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_system_prompt(personality="roast"):
    """Build the full system prompt with personality and tools."""
    p = PERSONALITIES.get(personality, PERSONALITIES["roast"])

    tool_lines = []
    for name, info in TOOLS.items():
        tool_lines.append(f"  - {name}({info['params']}): {info['description']}")

    return f"""You are NEXUS-AGENT v2, an autonomous AML investigation agent built by HASH at Bionic Banker.

PERSONALITY: {p['name']}
{p['instructions']}

TOOLS:
{chr(10).join(tool_lines)}

HOW TO WORK:
1. Score the transaction to see what the engine flagged
2. Check sender and receiver wallet histories (includes MEMORY from past investigations)
3. Use recall_memory to check if you've seen these wallets before in past cases
4. Trace fund flows if layering/structuring suspected
5. Check sanctions for CRITICAL cases
6. Produce your final report

RESPOND FORMAT (every turn):
THINK: [your reasoning about what to do next]
ACTION: tool_name parameter

When you have enough evidence, produce your FINAL REPORT instead of another ACTION:
THINK: [your final reasoning tying all evidence together]
REPORT:
🔥 COMMENTARY: Write 2-3 sentences in your personality voice. Be specific — mention dollar amounts, wallet addresses, time gaps, flags. Do NOT leave placeholders.
📋 SAR NARRATIVE: Write a professional 3-5 sentence investigation summary with specific evidence from your tool results. This is a real SAR-style narrative.
⚡ RISK VERDICT: State CRITICAL, HIGH, MEDIUM, or LOW — then one sentence justifying with evidence.
🎯 RECOMMENDED ACTION: State one of: freeze, escalate, monitor, dismiss — then explain why.
🧠 MEMORY NOTE: One sentence about what to remember about these wallets for future investigations.

CRITICAL: Fill in EVERY section with your ACTUAL analysis. Never output template brackets like [your reasoning]. Write the real content.

RULES:
- Use EXACT wallet addresses from tool results
- The COMMENTARY must match your personality and reference SPECIFIC evidence
- SAR NARRATIVE is always professional regardless of personality
- Check sanctions before closing CRITICAL cases
- Use recall_memory if a wallet seems familiar
- Maximum {MAX_LOOPS} tool calls per investigation
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_agent_response(text):
    """Parse THINK, ACTION, REPORT from agent output."""
    result = {"think": "", "action": None, "action_param": None, "report": None, "memory_note": None}

    think_match = re.search(r"THINK:\s*(.+?)(?=ACTION:|REPORT:|$)", text, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    action_match = re.search(r"ACTION:\s*(\w+)\s*\(?\s*([^)\n]+?)\s*\)?\s*$", text, re.MULTILINE)
    if action_match:
        result["action"] = action_match.group(1).strip()
        result["action_param"] = action_match.group(2).strip()

    if "REPORT:" in text or "COMMENTARY:" in text:
        idx = text.find("COMMENTARY:") if "COMMENTARY:" in text else text.find("REPORT:")
        result["report"] = text[idx:]

    # Extract memory note
    mem_match = re.search(r"MEMORY NOTE:\s*(.+?)(?=\n|$)", text)
    if mem_match:
        result["memory_note"] = mem_match.group(1).strip()

    return result


def investigate(tx_id, personality="roast", use_api=True):
    """Run one investigation."""
    global MEMORY

    print(f"\n{'━'*65}")
    print(f"🕵️ NEXUS-AGENT v2 — Investigating {tx_id}")
    print(f"   Personality: {PERSONALITIES.get(personality, {}).get('name', personality)}")
    print(f"{'━'*65}")

    system = build_system_prompt(personality)
    messages = [{"role": "user", "content": f"Investigate transaction {tx_id}. Build the evidence chain and deliver your report."}]

    for loop in range(MAX_LOOPS):
        if VERBOSE:
            print(f"\n{'─'*40} Loop {loop+1}/{MAX_LOOPS} {'─'*10}")

        if use_api and PROVIDER != "simulation":
            agent_text = call_llm(system, messages)
        else:
            agent_text = f"THINK: Simulation mode.\nREPORT:\n🔥 COMMENTARY: [sim]\n📋 SAR NARRATIVE: [sim]\n⚡ RISK VERDICT: UNKNOWN\n🎯 RECOMMENDED ACTION: Run with API"

        if VERBOSE:
            print(f"🤖 {agent_text[:300]}{'...' if len(agent_text) > 300 else ''}")

        parsed = parse_agent_response(agent_text)

        # Check for final report
        if parsed["report"]:
            print(f"\n{'━'*65}")
            print(f"📊 INVESTIGATION COMPLETE — {tx_id}")
            print(f"{'━'*65}")
            print(parsed["report"])
            print(f"{'━'*65}")
            print(f"🔄 Loops: {loop+1}/{MAX_LOOPS}")

            # Update memory
            if MEMORY:
                tx = TX_STORE.get(tx_id, {})
                if tx.get("sender"):
                    MEMORY.remember_wallet(tx["sender"], {
                        "risk_score": tx.get("risk_score"),
                        "labels": tx.get("flags", []),
                        "note": f"Investigated in {tx_id}",
                    })
                if tx.get("receiver"):
                    MEMORY.remember_wallet(tx["receiver"], {
                        "note": f"Received funds in {tx_id}",
                    })
                if tx.get("sender") and tx.get("receiver"):
                    MEMORY.remember_connection(tx["sender"], tx["receiver"])
                MEMORY.remember_investigation(tx_id, parsed.get("memory_note", parsed["report"][:200]))
                MEMORY.save()
                print(f"🧠 Memory updated")

            return {
                "tx_id": tx_id,
                "loops": loop + 1,
                "personality": personality,
                "report": parsed["report"],
                "memory_note": parsed.get("memory_note"),
            }

        # Execute tool
        if parsed["action"] and parsed["action"] in TOOLS:
            tool_name = parsed["action"]
            tool_param = parsed["action_param"]
            if VERBOSE:
                print(f"🔧 {tool_name}({tool_param})")
            tool_result = TOOLS[tool_name]["fn"](tool_param)
            if VERBOSE:
                print(f"📄 {tool_result[:200]}...")
            messages.append({"role": "assistant", "content": agent_text})
            messages.append({"role": "user", "content": f"TOOL RESULT ({tool_name}):\n{tool_result}"})
        elif parsed["action"]:
            messages.append({"role": "assistant", "content": agent_text})
            messages.append({"role": "user", "content": f"ERROR: Unknown tool '{parsed['action']}'. Available: {list(TOOLS.keys())}"})
        else:
            messages.append({"role": "assistant", "content": agent_text})
            messages.append({"role": "user", "content": "Continue investigating. Use a tool or produce your final REPORT."})

    print(f"\n⚠️ Max loops ({MAX_LOOPS}) hit.")
    return {"tx_id": tx_id, "loops": MAX_LOOPS, "report": None, "error": "max_loops"}


def investigate_all(source="sample", personality="roast", use_api=True):
    """Batch investigate all transactions from a source."""
    txns = load_transactions(source)
    if not txns:
        print("No transactions to investigate.")
        return []

    # Populate TX_STORE
    for tx in txns:
        TX_STORE[tx["id"]] = tx

    print(f"\n{'='*65}")
    print(f"NEXUS-AGENT v2 — Batch Investigation")
    print(f"{'='*65}")
    print(f"  Source: {source}")
    print(f"  Transactions: {len(txns)}")
    print(f"  Personality: {PERSONALITIES.get(personality, {}).get('name', personality)}")
    print(f"  Provider: {PROVIDER} ({MODEL})")

    results = []
    for tx in txns:
        r = investigate(tx["id"], personality, use_api)
        results.append(r)
        time.sleep(1)

    # Save results
    out = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "provider": PROVIDER,
        "personality": personality,
        "results": results,
    }
    out_path = os.path.join(WORKSPACE, "agent_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n💾 Results: {out_path}")

    # Detect patterns across all investigations
    if MEMORY:
        patterns = MEMORY.detect_patterns()
        if patterns:
            print(f"\n🔍 Cross-case patterns detected: {len(patterns)}")
            for p in patterns:
                print(f"   {p['type']}: {p.get('wallet', 'N/A')} ({p.get('appearances', p.get('connections', '?'))})")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEXUS-AGENT v2 — Investigation Agent with Memory")
    parser.add_argument("--tx", type=str, help="Investigate a specific transaction ID")
    parser.add_argument("--all", action="store_true", help="Investigate all transactions")
    parser.add_argument("--input", type=str, default="sample", help="Data source: 'sample', path to .csv, or path to .json")
    parser.add_argument("--personality", type=str, default="roast", choices=list(PERSONALITIES.keys()),
                        help="Investigation personality: roast, detective, auditor, explain")
    parser.add_argument("--sim", action="store_true", help="Simulation mode (no API)")
    parser.add_argument("--provider", type=str, choices=["groq", "gemini", "anthropic"])
    parser.add_argument("--no-memory", action="store_true", help="Disable memory (fresh investigations)")
    args = parser.parse_args()

    if args.provider:
        PROVIDER = args.provider
        if args.provider == "groq": MODEL = "llama-3.3-70b-versatile"
        elif args.provider == "gemini": MODEL = "gemini-2.5-flash"
        elif args.provider == "anthropic": MODEL = "claude-haiku-4-5-20251001"

    use_api = not args.sim

    cost_info = {"groq": "FREE", "gemini": "FREE tier", "anthropic": "⚠️ PAID", "simulation": "FREE (no API)"}
    print(f"\n💰 Provider: {PROVIDER} — {cost_info.get(PROVIDER, '?')}")
    print(f"   Model: {MODEL}")

    # Initialize memory
    if not args.no_memory:
        MEMORY = AgentMemory()
    else:
        print("🧠 Memory disabled")

    # Load transactions
    txns = load_transactions(args.input)
    for tx in txns:
        TX_STORE[tx["id"]] = tx

    if args.tx:
        investigate(args.tx, args.personality, use_api)
    elif args.all:
        investigate_all(args.input, args.personality, use_api)
    else:
        print("No --tx or --all. Running most suspicious.")
        if txns:
            most_sus = max(txns, key=lambda t: t.get("risk_score", 0))
            investigate(most_sus["id"], args.personality, use_api)
