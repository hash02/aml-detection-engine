"""
NEXUS-AGENT v3 — Multi-Model Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2 → v3 upgrade: SPLIT-BRAIN ARCHITECTURE

v2 = one model does everything (investigate + write report)
v3 = two models, each doing what they're best at:

  REASONER  (Groq/Llama 70B)  → fast, structured, follows instructions
    → handles: THINK/ACTION loops, tool calls, evidence gathering
    → why: speed matters here, creativity doesn't

  SYNTHESIZER (Gemini)  → creative, narrative, good at writing
    → handles: final report, personality voice, SAR narrative
    → why: writing quality matters here, speed doesn't

Same investigation. Two brains. Each doing their strongest work.
Cost: $0 (both free tier).

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL ROUTING — the core v3 idea
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Each role gets its own model + fallback chain
ROLES = {
    "reasoner": {
        "purpose": "Tool selection, evidence gathering, structured reasoning",
        "chain": [
            ("groq", "llama-3.3-70b-versatile"),      # fast, free, follows instructions
            ("gemini", "gemini-2.5-flash"),             # fallback
            ("anthropic", "claude-haiku-4-5-20251001"), # paid last resort
        ],
    },
    "synthesizer": {
        "purpose": "Report writing, personality voice, narrative quality",
        "chain": [
            ("gemini", "gemini-2.5-flash"),             # creative, good at writing
            ("groq", "llama-3.3-70b-versatile"),        # fallback
            ("anthropic", "claude-haiku-4-5-20251001"), # paid last resort
        ],
    },
}

MAX_LOOPS = 8
VERBOSE = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEMORY (inherited from v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentMemory:
    """Persistent knowledge graph — wallets, patterns, connections."""

    def __init__(self, memory_path=None):
        self.path = memory_path or os.path.join(WORKSPACE, "agent_memory.json")
        self.wallets = {}
        self.patterns = []
        self.investigations = []
        self.connections = defaultdict(set)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.wallets = data.get("wallets", {})
            self.patterns = data.get("patterns", [])
            self.investigations = data.get("investigations", [])
            for k, v in data.get("connections", {}).items():
                self.connections[k] = set(v)
            print(f"🧠 Memory loaded: {len(self.wallets)} wallets, {len(self.investigations)} investigations")
        else:
            print("🧠 Fresh memory — no prior investigations")

    def save(self):
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
        addr = address.strip().strip('"').strip("'")
        if addr not in self.wallets:
            self.wallets[addr] = {
                "first_seen": datetime.now().isoformat(),
                "investigations": [], "risk_scores": [],
                "labels": [], "notes": [],
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
        a1 = addr1.strip().strip('"').strip("'")
        a2 = addr2.strip().strip('"').strip("'")
        self.connections[a1].add(a2)
        self.connections[a2].add(a1)

    def remember_investigation(self, tx_id, summary):
        self.investigations.append({
            "tx_id": tx_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        })

    def recall_wallet(self, address):
        addr = address.strip().strip('"').strip("'")
        for known_addr, data in self.wallets.items():
            if addr in known_addr or known_addr in addr:
                connections = list(self.connections.get(known_addr, set()))
                return {
                    "known": True, "address": known_addr,
                    "times_seen": len(data.get("investigations", [])),
                    "risk_history": data.get("risk_scores", []),
                    "labels": data.get("labels", []),
                    "notes": data.get("notes", []),
                    "connections": connections[:10],
                }
        return {"known": False, "address": addr}

    def detect_patterns(self):
        patterns = []
        for addr, data in self.wallets.items():
            if len(data.get("risk_scores", [])) >= 2:
                patterns.append({
                    "type": "repeat_offender", "wallet": addr,
                    "appearances": len(data["risk_scores"]),
                    "avg_risk": sum(data["risk_scores"]) / len(data["risk_scores"]),
                })
        for addr, conns in self.connections.items():
            if len(conns) >= 3:
                patterns.append({"type": "hub_wallet", "wallet": addr, "connections": len(conns)})
        self.patterns = patterns
        return patterns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_transactions(source):
    if source == "sample":
        return get_sample_transactions()
    if not os.path.exists(source):
        print(f"❌ File not found: {source}")
        return []
    ext = os.path.splitext(source)[1].lower()
    if ext == ".json":
        with open(source) as f:
            data = json.load(f)
        if isinstance(data, list): return data
        if "transactions" in data: return data["transactions"]
        if "results" in data: return data["results"]
        return [data]
    elif ext == ".csv":
        txns = []
        with open(source, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx = {
                    "id": row.get("id", row.get("tx_id", row.get("hash", f"TXN-{len(txns)}"))),
                    "sender": row.get("sender", row.get("from", "unknown")),
                    "receiver": row.get("receiver", row.get("to", "unknown")),
                    "amount": float(row.get("amount", 0)),
                    "currency": row.get("currency", "USD"),
                    "timestamp": row.get("timestamp", ""),
                    "chain": row.get("chain", "unknown"),
                    "risk_score": float(row.get("risk_score", 0)),
                    "risk_level": row.get("risk_level", "UNKNOWN"),
                    "flags": [f.strip() for f in row.get("flags", "").split(",") if f.strip()],
                }
                txns.append(tx)
        print(f"📂 Loaded {len(txns)} transactions from CSV")
        return txns
    print(f"❌ Unsupported: {ext}")
    return []


def get_sample_transactions():
    return [
        {
            "id": "TXN-2024-00147",
            "sender": "0x7a3B...f291", "receiver": "0x9cD4...8e33",
            "amount": 847000, "currency": "USDT",
            "timestamp": "2024-03-15T02:14:33Z", "chain": "Ethereum",
            "risk_score": 185, "risk_level": "CRITICAL",
            "flags": ["novel_wallet_dump", "high_risk_country", "exit_rush"],
            "sender_profile": {"wallet_age_days": 3, "total_tx_count": 2, "known_labels": [], "country_exposure": ["KP"]},
            "receiver_profile": {"wallet_age_days": 891, "total_tx_count": 4200, "known_labels": ["exchange_deposit", "binance_hot"], "country_exposure": ["KY"]},
        },
        {
            "id": "TXN-2024-00392",
            "sender": "0x1eA2...c901", "receiver": "0x3fB7...d445",
            "amount": 9800, "currency": "ETH",
            "timestamp": "2024-03-15T14:22:01Z", "chain": "Ethereum",
            "risk_score": 140, "risk_level": "CRITICAL",
            "flags": ["mixer_touch", "peel_chain_linear", "exchange_avoidance"],
            "sender_profile": {"wallet_age_days": 730, "total_tx_count": 12, "known_labels": ["tornado_user"], "country_exposure": []},
            "receiver_profile": {"wallet_age_days": 2, "total_tx_count": 1, "known_labels": [], "country_exposure": []},
        },
        {
            "id": "TXN-2024-00510",
            "sender": "0x4dC8...a223", "receiver": "0x8eF1...b667",
            "amount": 4950, "currency": "USDC",
            "timestamp": "2024-03-16T08:45:12Z", "chain": "Polygon",
            "risk_score": 115, "risk_level": "HIGH",
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
            "sender": "0xDeAd...0001", "receiver": "0xBeeF...0002",
            "amount": 2300000, "currency": "ETH",
            "timestamp": "2024-03-17T03:00:05Z", "chain": "Ethereum",
            "risk_score": 395, "risk_level": "CRITICAL",
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
Be dramatic but accurate. Every detail you mention must come from the evidence.""",
    },
    "auditor": {
        "name": "📊 Strict Auditor",
        "instructions": """Your voice: precise, methodical, by-the-book. You cite regulation numbers.
Structure findings as: Finding → Evidence → Risk Assessment → Regulatory Reference.
Reference FINTRAC/PCMLTFA for Canadian context, FinCEN/BSA for US.
Everything professional. No roasts. No personality. Pure compliance.""",
    },
    "explain": {
        "name": "🎓 Plain English",
        "instructions": """Your voice: clear, simple, educational. Explain like you're teaching someone who's never seen a SAR before.
Use analogies. "Imagine someone opens a brand new bank account, deposits $847K, then immediately wires it overseas — in 3 days."
No jargon unless you define it first. For training, demos, and non-technical stakeholders.""",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TX_STORE = {}
MEMORY = None


def tool_score_transaction(tx_id):
    tx_id = tx_id.strip().strip('"').strip("'")
    tx = TX_STORE.get(tx_id)
    if not tx:
        return json.dumps({"error": f"Transaction {tx_id} not found"})
    return json.dumps({
        "id": tx["id"], "amount": tx.get("amount"), "currency": tx.get("currency"),
        "risk_score": tx.get("risk_score"), "risk_level": tx.get("risk_level"),
        "flags": tx.get("flags", []), "sender": tx.get("sender"),
        "receiver": tx.get("receiver"), "chain": tx.get("chain"),
        "timestamp": tx.get("timestamp"),
    }, indent=2)


def tool_check_history(wallet_address):
    wallet_address = wallet_address.strip().strip('"').strip("'")
    for tx in TX_STORE.values():
        if wallet_address in str(tx.get("sender", "")):
            profile = tx.get("sender_profile", {})
            result = {
                "wallet": wallet_address, "role": "sender",
                "age_days": profile.get("wallet_age_days", "unknown"),
                "total_transactions": profile.get("total_tx_count", "unknown"),
                "labels": profile.get("known_labels", []),
                "country_exposure": profile.get("country_exposure", []),
                "last_active_days_ago": profile.get("last_active_days_ago", "unknown"),
            }
            if MEMORY:
                mem = MEMORY.recall_wallet(wallet_address)
                if mem.get("known"):
                    result["memory"] = {"times_investigated": mem["times_seen"], "risk_history": mem["risk_history"], "known_connections": mem["connections"][:5]}
            return json.dumps(result, indent=2)
        if wallet_address in str(tx.get("receiver", "")):
            profile = tx.get("receiver_profile", {})
            result = {
                "wallet": wallet_address, "role": "receiver",
                "age_days": profile.get("wallet_age_days", "unknown"),
                "total_transactions": profile.get("total_tx_count", "unknown"),
                "labels": profile.get("known_labels", []),
                "country_exposure": profile.get("country_exposure", []),
            }
            if MEMORY:
                mem = MEMORY.recall_wallet(wallet_address)
                if mem.get("known"):
                    result["memory"] = {"times_investigated": mem["times_seen"], "risk_history": mem["risk_history"], "known_connections": mem["connections"][:5]}
            return json.dumps(result, indent=2)
    if MEMORY:
        mem = MEMORY.recall_wallet(wallet_address)
        if mem.get("known"):
            return json.dumps({"wallet": wallet_address, "source": "memory_only", **mem}, indent=2, default=str)
    return json.dumps({"error": f"No history for {wallet_address}"})


def tool_trace_flow(tx_id):
    tx_id = tx_id.strip().strip('"').strip("'")
    tx = TX_STORE.get(tx_id)
    if not tx:
        return json.dumps({"error": f"Transaction {tx_id} not found"})
    related = tx.get("related_transactions", [])
    return json.dumps({
        "origin_tx": tx_id, "sender": tx.get("sender"), "receiver": tx.get("receiver"),
        "amount": tx.get("amount"), "related_flows": related if related else "No related flows detected.",
        "hop_count": len(related),
        "total_related_volume": sum(r.get("amount", 0) for r in related) if related else 0,
    }, indent=2)


def tool_check_sanctions(wallet_address):
    wallet_address = wallet_address.strip().strip('"').strip("'")
    sanctioned = {
        "0xDeAd...0001": {"list": "OFAC SDN", "reason": "BitFinex 2016 hack proceeds", "added": "2022-02-08"},
    }
    for addr, info in sanctioned.items():
        if addr in wallet_address or wallet_address in addr:
            return json.dumps({"match": True, "wallet": wallet_address, **info})
    return json.dumps({"match": False, "wallet": wallet_address})


def tool_recall_memory(query):
    query = query.strip().strip('"').strip("'").lower()
    if not MEMORY:
        return json.dumps({"error": "No memory initialized"})
    results = {"query": query, "matches": []}
    for addr, data in MEMORY.wallets.items():
        if query in addr.lower() or any(query in l.lower() for l in data.get("labels", [])):
            results["matches"].append({"type": "wallet", "address": addr, "data": data})
    for inv in MEMORY.investigations:
        if query in inv.get("tx_id", "").lower() or query in inv.get("summary", "").lower():
            results["matches"].append({"type": "investigation", **inv})
    results["total_matches"] = len(results["matches"])
    return json.dumps(results, indent=2, default=str)


TOOLS = {
    "score_transaction": {"fn": tool_score_transaction, "description": "Get risk score and flags for a transaction ID", "params": "tx_id"},
    "check_history": {"fn": tool_check_history, "description": "Check wallet profile, age, labels, and MEMORY of past investigations", "params": "wallet_address"},
    "trace_flow": {"fn": tool_trace_flow, "description": "Trace related fund flows and hop patterns", "params": "tx_id"},
    "check_sanctions": {"fn": tool_check_sanctions, "description": "Check OFAC/sanctions lists", "params": "wallet_address"},
    "recall_memory": {"fn": tool_recall_memory, "description": "Search memory for past investigations, known wallets, patterns", "params": "query"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM CALLER — role-aware routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _raw_call(provider, model, system_prompt, messages, max_tokens):
    """Low-level provider call. Returns text or raises."""
    if provider == "groq" and GROQ_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1", timeout=30.0)
        msgs = [{"role": "system", "content": system_prompt}] + messages
        r = client.chat.completions.create(model=model, max_tokens=max_tokens, messages=msgs)
        return r.choices[0].message.content.strip()
    elif provider == "gemini" and GEMINI_KEY:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        convo = system_prompt + "\n\n"
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            convo += f"{role}: {m['content']}\n\n"
        convo += "Assistant:"
        r = client.models.generate_content(model=model, contents=convo)
        return r.text.strip()
    elif provider == "anthropic" and ANTHROPIC_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        r = client.messages.create(model=model, max_tokens=max_tokens, system=system_prompt, messages=messages)
        return r.content[0].text.strip()
    raise ValueError(f"No key for {provider}")


def call_role(role, system_prompt, messages, max_tokens=800):
    """Route a call based on role (reasoner or synthesizer).
    Each role has its own preferred model + fallback chain."""
    chain = ROLES[role]["chain"]

    for i, (provider, model) in enumerate(chain):
        for attempt in range(2):
            try:
                text = _raw_call(provider, model, system_prompt, messages, max_tokens)
                if i > 0:
                    cost = {"groq": "FREE", "gemini": "FREE", "anthropic": "⚠️ PAID"}.get(provider, "?")
                    print(f"      ✅ {role} fallback → {provider} ({cost})")
                return text, provider
            except Exception as e:
                err_s = str(e).lower()
                if "429" in str(e) or "rate" in err_s or "quota" in err_s:
                    if attempt == 0:
                        print(f"      [{provider} rate limited for {role}, retry...]")
                        time.sleep(8)
                    else:
                        print(f"      [{provider} exhausted for {role}, next provider...]")
                        break
                else:
                    print(f"      [{provider} error: {str(e)[:60]}]")
                    break

    return "LLM_ERROR: All providers exhausted for " + role, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPTS — separate for each brain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_reasoner_prompt():
    """Prompt for the REASONER — focused on tool use and evidence gathering."""
    tool_lines = [f"  - {n}({t['params']}): {t['description']}" for n, t in TOOLS.items()]
    return f"""You are the REASONER module of NEXUS-AGENT v3, an AML investigation agent.

YOUR JOB: Gather evidence using tools. Be systematic. Be thorough.
You do NOT write the final report — a separate module handles that.
Your job is to collect ALL the evidence needed for a complete investigation.

TOOLS:
{chr(10).join(tool_lines)}

INVESTIGATION PROTOCOL:
1. Score the transaction (risk level, flags)
2. Check sender wallet history
3. Check receiver wallet history
4. Check memory for any prior encounters with these wallets
5. Trace fund flows if layering/structuring suspected
6. Check sanctions for CRITICAL cases

RESPOND FORMAT:
THINK: [your reasoning about what evidence you still need]
ACTION: tool_name parameter

When you have gathered ALL necessary evidence, output:
THINK: [summary of all evidence gathered]
EVIDENCE_COMPLETE:
- Transaction: [id, amount, currency, risk level, flags]
- Sender: [address, age, labels, country exposure, memory hits]
- Receiver: [address, age, labels, country exposure, memory hits]
- Sanctions: [any matches]
- Fund flows: [any related transactions]
- Memory: [any prior investigations involving these wallets]
- Key finding: [the single most important discovery]

RULES:
- Use EXACT addresses from tool results
- Gather evidence from ALL relevant tools before declaring complete
- Maximum {MAX_LOOPS} tool calls
- Do NOT write a report — just collect and organize the facts"""


def build_synthesizer_prompt(personality="roast"):
    """Prompt for the SYNTHESIZER — focused on writing the report."""
    p = PERSONALITIES.get(personality, PERSONALITIES["roast"])
    return f"""You are the SYNTHESIZER module of NEXUS-AGENT v3, an AML investigation agent.

YOUR JOB: Take the evidence collected by the Reasoner and write a compelling investigation report.
You do NOT call tools. You only write.

PERSONALITY: {p['name']}
{p['instructions']}

You will receive a structured evidence package. Transform it into:

🔥 COMMENTARY: 2-3 sentences in your personality voice. Be SPECIFIC — mention dollar amounts, wallet addresses, time gaps, flags. Make it memorable.
📋 SAR NARRATIVE: Professional 3-5 sentence investigation summary. Cite specific evidence. This goes to regulators — it must be precise.
⚡ RISK VERDICT: CRITICAL/HIGH/MEDIUM/LOW — one sentence justification with evidence.
🎯 RECOMMENDED ACTION: freeze/escalate/monitor/dismiss — explain why.
🧠 MEMORY NOTE: One sentence about what to remember for future investigations.

RULES:
- Every claim must reference specific evidence from the package
- COMMENTARY must be in your personality voice — make it genuinely engaging
- SAR NARRATIVE is always professional regardless of personality
- Never use placeholder text — write the real content
- The COMMENTARY is what makes people actually read the report — make it count"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_reasoner_response(text):
    """Parse THINK, ACTION, EVIDENCE_COMPLETE from reasoner output."""
    result = {"think": "", "action": None, "action_param": None, "evidence": None}

    think_match = re.search(r"THINK:\s*(.+?)(?=ACTION:|EVIDENCE_COMPLETE:|$)", text, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    action_match = re.search(r"ACTION:\s*(\w+)\s*\(?\s*([^)\n]+?)\s*\)?\s*$", text, re.MULTILINE)
    if action_match:
        result["action"] = action_match.group(1).strip()
        result["action_param"] = action_match.group(2).strip()

    if "EVIDENCE_COMPLETE:" in text:
        idx = text.find("EVIDENCE_COMPLETE:")
        result["evidence"] = text[idx:]

    return result


def gather_evidence(tx_id):
    """PHASE 1: Reasoner gathers all evidence via tool calls."""
    print(f"\n  ┌─ PHASE 1: REASONING (evidence gathering)")
    print(f"  │  Model preference: Groq → Gemini → Anthropic")

    system = build_reasoner_prompt()
    messages = [{"role": "user", "content": f"Investigate transaction {tx_id}. Gather all evidence systematically."}]

    all_tool_results = []  # track everything the reasoner found
    reasoner_provider = None

    for loop in range(MAX_LOOPS):
        if VERBOSE:
            print(f"  │  ── Loop {loop+1}/{MAX_LOOPS}")

        text, provider = call_role("reasoner", system, messages)
        if not reasoner_provider:
            reasoner_provider = provider

        if VERBOSE:
            print(f"  │  🧠 {text[:200]}{'...' if len(text) > 200 else ''}")

        parsed = parse_reasoner_response(text)

        # Evidence complete — hand off to synthesizer
        if parsed["evidence"]:
            print(f"  │  ✅ Evidence gathered in {loop+1} loops")
            print(f"  └─ Reasoner: {reasoner_provider}")
            return parsed["evidence"], all_tool_results, loop + 1, reasoner_provider

        # Execute tool
        if parsed["action"] and parsed["action"] in TOOLS:
            tool_name = parsed["action"]
            tool_param = parsed["action_param"]
            if VERBOSE:
                print(f"  │  🔧 {tool_name}({tool_param})")
            tool_result = TOOLS[tool_name]["fn"](tool_param)
            all_tool_results.append({"tool": tool_name, "param": tool_param, "result": tool_result})
            if VERBOSE:
                print(f"  │  📄 {tool_result[:150]}...")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"TOOL RESULT ({tool_name}):\n{tool_result}"})
        elif parsed["action"]:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"ERROR: Unknown tool '{parsed['action']}'. Available: {list(TOOLS.keys())}"})
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Continue. Use a tool (ACTION) or declare EVIDENCE_COMPLETE."})

    print(f"  └─ ⚠️ Reasoner hit max loops ({MAX_LOOPS})")
    # Force an evidence summary from what we have
    evidence_summary = "EVIDENCE_COMPLETE (partial — max loops reached):\n"
    for tr in all_tool_results:
        evidence_summary += f"- {tr['tool']}({tr['param']}): {tr['result'][:200]}\n"
    return evidence_summary, all_tool_results, MAX_LOOPS, reasoner_provider


def write_report(evidence, personality="roast"):
    """PHASE 2: Synthesizer writes the report from evidence."""
    print(f"\n  ┌─ PHASE 2: SYNTHESIS (report writing)")
    print(f"  │  Model preference: Gemini → Groq → Anthropic")
    print(f"  │  Personality: {PERSONALITIES.get(personality, {}).get('name', personality)}")

    system = build_synthesizer_prompt(personality)
    messages = [{"role": "user", "content": f"Here is the evidence package from the investigation:\n\n{evidence}\n\nWrite your complete report."}]

    text, provider = call_role("synthesizer", system, messages, max_tokens=1000)

    print(f"  └─ Synthesizer: {provider}")
    return text, provider


def investigate(tx_id, personality="roast"):
    """Full v3 pipeline: REASON → SYNTHESIZE."""
    global MEMORY

    print(f"\n{'━'*65}")
    print(f"🕵️ NEXUS-AGENT v3 — Investigating {tx_id}")
    print(f"   Pipeline: Reasoner → Synthesizer")
    print(f"   Personality: {PERSONALITIES.get(personality, {}).get('name', personality)}")
    print(f"{'━'*65}")

    # Phase 1: gather evidence
    evidence, tool_results, reason_loops, reason_provider = gather_evidence(tx_id)

    # Phase 2: write report
    report, synth_provider = write_report(evidence, personality)

    # Display
    print(f"\n{'━'*65}")
    print(f"📊 INVESTIGATION COMPLETE — {tx_id}")
    print(f"{'━'*65}")
    print(report)
    print(f"{'━'*65}")
    print(f"🔄 Pipeline: {reason_loops} reasoning loops")
    print(f"🧠 Reasoner: {reason_provider} | Synthesizer: {synth_provider}")
    cost = {"groq": "FREE", "gemini": "FREE", "anthropic": "PAID"}.get(reason_provider, "?")
    cost2 = {"groq": "FREE", "gemini": "FREE", "anthropic": "PAID"}.get(synth_provider, "?")
    print(f"💰 Cost: {cost} + {cost2} = {'$0' if 'PAID' not in cost + cost2 else '⚠️ check'}")

    # Update memory
    if MEMORY:
        tx = TX_STORE.get(tx_id, {})
        if tx.get("sender"):
            MEMORY.remember_wallet(tx["sender"], {
                "risk_score": tx.get("risk_score"),
                "labels": tx.get("flags", []),
                "note": f"v3 investigation: {tx_id}",
            })
        if tx.get("receiver"):
            MEMORY.remember_wallet(tx["receiver"], {"note": f"Received in {tx_id}"})
        if tx.get("sender") and tx.get("receiver"):
            MEMORY.remember_connection(tx["sender"], tx["receiver"])

        # Extract memory note from report
        mem_match = re.search(r"MEMORY NOTE:\s*(.+?)(?=\n|$)", report)
        mem_note = mem_match.group(1).strip() if mem_match else report[:200]
        MEMORY.remember_investigation(tx_id, mem_note)
        MEMORY.save()
        print(f"🧠 Memory updated")

    return {
        "tx_id": tx_id,
        "personality": personality,
        "reason_loops": reason_loops,
        "reason_provider": reason_provider,
        "synth_provider": synth_provider,
        "report": report,
        "evidence": evidence,
    }


def investigate_all(source="sample", personality="roast"):
    """Batch investigate all transactions."""
    txns = load_transactions(source)
    if not txns:
        print("No transactions.")
        return []
    for tx in txns:
        TX_STORE[tx["id"]] = tx

    print(f"\n{'='*65}")
    print(f"NEXUS-AGENT v3 — Batch Pipeline Investigation")
    print(f"{'='*65}")
    print(f"  Transactions: {len(txns)}")
    print(f"  Personality: {PERSONALITIES.get(personality, {}).get('name', personality)}")
    print(f"  Reasoner chain:    {' → '.join(p for p,m in ROLES['reasoner']['chain'])}")
    print(f"  Synthesizer chain: {' → '.join(p for p,m in ROLES['synthesizer']['chain'])}")

    results = []
    for tx in txns:
        r = investigate(tx["id"], personality)
        results.append(r)
        time.sleep(2)  # rate limit breathing room

    out = {
        "timestamp": datetime.now().isoformat(),
        "agent_version": "v3",
        "architecture": "split-brain (reasoner + synthesizer)",
        "personality": personality,
        "results": results,
    }
    out_path = os.path.join(WORKSPACE, "agent_v3_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n💾 Results: {out_path}")

    if MEMORY:
        patterns = MEMORY.detect_patterns()
        if patterns:
            print(f"\n🔍 Cross-case patterns: {len(patterns)}")
            for p in patterns:
                print(f"   {p['type']}: {p.get('wallet', 'N/A')}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEXUS-AGENT v3 — Multi-Model Pipeline")
    parser.add_argument("--tx", type=str, help="Investigate specific transaction")
    parser.add_argument("--all", action="store_true", help="Investigate all transactions")
    parser.add_argument("--input", type=str, default="sample", help="Data source")
    parser.add_argument("--personality", type=str, default="roast", choices=list(PERSONALITIES.keys()))
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"NEXUS-AGENT v3 — Split-Brain Architecture")
    print(f"{'='*65}")
    print(f"  Reasoner:     {' → '.join(f'{p} ({m})' for p,m in ROLES['reasoner']['chain'])}")
    print(f"  Synthesizer:  {' → '.join(f'{p} ({m})' for p,m in ROLES['synthesizer']['chain'])}")
    print(f"  Keys: Groq={'✅' if GROQ_KEY else '❌'}  Gemini={'✅' if GEMINI_KEY else '❌'}  Anthropic={'✅' if ANTHROPIC_KEY else '❌'}")

    if not args.no_memory:
        MEMORY = AgentMemory()

    txns = load_transactions(args.input)
    for tx in txns:
        TX_STORE[tx["id"]] = tx

    if args.tx:
        investigate(args.tx, args.personality)
    elif args.all:
        investigate_all(args.input, args.personality)
    else:
        print("\nNo --tx or --all. Running most suspicious.")
        if txns:
            most_sus = max(txns, key=lambda t: t.get("risk_score", 0))
            investigate(most_sus["id"], args.personality)
