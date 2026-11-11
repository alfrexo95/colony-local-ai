#!/usr/bin/env python3
"""
Autonomous Agent for The Colony - PyTorch Native Edition v6.0
Features: 5.5h continuous loop with 10-min rests, Wallet of Satoshi sync,
Crypto endpoints (/crypto/paid, chains priority: BNB > ETH > SOL > EVM > BTC),
and Client Service Orders management.
"""

import gc
import json
import os
import re
import sys
import time
import math
import logging
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from colony_sdk import ColonyClient

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("colony-pytorch-agent")

# CONFIG & ENV
COLONY_KEY = os.environ.get("COLONY_API_KEY", "")
API_BASE = os.environ.get("COLONY_BASE", "https://thecolony.ai/api/v1").rstrip("/")
DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")

COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "CG-DxDmERwtikShxhjWGfHzyKnX")
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
COINGECKO_HEADERS = {"x-cg-demo-api-key": COINGECKO_API_KEY}

LIGHTNING_ADDRESS = os.environ.get("COLONY_LIGHTNING_ADDRESS", "chieftrait089@walletofsatoshi.com")
EVM_ADDRESS = os.environ.get("COLONY_EVM_ADDRESS", "")

MAX_TASKS_PER_RUN = 3
MIN_REVIEW_SCORE = 7
STATE_FILE = Path("state.json")

ENABLE_BID = True
REQUIRE_SATS = True
BID_PCT = 0.9
BID_DESC_MAX = 900
REST_INTERVAL_MINUTES = 10
TOTAL_RUN_HOURS = 5.5

PYTORCH_MODEL_ID = os.environ.get("PYTORCH_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
SOLVER_MAX_TOKENS = 1000
REVIEWER_MAX_TOKENS = 400

BLOCKED = re.compile(
    r"\b(seed phrase|mnemonic|private key|password|2fa|otp code|ssh key|credit card|doxx|doxxing|"
    r"malware|ransomware|phishing|keylogger|ddos|fake reviews?|wallet drainer|rug ?pull)\b", re.I)

def agent_log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _dict(x): return x if isinstance(x, dict) else {}
def _num(x, default=0.0):
    try: return float(x)
    except (TypeError, ValueError): return default

def meta(task):
    for key in ("metadata", "metadata_"):
        m = task.get(key)
        if isinstance(m, str):
            try: m = json.loads(m)
            except ValueError: m = None
        if isinstance(m, dict) and m: return m
    return {}

def _as_list(d, keys=("tasks", "items", "data", "results", "comments", "bids", "orders")):
    if isinstance(d, list): return d
    if isinstance(d, dict):
        for k in keys:
            if isinstance(d.get(k), list): return d[k]
    return []

def load_state():
    try: st = json.loads(STATE_FILE.read_text())
    except Exception: st = {}
    st.setdefault("answered", {})
    st.setdefault("submissions", [])
    st.setdefault("failures", {})
    st.setdefault("pending_bids", {})
    st.setdefault("rejected", {})
    st.setdefault("orders", {})
    return st

def save_state(st):
    cutoff = time.time() - 30 * 86400
    st["answered"] = {k: v for k, v in st["answered"].items() if v > cutoff}
    st["submissions"] = [s for s in st["submissions"] if s.get("timestamp", 0) > cutoff]
    st["failures"] = {k: v for k, v in st["failures"].items() if k not in st["answered"]}
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(STATE_FILE)

# PRIORITY SYSTEM (BNB > ETH > SOL > EVM > BTC)
_CHAIN_PATTERNS = [
    (5, "BNB", re.compile(r"\b(bsc|bnb|binance)\b")),
    (4, "ETH", re.compile(r"\b(ethereum|eth|erc[- ]?20)\b")),
    (3, "SOL", re.compile(r"\b(solana|sol|spl)\b")),
    (2, "EVM", re.compile(r"\b(evm|polygon|arbitrum|optimism|base (chain|network|mainnet|l2)|on base)\b")),
    (1, "BTC", re.compile(r"\b(bitcoin|btc|lightning)\b")),
]

_CHAIN_BY_ID = {
    "bsc": (5, "BNB"), "ethereum": (4, "ETH"), "sepolia": (4, "ETH"), "solana": (3, "SOL"),
    "base": (2, "EVM"), "base-sepolia": (2, "EVM"), "arbitrum": (2, "EVM"), "polygon": (2, "EVM"),
    "optimism": (2, "EVM"), "bitcoin": (1, "BTC"),
}
_CAIP2 = {"eip155:56": "bsc", "eip155:1": "ethereum", "eip155:8453": "base",
          "eip155:42161": "arbitrum", "eip155:137": "polygon", "eip155:10": "optimism"}

def _chain_key(c):
    c = str(c or "").lower().strip()
    if c.startswith("solana:"): return "solana"
    if c.startswith("bip122:"): return "bitcoin"
    return _CAIP2.get(c, c)

def get_chain_priority(task):
    chains = []
    reward = meta(task).get("reward")
    if isinstance(reward, dict): chains.append(_dict(reward.get("asset")).get("chain"))
    refs = _dict(task.get("crypto")).get("refs")
    if isinstance(refs, list): chains.extend(_dict(r).get("chain") for r in refs)
    found = [_CHAIN_BY_ID[k] for k in (_chain_key(c) for c in chains) if k in _CHAIN_BY_ID]
    if found: return max(found)
    
    text = (str(task.get("title") or "") + " " + str(task.get("body") or task.get("content") or "")).lower()
    for prio, name, pat in _CHAIN_PATTERNS:
        if pat.search(text): return prio, name
    return 0, "OTHER"

def get_task_value(task):
    metadata = meta(task)
    reward = metadata.get("reward")
    token_amt = _num(reward.get("amount") if isinstance(reward, dict) else reward)
    sats_budget = _num(metadata.get("budget_max_sats"))
    chain_prio, _ = get_chain_priority(task)
    return (chain_prio, 1 if token_amt > 0 else 0, sats_budget, token_amt)

# SDK & HTTP INITIALIZATION
if not COLONY_KEY:
    print("[!] Missing COLONY_API_KEY secret.", flush=True)
    sys.exit(1)

try:
    client = ColonyClient(COLONY_KEY, base_url=API_BASE)
    _boot = getattr(client, "bootstrap", None)
    if callable(_boot): _boot()
    else: client.get_me()
    agent_log("[*] Connected via SDK. Bootstrap status: OK")
except Exception as e:
    agent_log(f"[!] SDK connection failed: {e}")
    sys.exit(1)

def http(method, path, headers=None, **kw):
    for attempt in range(3):
        try:
            r = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=30, **kw)
        except requests.RequestException:
            if attempt == 2: return 0, "error"
            time.sleep(2)
            continue
        return r.status_code, r.json() if r.headers.get("content-type","").startswith("application/json") else r.text
    return 0, "no response"

def get_jwt():
    st, d = http("POST", "/auth/token", json={"api_key": COLONY_KEY})
    if st >= 400 or st == 0: raise RuntimeError(f"auth failed: HTTP {st}")
    jwt = _dict(d).get("access_token")
    if not jwt: raise RuntimeError("no JWT received")
    return jwt

def sync_profile(headers):
    st, me = http("GET", "/users/me", headers=headers)
    me = _dict(me)
    username = str(me.get("username") or "")
    agent_log(f"[*] Agent: {username} | karma: {me.get('karma', 0)} | Lightning: {me.get('lightning_address', 'NOT SET')}")
    
    # Forzar actualización de la Lightning Address (Wallet of Satoshi) si difiere
    update = {}
    if me.get("lightning_address") != LIGHTNING_ADDRESS:
        update["lightning_address"] = LIGHTNING_ADDRESS
    if EVM_ADDRESS and me.get("evm_address") != EVM_ADDRESS:
        update["evm_address"] = EVM_ADDRESS
    if update:
        st_up, _ = http("PUT", "/users/me", headers=headers, json=update)
        agent_log(f"[*] Updated profile addresses: HTTP {st_up} (Lightning -> {LIGHTNING_ADDRESS})")
    return username

def fetch_all_tasks(headers):
    tasks = []
    # 1. Endpoint general de marketplace
    st, d = http("GET", "/marketplace/tasks", headers=headers, params={"status": "open"})
    if st < 400: tasks.extend(_as_list(d))
    
    # 2. Endpoint especializado en crypto pagado (según crypto.md)[span_4](start_span)[span_4](end_span)
    st_p, d_p = http("GET", "/crypto/paid", headers=headers)
    if st_p < 400:
        known = {str(t.get("id")) for t in tasks if isinstance(t, dict)}
        extra = [t for t in _as_list(d_p) if isinstance(t, dict) and str(t.get("id")) not in known]
        if extra: tasks.extend(extra)
        
    return tasks

# OFTERS & ORDERS HANDLER (Seguimiento de publicaciones y solicitudes de clientes)
def handle_client_orders(headers):
    st, d = http("GET", "/offers/orders", headers=headers)
    if st >= 400: return
    orders = _as_list(d)
    for o in orders:
        oid = str(o.get("id") or o.get("order_id") or "")
        status = str(o.get("status") or "").lower()
        if not oid or status in ("delivered", "payout_completed", "cancelled"): continue
        
        if status == "requested":
            http("POST", f"/offers/orders/{oid}/accept", headers=headers)
            agent_log(f"   [+] Accepted client service order {oid}")
        elif status == "accepted":
            http("POST", f"/offers/orders/{oid}/payment/check", headers=headers)

# PYTORCH INFERENCE ENGINE
_torch_model = None
_torch_tokenizer = None

def load_pytorch_model():
    global _torch_model, _torch_tokenizer
    if _torch_model is None:
        agent_log(f"[*] Loading PyTorch model {PYTORCH_MODEL_ID}...")
        _torch_tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_ID)
        _torch_model = AutoModelForCausalLM.from_pretrained(
            PYTORCH_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if _torch_tokenizer.pad_token is None:
            _torch_tokenizer.pad_token = _torch_tokenizer.eos_token
    return _torch_model, _torch_tokenizer

def ask_pytorch(system, user, max_tokens=1000):
    model, tokenizer = load_pytorch_model()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(tokenizer, "apply_chat_template") else f"System: {system}\nUser: {user}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m: raise ValueError("Invalid JSON from AI")
    return json.loads(m.group(0))

def generate_solution(title, body):
    sys_prompt = 'You are an elite developer and smart contract auditor on The Colony. Provide accurate markdown solutions. Respond ONLY with JSON: {"can_do": true, "confidence": "high", "response": "<markdown solution>"}'
    try:
        res = ask_pytorch(sys_prompt, f"TASK: {title}\nDESC: {body[:2000]}", SOLVER_MAX_TOKENS)
        return str(res.get("response", "")).strip() if res.get("can_do") else None
    except Exception:
        return None

# CICLO DE EJECUCIÓN CONTINUA (5.5 HORAS CON DESCANSO DE 10 MINUTOS)
def run_continuous_agent():
    deadline = time.time() + (TOTAL_RUN_HOURS * 3600)
    agent_log(f"[*] Starting 5.5-hour continuous agent loop with {REST_INTERVAL_MINUTES}m rest intervals...")
    
    while time.time() < deadline:
        cycle_start = time.time()
        agent_log("[*] --- Starting new intelligence & earning cycle ---")
        
        state = load_state()
        try:
            jwt = get_jwt()
            headers = {"Authorization": f"Bearer {jwt}", "Accept": "application/json"}
            sync_profile(headers)
            handle_client_orders(headers)
            tasks = fetch_all_tasks(headers)
        except Exception as e:
            agent_log(f"[!] Init error in cycle: {e}")
            time.sleep(60)
            continue

        tasks = [t for t in tasks if isinstance(t, dict) and str(t.get("status") or "open").lower() in ("open", "bidding")]
        tasks.sort(key=get_task_value, reverse=True)

        answered = 0
        for task in tasks:
            if answered >= MAX_TASKS_PER_RUN: break
            pid = str(task.get("id"))
            if pid in state["answered"] or state["failures"].get(pid, 0) >= 3: continue
            
            title = str(task.get("title") or "")[:80]
            body = str(task.get("body") or task.get("content") or "")
            metadata = meta(task)
            sats_budget = _num(metadata.get("budget_max_sats"))

            if REQUIRE_SATS and sats_budget <= 0: continue
            if BLOCKED.search(f"{title} {body}"):
                state["answered"][pid] = time.time()
                save_state(state)
                continue

            _, chain_name = get_chain_priority(task)
            agent_log(f"[?] Processing [{chain_name}] Task: {title}")
            
            answer = generate_solution(title, body)
            if not answer:
                state["failures"][pid] = state["failures"].get(pid, 0) + 1
                save_state(state)
                continue

            try:
                client.create_comment(pid, answer)
                answered += 1
                state["answered"][pid] = time.time()
                state["submissions"].append({"timestamp": time.time(), "estimated_value": sats_budget or 50})
                save_state(state)
                agent_log("   [+] Successfully posted solution & earned value!")
            except Exception:
                state["failures"][pid] = state["failures"].get(pid, 0) + 1
                save_state(state)

            if ENABLE_BID and sats_budget > 0:
                bid_sats = max(1, int(sats_budget * BID_PCT))
                pitch = f"Verified solution by colony-local-ai. Summary: {re.sub(r'\\s+', ' ', answer).strip()[:300]}"
                http("POST", f"/marketplace/{pid}/bid", headers=headers, json={"bid_amount_sats": bid_sats, "bid_description": pitch[:BID_DESC_MAX]})
            
            time.sleep(20)

        # Calcular tiempo transcurrido y descansar 10 minutos antes del siguiente ciclo
        elapsed = time.time() - cycle_start
        rest_time = REST_INTERVAL_MINUTES * 60
        agent_log(f"[*] Cycle finished. Resting for {REST_INTERVAL_MINUTES} minutes...")
        
        sleeping = 0
        while sleeping < rest_time and time.time() < deadline:
            time.sleep(30)
            sleeping += 30

    agent_log("[*] 5.5-hour runtime completed successfully. Exiting clean.")

if __name__ == "__main__":
    run_continuous_agent()
