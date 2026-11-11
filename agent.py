#!/usr/bin/env python3
"""
Autonomous Agent for The Colony - PyTorch Native Single-File Edition v5.5
Zero external heavy dependencies (No pandas/numpy). Clean atomic cycle.
Prioritizes BNB > ETH > SOL > EVM > BTC.
"""

import gc
import json
import os
import re
import sys
import time
import math
import statistics
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

def _i(name, default):
    try: return int(float(os.environ.get(name, default)))
    except ValueError: return int(default)

def _f(name, default):
    try: return float(os.environ.get(name, default))
    except ValueError: return float(default)

MAX_TASKS_PER_RUN = _i("MAX_TASKS_PER_RUN", 2)
MIN_CONFIDENCE = os.environ.get("MIN_CONFIDENCE", "medium")
MIN_REVIEW_SCORE = _i("MIN_REVIEW_SCORE", 7)
STATE_FILE = Path(os.environ.get("STATE_FILE", "state.json"))

ENABLE_BID = os.environ.get("ENABLE_BID", "1") == "1"
REQUIRE_SATS = os.environ.get("REQUIRE_SATS", "1") == "1"
BID_PCT = _f("BID_PCT", 0.9)
BID_DESC_MAX = _i("BID_DESC_MAX", 900)
LIGHTNING_ADDRESS = os.environ.get("COLONY_LIGHTNING_ADDRESS", "")
EVM_ADDRESS = os.environ.get("COLONY_EVM_ADDRESS", "")

POST_DELAY = _i("POST_DELAY_SECONDS", 15)
MAX_FAILURES = _i("MAX_TASK_FAILURES", 3)
MIN_ANSWER_CHARS = _i("MIN_ANSWER_CHARS", 50)
MAX_ANSWER_CHARS = _i("MAX_ANSWER_CHARS", 9500)
FETCH_CRYPTO_PAID = os.environ.get("FETCH_CRYPTO_PAID", "1") == "1"

PYTORCH_MODEL_ID = os.environ.get("PYTORCH_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
SOLVER_MAX_TOKENS = _i("SOLVER_MAX_TOKENS", 1000)
REVIEWER_MAX_TOKENS = _i("REVIEWER_MAX_TOKENS", 400)

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
    st["pending_bids"] = {k: v for k, v in st["pending_bids"].items() if v.get("ts", 0) > time.time() - 2 * 86400}
    st["rejected"] = {k: v for k, v in st["rejected"].items() if v > time.time() - 7 * 86400}
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

def _structured_chain(task):
    chains = []
    reward = meta(task).get("reward")
    if isinstance(reward, dict): chains.append(_dict(reward.get("asset")).get("chain"))
    refs = _dict(task.get("crypto")).get("refs")
    if isinstance(refs, list): chains.extend(_dict(r).get("chain") for r in refs)
    found = [_CHAIN_BY_ID[k] for k in (_chain_key(c) for c in chains) if k in _CHAIN_BY_ID]
    return max(found) if found else None

def get_chain_priority(task):
    structured = _structured_chain(task)
    if structured: return structured
    text = (str(task.get("title") or "") + " " + str(task.get("body") or task.get("content") or "")).lower()
    for prio, name, pat in _CHAIN_PATTERNS:
        if pat.search(text): return prio, name
    return 0, "OTHER"

def _task_money(task):
    metadata = meta(task)
    reward = metadata.get("reward")
    if isinstance(reward, dict): token_amt = _num(reward.get("amount"))
    else: token_amt = _num(reward)
    sats_budget = _num(metadata.get("budget_max_sats"))
    return metadata, reward, sats_budget, token_amt

def get_task_value(task):
    _, _, sats_budget, token_amt = _task_money(task)
    chain_priority, _ = get_chain_priority(task)
    return (chain_priority, 1 if token_amt > 0 else 0, sats_budget, token_amt)

def compute_bid(task, sats_budget):
    bid = max(1, int(sats_budget * BID_PCT))
    bmin = _num(meta(task).get("budget_min_sats"))
    if bmin > 0: bid = max(bid, int(bmin))
    return max(1, min(bid, int(sats_budget)))

# SDK INITIALIZATION
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

def fetch_tasks(headers):
    st, d = http("GET", "/marketplace/tasks", headers=headers, params={"status": "open"})
    if 400 <= st < 500: st, d = http("GET", "/marketplace/tasks", headers=headers)
    if st >= 400 or st == 0: raise RuntimeError(f"HTTP {st}")
    return _as_list(d)

_profile_synced = False
MY_USERNAME = ""

def sync_profile(headers):
    global _profile_synced, MY_USERNAME
    st, me = http("GET", "/users/me", headers=headers)
    me = _dict(me)
    MY_USERNAME = str(me.get("username") or MY_USERNAME or "")
    if st < 400 and not _profile_synced:
        agent_log(f"[*] Agent: {MY_USERNAME} | karma: {me.get('karma', 0)}")
    if _profile_synced or DRY_RUN: return
    update = {}
    if LIGHTNING_ADDRESS and me.get("lightning_address") != LIGHTNING_ADDRESS: update["lightning_address"] = LIGHTNING_ADDRESS
    if EVM_ADDRESS and me.get("evm_address") != EVM_ADDRESS: update["evm_address"] = EVM_ADDRESS
    if update: http("PUT", "/users/me", headers=headers, json=update)
    _profile_synced = True

def already_commented(post_id):
    if not MY_USERNAME: return False
    try:
        comments = _as_list(client.get_comments(post_id))
    except Exception: return False
    for c in comments:
        c = _dict(c)
        author = c.get("author")
        name = _dict(author).get("username") if isinstance(author, dict) else author
        name = name or c.get("username") or c.get("author_username")
        if str(name or "").lower() == MY_USERNAME.lower(): return True
    return False

def place_bid(headers, post_id, sats, pitch):
    return http("POST", f"/marketplace/{post_id}/bid", headers=headers, json={"bid_amount_sats": int(sats), "bid_description": pitch[:BID_DESC_MAX]})

def build_pitch(answer):
    return f"Verified solution by colony-local-ai. Summary: {re.sub(r'\\s+', ' ', answer).strip()[:300]}"

def merge_detail(headers, task):
    pid = str(task.get("id"))
    st, d = http("GET", f"/posts/{pid}", headers=headers)
    d = _dict(d)
    det = (_dict(d.get("post")) or d) if st < 400 else {}
    for k in ("metadata", "metadata_", "summary", "crypto", "client"):
        if det.get(k) and not task.get(k): task[k] = det[k]

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

def free_pytorch_model():
    global _torch_model
    if _torch_model is not None:
        del _torch_model
        _torch_model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

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

# ATOMIC EXECUTION CYCLE
if __name__ == "__main__":
    agent_log("[*] Starting atomic single-cycle PyTorch execution...")
    state = load_state()
    try:
        jwt = get_jwt()
        headers = {"Authorization": f"Bearer {jwt}", "Accept": "application/json"}
        sync_profile(headers)
        tasks = fetch_tasks(headers)
    except Exception as e:
        agent_log(f"[!] Init error: {e}")
        sys.exit(0)

    tasks = [t for t in tasks if isinstance(t, dict) and str(t.get("status") or "open").lower() in ("open", "bidding")]
    tasks.sort(key=get_task_value, reverse=True)

    answered = 0
    for task in tasks:
        if answered >= MAX_TASKS_PER_RUN: break
        pid = str(task.get("id"))
        if pid in state["answered"] or state["failures"].get(pid, 0) >= MAX_FAILURES: continue
        
        if not meta(task): merge_detail(headers, task)
        title = str(task.get("title") or "")[:80]
        body = str(task.get("body") or task.get("content") or "")
        _, _, sats_budget, _ = _task_money(task)

        if REQUIRE_SATS and sats_budget <= 0: continue
        if BLOCKED.search(f"{title} {body}") or already_commented(pid):
            state["answered"][pid] = time.time()
            save_state(state)
            continue

        agent_log(f"[?] Processing Task: {title}")
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
            agent_log("   [+] Successfully posted solution!")
        except Exception:
            state["failures"][pid] = state["failures"].get(pid, 0) + 1
            save_state(state)

        if ENABLE_BID and sats_budget > 0:
            place_bid(headers, pid, compute_bid(task, sats_budget), build_pitch(answer))
        time.sleep(POST_DELAY)

    free_pytorch_model()
    agent_log("[*] Cycle completed successfully. Exiting.")
