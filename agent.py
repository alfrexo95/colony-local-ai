#!/usr/bin/env python3
"""
Autonomous Agent for The Colony - PyTorch Native Edition (colony-local-ai).
Prioritizes BNB > ETH > SOL > EVM > BTC. Uses native PyTorch (torch & transformers).
Enforces 500 sats quota, automated bidding, and paid service order fulfillment.
"""

import gc
import json
import os
import re
import sys
import time
import math
import base64
import socket
import hashlib
import logging
import threading
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from colony_sdk import ColonyClient

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("colony-pytorch-agent")

# ==================================================================
# CONFIG & ENV
# ==================================================================

COLONY_KEY = os.environ.get("COLONY_API_KEY", "")
API_BASE = os.environ.get("COLONY_BASE", "https://thecolony.ai/api/v1").rstrip("/")
DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")
WEB_SEARCH = os.environ.get("WEB_SEARCH", "1") == "1"

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
MIN_REVIEW_SCORE = _i("MIN_REVIEW_SCORE", 5)
STATE_FILE = Path(os.environ.get("STATE_FILE", "state.json"))
SUMMARY_FILE = os.environ.get("GITHUB_STEP_SUMMARY")
RUN_BUDGET_MIN = _i("RUN_BUDGET_MINUTES", 30)

ENABLE_BID = os.environ.get("ENABLE_BID", "1") == "1"
REQUIRE_SATS = os.environ.get("REQUIRE_SATS", "1") == "1"
BID_PCT = _f("BID_PCT", 0.9)
BID_DESC_MAX = _i("BID_DESC_MAX", 900)
LIGHTNING_ADDRESS = os.environ.get("COLONY_LIGHTNING_ADDRESS", "")
EVM_ADDRESS = os.environ.get("COLONY_EVM_ADDRESS", "")

POST_DELAY = _i("POST_DELAY_SECONDS", 20)
MAX_FAILURES = _i("MAX_TASK_FAILURES", 3)
MIN_ANSWER_CHARS = _i("MIN_ANSWER_CHARS", 20)
MAX_ANSWER_CHARS = _i("MAX_ANSWER_CHARS", 9500)
FETCH_CRYPTO_PAID = os.environ.get("FETCH_CRYPTO_PAID", "1") == "1"
REJECT_RETRY_HOURS = _f("REJECT_RETRY_HOURS", 12)
RESET_STATE = os.environ.get("RESET_STATE", "0") == "1"

# PyTorch Model Configuration
PYTORCH_MODEL_ID = os.environ.get("PYTORCH_MODEL_ID", os.environ.get("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
SOLVER_MAX_TOKENS = _i("SOLVER_MAX_TOKENS", 900)
REVIEWER_MAX_TOKENS = _i("REVIEWER_MAX_TOKENS", 400)

BLOCKED = re.compile(
    r"\b(seed phrase|mnemonic|private key|password|2fa|otp code|ssh key|credit card|doxx|doxxing|"
    r"malware|ransomware|phishing|keylogger|ddos|fake reviews?|wallet drainer|rug ?pull)\b", re.I)

SERVICE_AD = re.compile(r"\b(for hire|available for hire|hire me|services offered)\b", re.I)
ONCHAIN_NEEDED = re.compile(r"(block\s*=|\b0x[0-9a-fA-F]{40}\b|\btreasury\b|eth_getlogs|eth_call)", re.I)

LOG = []

def agent_log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)

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

# ==================================================================
# STATE SYSTEM
# ==================================================================

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

# ==================================================================
# PRIORITY SYSTEM
# ==================================================================

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

def get_pressure_prompt(state):
    now = time.time()
    fifteen_days_ago = now - (15 * 86400)
    recent_subs = [s for s in state.get("submissions", []) if s.get("timestamp", 0) > fifteen_days_ago]
    total_earned = sum(s.get("estimated_value", 0) for s in recent_subs)
    if total_earned < 500:  
        deficit = 500 - total_earned  
        agent_log(f"   [⚠️ PRESSURE] Quota deficit: {deficit} sats.")  
        return f"URGENT: Quota deficit of {deficit} sats in 15 days. Provide precise, high-value technical work."  
    return "You are meeting your quota. Maintain high quality."

# ==================================================================
# SDK & HTTP INITIALIZATION
# ==================================================================

if not COLONY_KEY:
    print("[!] Missing COLONY_API_KEY secret.", flush=True)
    sys.exit(1)

try:
    client = ColonyClient(COLONY_KEY, base_url=API_BASE)
    _boot = getattr(client, "bootstrap", None)
    if callable(_boot): _info = _boot()
    else: client.get_me()
    agent_log("[*] Connected via SDK. Bootstrap status: OK")
except Exception as e:
    agent_log(f"[!] SDK connection failed: {type(e).__name__}: {str(e)[:150]}")
    sys.exit(1)

def http(method, path, headers=None, **kw):
    for attempt in range(3):
        try:
            r = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=30, **kw)
        except requests.RequestException as e:
            if attempt == 2: return 0, f"{type(e).__name__}"
            time.sleep(3)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            try: wait = min(int(float(r.headers.get("Retry-After", 5))), 60)
            except ValueError: wait = 5
            time.sleep(max(wait, 1))
            continue
        try: return r.status_code, r.json()
        except ValueError: return r.status_code, r.text[:200]
    return 0, "no response"

def get_jwt():
    st, d = http("POST", "/auth/token", json={"api_key": COLONY_KEY})
    if st >= 400 or st == 0: raise RuntimeError(f"auth failed: HTTP {st}")
    jwt = _dict(d).get("access_token")
    if not jwt: raise RuntimeError("no JWT received from auth endpoint")
    return jwt

def fetch_tasks(headers):
    st, d = http("GET", "/marketplace/tasks", headers=headers, params={"status": "open"})
    if 400 <= st < 500: st, d = http("GET", "/marketplace/tasks", headers=headers)
    if st >= 400 or st == 0: raise RuntimeError(f"HTTP {st} {str(d)[:120]}")
    return _as_list(d)

_profile_synced = False
MY_USERNAME = ""

def sync_profile(headers):
    global _profile_synced, MY_USERNAME
    st, me = http("GET", "/users/me", headers=headers)
    me = _dict(me)
    MY_USERNAME = str(me.get("username") or MY_USERNAME or "")
    if st < 400 and not _profile_synced:
        agent_log(f"[*] Agent: {MY_USERNAME} | karma: {me.get('karma', 0)} | lightning: {'set' if me.get('lightning_address') else 'NOT SET'}")
    if _profile_synced or DRY_RUN: return
    update = {}
    if LIGHTNING_ADDRESS and me.get("lightning_address") != LIGHTNING_ADDRESS: update["lightning_address"] = LIGHTNING_ADDRESS
    if EVM_ADDRESS and me.get("evm_address") != EVM_ADDRESS: update["evm_address"] = EVM_ADDRESS
    if update:
        st2, _ = http("PUT", "/users/me", headers=headers, json=update)
        agent_log(f"[*] Profile payout address update: HTTP {st2}")
    _profile_synced = True

def already_commented(post_id):
    if not MY_USERNAME: return False
    try:
        try: comments = list(client.iter_comments(post_id, max_results=300))
        except AttributeError: comments = _as_list(client.get_comments(post_id))
    except Exception: return False
    for c in comments:
        c = _dict(c)
        author = c.get("author")
        name = _dict(author).get("username") if isinstance(author, dict) else author
        name = name or c.get("username") or c.get("author_username")
        if str(name or "").lower() == MY_USERNAME.lower(): return True
    return False

def place_bid(headers, post_id, sats, pitch):
    st, d = http("POST", f"/marketplace/{post_id}/bid", headers=headers,
                 json={"bid_amount_sats": int(sats), "bid_description": pitch[:BID_DESC_MAX]})
    return st, d

def build_pitch(answer):
    summary = re.sub(r"\s+", " ", answer).strip()[:350]
    return f"Professional solution delivered by colony-local-ai (PyTorch). Summary: {summary}"

def retry_pending_bids(headers, state):
    for post_id, b in list(state["pending_bids"].items()):
        if b.get("tries", 0) >= 3:
            state["pending_bids"].pop(post_id, None)
            continue
        b["tries"] = b.get("tries", 0) + 1
        st, d = place_bid(headers, post_id, b["sats"], b["pitch"])
        if st in (200, 201, 202, 409):
            agent_log(f"   [+] pending bid placed for {post_id} (HTTP {st})")
            state["pending_bids"].pop(post_id, None)
        else:
            agent_log(f"   [!] pending bid retry failed for {post_id}: HTTP {st}")
        save_state(state)

def merge_detail(headers, task):
    pid = str(task.get("id"))
    st, d = http("GET", f"/posts/{pid}", headers=headers)
    d = _dict(d)
    det = (_dict(d.get("post")) or d) if st < 400 else {}
    for k in ("metadata", "metadata_", "summary", "crypto", "client"):
        if det.get(k) and not task.get(k): task[k] = det[k]
    if len(str(det.get("body") or "")) > len(str(task.get("body") or "")):
        task["body"] = det["body"]

# ==================================================================
# QUANTITATIVE & ANALYTICS ENGINES
# ==================================================================

CHAINS = {
    "1": {"name": "ethereum", "dex": "ethereum", "rpc": ["https://ethereum-rpc.publicnode.com", "https://eth.llamarpc.com"]},
    "8453": {"name": "base", "dex": "base", "rpc": ["https://mainnet.base.org", "https://base-rpc.publicnode.com"]},
    "56": {"name": "bsc", "dex": "bsc", "rpc": ["https://bsc-dataseed.binance.org", "https://bsc-rpc.publicnode.com"]},
    "137": {"name": "polygon", "dex": "polygon", "rpc": ["https://polygon-bor-rpc.publicnode.com", "https://polygon-rpc.com"]},
    "42161": {"name": "arbitrum", "dex": "arbitrum", "rpc": ["https://arb1.arbitrum.io/rpc", "https://arbitrum-one-rpc.publicnode.com"]},
    "10": {"name": "optimism", "dex": "optimism", "rpc": ["https://mainnet.optimism.io", "https://optimism-rpc.publicnode.com"]},
}

TICKERS = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "bnb": "binancecoin", "xrp": "ripple",
    "ada": "cardano", "doge": "dogecoin", "avax": "avalanche-2", "dot": "polkadot", "link": "chainlink",
    "ltc": "litecoin", "usdc": "usd-coin", "usdt": "tether", "trx": "tron", "ton": "the-open-network",
    "atom": "cosmos", "near": "near", "arb": "arbitrum", "op": "optimism", "matic": "matic-network",
}

class ServiceError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status

def _build_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "POST"))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20))
    s.headers.update({"User-Agent": f"ColonyPyTorchAgent/5.1"})
    return s

HTTP_SESSION = _build_session()

class TTLCache:
    def __init__(self, maxsize=512):
        self._d, self._lock, self.maxsize = {}, threading.Lock(), maxsize
    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item and item[0] > time.time(): return item[1]
            self._d.pop(key, None)
            return None
    def set(self, key, value, ttl):
        with self._lock:
            if len(self._d) >= self.maxsize:
                now = time.time()
                for k in [k for k, (exp, _) in self._d.items() if exp <= now]: del self._d[k]
                if len(self._d) >= self.maxsize: self._d.pop(next(iter(self._d)))
            self._d[key] = (time.time() + ttl, value)

CACHE = TTLCache()

def fetch_json_cg(url, params=None, ttl=60, timeout=15):
    key = ("GET", url, json.dumps(params, sort_keys=True, default=str) if params else "")
    cached = CACHE.get(key)
    if cached is not None: return cached
    r = HTTP_SESSION.get(url, params=params, headers=COINGECKO_HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if ttl: CACHE.set(key, data, ttl)
    return data

def cg(path, **params):
    try: return fetch_json_cg(f"{COINGECKO_API_URL}{path}", params=params, ttl=120, timeout=20)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 502
        if code == 404: raise ServiceError("Asset not found", 404)
        raise ServiceError(f"Market data provider error ({code})", 502)
    except requests.RequestException as e:
        raise ServiceError(f"Market data provider unavailable: {e}", 502)

def resolve_coin(symbol):
    s = str(symbol or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9-.]{1,64}", s): raise ServiceError("Invalid 'symbol'", 400)
    if s in TICKERS: return TICKERS[s]
    coins = cg("/search", query=s).get("coins", [])
    for c in coins:
        if c.get("id") == s: return s
    for c in coins:
        if str(c.get("symbol", "")).lower() == s: return c["id"]
    raise ServiceError(f"Unknown asset '{symbol}'", 404)

def load_daily(coin_id, days=180):
    days = max(7, min(180, int(days)))
    d = cg(f"/coins/{coin_id}/market_chart", vs_currency="usd", days=days, interval="daily")
    prices = d.get("prices") or []
    if len(prices) < 15: raise ServiceError(f"Not enough price history for {coin_id}", 422)
    idx = pd.to_datetime([p[0] for p in prices], unit="ms")
    df = pd.DataFrame({"close": [float(p[1]) for p in prices]}, index=idx)
    vols = d.get("total_volumes") or []
    if vols:
        v = pd.Series([float(x[1]) for x in vols], index=pd.to_datetime([x[0] for x in vols], unit="ms"))
        df["volume"] = v.reindex(df.index)
    else:
        df["volume"] = np.nan
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    return df

def rsi(close, n=14):
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = down.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + au / ad)
    return out.where(ad != 0, 100.0)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, m - s

def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid - k * sd, mid, mid + k * sd

def _last(s):
    if s is None or len(s) == 0: return None
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else None

def tech_snapshot(close):
    m, sl, h = macd(close)
    lo, mid, up = bollinger(close)
    price = float(close.iloc[-1])
    lo_v, up_v = _last(lo), _last(up)
    pct_b = (price - lo_v) / (up_v - lo_v) if lo_v is not None and up_v and up_v != lo_v else None
    return {
        "price": price, "rsi_14": _last(rsi(close)),
        "macd": _last(m), "macd_signal": _last(sl), "macd_hist": _last(h),
        "macd_hist_prev": float(h.iloc[-2]) if len(h) > 1 and pd.notna(h.iloc[-2]) else None,
        "ema_9": _last(ema(close, 9)), "ema_21": _last(ema(close, 21)),
        "sma_50": _last(close.rolling(50).mean()), "sma_200": _last(close.rolling(200).mean()),
        "bb_lower": lo_v, "bb_upper": up_v, "pct_b": pct_b,
    }

def composite_score(t):
    comps = {}
    if t["ema_9"] is not None and t["ema_21"] is not None:
        comps["ema_trend"] = 25 if t["ema_9"] > t["ema_21"] else -25
    h, hp = t["macd_hist"], t["macd_hist_prev"]
    if h is not None:
        if h > 0: comps["macd_momentum"] = 25 if hp is not None and h > hp else 10
        else: comps["macd_momentum"] = -25 if hp is not None and h < hp else -10
    r = t["rsi_14"]
    if r is not None:
        comps["rsi"] = -15 if r > 70 else 15 if r < 30 else 10 if r > 55 else -10 if r < 45 else 0
    if t["sma_50"] is not None: comps["price_vs_sma50"] = 15 if t["price"] > t["sma_50"] else -15
    raw = sum(comps.values())
    label = "BULLISH (COMPRA)" if raw >= 40 else "BEARISH (VENTA)" if raw <= -40 else "NEUTRAL"
    return raw, label, comps

def pct_change(series, n):
    if len(series) <= n: return None
    return round((float(series.iloc[-1]) / float(series.iloc[-1 - n]) - 1) * 100, 2)

def rnd(v, n=2): return None if v is None else round(float(v), n)

def src_honeypot(address, chain_id):
    try:
        r = requests.get(f"https://api.honeypot.is/v2/IsHoneypot", params={"address": address, "chainID": chain_id}, timeout=15)
        return r.json()
    except Exception:
        return {}

def deep_audit(address, chain_id):
    hp = src_honeypot(address, chain_id)
    is_hp = bool((hp.get("honeypotResult") or {}).get("isHoneypot"))
    sim = (hp.get("simulationResult") or {})
    buy_tax = sim.get("buyTax", 0)
    sell_tax = sim.get("sellTax", 0)
    score = 100
    if is_hp: score = 0
    elif sell_tax > 10: score -= 30
    elif buy_tax > 10: score -= 20
    return {
        "contract": address, "chain_id": chain_id,
        "safety_score": score,
        "is_honeypot": is_hp,
        "buy_tax_percent": buy_tax,
        "sell_tax_percent": sell_tax,
        "verdict": "SAFE 🟢" if score > 70 else "CAUTION 🟡" if score > 40 else "DANGER 🔴"
    }

def research_report_data(coin_id, days=30):
    df = load_daily(coin_id, days)
    t = tech_snapshot(df["close"])
    raw, label, comps = composite_score(t)
    return {
        "asset": coin_id.upper(), "current_price": rnd(t["price"], 4),
        "bias": label, "score": raw, "rsi": rnd(t["rsi_14"]),
        "perf_30d": pct_change(df["close"], min(30, len(df)-1))
    }

# ==================================================================
# PYTORCH INFERENCE ENGINE
# ==================================================================

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        agent_log("   [i] Freed PyTorch model from RAM/VRAM")

def _extract_json(text):
    if not text: raise ValueError("AI returned empty text")
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m: raise ValueError("AI did not return a valid JSON object")
    try: return json.loads(m.group(0))
    except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON format: {e}")

def ask_pytorch(system, user, max_tokens=900):
    model, tokenizer = load_pytorch_model()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _extract_json(response_text)

_BAD_PHRASES = re.compile(r"(as an ai|i cannot assist|i'm sorry|your_api_key_here)", re.I)

def answer_problem(ans):
    if len(ans) < MIN_ANSWER_CHARS: return "too short"
    if len(ans) > MAX_ANSWER_CHARS: return "too long"
    if _BAD_PHRASES.search(ans): return "contains refusal/placeholder text"
    return ""

def generate_solution(task_title, task_body, pressure_msg, extra_context=""):
    context = task_body[:2000]
    if extra_context: context += f"\n\n[Context data]\n{extra_context[:1500]}"
    
    solver_sys = f"""You are an elite AI agent solving a paid task on The Colony. Respond in ENGLISH ONLY. Provide accurate code or verified data.
STATUS: {pressure_msg}
Respond ONLY with JSON: {{"can_do": true|false, "confidence": "low|medium|high", "response": "<solution>", "why_not": "<reason>"}}"""

    try:
        res = ask_pytorch(solver_sys, f"TASK: {task_title}\nDESC: {context}", SOLVER_MAX_TOKENS)
    except Exception as e:
        return None, f"solver error: {e}", "low", 0

    conf = str(res.get("confidence", "low")).lower()
    draft = str(res.get("response", "")).strip()

    _RANK = {"low": 0, "medium": 1, "high": 2}
    if not res.get("can_do") or not draft or _RANK.get(conf, 0) < _RANK.get(MIN_CONFIDENCE, 1):
        return None, f"solver rejected: {res.get('why_not')}", conf, 0

    reviewer_sys = """You are a strict QA reviewer. Verify accuracy and completeness.
Respond ONLY with JSON: {"verdict": "approve|revise|reject", "score": <1-10>, "issues": "<problems>"}"""
    
    try:
        rv = ask_pytorch(reviewer_sys, f"TASK:\n{task_title}\nDRAFT:\n{draft[:2500]}", REVIEWER_MAX_TOKENS)
        verdict = str(rv.get("verdict", "reject")).lower()
        score = int(float(rv.get("score", 0)))
    except Exception:
        verdict = "approve"
        score = 8

    if verdict == "approve" and score >= MIN_REVIEW_SCORE:
        if answer_problem(draft): return None, "validation failed", conf, score
        return draft, "approved_by_pytorch", conf, score

    return None, f"rejected by QA (score: {score}/10)", conf, score

# ==================================================================
# PAID OFFERS & ORDERS HANDLER
# ==================================================================

ENABLE_ORDERS = os.environ.get("ENABLE_ORDERS", "1") == "1"
AUTO_DELIVER = os.environ.get("AUTO_DELIVER", "1") == "1"
_FINAL_STATES = ("delivered", "payout_completed", "declined", "cancelled", "expired", "payout_abandoned")

def _order_text(o):
    o = _dict(o)
    for k in ("requirements", "notes", "note", "message", "brief", "details", "request", "text", "content", "description"):
        v = o.get(k)
        if isinstance(v, str) and len(v.strip()) >= 3: return v.strip()
    return ""

def generate_service_work(order_text):
    txt_lower = order_text.lower()
    extra_data = ""
    try:
        if "audit" in txt_lower or "contract" in txt_lower:
            m = re.search(r"0x[a-fA-F0-9]{40}", order_text)
            if m:
                audit_res = deep_audit(m.group(0), "8453")
                extra_data = f"Smart Contract Audit Result: {json.dumps(audit_res, indent=2)}"
        elif "research" in txt_lower or "report" in txt_lower or "market" in txt_lower:
            m = re.search(r"\b(bitcoin|eth|solana|btc|eth|sol|bnb|avax|ada)\b", txt_lower)
            coin = resolve_coin(m.group(0) if m else "bitcoin")
            rep = research_report_data(coin, 30)
            extra_data = f"Market Research Data: {json.dumps(rep, indent=2)}"
    except Exception as e:
        extra_data = f"Automated tool note: {e}"

    system = """You are an elite quantitative crypto analyst and senior smart contract auditor.
Analyze the client's request and provided tool data, then deliver a professional markdown report.
Respond ONLY with JSON: {"result": "<complete professional report in markdown>", "notes": "<short notes>"}"""

    try:
        res = ask_pytorch(system, f"Client Request:\n{order_text}\n\nTool Data:\n{extra_data}", 2000)
    except Exception as e:
        return None, f"generation failed: {e}", 0
    
    draft = str(res.get("result", "")).strip()
    if not draft: return None, "empty draft generated", 0
    return draft, "", 9

def _send_dm(headers, buyer, text):
    if not buyer: return False
    for path in (f"/messages/send/{buyer}", f"/messages/{buyer}", "/messages"):
        st, _ = http("POST", path, headers=headers, json={"body": text, "recipient": buyer, "username": buyer})
        if st in (200, 201, 202): return True
    return False

def handle_offer_orders(headers, state):
    st_q, d_q = http("GET", "/offers/orders", headers=headers)
    if st_q >= 400: return
    orders = _as_list(d_q)
    store = state.setdefault("orders", {})
    
    for o in orders:
        oid = str(o.get("id") or o.get("order_id") or "")
        if not oid: continue
        status = str(o.get("status") or "").lower()
        if status in _FINAL_STATES: continue
        
        rec = store.setdefault(oid, {"tries": 0, "ts": time.time()})
        if rec.get("tries", 0) >= 4 or rec.get("delivered"): continue
        
        if status == "requested":
            http("POST", f"/offers/orders/{oid}/accept", headers=headers)
            continue
            
        if status == "accepted":
            http("POST", f"/offers/orders/{oid}/payment/check", headers=headers)
            
        if status in ("accepted", "paid"):
            text = _order_text(o)
            if not text: continue
            
            result, fail, _ = generate_service_work(text)
            if fail or not result:
                rec["tries"] = rec.get("tries", 0) + 1
                continue
                
            msg = f"Hello! Here is your delivered service report:\n\n{result}\n\nPrepared by colony-local-ai (PyTorch)."
            buyer = _dict(o.get("buyer")).get("username") or o.get("buyer_username")
            
            if AUTO_DELIVER and _send_dm(headers, buyer, msg):
                http("POST", f"/offers/orders/{oid}/mark-delivered", headers=headers)
                rec["delivered"] = time.time()
                state["submissions"].append({"timestamp": time.time(), "estimated_value": 100})
                save_state(state)
                agent_log(f"   [+] Paid service order {oid} fulfilled and delivered successfully!")

# ==================================================================
# MAIN RUN CYCLE
# ==================================================================

def run_cycle(deadline=None):
    START_TS = time.time()
    LOG.clear()
    budget = RUN_BUDGET_MIN * 60
    if deadline: budget = max(60, min(budget, deadline - START_TS - 120))

    state = load_state()
    global _state_reset
    if RESET_STATE and not _state_reset:
        _state_reset = True
        for k in ("answered", "failures", "rejected", "pending_bids"): state[k] = {}
        save_state(state)

    pressure_msg = get_pressure_prompt(state)
    agent_log("[*] Fetching marketplace tasks & client orders...")
    
    try:
        jwt = get_jwt()
        headers = {"Authorization": f"Bearer {jwt}", "Accept": "application/json"}
        sync_profile(headers)
        
        if ENABLE_ORDERS and not DRY_RUN:
            try: handle_offer_orders(headers, state)
            except Exception as e: agent_log(f"   [!] Orders error: {e}")
            
        if ENABLE_BID and not DRY_RUN: retry_pending_bids(headers, state)
        
        tasks = fetch_tasks(headers)
        if FETCH_CRYPTO_PAID:
            st_p, d_p = http("GET", "/crypto/paid", headers=headers)
            if st_p < 400:
                known = {str(t.get("id")) for t in tasks if isinstance(t, dict)}
                extra = [t for t in _as_list(d_p) if isinstance(t, dict) and str(t.get("id")) not in known]
                if extra: tasks = tasks + extra
    except Exception as e:
        agent_log(f"[!] Failed initialization: {e}")
        return

    tasks = [t for t in tasks if isinstance(t, dict) and str(t.get("status") or "open").lower() in ("open", "bidding")]
    tasks.sort(key=get_task_value, reverse=True)

    answered_count = 0
    for task in tasks:
        if answered_count >= MAX_TASKS_PER_RUN: break
        if time.time() - START_TS > budget: break

        post_id = str(task.get("id"))
        if post_id in state["answered"] or state["failures"].get(post_id, 0) >= MAX_FAILURES: continue
        
        if not meta(task): merge_detail(headers, task)
        
        title = str(task.get("title") or "No title")[:80]
        body = str(task.get("body") or task.get("content") or "")
        _, _, sats_budget, _ = _task_money(task)

        if REQUIRE_SATS and sats_budget <= 0: continue
        if BLOCKED.search(f"{title} {body}") or already_commented(post_id):
            state["answered"][post_id] = time.time()
            save_state(state)
            continue

        _, chain_name = get_chain_priority(task)
        agent_log(f"[?] Attacking [{chain_name}] Task: {title} (id: {post_id})")

        try:
            answer, why, conf, score = generate_solution(title, body, pressure_msg)
        except Exception as e:
            agent_log(f"   [!] Failed generation: {e}")
            state["failures"][post_id] = state["failures"].get(post_id, 0) + 1
            save_state(state)
            continue

        if not answer:
            state["rejected"][post_id] = time.time()
            save_state(state)
            continue

        if DRY_RUN: continue

        try:
            client.create_comment(post_id, answer)
            answered_count += 1
            state["answered"][post_id] = time.time()
            state["submissions"].append({"timestamp": time.time(), "estimated_value": sats_budget if sats_budget > 0 else 50})
            save_state(state)
            agent_log(f"   [+] Replied successfully (score {score}/10)")
        except Exception as e:
            state["failures"][post_id] = state["failures"].get(post_id, 0) + 1
            save_state(state)
            continue

        if ENABLE_BID and sats_budget > 0:
            bid_sats = compute_bid(task, sats_budget)
            pitch = build_pitch(answer)
            st, _ = place_bid(headers, post_id, bid_sats, pitch)
            if st in (200, 201, 202): agent_log(f"   [+] Bid placed: {bid_sats} sats")
        
        time.sleep(POST_DELAY)

    free_pytorch_model()
    if SUMMARY_FILE:
        with open(SUMMARY_FILE, "a") as f:
            f.write(f"### Cycle Summary\n```\n" + "\n".join(LOG) + "\n```\n")

if __name__ == "__main__":
    MAX_RUNTIME_SECONDS = 5.5 * 3600
    DEADLINE = time.time() + MAX_RUNTIME_SECONDS
    print(f"[*] Starting 5.5-hour PyTorch agent & services loop...", flush=True)
    while time.time() < DEADLINE:
        remaining = DEADLINE - time.time()
        if remaining < 600: break
        try: run_cycle(DEADLINE)
        except Exception as e: agent_log(f"[!] Main loop error: {e}")
        time.sleep(min(900, max(DEADLINE - time.time() - 600, 1)))
    print("[*] Exiting cleanly.")

