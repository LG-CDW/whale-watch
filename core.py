"""
Whale Watch core engine.

Everything that does not depend on the user interface lives here, so the
Streamlit app (whale_watch.py) and the scheduled alert job (alerts_job.py)
share exactly the same logic:

  * market data from Yahoo Finance (free, usually ~15 min delayed)
  * Black-Scholes pricing and Greeks
  * the unusual-flow scanner, pattern detection and conviction score
  * next-day open-interest confirmation and the flag scorecard
  * dealer gamma exposure (GEX), gamma flip, walls, max pain
  * storage (local files, or a "data" branch in your GitHub repo)
  * Telegram / Discord alerts

For education only. Not financial advice.
"""
from __future__ import annotations

import base64
import json
import math
import os
import tempfile
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
RISK_FREE_RATE = 0.04
MIN_T = 1.0 / (365 * 24)  # floor of one hour so same-day Greeks stay finite

# Yahoo uses ^ prefixes for index options.
INDEX_SYMBOLS = {"SPX": "^SPX", "SPXW": "^SPX", "NDX": "^NDX", "NDXP": "^NDX",
                 "RUT": "^RUT", "VIX": "^VIX", "XSP": "^XSP", "DJX": "^DJI"}
PRICE_FALLBACK = {"^SPX": "^GSPC"}
# Underlyings where big out-of-the-money puts are usually portfolio insurance.
HEDGE_UNDERLYINGS = {"SPX", "SPXW", "XSP", "NDX", "NDXP", "RUT", "SPY", "QQQ", "IWM", "DIA"}
VOL_PRODUCTS = {"VIX", "UVXY", "VXX", "VIXY"}

# NYSE full-day closures and 1pm early closes. Update once a year.
MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
EARLY_CLOSES = {"2026-11-27", "2026-12-24", "2027-11-26"}

DEFAULT_SETTINGS = {
    "watchlist": ["SPX", "SPY", "QQQ", "NVDA", "TSLA"],
    "max_expirations": 3,
    "min_premium": 1_000_000,
    "min_premium_overrides": {"SPX": 2_000_000},
    "alert_min_score": 65,
    "alert_categories": ["Directional bet"],
    # ---- signal engine (paper trading) ----
    "signal_min_score": 70,
    "signal_categories": ["Directional bet"],
    "signal_target_pct": 50,
    "signal_stop_pct": 40,
    "signal_breakeven_after_pct": 30,
    "signal_size_usd": 1000,
    "signal_max_open": 10,
    "signal_min_price": 0.20,
    "signal_max_chase_pct": 25,
    "signal_last_entry_0dte": "14:30",
    "signal_exit_time": "15:30",
    "signal_max_hold_days": 5,
    "signal_alerts": True,
    "signal_exit_on_whale_exit": True,
    # ---- whale position tracking ----
    "whale_min_score": 70,
    "whale_alerts": True,
    "whale_max_tracked": 60,
}

CATEGORIES = ["Directional bet", "Seller-driven", "Spread / roll / combo",
              "Likely hedge", "May be closing", "Unclear"]


# ======================================================================= time
def now_et() -> datetime:
    return datetime.now(ET)


def market_status(now: datetime | None = None) -> tuple[str, str]:
    """Returns (status, detail). status is 'open' or 'closed'."""
    now = now or now_et()
    d = now.date().isoformat()
    if now.weekday() >= 5:
        return "closed", "Weekend"
    if d in MARKET_HOLIDAYS:
        return "closed", "Market holiday"
    close = dtime(13, 0) if d in EARLY_CLOSES else dtime(16, 0)
    if now.time() < dtime(9, 30):
        return "closed", "Pre-market"
    if now.time() >= close:
        return "closed", "After hours"
    return "open", "Early close 1pm ET" if d in EARLY_CLOSES else "Regular session"


def to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return pd.to_datetime(x).date()


def years_left(expiration, now: datetime | None = None) -> float:
    """Years until 4pm ET on the expiration date (0 once expired)."""
    now = now or now_et()
    e = to_date(expiration)
    close = datetime(e.year, e.month, e.day, 16, 0, tzinfo=ET)
    return max(0.0, (close - now).total_seconds() / (365.0 * 86400))


def base_symbol(symbol: str) -> str:
    return symbol.strip().upper().lstrip("^")


def yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return INDEX_SYMBOLS.get(s, s)


def is_index(symbol: str) -> bool:
    return yahoo_symbol(symbol).startswith("^")


# ======================================================================= math
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_greeks(spot: float, strike: float, years: float, iv_pct: float, opt_type: str) -> dict:
    """Black-Scholes price per share plus delta, gamma, theta (per day) and vega (per 1 vol point)."""
    call = opt_type == "C"
    intrinsic = max(0.0, spot - strike) if call else max(0.0, strike - spot)
    vol = (iv_pct or 0) / 100.0
    if years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        itm = intrinsic > 0
        return {"price": intrinsic, "delta": (1.0 if call else -1.0) if itm else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    t = max(years, MIN_T)
    sq = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + vol * vol / 2) * t) / sq
    d2 = d1 - sq
    disc = math.exp(-RISK_FREE_RATE * t)
    pdf = norm_pdf(d1)
    if call:
        price = spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = -spot * pdf * vol / (2 * math.sqrt(t)) - RISK_FREE_RATE * strike * disc * norm_cdf(d2)
    else:
        price = strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1
        theta = -spot * pdf * vol / (2 * math.sqrt(t)) + RISK_FREE_RATE * strike * disc * norm_cdf(-d2)
    return {"price": max(price, intrinsic if years <= 0 else 0.0), "delta": delta,
            "gamma": pdf / (spot * sq), "theta": theta / 365.0, "vega": spot * pdf * math.sqrt(t) / 100.0}


def option_value(spot, strike, years, iv_pct, opt_type) -> float:
    return bs_greeks(spot, strike, years, iv_pct, opt_type)["price"]


def gamma_array(spot: float, strikes: np.ndarray, years: np.ndarray, ivs: np.ndarray) -> np.ndarray:
    t = np.maximum(years, MIN_T)
    sq = ivs * np.sqrt(t)
    d1 = (np.log(spot / strikes) + (RISK_FREE_RATE + ivs ** 2 / 2) * t) / sq
    return np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (spot * sq)


# ======================================================================= formatting
def fmt_money(x, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    sign = ("-" if x < 0 else "+" if x > 0 else "") if signed else ("-" if x < 0 else "")
    a = abs(x)
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:,.0f}"


def contract_label(symbol, exp, strike, opt_type) -> str:
    e = to_date(exp)
    return f"{base_symbol(symbol)} {strike:g}{opt_type} {e:%m/%d/%y}"


def contract_key(symbol, exp, strike, opt_type) -> str:
    return f"{base_symbol(symbol)}|{to_date(exp).isoformat()}|{float(strike):g}|{opt_type}"


def parse_key(key: str):
    sym, exp, strike, typ = key.split("|")
    return sym, exp, float(strike), typ


# ======================================================================= market data
def fetch_price(symbol: str):
    ysym = yahoo_symbol(symbol)
    for s in [ysym] + ([PRICE_FALLBACK[ysym]] if ysym in PRICE_FALLBACK else []):
        t = yf.Ticker(s)
        try:
            p = t.fast_info.last_price
            if p and p > 0 and not math.isnan(p):
                return float(p)
        except Exception:
            pass
        try:
            h = t.history(period="5d", interval="1m")
            if h is not None and not h.empty:
                return float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass
    return None


def fetch_adv(symbol: str):
    """20-day average daily dollar volume of the stock. None for indexes."""
    if is_index(symbol):
        return None
    try:
        h = yf.Ticker(yahoo_symbol(symbol)).history(period="2mo")
        if h is None or h.empty:
            return None
        dv = (h["Close"] * h["Volume"]).dropna().tail(20)
        return float(dv.mean()) if len(dv) else None
    except Exception:
        return None


def fetch_expirations(symbol: str) -> list[str]:
    return list(yf.Ticker(yahoo_symbol(symbol)).options or [])


def fetch_chain(symbol: str, exp: str):
    ch = yf.Ticker(yahoo_symbol(symbol)).option_chain(exp)
    return ch.calls, ch.puts


def fetch_close_on(symbol: str, d: str):
    """Closing price of the underlying on a date (or the next trading day)."""
    try:
        start = to_date(d)
        ysym = yahoo_symbol(symbol)
        for s in [ysym] + ([PRICE_FALLBACK[ysym]] if ysym in PRICE_FALLBACK else []):
            h = yf.Ticker(s).history(start=start.isoformat(), end=(start + timedelta(days=6)).isoformat())
            if h is not None and not h.empty:
                return float(h["Close"].dropna().iloc[0])
    except Exception:
        pass
    return None


def _col(d: pd.DataFrame, name: str) -> pd.Series:
    if name in d.columns:
        return pd.to_numeric(d[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=d.index)


def chain_frame(symbol: str, exp: str, calls, puts, spot, now: datetime) -> pd.DataFrame:
    """Standardised table of every contract in one expiration."""
    parts = []
    dte = (to_date(exp) - now.date()).days
    for opt_type, d in (("C", calls), ("P", puts)):
        if d is None or len(d) == 0:
            continue
        f = pd.DataFrame({
            "Symbol": base_symbol(symbol),
            "Expiration": exp,
            "DTE": dte,
            "Strike": _col(d, "strike"),
            "Type": opt_type,
            "Volume": _col(d, "volume"),
            "OI": _col(d, "openInterest"),
            "Last": _col(d, "lastPrice"),
            "Bid": _col(d, "bid"),
            "Ask": _col(d, "ask"),
            "IV": (_col(d, "impliedVolatility") * 100).round(1),
        })
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    f["Premium"] = (f["Volume"] * f["Last"] * 100).round(0)
    noq = (f["Bid"] <= 0) | (f["Ask"] <= 0) | (f["Ask"] < f["Bid"])
    mid = (f["Bid"] + f["Ask"]) / 2
    f["Fill"] = np.select(
        [noq, f["Last"] >= f["Ask"] - 0.005, f["Last"] <= f["Bid"] + 0.005, f["Last"] >= mid],
        ["No quote", "At/above ask", "At/below bid", "Leans ask"], "Leans bid")
    if spot:
        f["OTM %"] = np.where(f["Type"] == "C", (f["Strike"] - spot) / spot, (spot - f["Strike"]) / spot) * 100
    else:
        f["OTM %"] = np.nan
    f["OTM %"] = f["OTM %"].round(2)
    return f


# ======================================================================= scanner
BUY_FILLS = ("At/above ask", "Leans ask")
SELL_FILLS = ("At/below bid", "Leans bid")


def detect_patterns(cands: pd.DataFrame, full: pd.DataFrame, spot, tol: float = 0.04, min_leg: int = 500,
                    unusual_x: float = 3.0) -> list[str]:
    """Flags contracts whose volume closely matches another *unusually busy* contract: likely legs
    of one trade. Both legs must trade at least `unusual_x` times the typical volume of nearby
    strikes, so coincidental matches on busy chains like SPX are ignored."""
    if full.empty:
        return [""] * len(cands)
    f = full.copy()
    near = f if not spot else f[(f["Strike"] - spot).abs() / spot <= 0.05]
    base = near[near["Volume"] > 0].groupby(["Expiration", "Type"])["Volume"].median()
    f["_base"] = [base.get((e, t), 0.0) for e, t in zip(f["Expiration"], f["Type"])]
    big = f[(f["Volume"] >= min_leg) & (f["Volume"] >= unusual_x * f["_base"])]
    if spot:
        big = big[(big["Strike"] - spot).abs() / spot <= 0.15]
    labels = []
    for _, r in cands.iterrows():
        v = r["Volume"]
        label = ""
        b = base.get((r["Expiration"], r["Type"]), 0.0)
        if v >= min_leg and v >= unusual_x * b and not big.empty:
            pool = big[((big["Volume"] - v).abs() <= max(tol * v, 25))
                       & ~((big["Expiration"] == r["Expiration"]) & (big["Strike"] == r["Strike"]) & (big["Type"] == r["Type"]))]
            width = (spot or r["Strike"]) * 0.05
            same = pool[(pool["Expiration"] == r["Expiration"]) & ((pool["Strike"] - r["Strike"]).abs() <= width)]
            if not same.empty:
                o = same.loc[(same["Volume"] - v).abs().idxmin()]
                if o["Type"] == r["Type"]:
                    label = f"Possible spread with {o['Strike']:g}{o['Type']}"
                elif o["Strike"] == r["Strike"]:
                    label = f"Possible straddle with {o['Strike']:g}{o['Type']}"
                else:
                    label = f"Possible combo with {o['Strike']:g}{o['Type']}"
            else:
                roll = pool[(pool["Strike"] == r["Strike"]) & (pool["Type"] == r["Type"])]
                if not roll.empty:
                    o = roll.loc[(roll["Volume"] - v).abs().idxmin()]
                    label = f"Possible roll with {to_date(o['Expiration']):%m/%d}"
        labels.append(label)
    return labels


def scan_symbol(symbol: str, max_expirations: int, min_premium: float, now: datetime | None = None):
    """Scan one underlying. Returns (DataFrame of big contracts, error text or None)."""
    now = now or now_et()
    sym = base_symbol(symbol)
    try:
        exps = fetch_expirations(sym)
    except Exception as e:
        return pd.DataFrame(), f"{sym}: couldn't load option expirations ({e})."
    exps = [e for e in exps if (to_date(e) - now.date()).days >= 0][:max_expirations]
    if not exps:
        return pd.DataFrame(), f"{sym}: no listed options found."
    spot = fetch_price(sym)
    adv = fetch_adv(sym)
    frames = []
    for e in exps:
        try:
            calls, puts = fetch_chain(sym, e)
        except Exception:
            continue
        f = chain_frame(sym, e, calls, puts, spot, now)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(), f"{sym}: option chains came back empty (Yahoo may be rate-limiting; try again)."
    full = pd.concat(frames, ignore_index=True)
    c = full[(full["Premium"] >= min_premium) & (full["Volume"] > 0)].copy()
    if c.empty:
        return c, None
    c["Spot"] = spot
    deltas, gammas = [], []
    for _, r in c.iterrows():
        g = bs_greeks(spot or r["Strike"], r["Strike"], years_left(r["Expiration"], now), r["IV"], r["Type"])
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
    c["Delta"] = np.round(deltas, 3)
    c["Gamma"] = gammas
    c["Hedge $"] = (c["Volume"] * c["Delta"].abs() * 100 * (spot or 0)).round(0)
    c["% ADV"] = (c["Hedge $"] / adv * 100).round(2) if adv else np.nan
    buy = c["Fill"].isin(BUY_FILLS)
    sell = c["Fill"].isin(SELL_FILLS)
    call = c["Type"] == "C"
    c["Bias"] = np.select([(call & buy) | (~call & sell), (call & sell) | (~call & buy)], ["Bullish", "Bearish"], "Unclear")
    c["Signed hedge $"] = np.select([c["Bias"] == "Bullish", c["Bias"] == "Bearish"], [c["Hedge $"], -c["Hedge $"]], 0.0)
    c["Pattern"] = detect_patterns(c, full, spot)
    c["Vol/OI"] = np.where(c["OI"] > 0, (c["Volume"] / c["OI"].replace(0, np.nan)).round(1), np.inf)
    c["Key"] = [contract_key(sym, r["Expiration"], r["Strike"], r["Type"]) for _, r in c.iterrows()]
    c["Contract"] = [contract_label(sym, r["Expiration"], r["Strike"], r["Type"]) for _, r in c.iterrows()]
    return c.reset_index(drop=True), None


def min_premium_for(symbol: str, settings: dict) -> float:
    o = settings.get("min_premium_overrides") or {}
    return float(o.get(base_symbol(symbol), settings.get("min_premium", 1_000_000)))


def _prior_entry(entries: list, today: str):
    prior = [e for e in entries if e["d"] < today]
    return max(prior, key=lambda e: e["d"]) if prior else None


def oi_verdict(oi_change: float, volume: float) -> tuple[str, bool | None]:
    if volume <= 0:
        return "n/a", None
    ratio = oi_change / volume
    if ratio >= 0.5:
        return f"Confirmed opened (+{oi_change:,.0f} OI)", True
    if ratio <= -0.3:
        return f"Closed out ({oi_change:,.0f} OI)", False
    return f"Unconfirmed ({oi_change:+,.0f} OI)", None


def score_flow(df: pd.DataFrame, history: dict, today: str, settings: dict) -> pd.DataFrame:
    """Adds Category, Score (0-100), Confirmation, Days seen and a plain-English Why column."""
    if df is None or df.empty:
        return df
    df = df.copy()
    cats, scores, whys, confs, seen = [], [], [], [], []
    for _, r in df.iterrows():
        reasons, pts = [], 0.0
        minp = min_premium_for(r["Symbol"], settings)
        entries = history.get(r["Key"], [])
        prior = _prior_entry(entries, today)
        days = len({e["d"] for e in entries} | {today})
        confirmed = None
        conf_text = ""
        if prior:
            conf_text, confirmed = oi_verdict(r["OI"] - prior["oi"], prior["v"])
            conf_text = f"{conf_text} since {prior['d'][5:]}"
        elif r["DTE"] <= 0:
            conf_text = "Expires today, can't be confirmed"
        else:
            conf_text = "First seen today, check tomorrow"

        # --- size
        ratio = max(r["Premium"] / max(minp, 1), 1e-9)
        pts += float(np.clip(12 * math.log10(ratio) + 12, 0, 30))
        reasons.append(f"Premium {fmt_money(r['Premium'])}")
        # --- new money
        vo = r["Vol/OI"]
        if vo >= 10:
            pts += 20
        elif vo >= 3:
            pts += 15
        elif vo >= 1:
            pts += 10
        if vo >= 1:
            reasons.append("Volume above open interest" if math.isinf(vo) else f"Volume {vo:.1f}x open interest")
        # --- aggression
        fill_pts = {"At/above ask": 20, "Leans ask": 10, "At/below bid": 8, "Leans bid": 4}.get(r["Fill"], 3)
        pts += fill_pts
        reasons.append(r["Fill"].replace("At/above ask", "Paid at/above ask").replace("At/below bid", "Sold at/below bid"))
        # --- urgency
        otm = r["OTM %"] if not pd.isna(r["OTM %"]) else 0
        if r["DTE"] <= 2 and otm >= 0.5:
            pts += 10
            reasons.append(f"{int(r['DTE'])} day(s) left")
        if otm >= 3:
            pts += 5
            reasons.append(f"{otm:.1f}% out of the money")
        # --- market impact (single stocks)
        adv = r.get("% ADV", np.nan)
        if adv is not None and not pd.isna(adv):
            if adv >= 5:
                pts += 15
            elif adv >= 1:
                pts += 7
            if adv >= 1:
                reasons.append(f"Hedging ≈ {adv:.1f}% of daily volume")
        # --- confirmation and repetition
        if confirmed is True:
            pts += 15
            reasons.append("Earlier flag confirmed as new position")
        elif confirmed is False:
            pts -= 10
        if days >= 2:
            pts += min(10, 5 * (days - 1))
            reasons.append(f"Flagged on {days} days")

        # --- category (penalties for likely non-directional flow)
        sym = r["Symbol"]
        if r["Pattern"]:
            cat = "Spread / roll / combo"
            pts -= 25
            reasons.append(r["Pattern"])
        elif (sym in HEDGE_UNDERLYINGS and r["Type"] == "P" and otm >= 2 and r["DTE"] >= 7) or \
                (sym in VOL_PRODUCTS and r["Type"] == "C" and r["Fill"] in BUY_FILLS):
            cat = "Likely hedge"
            pts -= 20
            reasons.append("Looks like portfolio insurance")
        elif vo < 1 and confirmed is not True:
            cat = "May be closing"
            pts -= 15
        elif r["Fill"] in SELL_FILLS:
            cat = "Seller-driven"
        elif r["Fill"] in BUY_FILLS:
            cat = "Directional bet"
        else:
            cat = "Unclear"
        cats.append(cat)
        scores.append(int(np.clip(round(pts), 0, 100)))
        whys.append(", ".join(x for x in reasons if x))
        confs.append(conf_text)
        seen.append(days)
    df["Category"] = cats
    df["Score"] = scores
    df["Confirmation"] = confs
    df["Days seen"] = seen
    df["Why"] = whys
    return df.sort_values(["Score", "Premium"], ascending=False).reset_index(drop=True)


# ======================================================================= history
def record_history(history: dict, df: pd.DataFrame, today: str, keep_days: int = 45, max_keys: int = 6000) -> dict:
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            entry = {"d": today, "v": int(r["Volume"]), "oi": int(r["OI"]), "px": round(float(r["Last"]), 3),
                     "fill": r["Fill"], "bias": r["Bias"], "score": int(r.get("Score", 0)),
                     "cat": r.get("Category", ""), "spot": float(r["Spot"]) if r["Spot"] else None}
            entries = [e for e in history.get(r["Key"], []) if e["d"] != today]
            entries.append(entry)
            history[r["Key"]] = entries
    cutoff = (to_date(today) - timedelta(days=keep_days)).isoformat()
    for k in list(history):
        history[k] = [e for e in history[k] if e["d"] >= cutoff]
        if not history[k]:
            del history[k]
    if len(history) > max_keys:
        ranked = sorted(history, key=lambda k: max(e["d"] for e in history[k]), reverse=True)
        history = {k: history[k] for k in ranked[:max_keys]}
    return history


def merge_history(remote: dict, local: dict) -> dict:
    out = {k: list(v) for k, v in (remote or {}).items()}
    for k, entries in (local or {}).items():
        byd = {e["d"]: e for e in out.get(k, [])}
        for e in entries:
            byd[e["d"]] = e
        out[k] = sorted(byd.values(), key=lambda e: e["d"])
    return out


def confirm_report(history: dict, today: str, chain_fn, close_fn, price_fn,
                   lookback_days: int = 10, min_score: int = 0, max_chains: int = 30):
    """For contracts flagged on earlier days: did open interest rise, and how did they do since?"""
    cutoff = (to_date(today) - timedelta(days=lookback_days)).isoformat()
    flags = []
    for key, entries in history.items():
        earlier = [e for e in entries if cutoff <= e["d"] < today and e.get("score", 0) >= min_score]
        if earlier:
            flags.append((key, max(earlier, key=lambda e: e["d"]), entries))
    if not flags:
        return pd.DataFrame(), {}
    flags.sort(key=lambda x: x[1]["d"], reverse=True)

    chain_cache, rows, fetched = {}, [], 0
    for key, flag, entries in flags:
        sym, exp, strike, typ = parse_key(key)
        later = [e for e in entries if e["d"] > flag["d"]]
        oi_now = px_now = None
        source = ""
        expired = to_date(exp) < to_date(today)
        if expired:
            settle = close_fn(sym, exp)
            if settle is not None:
                px_now = max(0.0, settle - strike) if typ == "C" else max(0.0, strike - settle)
                source = f"Expired, settled vs {settle:,.2f}"
        else:
            ck = (sym, exp)
            if ck not in chain_cache and fetched < max_chains:
                fetched += 1
                try:
                    calls, puts = chain_fn(sym, exp)
                    chain_cache[ck] = (calls, puts)
                except Exception:
                    chain_cache[ck] = None
            ch = chain_cache.get(ck)
            if ch:
                d = ch[0] if typ == "C" else ch[1]
                row = d[np.isclose(pd.to_numeric(d["strike"], errors="coerce"), strike)] if d is not None and len(d) else None
                if row is not None and len(row):
                    row = row.iloc[0]
                    oi_now = float(row.get("openInterest", 0) or 0)
                    bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
                    px_now = (bid + ask) / 2 if bid > 0 and ask > 0 else float(row.get("lastPrice", 0) or 0)
                    source = "Live chain"
        if later:
            oi_after = later[0]["oi"]
        else:
            oi_after = oi_now
        if oi_after is not None:
            verdict, ok = oi_verdict(oi_after - flag["oi"], flag["v"])
        else:
            verdict, ok = ("Expired before it could be checked" if expired else "No data yet"), None
        ret = None
        if px_now is not None and flag["px"] > 0:
            raw = (px_now - flag["px"]) / flag["px"]
            ret = -raw if flag.get("bias") and flag["fill"] in SELL_FILLS else raw
        rows.append({
            "Flagged": flag["d"],
            "Contract": contract_label(sym, exp, strike, typ),
            "Category": flag.get("cat", ""),
            "Score": flag.get("score", 0),
            "Bias": flag.get("bias", ""),
            "Volume": flag["v"],
            "OI then": flag["oi"],
            "OI after": oi_after,
            "Verdict": verdict,
            "Opened": ok,
            "_checked": oi_after is not None,
            "Price then": flag["px"],
            "Price now": None if px_now is None else round(px_now, 2),
            "Follow return %": None if ret is None else round(ret * 100, 1),
            "Source": source,
        })
    rep = pd.DataFrame(rows)
    checked = rep[rep["_checked"]]
    directional = rep[(rep["Category"] == "Directional bet") & rep["Follow return %"].notna()]
    summary = {
        "flags": len(rep),
        "checked": len(checked),
        "opened_pct": float((checked["Opened"] == True).mean() * 100) if len(checked) else None,  # noqa: E712
        "dir_count": len(directional),
        "dir_avg_return": float(directional["Follow return %"].mean()) if len(directional) else None,
        "dir_win_rate": float((directional["Follow return %"] > 0).mean() * 100) if len(directional) else None,
    }
    return rep.drop(columns=["_checked"]), summary


# ======================================================================= dealer positioning
def dealer_positioning(symbol: str, max_days: int = 45, max_expirations: int = 10, now: datetime | None = None):
    """Estimate dealer gamma exposure using the common convention that dealers are
    long the calls and short the puts customers trade. It is a model, not a fact."""
    now = now or now_et()
    sym = base_symbol(symbol)
    spot = fetch_price(sym)
    if not spot:
        return None, f"{sym}: couldn't get a price."
    try:
        exps = [e for e in fetch_expirations(sym) if 0 <= (to_date(e) - now.date()).days <= max_days][:max_expirations]
    except Exception as e:
        return None, f"{sym}: couldn't load expirations ({e})."
    if not exps:
        return None, f"{sym}: no expirations within {max_days} days."
    frames = []
    for e in exps:
        try:
            calls, puts = fetch_chain(sym, e)
        except Exception:
            continue
        f = chain_frame(sym, e, calls, puts, spot, now)
        if not f.empty:
            frames.append(f)
    if not frames:
        return None, f"{sym}: option chains came back empty."
    full = pd.concat(frames, ignore_index=True)
    full["T"] = [years_left(e, now) for e in full["Expiration"]]
    g = full[(full["OI"] > 0) & (full["IV"] > 1) & (full["IV"] < 500)].copy()
    if g.empty:
        return None, f"{sym}: no usable open interest."
    sign = np.where(g["Type"] == "C", 1.0, -1.0)
    strikes, T, iv, oi = g["Strike"].to_numpy(float), g["T"].to_numpy(float), (g["IV"] / 100).to_numpy(float), g["OI"].to_numpy(float)

    def total_gex(s):
        return float(np.sum(gamma_array(s, strikes, T, iv) * oi * 100 * s * s * 0.01 * sign))

    g["GEX"] = gamma_array(spot, strikes, T, iv) * oi * 100 * spot * spot * 0.01 * sign
    by = g.groupby(["Strike", "Type"])["GEX"].sum().unstack(fill_value=0.0)
    by = by.reindex(columns=["C", "P"], fill_value=0.0)
    by["Net"] = by["C"] + by["P"]
    by = by.reset_index()

    grid = np.linspace(spot * 0.88, spot * 1.12, 97)
    curve = np.array([total_gex(s) for s in grid])
    flip = None
    sc = np.where(np.diff(np.sign(curve)) != 0)[0]
    if len(sc):
        i = sc[np.argmin(np.abs(grid[sc] - spot))]
        x0, x1, y0, y1 = grid[i], grid[i + 1], curve[i], curve[i + 1]
        flip = float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)

    call_wall = float(by.loc[by["C"].idxmax(), "Strike"]) if (by["C"] > 0).any() else None
    put_wall = float(by.loc[by["P"].idxmin(), "Strike"]) if (by["P"] < 0).any() else None

    near = sorted(full["Expiration"].unique(), key=lambda e: to_date(e))[0]
    n = full[full["Expiration"] == near]
    ks = np.sort(n["Strike"].unique())
    pain = None
    if len(ks):
        nc, npt = n[n["Type"] == "C"], n[n["Type"] == "P"]
        costs = [float((nc["OI"] * np.maximum(0, k - nc["Strike"])).sum() + (npt["OI"] * np.maximum(0, npt["Strike"] - k)).sum()) for k in ks]
        pain = float(ks[int(np.argmin(costs))])
    em = None
    if len(ks):
        atm = ks[np.argmin(np.abs(ks - spot))]
        legs = n[n["Strike"] == atm]
        if len(legs) >= 2:
            mids = np.where((legs["Bid"] > 0) & (legs["Ask"] > 0), (legs["Bid"] + legs["Ask"]) / 2, legs["Last"])
            em = float(np.sum(mids))
    calls_oi = full.loc[full["Type"] == "C", "OI"].sum()
    puts_oi = full.loc[full["Type"] == "P", "OI"].sum()
    calls_v = full.loc[full["Type"] == "C", "Volume"].sum()
    puts_v = full.loc[full["Type"] == "P", "Volume"].sum()
    total = total_gex(spot)
    return {
        "symbol": sym, "spot": spot, "total_gex": total, "flip": flip,
        "call_wall": call_wall, "put_wall": put_wall, "max_pain": pain, "max_pain_exp": near,
        "expected_move": em, "expected_move_exp": near,
        "pcr_oi": float(puts_oi / calls_oi) if calls_oi else None,
        "pcr_vol": float(puts_v / calls_v) if calls_v else None,
        "expirations": exps, "by_strike": by,
        "curve": pd.DataFrame({"Price": grid, "GEX": curve}),
        "regime": "Positive gamma" if total >= 0 else "Negative gamma",
        "as_of": now.isoformat(timespec="minutes"),
    }, None


# ======================================================================= storage
class LocalStorage:
    """JSON files in a folder. Used on your own computer or when cloud sync is off."""
    kind = "local"

    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def describe(self) -> str:
        return "Saved on this machine only"

    def get_json(self, name: str, default=None):
        p = os.path.join(self.folder, name)
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def put_json(self, name: str, data, merge=None, message: str | None = None) -> bool:
        p = os.path.join(self.folder, name)
        fd, tmp = tempfile.mkstemp(dir=self.folder, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), default=str)
        os.replace(tmp, p)
        return True


class GitHubStorage:
    """JSON files on a separate 'data' branch of your GitHub repo, so history, trades and
    settings survive restarts and are shared by the app and the scheduled alert job.
    The app's code lives on 'main', so data commits never redeploy the app."""
    kind = "github"
    API = "https://api.github.com"

    def __init__(self, token: str, repo: str, branch: str = "data", session=None):
        self.repo, self.branch = repo, branch
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28",
                               "Accept": "application/vnd.github+json"})
        self._branch_ok = False

    def describe(self) -> str:
        return f"Synced to github.com/{self.repo} (branch '{self.branch}')"

    def _url(self, name):
        return f"{self.API}/repos/{self.repo}/contents/data/{name}"

    def _ensure_branch(self):
        if self._branch_ok:
            return
        r = self.s.get(f"{self.API}/repos/{self.repo}/branches/{self.branch}", timeout=20)
        if r.status_code == 404:
            info = self.s.get(f"{self.API}/repos/{self.repo}", timeout=20)
            info.raise_for_status()
            default = info.json()["default_branch"]
            ref = self.s.get(f"{self.API}/repos/{self.repo}/git/ref/heads/{default}", timeout=20)
            ref.raise_for_status()
            c = self.s.post(f"{self.API}/repos/{self.repo}/git/refs",
                            json={"ref": f"refs/heads/{self.branch}", "sha": ref.json()["object"]["sha"]}, timeout=20)
            if c.status_code not in (201, 422):
                c.raise_for_status()
        elif r.status_code != 200:
            r.raise_for_status()
        self._branch_ok = True

    def check(self) -> tuple[bool, str]:
        try:
            self._ensure_branch()
            return True, "Connected"
        except Exception as e:
            return False, str(e)

    def get_json(self, name: str, default=None):
        try:
            r = self.s.get(self._url(name), params={"ref": self.branch},
                           headers={"Accept": "application/vnd.github.raw+json"}, timeout=30)
            if r.status_code == 200:
                return json.loads(r.content.decode("utf-8"))
        except Exception:
            pass
        return default

    def _sha(self, name):
        r = self.s.get(self._url(name), params={"ref": self.branch}, timeout=30)
        if r.status_code == 200:
            return r.json().get("sha")
        return None

    def put_json(self, name: str, data, merge=None, message: str | None = None) -> bool:
        self._ensure_branch()
        for attempt in range(3):
            if attempt and merge:
                data = merge(self.get_json(name, None), data)
            body = {"message": message or f"Update {name}", "branch": self.branch,
                    "content": base64.b64encode(json.dumps(data, separators=(",", ":"), default=str).encode()).decode()}
            sha = self._sha(name)
            if sha:
                body["sha"] = sha
            r = self.s.put(self._url(name), json=body, timeout=60)
            if r.status_code in (200, 201):
                return True
            if r.status_code not in (409, 422):
                return False
        return False


def make_storage(token: str | None, repo: str | None, local_dir: str):
    if token and repo:
        return GitHubStorage(token, repo)
    return LocalStorage(local_dir)


def load_settings(storage) -> dict:
    s = dict(DEFAULT_SETTINGS)
    s.update(storage.get_json("settings.json", {}) or {})
    return s


# ======================================================================= alerts
def alert_text(r) -> str:
    vo = "new" if math.isinf(r["Vol/OI"]) else f"{r['Vol/OI']:.1f}x OI"
    lines = [
        f"🐋 {r['Contract']}  |  Score {r['Score']}",
        f"{r['Category']}, {r['Bias'].lower()}",
        f"Premium {fmt_money(r['Premium'])}, {int(r['Volume']):,} contracts ({vo})",
        f"{r['Fill']}, {r['OTM %']:.1f}% OTM, {int(r['DTE'])} DTE",
        f"Spot {r['Spot']:,.2f}" if r["Spot"] else "",
        f"Why: {r['Why']}",
    ]
    return "\n".join(x for x in lines if x)


def send_alert(text: str, telegram_token=None, telegram_chat=None, discord_url=None) -> list[str]:
    errors = []
    if telegram_token and telegram_chat:
        try:
            r = requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                              json={"chat_id": telegram_chat, "text": text, "disable_web_page_preview": True}, timeout=20)
            if r.status_code != 200:
                errors.append(f"Telegram: {r.status_code} {r.text[:120]}")
        except Exception as e:
            errors.append(f"Telegram: {e}")
    if discord_url:
        try:
            r = requests.post(discord_url, json={"content": text[:1990]}, timeout=20)
            if r.status_code not in (200, 204):
                errors.append(f"Discord: {r.status_code} {r.text[:120]}")
        except Exception as e:
            errors.append(f"Discord: {e}")
    if not ((telegram_token and telegram_chat) or discord_url):
        errors.append("No alert channel configured.")
    return errors


# ======================================================================= signal engine (paper trading)
def _hm(text: str) -> dtime:
    h, m = str(text).split(":")
    return dtime(int(h), int(m))


def quote_from_chain(calls, puts, strike: float, opt_type: str):
    d = calls if opt_type == "C" else puts
    if d is None or len(d) == 0:
        return None
    row = d[np.isclose(pd.to_numeric(d["strike"], errors="coerce"), strike)]
    if not len(row):
        return None
    row = row.iloc[0]
    return {"bid": float(row.get("bid", 0) or 0), "ask": float(row.get("ask", 0) or 0),
            "last": float(row.get("lastPrice", 0) or 0),
            "oi": float(row.get("openInterest", 0) or 0), "volume": float(row.get("volume", 0) or 0)}


def make_quote_fn(chain_fn):
    cache = {}

    def quote(sym, exp, strike, opt_type):
        k = (sym, exp)
        if k not in cache:
            try:
                cache[k] = chain_fn(sym, exp)
            except Exception:
                cache[k] = None
        ch = cache[k]
        return quote_from_chain(ch[0], ch[1], strike, opt_type) if ch else None
    return quote


def open_signals(flow: pd.DataFrame, signals: list, settings: dict, now: datetime | None = None):
    """Opens a paper position following each qualifying whale trade. Returns (signals, new ones)."""
    now = now or now_et()
    new = []
    if flow is None or flow.empty or market_status(now)[0] != "open":
        return signals, new
    today = now.date().isoformat()
    open_now = [x for x in signals if x["status"] == "open"]
    open_keys = {x["key"] for x in open_now}
    taken_today = {x["key"] for x in signals if x["entry_time"][:10] == today}
    cands = flow[(flow["Score"] >= settings["signal_min_score"])
                 & flow["Category"].isin(settings["signal_categories"])].sort_values("Score", ascending=False)
    last_0dte, exit_t = _hm(settings["signal_last_entry_0dte"]), _hm(settings["signal_exit_time"])
    for _, r in cands.iterrows():
        if len(open_now) + len(new) >= int(settings["signal_max_open"]):
            break
        if r["Key"] in open_keys or r["Key"] in taken_today:
            continue
        if r["DTE"] <= 0 and (now.time() >= last_0dte or now.time() >= exit_t):
            continue
        if r["Fill"] not in BUY_FILLS:  # only follow buyers: we buy the same contract
            continue
        quoted = r["Bid"] > 0 and r["Ask"] > 0
        entry = float(r["Ask"] if quoted else r["Last"])
        if entry < float(settings["signal_min_price"]):
            continue
        if r["Last"] > 0 and entry > r["Last"] * (1 + settings["signal_max_chase_pct"] / 100):
            continue
        qty = max(1, int(float(settings["signal_size_usd"]) // (entry * 100)))
        sig = {
            "id": f"{r['Key']}|{now:%Y%m%d%H%M}", "key": r["Key"], "contract": r["Contract"],
            "symbol": r["Symbol"], "exp": str(r["Expiration"]), "strike": float(r["Strike"]), "type": r["Type"],
            "score": int(r["Score"]), "category": r["Category"], "bias": r["Bias"], "why": r["Why"],
            "whale_premium": float(r["Premium"]), "entry_time": now.isoformat(timespec="minutes"),
            "entry_px": round(entry, 2), "qty": qty,
            "target_px": round(entry * (1 + settings["signal_target_pct"] / 100), 2),
            "stop_px": round(entry * (1 - settings["signal_stop_pct"] / 100), 2),
            "breakeven_armed": False, "last_px": round(entry, 2), "max_px": round(entry, 2),
            "last_time": now.isoformat(timespec="minutes"), "spot_at_entry": r["Spot"],
            "status": "open", "exit_time": None, "exit_px": None, "exit_reason": None, "pnl": None, "ret_pct": None,
        }
        new.append(sig)
        taken_today.add(r["Key"])
    return signals + new, new


def _close(sig: dict, px: float, reason: str, now: datetime):
    sig.update({"status": "closed", "exit_px": round(px, 2), "exit_reason": reason,
                "exit_time": now.isoformat(timespec="minutes"),
                "pnl": round((px - sig["entry_px"]) * 100 * sig["qty"], 2),
                "ret_pct": round((px - sig["entry_px"]) / sig["entry_px"] * 100, 1) if sig["entry_px"] else None})


def update_signals(signals: list, quote_fn, close_fn, settings: dict, now: datetime | None = None):
    """Marks open signals to market and closes any that hit an exit rule. Returns (signals, exits)."""
    now = now or now_et()
    exits = []
    is_open = market_status(now)[0] == "open"
    exit_t = _hm(settings["signal_exit_time"])
    be_pct = float(settings.get("signal_breakeven_after_pct") or 0)
    for sig in signals:
        if sig["status"] != "open":
            continue
        exp_d = to_date(sig["exp"])
        after_close = now.time() >= dtime(16, 0)
        if exp_d < now.date() or (exp_d == now.date() and after_close):
            settle = close_fn(sig["symbol"], sig["exp"])
            if settle is None:
                continue
            k = sig["strike"]
            _close(sig, max(0.0, settle - k) if sig["type"] == "C" else max(0.0, k - settle), "Expired", now)
            exits.append(sig)
            continue
        q = quote_fn(sig["symbol"], sig["exp"], sig["strike"], sig["type"])
        if not q:
            continue
        quoted = q["bid"] > 0 and q["ask"] > 0
        mark = (q["bid"] + q["ask"]) / 2 if quoted else q["last"]
        sell_px = q["bid"] if quoted else q["last"]
        sig["last_px"] = round(mark, 2)
        sig["max_px"] = round(max(sig.get("max_px") or 0, mark), 2)
        sig["last_time"] = now.isoformat(timespec="minutes")
        if not is_open:
            continue
        if be_pct > 0 and not sig.get("breakeven_armed") and sig["max_px"] >= sig["entry_px"] * (1 + be_pct / 100):
            sig["breakeven_armed"] = True
            sig["stop_px"] = max(sig["stop_px"], sig["entry_px"])
        held = (now.date() - to_date(sig["entry_time"][:10])).days
        reason = None
        if sell_px >= sig["target_px"]:
            reason = "Target hit"
        elif sell_px <= sig["stop_px"]:
            reason = "Breakeven stop" if sig.get("breakeven_armed") and sig["stop_px"] >= sig["entry_px"] else "Stop hit"
        elif exp_d == now.date() and now.time() >= exit_t:
            reason = "Time exit (expiration day)"
        elif held >= int(settings["signal_max_hold_days"]) and now.time() >= exit_t:
            reason = "Max hold reached"
        if reason:
            _close(sig, sell_px, reason, now)
            exits.append(sig)
    return signals, exits


def merge_signals(remote: list, local: list) -> list:
    byid = {x["id"]: x for x in (remote or [])}
    for x in local or []:
        cur = byid.get(x["id"])
        if cur is None:
            byid[x["id"]] = x
        elif cur["status"] == "closed" and x["status"] != "closed":
            continue
        elif x["status"] == "closed" and cur["status"] != "closed":
            byid[x["id"]] = x
        elif (x.get("last_time") or "") >= (cur.get("last_time") or ""):
            byid[x["id"]] = x
    return sorted(byid.values(), key=lambda s: s["entry_time"])


def signal_stats(signals: list) -> dict:
    closed = [x for x in signals if x["status"] == "closed" and x["pnl"] is not None]
    if not closed:
        return {"trades": 0}
    pnl = np.array([x["pnl"] for x in closed])
    rets = np.array([x["ret_pct"] or 0 for x in closed])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    curve = sorted(closed, key=lambda x: x["exit_time"])
    return {
        "trades": len(closed), "win_rate": float((pnl > 0).mean() * 100), "avg_ret": float(rets.mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "best": float(rets.max()), "worst": float(rets.min()),
        "equity": pd.DataFrame({"Time": [pd.to_datetime(x["exit_time"]) for x in curve],
                                "Equity": np.cumsum([x["pnl"] for x in curve])}),
    }


def signal_entry_text(sig: dict) -> str:
    return (f"🟢 ENTRY (paper) {sig['contract']}\n"
            f"Buy {sig['qty']} @ ${sig['entry_px']:.2f}  |  score {sig['score']}\n"
            f"Target ${sig['target_px']:.2f}  |  Stop ${sig['stop_px']:.2f}\n"
            f"Whale: {fmt_money(sig['whale_premium'])} {sig['bias'].lower()}\n"
            f"Why: {sig['why']}")


def signal_exit_text(sig: dict) -> str:
    icon = "✅" if (sig["pnl"] or 0) > 0 else "🔴"
    return (f"{icon} EXIT (paper) {sig['contract']}: {sig['exit_reason']}\n"
            f"Sold @ ${sig['exit_px']:.2f} (entry ${sig['entry_px']:.2f})  |  {sig['ret_pct']:+.1f}%  |  {fmt_money(sig['pnl'], signed=True)}")


# ======================================================================= whale position tracking
# Infers what the whale did after the big print, from public footprints:
#   entry confirmed  -> open interest rose overnight by at least half the whale's size
#   exit             -> open interest later fell by a big share of the confirmed position;
#                       the exit price is the contract's price on the day it happened
#   exit warning     -> heavy selling at the bid in that contract today (confirmed tomorrow)
# Open interest is shared by every trader in the contract, so these are estimates.
WHALE_ACTIVE = ("Unconfirmed", "Holding", "Partly exited")


def fill_side(last: float, bid: float, ask: float) -> str:
    if bid <= 0 or ask <= 0 or ask < bid:
        return "No quote"
    if last >= ask - 0.005:
        return "At/above ask"
    if last <= bid + 0.005:
        return "At/below bid"
    return "Leans ask" if last >= (bid + ask) / 2 else "Leans bid"


def _w_event(w, now, kind, text, contracts=None, px=None):
    w["events"].append({"t": now.isoformat(timespec="minutes"), "kind": kind, "text": text,
                        "contracts": contracts, "px": None if px is None else round(px, 2)})
    return {"whale": w, "kind": kind, "text": text}


def _w_pnl(w):
    held = w["est_contracts"] if w["est_contracts"] is not None else w["size"]
    realized = sum((e["px"] - w["entry_px"]) * 100 * e["contracts"] for e in w["events"]
                   if e["kind"] in ("exit", "expired") and e["contracts"] and e["px"] is not None)
    rem = max(0, held - w["exited"])
    unreal = rem * (w["last_px"] - w["entry_px"]) * 100 if w.get("last_px") is not None else 0
    w["pnl_est"] = round(realized + unreal, 0)
    cost = w["entry_px"] * 100 * held
    w["ret_est"] = round(w["pnl_est"] / cost * 100, 1) if cost else None
    sold = [e for e in w["events"] if e["kind"] in ("exit", "expired") and e["contracts"]]
    n = sum(e["contracts"] for e in sold)
    w["avg_exit_px"] = round(sum(e["px"] * e["contracts"] for e in sold) / n, 2) if n else None


def open_whales(flow: pd.DataFrame, whales: list, settings: dict, now: datetime | None = None):
    """Starts tracking each high-score directional whale. A later buy in the same contract is an add-on."""
    now = now or now_et()
    events = []
    if flow is None or flow.empty:
        return whales, events
    today = now.date().isoformat()
    active = {w["key"]: w for w in whales if w["status"] in WHALE_ACTIVE}
    started_today = {w["key"] for w in whales if w["entry_date"] == today}
    cands = flow[(flow["Score"] >= settings["whale_min_score"]) & (flow["Category"] == "Directional bet")]
    for _, r in cands.sort_values("Score", ascending=False).iterrows():
        k = r["Key"]
        obs = {"d": today, "oi": float(r["OI"]), "vol": float(r["Volume"]), "last": float(r["Last"]),
               "bid": float(r["Bid"]), "ask": float(r["Ask"]), "fill": r["Fill"]}
        if k in active:
            w = active[k]
            if w["entry_date"] != today and not w.get("pending_add") and r["Fill"] in BUY_FILLS:
                w["pending_add"] = {"d": today, "v": float(r["Volume"]), "oi": float(r["OI"]), "px": float(r["Last"])}
                events.append(_w_event(w, now, "add", f"Bought more: ~{int(r['Volume']):,} @ ${r['Last']:.2f} (confirm tomorrow)",
                                       int(r["Volume"]), float(r["Last"])))
                w["updated"] = now.isoformat(timespec="seconds")
            continue
        if k in started_today or len([w for w in whales if w["status"] in WHALE_ACTIVE]) >= int(settings["whale_max_tracked"]):
            continue
        w = {"id": f"{k}|{today}", "key": k, "contract": r["Contract"], "symbol": r["Symbol"], "exp": str(r["Expiration"]),
             "strike": float(r["Strike"]), "type": r["Type"], "bias": r["Bias"], "score": int(r["Score"]), "why": r["Why"],
             "entry_date": today, "entry_time": now.isoformat(timespec="minutes"), "entry_px": round(float(r["Last"]), 2),
             "size": int(r["Volume"]), "premium": float(r["Premium"]), "oi_at_flag": float(r["OI"]),
             "est_contracts": None, "exited": 0, "status": "Unconfirmed", "last_px": round(float(r["Last"]), 2),
             "pnl_est": 0.0, "ret_est": 0.0, "avg_exit_px": None, "obs": [obs], "events": [], "hint_day": None,
             "pending_add": None, "updated": now.isoformat(timespec="seconds")}
        events.append(_w_event(w, now, "entry", f"Whale bought ~{w['size']:,} @ ${w['entry_px']:.2f} "
                                                f"({fmt_money(w['premium'])}), score {w['score']}", w["size"], w["entry_px"]))
        whales.append(w)
        started_today.add(k)
        active[k] = w
    return whales, events


def update_whales(whales: list, quote_fn, close_fn, now: datetime | None = None):
    """Daily footprint check for every tracked whale. Returns (whales, new events)."""
    now = now or now_et()
    today = now.date().isoformat()
    events = []
    for w in whales:
        if w["status"] not in WHALE_ACTIVE:
            continue
        exp_d = to_date(w["exp"])
        if exp_d < now.date() or (exp_d == now.date() and now.time() >= dtime(16, 0)):
            settle = close_fn(w["symbol"], w["exp"])
            if settle is None:
                continue
            px = max(0.0, settle - w["strike"]) if w["type"] == "C" else max(0.0, w["strike"] - settle)
            held = w["est_contracts"] if w["est_contracts"] is not None else w["size"]
            rem = max(0, held - w["exited"])
            w["last_px"] = round(px, 2)
            if rem:
                note = " (0DTE, so whether they held to the end can't be confirmed)" if w["est_contracts"] is None else ""
                events.append(_w_event(w, now, "expired", f"Expired at ${px:.2f} with ~{rem:,} still held{note}", rem, px))
                w["exited"] += rem
            w["status"] = "Expired"
            _w_pnl(w)
            w["updated"] = now.isoformat(timespec="seconds")
            continue

        q = quote_fn(w["symbol"], w["exp"], w["strike"], w["type"])
        if not q:
            continue
        quoted = q["bid"] > 0 and q["ask"] > 0
        w["last_px"] = round((q["bid"] + q["ask"]) / 2 if quoted else q["last"], 2)
        obs = {"d": today, "oi": q["oi"], "vol": q["volume"], "last": q["last"], "bid": q["bid"], "ask": q["ask"],
               "fill": fill_side(q["last"], q["bid"], q["ask"])}
        w["obs"] = [o for o in w["obs"] if o["d"] != today] + [obs]
        prev = [o for o in w["obs"] if o["d"] < today]
        prev = prev[-1] if prev else None

        if prev is not None:
            if w["est_contracts"] is None:
                inc = q["oi"] - w["oi_at_flag"]
                if inc >= 0.5 * w["size"]:
                    w["est_contracts"] = int(min(w["size"], inc))
                    w["status"] = "Holding"
                    events.append(_w_event(w, now, "confirmed",
                                           f"Confirmed: open interest rose {inc:+,.0f} overnight, so ~{w['est_contracts']:,} are being held"))
                else:
                    w["status"] = "Not held overnight"
                    events.append(_w_event(w, now, "not_held",
                                           f"Open interest only moved {inc:+,.0f}: a day trade or closing an older position"))
            else:
                if w.get("pending_add") and today > w["pending_add"]["d"]:
                    pa = w["pending_add"]
                    inc = q["oi"] - pa["oi"]
                    if inc >= 0.5 * pa["v"]:
                        add = int(min(pa["v"], inc))
                        held = w["est_contracts"]
                        w["entry_px"] = round((w["entry_px"] * held + pa["px"] * add) / (held + add), 2)
                        w["est_contracts"] = held + add
                        events.append(_w_event(w, now, "add_confirmed", f"Add-on confirmed: ~{add:,} more, now ~{w['est_contracts']:,}"))
                    w["pending_add"] = None
                delta = q["oi"] - prev["oi"]
                rem = w["est_contracts"] - w["exited"]
                if delta <= -max(0.2 * w["est_contracts"], 1) and rem > 0:
                    sold = int(min(rem, -delta))
                    px = (prev["bid"] + prev["ask"]) / 2 if prev["bid"] > 0 and prev["ask"] > 0 else prev["last"]
                    w["exited"] += sold
                    left = w["est_contracts"] - w["exited"]
                    w["status"] = "Exited" if left <= 0.1 * w["est_contracts"] else "Partly exited"
                    pct = sold / w["est_contracts"] * 100
                    ret = (px - w["entry_px"]) / w["entry_px"] * 100 if w["entry_px"] else 0
                    events.append(_w_event(w, now, "exit",
                                           f"Whale sold ~{sold:,} ({pct:.0f}% of position) around ${px:.2f} on {prev['d'][5:]}, "
                                           f"{ret:+.0f}% from entry", sold, px))
            # same-day warning: heavy selling at the bid
            rem = (w["est_contracts"] or 0) - w["exited"]
            if w["status"] in ("Holding", "Partly exited") and w.get("hint_day") != today and \
                    obs["fill"] in SELL_FILLS and q["volume"] >= 0.5 * max(rem, 1):
                w["hint_day"] = today
                events.append(_w_event(w, now, "exit_hint",
                                       f"Possible exit in progress: {q['volume']:,.0f} traded today, mostly at the bid. "
                                       "Open interest will confirm tomorrow"))
        _w_pnl(w)
        w["updated"] = now.isoformat(timespec="seconds")
    return whales, events


def merge_whales(remote: list, local: list) -> list:
    byid = {x["id"]: x for x in (remote or [])}
    for x in local or []:
        cur = byid.get(x["id"])
        if cur is None or (x.get("updated") or "") >= (cur.get("updated") or ""):
            byid[x["id"]] = x
    return sorted(byid.values(), key=lambda w: w["entry_time"])


def prune_whales(whales: list, today: str, keep_days: int = 60) -> list:
    cutoff = (to_date(today) - timedelta(days=keep_days)).isoformat()
    return [w for w in whales if w["status"] in WHALE_ACTIVE or w["entry_date"] >= cutoff]


def apply_whale_exits(signals: list, whales: list, quote_fn, settings: dict, now: datetime | None = None):
    """Closes paper signals whose whale has sold at least half its position since the signal opened."""
    now = now or now_et()
    exits = []
    if not settings.get("signal_exit_on_whale_exit") or market_status(now)[0] != "open":
        return signals, exits
    by_key = {}
    for w in whales:
        by_key.setdefault(w["key"], []).append(w)
    for sig in signals:
        if sig["status"] != "open":
            continue
        for w in by_key.get(sig["key"], []):
            held = w["est_contracts"] or 0
            exit_evts = [e for e in w["events"] if e["kind"] == "exit" and e["t"] >= sig["entry_time"]]
            if held and exit_evts and w["exited"] >= 0.5 * held:
                q = quote_fn(sig["symbol"], sig["exp"], sig["strike"], sig["type"])
                if not q:
                    continue
                px = q["bid"] if q["bid"] > 0 else q["last"]
                _close(sig, px, "Whale exited", now)
                exits.append(sig)
                break
    return signals, exits


WHALE_ICONS = {"entry": "🐋 WHALE IN", "confirmed": "✅ WHALE HOLDING", "not_held": "↩️ WHALE DIDN'T HOLD",
               "add": "➕ WHALE ADDING", "add_confirmed": "➕ ADD CONFIRMED", "exit": "🚪 WHALE OUT",
               "exit_hint": "👀 POSSIBLE WHALE EXIT", "expired": "⌛ WHALE EXPIRED"}


def whale_event_text(ev: dict) -> str:
    w = ev["whale"]
    line = f"{WHALE_ICONS.get(ev['kind'], '🐋')}: {w['contract']}\n{ev['text']}"
    if ev["kind"] in ("exit", "expired", "exit_hint") and w.get("ret_est") is not None:
        line += f"\nWhale's estimated result so far: {w['ret_est']:+.0f}% ({fmt_money(w['pnl_est'], signed=True)})"
    return line
