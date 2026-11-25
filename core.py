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
