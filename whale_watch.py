"""
Whale Watch: live options P/L tracker and unusual-activity scanner.

One-time setup:
    pip install -r requirements.txt
Run:
    streamlit run whale_watch.py
It opens in your browser at http://localhost:8501. Your trades are saved to
trades.csv next to this file.

Data comes from Yahoo Finance through the free, unofficial yfinance library.
Prices are usually about 15 minutes delayed. For education only, not
financial advice.
"""
from __future__ import annotations

import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Whale Watch", page_icon="🐋", layout="wide")

ET = ZoneInfo("America/New_York")
RISK_FREE_RATE = 0.04
TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.csv")
COLUMNS = ["symbol", "expiration", "strike", "type", "contracts", "price", "side", "iv", "note"]

# The NVDA trades from the screenshot, used the first time the app runs.
SEED = [
    {"symbol": "NVDA", "expiration": "2026-11-23", "strike": 125.0, "type": "C", "contracts": 360,
     "price": 5.98, "side": "sell", "iv": 105.0, "note": "Nov 20 3:54pm block, filled at bid"},
    {"symbol": "NVDA", "expiration": "2026-11-23", "strike": 125.0, "type": "C", "contracts": 587,
     "price": 6.05, "side": "buy", "iv": 107.0, "note": "Nov 20 3:54pm block, filled at ask"},
    {"symbol": "NVDA", "expiration": "2026-11-23", "strike": 125.0, "type": "C", "contracts": 393,
     "price": 6.147, "side": "buy", "iv": 108.0, "note": "Nov 20 3:39pm sweep, filled at ask"},
    {"symbol": "NVDA", "expiration": "2026-11-23", "strike": 125.0, "type": "C", "contracts": 193,
     "price": 6.149, "side": "buy", "iv": 108.0, "note": "Nov 20 3:38pm sweep, filled at ask"},
]


# ---------------------------------------------------------------- pricing math
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def option_value(spot: float, strike: float, years: float, iv_pct: float, opt_type: str) -> float:
    """Black-Scholes value per share. Uses intrinsic value at or after expiration."""
    intrinsic = max(0.0, spot - strike) if opt_type == "C" else max(0.0, strike - spot)
    vol = iv_pct / 100.0
    if years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return intrinsic
    sq = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + vol * vol / 2) * years) / sq
    d2 = d1 - sq
    disc = math.exp(-RISK_FREE_RATE * years)
    if opt_type == "C":
        return spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    return strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)


def years_left(expiration, now: datetime) -> float:
    """Time until 4pm Eastern on the expiration date, in years."""
    close = datetime(expiration.year, expiration.month, expiration.day, 16, 0, tzinfo=ET)
    return max(0.0, (close - now).total_seconds() / (365.0 * 86400))


def money(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    sign = "-" if x < 0 else ("+" if x > 0 else "")
    return f"{sign}${abs(x):,.0f}"


# ---------------------------------------------------------------- market data
INDEX_SYMBOLS = {"SPX": "^SPX", "SPXW": "^SPX", "NDX": "^NDX", "RUT": "^RUT", "VIX": "^VIX", "XSP": "^XSP"}


def yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return INDEX_SYMBOLS.get(s, s)


@st.cache_data(ttl=15, show_spinner=False)
def get_price(symbol: str):
    ticker = yf.Ticker(yahoo_symbol(symbol))
    try:
        p = ticker.fast_info.last_price
        if p and p > 0:
            return float(p)
    except Exception:
        pass
    try:
        hist = ticker.history(period="5d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


@st.cache_data(ttl=120, show_spinner=False)
def scan_symbol(symbol: str, max_expirations: int, min_premium: float):
    """Returns (DataFrame of big contracts, error message or None)."""
    ticker = yf.Ticker(yahoo_symbol(symbol))
    try:
        expirations = list(ticker.options)[:max_expirations]
    except Exception as e:
        return pd.DataFrame(), f"{symbol}: couldn’t load option expirations ({e})."
    if not expirations:
        return pd.DataFrame(), f"{symbol}: no listed options found."
    spot = get_price(symbol)
    frames = []
    for exp in expirations:
        try:
            chain = ticker.option_chain(exp)
        except Exception:
            continue
        for opt_type, d in (("C", chain.calls), ("P", chain.puts)):
            if d is None or d.empty:
                continue
            d = d.copy()
            vol = d["volume"].fillna(0)
            oi = d["openInterest"].fillna(0)
            last = d["lastPrice"].fillna(0)
            bid = d["bid"].fillna(0)
            ask = d["ask"].fillna(0)
            mid = (bid + ask) / 2
            out = pd.DataFrame({
                "Track": False,
                "Symbol": symbol,
                "Expiration": exp,
                "Strike": d["strike"],
                "Type": opt_type,
                "Volume": vol.astype(int),
                "Open interest": oi.astype(int),
                "Vol/OI": np.where(oi > 0, (vol / oi.replace(0, np.nan)).round(1), np.inf),
                "Last": last,
                "IV %": (d["impliedVolatility"].fillna(0) * 100).round(0),
                "Premium $": (vol * last * 100).round(0),
                "Fill side": np.where((bid > 0) & (ask > 0),
                                      np.where(last >= mid, "Ask (likely bought)", "Bid (likely sold)"),
                                      "Unknown"),
                "Stock": spot,
            })
            out = out[out["Premium $"] >= min_premium].copy()
            if out.empty:
                continue
            days = (pd.Timestamp(exp) - pd.Timestamp(datetime.now(ET).date())).days
            if spot:
                otm = np.where(opt_type == "C", (out["Strike"] - spot) / spot, (spot - out["Strike"]) / spot) * 100
            else:
                otm = np.zeros(len(out))
            out.insert(5, "OTM %", np.round(otm, 1))
            flags = []
            for (_, r), o in zip(out.iterrows(), otm):
                f = []
                if r["Vol/OI"] >= 5:
                    f.append("Volume 5x+ open interest")
                if o >= 2:
                    f.append("Far out of the money")
                if days <= 1:
                    f.append("0-1 days left")
                if r["Premium $"] >= 5 * min_premium:
                    f.append("Premium 5x your minimum")
                if str(r["Fill side"]).startswith("Ask"):
                    f.append("Bought at ask")
                flags.append(", ".join(f))
            out["Why it’s unusual"] = flags
            out["Score"] = [len([x for x in fl.split(", ") if x]) for fl in flags]
            frames.append(out)
    if not frames:
        return pd.DataFrame(), None
    return pd.concat(frames, ignore_index=True), None


# ---------------------------------------------------------------- trade storage
def load_trades() -> pd.DataFrame:
    df = None
    if os.path.exists(TRADES_FILE):
        try:
            df = pd.read_csv(TRADES_FILE)
        except Exception:
            df = None
    if df is None:
        df = pd.DataFrame(SEED)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS].copy()
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.date
    return df


def save_trades(df: pd.DataFrame) -> None:
    out = df.copy()
    out["expiration"] = out["expiration"].astype(str)
    out.to_csv(TRADES_FILE, index=False)


def clean_rows(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        try:
            sym = str(r["symbol"]).strip().upper()
            exp = r["expiration"]
            if not sym or sym in ("NAN", "NONE") or exp is None or pd.isna(exp):
                continue
            if not hasattr(exp, "year"):
                exp = pd.to_datetime(exp).date()
            rows.append({
                "symbol": sym,
                "expiration": exp,
                "strike": float(r["strike"]),
                "type": "P" if str(r["type"]).upper().startswith("P") else "C",
                "contracts": int(float(r["contracts"])),
                "price": float(r["price"]),
                "side": "sell" if str(r["side"]).lower().startswith("s") else "buy",
                "iv": float(r["iv"]) if pd.notna(r["iv"]) else 60.0,
                "note": "" if pd.isna(r["note"]) else str(r["note"]),
            })
        except Exception:
            continue
    return rows


def position_table(trades, now: datetime) -> pd.DataFrame:
    out = []
    for t in trades:
        spot = get_price(t["symbol"])
        sign = -1 if t["side"] == "sell" else 1
        cost = t["price"] * 100 * t["contracts"]
        if spot is None:
            value = pnl = float("nan")
        else:
            value = option_value(spot, t["strike"], years_left(t["expiration"], now), t["iv"], t["type"]) * 100 * t["contracts"]
            pnl = sign * (value - cost)
        breakeven = t["strike"] + t["price"] if t["type"] == "C" else t["strike"] - t["price"]
        out.append({
            "Contract": f'{t["symbol"]} {t["strike"]:g}{t["type"]} {t["expiration"]:%m/%d/%y}',
            "Side": "Sold" if t["side"] == "sell" else "Bought",
            "Size": f'{t["contracts"]:,} @ ${t["price"]:.2f}',
            "Stock now": spot,
            "Breakeven": breakeven,
            "Premium": cost,
            "Value now": value,
            "P/L": pnl,
            "Note": t["note"],
            "_symbol": t["symbol"],
        })
    return pd.DataFrame(out)


def color_pl(v):
    if isinstance(v, (int, float)) and not math.isnan(v):
        if v > 0:
            return "color: #0F7D53; font-weight: 600"
        if v < 0:
            return "color: #C2352B; font-weight: 600"
    return ""


# ---------------------------------------------------------------- app
if "base" not in st.session_state:
    st.session_state.base = load_trades()
    st.session_state.ver = 0
    st.session_state.saved_csv = st.session_state.base.to_csv(index=False)

with st.sidebar:
    st.header("Settings")
    refresh = st.select_slider("Refresh prices every", options=[10, 15, 30, 60, 120], value=30,
                               format_func=lambda s: f"{s} seconds")
    st.divider()
    st.subheader("Scanner")
    tickers_text = st.text_input("Tickers to scan", "SPX, SPY, NVDA, TSLA", help="Index options work too: SPX, NDX, RUT, VIX, XSP.")
    max_exp = st.slider("Nearest expirations to check", 1, 8, 3)
    min_prem = st.number_input("Minimum premium ($)", min_value=0, value=1_000_000, step=250_000,
                               help="SPX trades are big. $1M+ filters out the everyday noise.")
    only_new = st.checkbox("Only likely new positions (volume above open interest)", value=True)
    which = st.radio("Show", ["Calls and puts", "Calls only", "Puts only"], horizontal=False)

st.title("🐋 Whale Watch")
st.caption("Live P/L on whale trades you track, plus a scanner for unusually big options activity. "
           "Yahoo data is free and usually about 15 minutes delayed. Not financial advice.")

tab_trades, tab_scan = st.tabs(["My tracked trades", "Unusual activity scanner"])


@st.fragment(run_every=refresh)
def live_panel():
    df = st.session_state.get("current", st.session_state.base)
    now = datetime.now(ET)
    trades = clean_rows(df)
    if not trades:
        st.info("No trades yet. Add one in the table below or track one from the scanner.")
        return
    pos = position_table(trades, now)
    symbols = list(dict.fromkeys(pos["_symbol"]))
    cols = st.columns(min(len(symbols), 5) + 1)
    total = pos["P/L"].sum(skipna=True) if pos["P/L"].notna().any() else float("nan")
    cols[0].metric("Total P/L", money(total))
    for c, s in zip(cols[1:], symbols[:5]):
        p = get_price(s)
        c.metric(s, f"${p:,.2f}" if p else "n/a")

    shown = pos.drop(columns=["_symbol"])
    styled = (shown.style
              .format({"Stock now": lambda v: "n/a" if pd.isna(v) else f"${v:,.2f}",
                       "Breakeven": "${:,.2f}", "Premium": "${:,.0f}",
                       "Value now": lambda v: "n/a" if pd.isna(v) else f"${v:,.0f}",
                       "P/L": money})
              .map(color_pl, subset=["P/L"]))
    st.dataframe(styled, hide_index=True)
    st.caption(f"Updated {now:%-I:%M:%S %p} ET. Before expiration, values are Black-Scholes estimates from each trade’s IV."
               if os.name != "nt" else
               f"Updated {now:%I:%M:%S %p} ET. Before expiration, values are Black-Scholes estimates from each trade’s IV.")

    # Payoff chart for one symbol
    sym = st.selectbox("Payoff chart for", symbols, key="chart_sym")
    sym_trades = [t for t in trades if t["symbol"] == sym]
    strikes = [t["strike"] for t in sym_trades]
    spot = get_price(sym) or strikes[0]
    lo, hi = min(strikes + [spot]) * 0.85, max(strikes + [spot]) * 1.15
    xs = np.linspace(lo, hi, 120)
    now_pl, exp_pl = [], []
    for x in xs:
        a = b = 0.0
        for t in sym_trades:
            sign = -1 if t["side"] == "sell" else 1
            cost = t["price"] * 100 * t["contracts"]
            a += sign * (option_value(x, t["strike"], years_left(t["expiration"], now), t["iv"], t["type"]) * 100 * t["contracts"] - cost)
            b += sign * (option_value(x, t["strike"], 0, t["iv"], t["type"]) * 100 * t["contracts"] - cost)
        now_pl.append(a)
        exp_pl.append(b)
    chart = pd.DataFrame({"Stock price": xs.round(2), "P/L now": now_pl, "P/L at expiration": exp_pl}).set_index("Stock price")
    st.line_chart(chart)


with tab_trades:
    live_box = st.container()
    st.subheader("Edit trades")
    st.caption("Add a row for any whale trade you see. Side: “buy” if it filled at the ask, “sell” if at the bid. "
               "Changes save automatically to trades.csv.")
    edited = st.data_editor(
        st.session_state.base,
        key=f"editor_{st.session_state.ver}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", required=True),
            "expiration": st.column_config.DateColumn("Expiration", format="MM/DD/YY", required=True),
            "strike": st.column_config.NumberColumn("Strike", min_value=0.0, step=0.5),
            "type": st.column_config.SelectboxColumn("Call/Put", options=["C", "P"], default="C"),
            "contracts": st.column_config.NumberColumn("Contracts", min_value=1, step=1),
            "price": st.column_config.NumberColumn("Price paid", min_value=0.0, step=0.01, format="$%.3f"),
            "side": st.column_config.SelectboxColumn("Side", options=["buy", "sell"], default="buy"),
            "iv": st.column_config.NumberColumn("IV %", min_value=1.0, step=1.0, default=60.0),
            "note": st.column_config.TextColumn("Note"),
        },
    )
    st.session_state.current = edited
    csv_now = edited.to_csv(index=False)
    if csv_now != st.session_state.saved_csv:
        save_trades(edited)
        st.session_state.saved_csv = csv_now
    with live_box:
        live_panel()


with tab_scan:
    st.markdown(
        "Finds contracts where today’s traded premium is above your minimum. **Volume above open interest** "
        "usually means new positions opened today. **Fill side** compares the last trade to the bid and ask, "
        "a rough guess at whether buyers or sellers were in control. Rows with the most red flags "
        "(huge vs open interest, far out of the money, expiring within a day, bought at the ask) sort to the top. Yahoo shows the day’s total volume, "
        "not individual sweeps or blocks, so one row can be many traders."
    )
    tickers = [t.strip().upper() for t in tickers_text.replace(";", ",").split(",") if t.strip()]
    scan_params = (tuple(tickers), int(max_exp), float(min_prem))
    clicked = st.button("Scan now", type="primary")
    settings_changed = "scan" in st.session_state and st.session_state.get("scan_params") != scan_params
    if clicked or settings_changed:
        if clicked:
            scan_symbol.clear()  # always fetch fresh data on a manual scan
        st.session_state.scan_params = scan_params
        frames, errors = [], []
        with st.spinner("Scanning option chains…"):
            for sym in tickers:
                df, err = scan_symbol(sym, max_exp, float(min_prem))
                if err:
                    errors.append(err)
                if not df.empty:
                    frames.append(df)
        st.session_state.scan = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        st.session_state.scan_errors = errors
        st.session_state.scan_time = datetime.now(ET)

    for err in st.session_state.get("scan_errors", []):
        st.warning(err)

    results = st.session_state.get("scan")
    if results is None:
        st.info("Pick tickers in the sidebar, then select Scan now.")
    elif results.empty:
        st.info("Nothing above your minimum premium. Try a lower minimum or more tickers.")
    else:
        view = results.copy()
        if only_new:
            view = view[view["Volume"] > view["Open interest"]]
        if which == "Calls only":
            view = view[view["Type"] == "C"]
        elif which == "Puts only":
            view = view[view["Type"] == "P"]
        view = view.sort_values(["Score", "Premium $"], ascending=False).reset_index(drop=True)
        st.caption(f"{len(view)} contracts, scanned {st.session_state.scan_time:%I:%M %p} ET for "
                   f"{', '.join(st.session_state.scan_params[0])} at ${st.session_state.scan_params[2]:,.0f}+ premium. "
                   "Tick Track on any row to follow it in My tracked trades.")
        picked = st.data_editor(
            view,
            key="scan_editor",
            hide_index=True,
            disabled=[c for c in view.columns if c != "Track"],
            column_config={
                "Track": st.column_config.CheckboxColumn("Track"),
                "Premium $": st.column_config.NumberColumn("Premium $", format="$%d"),
                "Last": st.column_config.NumberColumn("Last", format="$%.2f"),
                "Stock": st.column_config.NumberColumn("Stock", format="$%.2f"),
            },
        )
        chosen = picked[picked["Track"]]
        if st.button(f"Track {len(chosen)} checked contract(s)", disabled=chosen.empty):
            new_rows = pd.DataFrame([{
                "symbol": r["Symbol"],
                "expiration": pd.to_datetime(r["Expiration"]).date(),
                "strike": float(r["Strike"]),
                "type": r["Type"],
                "contracts": int(r["Volume"]),
                "price": float(r["Last"]),
                "side": "sell" if str(r["Fill side"]).startswith("Bid") else "buy",
                "iv": float(r["IV %"]) if r["IV %"] > 0 else 60.0,
                "note": f'Scanner {datetime.now(ET):%m/%d}: whole day’s volume, vol/OI {r["Vol/OI"]}',
            } for _, r in chosen.iterrows()])
            combined = pd.concat([st.session_state.current, new_rows], ignore_index=True)
            save_trades(combined)
            st.session_state.base = load_trades()
            st.session_state.saved_csv = st.session_state.base.to_csv(index=False)
            st.session_state.ver += 1
            st.toast(f"Tracking {len(new_rows)} new contract(s).")
            st.rerun()
