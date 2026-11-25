"""
Whale Watch: options flow intelligence.

Run locally:   pip install -r requirements.txt   then   streamlit run whale_watch.py
Deploy free:   Streamlit Community Cloud (see README.md)

Data: Yahoo Finance via yfinance (free, usually ~15 min delayed, unofficial).
For education only. Not financial advice.
"""
from __future__ import annotations

import math
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Whale Watch", page_icon="🐋", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
  .ww-pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600;}
  .ww-open {background:rgba(16,150,90,.15);color:#0f8a54;}
  .ww-closed {background:rgba(140,140,160,.18);color:inherit;}
</style>
""", unsafe_allow_html=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================= setup
def secret(name: str):
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.environ.get(name)


@st.cache_resource
def get_storage():
    return core.make_storage(secret("GITHUB_TOKEN"), secret("GITHUB_REPO"), os.path.join(APP_DIR, "data"))


storage = get_storage()


@st.cache_data(ttl=15, show_spinner=False)
def price(sym):
    return core.fetch_price(sym)


@st.cache_data(ttl=90, show_spinner=False)
def scan_cached(sym, max_exp, minp, bucket):
    return core.scan_symbol(sym, max_exp, minp)


@st.cache_data(ttl=600, show_spinner=False)
def gex_cached(sym, max_days, bucket):
    return core.dealer_positioning(sym, max_days)


@st.cache_data(ttl=300, show_spinner=False)
def chain_cached(sym, exp):
    return core.fetch_chain(sym, exp)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def close_cached(sym, d):
    return core.fetch_close_on(sym, d)


@st.cache_data(ttl=45, show_spinner=False)
def quote_chain(sym, exp, bucket):
    return core.fetch_chain(sym, exp)


def load_signals(refresh: bool = False) -> list:
    ss = st.session_state
    if "signals" not in ss or refresh:
        remote = storage.get_json("signals.json", []) or []
        ss.signals = core.merge_signals(remote, ss.get("signals", []))
        ss.signals_loaded = datetime.now()
    return ss.signals


def load_whales(refresh: bool = False) -> list:
    ss = st.session_state
    if "whales" not in ss or refresh:
        remote = storage.get_json("whales.json", []) or []
        ss.whales = core.merge_whales(remote, ss.get("whales", []))
        ss.whales_loaded = datetime.now()
    return ss.whales


def save_whales(note: str = "Whales from app"):
    storage.put_json("whales.json", st.session_state.whales, merge=core.merge_whales, message=note)


def save_signals(note: str = "Signals from app"):
    storage.put_json("signals.json", st.session_state.signals, merge=core.merge_signals, message=note)


def time_bucket(seconds: int) -> int:
    return int(datetime.now().timestamp() // seconds)


def init_state():
    ss = st.session_state
    if "settings" not in ss:
        ss.settings = core.load_settings(storage)
    if "history" not in ss:
        ss.history = storage.get_json("history.json", {}) or {}
        ss.history_loaded = datetime.now()
    if "trades" not in ss:
        raw = storage.get_json("trades.json", None)
        if raw is None and os.path.exists(os.path.join(APP_DIR, "trades.csv")):
            raw = pd.read_csv(os.path.join(APP_DIR, "trades.csv")).to_dict("records")
        ss.trades = trades_frame(raw if raw is not None else SEED_TRADES)
        ss.trades_saved = trades_json(ss.trades)
        ss.trades_ver = 0
    ss.setdefault("last_history_save", datetime.min)


TRADE_COLS = ["symbol", "expiration", "strike", "type", "contracts", "price", "side", "iv", "note"]
SEED_TRADES = [
    {"symbol": "NVDA", "expiration": "2026-11-23", "strike": 125.0, "type": "C", "contracts": 587, "price": 6.05,
     "side": "buy", "iv": 107.0, "note": "Example: Nov 20 block at ask"},
]


def trades_frame(records) -> pd.DataFrame:
    df = pd.DataFrame(records or [], columns=TRADE_COLS)
    for c in TRADE_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[TRADE_COLS].copy()
    df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.date
    for c in ("strike", "price", "iv"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["contracts"] = pd.to_numeric(df["contracts"], errors="coerce")
    return df


def trades_json(df: pd.DataFrame) -> list:
    out = df.copy()
    out["expiration"] = out["expiration"].astype(str)
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict("records")


def save_trades(df: pd.DataFrame):
    data = trades_json(df)
    if data != st.session_state.trades_saved:
        storage.put_json("trades.json", data, message="Update tracked trades")
        st.session_state.trades_saved = data


def add_trades(rows: list[dict]):
    cur = st.session_state.get("trades_current", st.session_state.trades)
    combined = pd.concat([cur, trades_frame(rows)], ignore_index=True)
    st.session_state.trades = combined
    save_trades(combined)
    st.session_state.trades_ver += 1


def save_history(force: bool = False):
    ss = st.session_state
    if force or (datetime.now() - ss.last_history_save).total_seconds() > 900:
        storage.put_json("history.json", ss.history, merge=core.merge_history, message="Scan from app")
        ss.last_history_save = datetime.now()


init_state()
S = st.session_state.settings


# ======================================================================= sidebar
with st.sidebar:
    st.markdown("### 🐋 Whale Watch")
    st.caption(("☁️ " if storage.kind == "github" else "💾 ") + storage.describe())
    st.divider()
    st.markdown("**Scanner**")
    tickers_text = st.text_input("Watchlist", ", ".join(S["watchlist"]),
                                 help="Stocks, ETFs and index options (SPX, NDX, RUT, VIX, XSP).")
    max_exp = st.slider("Nearest expirations", 1, 8, int(S["max_expirations"]),
                        help="2-3 covers 0DTE and this week, where most whale bets land.")
    min_prem = st.number_input("Minimum premium ($)", min_value=0, value=int(S["min_premium"]), step=250_000)
    overrides_text = st.text_input(
        "Per-ticker minimums", ", ".join(f"{k}={int(v)}" for k, v in (S.get("min_premium_overrides") or {}).items()),
        help="Bigger names need bigger thresholds. Example: SPX=2000000, TSLA=500000")
    st.markdown("**Filters**")
    cat_filter = st.multiselect("Show categories", core.CATEGORIES, default=core.CATEGORIES)
    min_score = st.slider("Minimum score", 0, 100, 0)
    side_filter = st.radio("Contracts", ["Calls and puts", "Calls only", "Puts only"], horizontal=True)
    st.markdown("**Live updates**")
    auto = st.toggle("Auto-rescan while market is open", value=False)
    auto_every = st.select_slider("Every", [60, 120, 300, 600], value=120, format_func=lambda s: f"{s // 60} min")
    refresh = st.select_slider("Trade prices refresh", [10, 15, 30, 60], value=30, format_func=lambda s: f"{s} s")


def parse_overrides(text: str) -> dict:
    out = {}
    for part in text.replace(";", ",").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[core.base_symbol(k)] = float(v.strip().replace("$", "").replace("_", ""))
            except ValueError:
                pass
    return out


tickers = [core.base_symbol(t) for t in tickers_text.replace(";", ",").split(",") if t.strip()]
live_settings = dict(S, watchlist=tickers, max_expirations=max_exp, min_premium=float(min_prem),
                     min_premium_overrides=parse_overrides(overrides_text))


# ======================================================================= header
now = core.now_et()
mstatus, mdetail = core.market_status(now)
h1, h2 = st.columns([3, 2])
with h1:
    st.title("Whale Watch")
    st.markdown(f'<span class="ww-pill {"ww-open" if mstatus == "open" else "ww-closed"}">'
                f'{"● Market open" if mstatus == "open" else "Market closed"}: {mdetail}</span>'
                f'&nbsp; <span style="opacity:.7;font-size:.85rem">{now:%a %b %d, %I:%M %p} ET. '
                f'Yahoo data, usually ~15 min delayed.</span>', unsafe_allow_html=True)
with h2:
    c1, c2, c3 = st.columns(3)
    spx, vix = price("SPX"), price("VIX")
    c1.metric("SPX", f"{spx:,.2f}" if spx else "n/a")
    c2.metric("VIX", f"{vix:,.2f}" if vix else "n/a")
    flow_df = st.session_state.get("flow")
    if flow_df is not None and not flow_df.empty:
        net = flow_df.loc[flow_df["Category"].isin(["Directional bet", "Seller-driven"]), "Signed hedge $"].sum()
        c3.metric("Net whale tilt", core.fmt_money(net, signed=True), "bullish" if net > 0 else "bearish" if net < 0 else "flat",
                  delta_color="normal" if net >= 0 else "inverse")
    else:
        c3.metric("Net whale tilt", "scan first")

tab_sig, tab_whale, tab_flow, tab_gex, tab_stick, tab_trades, tab_alerts, tab_guide = st.tabs(
    ["🎯 Signals", "🐋 Whale moves", "🔎 Flow scanner", "🧲 Dealer positioning", "✅ Did it stick?", "📈 My trades", "🔔 Alerts", "📘 Guide"])


# ======================================================================= signals (paper trading)
@st.fragment(run_every=60)
def signals_panel():
    ss = st.session_state
    sigs = load_signals(refresh=(datetime.now() - ss.get("signals_loaded", datetime.min)).total_seconds() > 300)
    qf = core.make_quote_fn(lambda sym, exp: quote_chain(sym, exp, time_bucket(45)))
    now_ = core.now_et()
    sigs, exits = core.update_signals(sigs, qf, close_cached, S, now_)
    sigs, wexits = core.apply_whale_exits(sigs, load_whales(), qf, S, now_)
    exits += wexits
    ss.signals = sigs
    if exits:
        save_signals(f"{len(exits)} exit(s) from app")
        for x in exits:
            st.toast(f"{'✅' if x['pnl'] > 0 else '🔴'} Exit: {x['contract']} {x['ret_pct']:+.1f}% ({x['exit_reason']})")

    open_s = [x for x in sigs if x["status"] == "open"]
    closed = [x for x in sigs if x["status"] == "closed"]
    st_ = core.signal_stats(sigs)
    unreal = sum((x["last_px"] - x["entry_px"]) * 100 * x["qty"] for x in open_s)
    m = st.columns(6)
    m[0].metric("Open", len(open_s), core.fmt_money(unreal, signed=True) + " open P/L" if open_s else None,
                delta_color="normal" if unreal >= 0 else "inverse")
    m[1].metric("Closed trades", st_["trades"])
    m[2].metric("Win rate", f"{st_['win_rate']:.0f}%" if st_["trades"] else "n/a")
    m[3].metric("Avg return / trade", f"{st_['avg_ret']:+.1f}%" if st_["trades"] else "n/a")
    m[4].metric("Paper P/L", core.fmt_money(st_["total_pnl"], signed=True) if st_["trades"] else "n/a")
    pf = st_.get("profit_factor")
    m[5].metric("Profit factor", "n/a" if not st_["trades"] else ("∞" if pf == float("inf") else f"{pf:.2f}"),
                help="Gross wins ÷ gross losses. Above 1.0 means the rules made money.")

    whale_status = {}
    for w in load_whales():
        whale_status[w["key"]] = w["status"] + (f" ({w['exited'] / w['est_contracts'] * 100:.0f}% sold)"
                                                if w["est_contracts"] and w["exited"] else "")
    st.markdown("##### Open signals")
    if not open_s:
        st.caption(f"No open signals. New ones open when a scan finds a {', '.join(S['signal_categories']).lower()} "
                   f"scoring {S['signal_min_score']}+ while the market is open.")
    else:
        rows = []
        for x in open_s:
            ret = (x["last_px"] - x["entry_px"]) / x["entry_px"] * 100 if x["entry_px"] else 0
            rows.append({"Close": False, "Contract": x["contract"], "Score": x["score"],
                         "Opened": x["entry_time"][5:16].replace("T", " "), "Entry": x["entry_px"], "Now": x["last_px"],
                         "Return %": round(ret, 1), "Target": x["target_px"],
                         "Stop": x["stop_px"], "Stop at breakeven": bool(x.get("breakeven_armed")),
                         "Qty": x["qty"], "P/L": core.fmt_money((x["last_px"] - x["entry_px"]) * 100 * x["qty"], signed=True),
                         "Whale": whale_status.get(x["key"], "Not tracked"), "Why": x["why"]})
        ed = st.data_editor(pd.DataFrame(rows), hide_index=True, key=f"open_sig_{len(open_s)}_{st_['trades']}",
                            disabled=[c for c in rows[0] if c != "Close"],
                            column_config={
                                "Close": st.column_config.CheckboxColumn("Close", width="small"),
                                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                                "Now": st.column_config.NumberColumn(format="$%.2f"),
                                "Target": st.column_config.NumberColumn(format="$%.2f"),
                                "Stop": st.column_config.NumberColumn(format="$%.2f"),
                                "Return %": st.column_config.NumberColumn(format="%+.1f%%"),
                                "Why": st.column_config.TextColumn(width="large")})
        picks = [open_s[i] for i, v in enumerate(ed["Close"].tolist()) if v]
        if st.button(f"Close {len(picks)} checked at current bid", disabled=not picks):
            for x in picks:
                q = qf(x["symbol"], x["exp"], x["strike"], x["type"])
                px = (q["bid"] if q and q["bid"] > 0 else (q["last"] if q else x["last_px"]))
                core._close(x, px, "Closed manually", core.now_et())
            save_signals("Manual close")
            st.rerun()

    if st_["trades"]:
        eq = st_["equity"]
        line = alt.Chart(eq).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Time:T", title=None), y=alt.Y("Equity:Q", title="Cumulative paper P/L ($)", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Time:T"), alt.Tooltip("Equity:Q", format="$,.0f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9aa3b2").encode(y="y:Q")
        st.altair_chart((line + zero).properties(height=240), width="stretch")

    st.markdown("##### Closed signals")
    if closed:
        cl = pd.DataFrame([{"Contract": x["contract"], "Score": x["score"],
                            "Opened": x["entry_time"][5:16].replace("T", " "), "Closed": (x["exit_time"] or "")[5:16].replace("T", " "),
                            "Entry": x["entry_px"], "Exit": x["exit_px"], "Return %": x["ret_pct"],
                            "P/L": x["pnl"], "Reason": x["exit_reason"], "Qty": x["qty"]} for x in reversed(closed)])
        st.dataframe(cl, hide_index=True, column_config={
            "Entry": st.column_config.NumberColumn(format="$%.2f"), "Exit": st.column_config.NumberColumn(format="$%.2f"),
            "Return %": st.column_config.NumberColumn(format="%+.1f%%"), "P/L": st.column_config.NumberColumn(format="$%.0f")})
        st.download_button("Download trade log (CSV)", pd.DataFrame(sigs).to_csv(index=False),
                           file_name="whale_signals.csv", mime="text/csv")
    else:
        st.caption("None yet.")
    st.caption(f"Checked {now_:%I:%M:%S %p} ET; refreshes every minute while this page is open, and every 15 minutes "
               "by the alert job when it's set up. Paper trades only: entries assume you pay the ask, exits assume you "
               "sell at the bid, on ~15-minute-delayed Yahoo data. Real fills, slippage and fees will differ.")


with tab_sig:
    st.markdown("Follows whale trades that score high enough, with automatic exits, as **paper trades**. "
                "Let it run for a few weeks: if the win rate and profit factor hold up, you'll know the "
                "signals are worth your attention. If they don't, you've lost nothing.")
    with st.expander("⚙️ Signal rules", expanded=False):
        with st.form("signal_rules"):
            f1, f2, f3 = st.columns(3)
            r_score = f1.slider("Enter when score is at least", 40, 100, int(S["signal_min_score"]))
            r_cats = f1.multiselect("Categories to follow", ["Directional bet"], default=["Directional bet"],
                                    help="Only buyer-driven bets can be followed by buying the same contract.")
            r_size = f1.number_input("Paper size per trade ($)", 100, 1_000_000, int(S["signal_size_usd"]), step=100)
            r_tgt = f2.number_input("Profit target (%)", 5, 1000, int(S["signal_target_pct"]), step=5)
            r_stop = f2.number_input("Stop loss (%)", 5, 100, int(S["signal_stop_pct"]), step=5)
            r_be = f2.number_input("Move stop to breakeven after gain of (%)", 0, 500, int(S["signal_breakeven_after_pct"]),
                                   step=5, help="0 turns this off.")
            r_exit = f3.text_input("Time exit on expiration day (ET, HH:MM)", S["signal_exit_time"])
            r_0dte = f3.text_input("No new 0DTE entries after (ET, HH:MM)", S["signal_last_entry_0dte"])
            r_hold = f3.number_input("Max hold (days)", 0, 60, int(S["signal_max_hold_days"]))
            g1, g2, g3 = st.columns(3)
            r_max = g1.number_input("Max open at once", 1, 50, int(S["signal_max_open"]))
            r_minpx = g2.number_input("Skip options cheaper than ($)", 0.0, 20.0, float(S["signal_min_price"]), step=0.05)
            r_chase = g3.number_input("Skip if price ran more than (%) past the whale's fill", 0, 300,
                                      int(S["signal_max_chase_pct"]), step=5)
            r_alert = st.checkbox("Send entry and exit alerts to my phone", value=bool(S.get("signal_alerts", True)))
            r_follow = st.checkbox("Exit when the whale exits (it sells at least half its position)",
                                   value=bool(S.get("signal_exit_on_whale_exit", True)))
            if st.form_submit_button("Save signal rules", type="primary"):
                try:
                    core._hm(r_exit), core._hm(r_0dte)
                    S.update({"signal_min_score": r_score, "signal_categories": r_cats or ["Directional bet"],
                              "signal_size_usd": r_size, "signal_target_pct": r_tgt, "signal_stop_pct": r_stop,
                              "signal_breakeven_after_pct": r_be, "signal_exit_time": r_exit.strip(),
                              "signal_last_entry_0dte": r_0dte.strip(), "signal_max_hold_days": r_hold,
                              "signal_max_open": r_max, "signal_min_price": r_minpx, "signal_max_chase_pct": r_chase,
                              "signal_alerts": r_alert, "signal_exit_on_whale_exit": r_follow})
                    storage.put_json("settings.json", S, message="Update signal rules")
                    st.success("Saved. New signals use these rules; open ones keep the targets and stops they started with.")
                except Exception:
                    st.error("Times must look like 15:30.")
        st.caption(f"Now: enter at score {S['signal_min_score']}+, target +{S['signal_target_pct']}%, stop "
                   f"-{S['signal_stop_pct']}%, breakeven after +{S['signal_breakeven_after_pct']}%, exit by "
                   f"{S['signal_exit_time']} ET on expiration day, max {S['signal_max_hold_days']} days, "
                   f"${S['signal_size_usd']:,} per trade.")
    signals_panel()
    with st.expander("Reset paper trading"):
        st.caption("Deletes every open and closed signal. Use this after changing rules, so stats reflect the new rules only.")
        if st.checkbox("Yes, I want to delete all signals") and st.button("Delete all signals"):
            st.session_state.signals = []
            storage.put_json("signals.json", [], message="Reset signals")
            st.rerun()


# ======================================================================= whale moves
@st.fragment(run_every=120)
def whale_panel():
    ss = st.session_state
    whales = load_whales(refresh=(datetime.now() - ss.get("whales_loaded", datetime.min)).total_seconds() > 300)
    qf = core.make_quote_fn(lambda sym, exp: quote_chain(sym, exp, time_bucket(45)))
    now_ = core.now_et()
    whales, evs = core.update_whales(whales, qf, close_cached, now_)
    ss.whales = whales
    if evs:
        save_whales(f"{len(evs)} whale event(s) from app")
        for ev in evs:
            st.toast(core.whale_event_text(ev).split("\n")[0])

    active = [w for w in whales if w["status"] in core.WHALE_ACTIVE]
    done = [w for w in whales if w["status"] in ("Exited", "Expired") and w["est_contracts"]]
    m = st.columns(5)
    m[0].metric("Tracking", len(active))
    m[1].metric("Confirmed holding", sum(1 for w in active if w["status"] in ("Holding", "Partly exited")))
    m[2].metric("Didn't hold overnight", sum(1 for w in whales if w["status"] == "Not held overnight"),
                help="Open interest didn't rise: a day trade or someone closing an older position.")
    m[3].metric("Whale win rate (est.)", f"{sum(1 for w in done if (w['pnl_est'] or 0) > 0) / len(done) * 100:.0f}%" if done else "n/a",
                f"{len(done)} finished", delta_color="off")
    m[4].metric("Whale avg return (est.)", f"{np.mean([w['ret_est'] or 0 for w in done]):+.0f}%" if done else "n/a")

    if not whales:
        st.info("No whales tracked yet. Every directional bet scoring "
                f"{S['whale_min_score']}+ from a scan (yours or the alert job's) is tracked here automatically.")
        return
    statuses = ["Unconfirmed", "Holding", "Partly exited", "Exited", "Expired", "Not held overnight"]
    show_st = st.multiselect("Show", statuses, default=["Unconfirmed", "Holding", "Partly exited", "Exited", "Expired"])
    rows = []
    for w in sorted(whales, key=lambda w: w["entry_time"], reverse=True):
        if w["status"] not in show_st:
            continue
        held = w["est_contracts"]
        last_ev = w["events"][-1]["text"] if w["events"] else ""
        rows.append({"Contract": w["contract"], "Entered": w["entry_time"][5:16].replace("T", " "),
                     "Whale bought": f"{w['size']:,} @ ${w['entry_px']:.2f}", "Premium": core.fmt_money(w["premium"]),
                     "Status": w["status"], "Est. held": "?" if held is None else f"{held:,}",
                     "Sold %": None if not held else round(w["exited"] / held * 100),
                     "Avg exit": w["avg_exit_px"], "Now": w["last_px"],
                     "Est. return %": w["ret_est"], "Est. P/L": core.fmt_money(w["pnl_est"], signed=True),
                     "Latest": last_ev, "Score": w["score"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(520, 40 + 36 * len(rows)), column_config={
            "Sold %": st.column_config.ProgressColumn("Sold %", min_value=0, max_value=100, format="%d%%"),
            "Avg exit": st.column_config.NumberColumn(format="$%.2f"), "Now": st.column_config.NumberColumn(format="$%.2f"),
            "Est. return %": st.column_config.NumberColumn(format="%+.0f%%"),
            "Latest": st.column_config.TextColumn(width="large")})

    st.markdown("##### Whale timeline")
    labels = {f"{w['contract']}  (entered {w['entry_date'][5:]}, {w['status'].lower()})": w for w in reversed(whales)}
    pick = st.selectbox("Pick a whale", list(labels))
    w = labels[pick]
    obs = pd.DataFrame(w["obs"])
    if len(obs):
        obs["Date"] = pd.to_datetime(obs["d"])
        obs["Price"] = np.where((obs["bid"] > 0) & (obs["ask"] > 0), (obs["bid"] + obs["ask"]) / 2, obs["last"])
        marks = [{"Date": pd.to_datetime(w["entry_date"]), "Price": w["entry_px"], "Event": "Whale entry"}]
        for e in w["events"]:
            if e["kind"] in ("exit", "expired") and e["px"] is not None:
                d = e["t"][:10]
                if e["kind"] == "exit":
                    prior = [o["d"] for o in w["obs"] if o["d"] < d]
                    d = prior[-1] if prior else d
                marks.append({"Date": pd.to_datetime(d), "Price": e["px"], "Event": "Whale exit" if e["kind"] == "exit" else "Expiration"})
        base = alt.Chart(obs).mark_line(strokeWidth=2.5, point=True).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("Price:Q", title="Option price ($)"),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Price:Q", format="$.2f"), alt.Tooltip("oi:Q", title="Open interest", format=","),
                     alt.Tooltip("vol:Q", title="Volume", format=",")])
        pts = alt.Chart(pd.DataFrame(marks)).mark_point(size=220, filled=True).encode(
            x="Date:T", y="Price:Q", shape=alt.Shape("Event:N", legend=alt.Legend(orient="top")),
            color=alt.Color("Event:N", scale=alt.Scale(domain=["Whale entry", "Whale exit", "Expiration"],
                                                       range=["#13a36b", "#d9483b", "#9aa3b2"])),
            tooltip=["Event", alt.Tooltip("Price:Q", format="$.2f")])
        st.altair_chart((base + pts).properties(height=260), width="stretch")
        oi = alt.Chart(obs).mark_bar(opacity=.6).encode(x=alt.X("Date:T", title=None),
                                                        y=alt.Y("oi:Q", title="Open interest", axis=alt.Axis(format="~s")))
        st.altair_chart(oi.properties(height=120), width="stretch")
    for e in reversed(w["events"]):
        st.markdown(f"**{e['t'][5:16].replace('T', ' ')}**: {core.WHALE_ICONS.get(e['kind'], '🐋')}: {e['text']}")
    st.caption(f"Why it was flagged: {w['why']}")


with tab_whale:
    st.markdown("Follows each high-score whale after the big print. **Entries** are confirmed when open interest "
                "rises overnight. **Exits** are detected when open interest later falls by a big share of the position, "
                "and the exit price is taken from that day. **Possible exit** warnings fire the same day when heavy "
                "selling hits the whale's contract.")
    whale_panel()
    st.caption("Open interest is shared by everyone trading the contract, so whale sizes, exits and returns are "
               "estimates from public footprints, not the whale's actual account. Small contracts with one dominant "
               "trader give the clearest read; busy SPX 0DTE contracts the weakest.")


# ======================================================================= flow scanner
def run_scan(force: bool):
    ss = st.session_state
    frames, errors = [], []
    bucket = time_bucket(1 if force else 90)
    with st.spinner(f"Scanning {', '.join(tickers)}…"):
        for t in tickers:
            df, err = scan_cached(t, max_exp, core.min_premium_for(t, live_settings), bucket)
            if err:
                errors.append(err)
            if df is not None and not df.empty:
                frames.append(df)
    today = core.now_et().date().isoformat()
    if (datetime.now() - ss.history_loaded).total_seconds() > 600:
        ss.history = core.merge_history(storage.get_json("history.json", {}) or {}, ss.history)
        ss.history_loaded = datetime.now()
    flow = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not flow.empty:
        flow = core.score_flow(flow, ss.history, today, live_settings)
        ss.history = core.record_history(ss.history, flow, today)
        save_history(force=force)
        whales, wev = core.open_whales(flow, load_whales(), ss.settings, core.now_et())
        if wev:
            ss.whales = whales
            save_whales(f"{len(wev)} whale event(s) from app scan")
        sigs, new = core.open_signals(flow, load_signals(), ss.settings, core.now_et())
        if new:
            ss.signals = sigs
            save_signals(f"{len(new)} new signal(s) from app scan")
            for sig in new:
                st.toast(f"🟢 New signal: {sig['contract']} @ ${sig['entry_px']:.2f}")
    ss.flow, ss.flow_errors, ss.flow_time = flow, errors, core.now_et()
    ss.scan_params = (tuple(tickers), max_exp, float(min_prem), overrides_text)


@st.fragment(run_every=auto_every if (auto and mstatus == "open") else None)
def flow_panel():
    ss = st.session_state
    params = (tuple(tickers), max_exp, float(min_prem), overrides_text)
    b1, b2, _ = st.columns([1, 1.4, 4])
    clicked = b1.button("Scan now", type="primary", width="stretch")
    if b2.button("Save as alert watchlist", width="stretch",
                 help="The scheduled job scans this watchlist and these thresholds."):
        ss.settings.update({k: live_settings[k] for k in ("watchlist", "max_expirations", "min_premium", "min_premium_overrides")})
        storage.put_json("settings.json", ss.settings, message="Update settings")
        st.toast("Saved. The alert job will use this watchlist.")
    stale = "flow_time" in ss and (core.now_et() - ss.flow_time).total_seconds() >= auto_every - 5
    if clicked or ("flow" in ss and ss.get("scan_params") != params) or (auto and mstatus == "open" and stale):
        run_scan(force=clicked)
        st.rerun()  # refresh the header tilt and other tabs with the new results

    for e in ss.get("flow_errors", []):
        st.warning(e)
    flow = ss.get("flow")
    if flow is None:
        st.info("Set your watchlist in the sidebar, then select **Scan now**. Best during market hours.")
        return
    if flow.empty:
        st.info("Nothing above your premium thresholds right now. Lower the minimum or add tickers.")
        return

    view = flow[flow["Category"].isin(cat_filter) & (flow["Score"] >= min_score)]
    if side_filter == "Calls only":
        view = view[view["Type"] == "C"]
    elif side_filter == "Puts only":
        view = view[view["Type"] == "P"]

    directional = view[view["Category"].isin(["Directional bet", "Seller-driven"])]
    bull = directional.loc[directional["Signed hedge $"] > 0, "Signed hedge $"].sum()
    bear = -directional.loc[directional["Signed hedge $"] < 0, "Signed hedge $"].sum()
    m = st.columns(5)
    m[0].metric("Contracts flagged", f"{len(view):,}")
    m[1].metric("High conviction (70+)", f"{(view['Score'] >= 70).sum():,}")
    m[2].metric("Bullish hedging flow", core.fmt_money(bull))
    m[3].metric("Bearish hedging flow", core.fmt_money(bear))
    m[4].metric("Premium scanned", core.fmt_money(view["Premium"].sum()))
    st.caption(f"Scanned {ss.flow_time:%I:%M:%S %p} ET for {', '.join(ss.scan_params[0])}. "
               "Hedging flow = shares dealers must trade to hedge (delta-adjusted), a better gauge of market impact than premium.")

    if not directional.empty:
        agg = directional.groupby(["Symbol", "Bias"], as_index=False)["Hedge $"].sum()
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("Hedge $:Q", title="Delta-adjusted flow ($)", axis=alt.Axis(format="~s")),
            y=alt.Y("Symbol:N", sort="-x", title=None),
            color=alt.Color("Bias:N", scale=alt.Scale(domain=["Bullish", "Bearish", "Unclear"],
                                                      range=["#13a36b", "#d9483b", "#9aa3b2"]), legend=alt.Legend(orient="top")),
            tooltip=["Symbol", "Bias", alt.Tooltip("Hedge $:Q", format="$,.0f")],
        ).properties(height=max(120, 34 * agg["Symbol"].nunique()))
        st.altair_chart(chart, width="stretch")

    show = pd.DataFrame({
        "Track": False,
        "Score": view["Score"],
        "Category": view["Category"],
        "Contract": view["Contract"],
        "Bias": view["Bias"],
        "Premium": view["Premium"].map(core.fmt_money),
        "Volume": view["Volume"].astype(int),
        "OI": view["OI"].astype(int),
        "Vol/OI": view["Vol/OI"].map(lambda v: "new" if math.isinf(v) else f"{v:.1f}x"),
        "Fill": view["Fill"],
        "OTM %": view["OTM %"],
        "DTE": view["DTE"].astype(int),
        "Hedge $": view["Hedge $"].map(core.fmt_money),
        "% ADV": view["% ADV"],
        "IV": view["IV"],
        "Confirmation": view["Confirmation"],
        "Days seen": view["Days seen"],
        "Why": view["Why"],
    })
    picked = st.data_editor(
        show, key=f"flow_editor_{ss.flow_time.timestamp():.0f}", hide_index=True, height=520,
        disabled=[c for c in show.columns if c != "Track"],
        column_config={
            "Track": st.column_config.CheckboxColumn("Track", width="small"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d", width="small"),
            "OTM %": st.column_config.NumberColumn("OTM %", format="%.1f%%"),
            "% ADV": st.column_config.NumberColumn("% ADV", format="%.2f%%", help="Hedging flow as % of the stock's average daily dollar volume."),
            "IV": st.column_config.NumberColumn("IV", format="%.0f%%"),
            "Why": st.column_config.TextColumn("Why it's flagged", width="large"),
        },
    )
    chosen = view.loc[picked["Track"].to_numpy()]
    if st.button(f"Track {len(chosen)} checked contract(s)", disabled=chosen.empty):
        add_trades([{
            "symbol": r["Symbol"], "expiration": r["Expiration"], "strike": float(r["Strike"]), "type": r["Type"],
            "contracts": int(r["Volume"]), "price": float(r["Last"]),
            "side": "sell" if r["Fill"] in core.SELL_FILLS else "buy",
            "iv": float(r["IV"]) if r["IV"] > 0 else 60.0,
            "note": f"Scanner {core.now_et():%m/%d}: score {r['Score']}, {r['Category'].lower()}, day volume",
        } for _, r in chosen.iterrows()])
        st.toast(f"Tracking {len(chosen)} contract(s). See My trades.")
        st.rerun()

    st.download_button("Download results (CSV)", view.drop(columns=["Gamma"], errors="ignore").to_csv(index=False),
                       file_name=f"whale_flow_{ss.flow_time:%Y%m%d_%H%M}.csv", mime="text/csv")


with tab_flow:
    flow_panel()


# ======================================================================= dealer positioning
with tab_gex:
    g1, g2, g3 = st.columns([1, 1.2, 3])
    gsym = core.base_symbol(g1.text_input("Underlying", "SPX", key="gex_sym") or "SPX")
    gdays = g2.select_slider("Expirations within", [7, 14, 30, 45, 60, 90], value=45, format_func=lambda d: f"{d} days")
    g3.write("")
    go = g3.button("Calculate", type="primary")
    if go or st.session_state.get("gex_for") == (gsym, gdays):
        with st.spinner(f"Loading {gsym} option chains…"):
            res, err = gex_cached(gsym, gdays, time_bucket(1 if go else 600))
        st.session_state.gex_for = (gsym, gdays)
        if err:
            st.error(err)
        elif res:
            spot = res["spot"]

            def lvl(x):
                return "n/a" if x is None else f"{x:,.2f}"

            def dist(x):
                return None if x is None else f"{(x - spot) / spot * 100:+.2f}% from spot"

            r1 = st.columns(4)
            r1[0].metric("Spot", lvl(spot))
            r1[1].metric("Dealer gamma (per 1% move)", core.fmt_money(res["total_gex"], signed=True), res["regime"],
                         delta_color="normal" if res["total_gex"] >= 0 else "inverse")
            r1[2].metric("Gamma flip", lvl(res["flip"]), dist(res["flip"]), delta_color="off")
            em = res["expected_move"]
            r1[3].metric(f"Expected move ({core.to_date(res['expected_move_exp']):%m/%d})",
                         "n/a" if em is None else f"±{em:,.2f}", None if em is None else f"±{em / spot * 100:.2f}%", delta_color="off")
            r2 = st.columns(4)
            r2[0].metric("Call wall", lvl(res["call_wall"]), dist(res["call_wall"]), delta_color="off")
            r2[1].metric("Put wall", lvl(res["put_wall"]), dist(res["put_wall"]), delta_color="off")
            r2[2].metric(f"Max pain ({core.to_date(res['max_pain_exp']):%m/%d})", lvl(res["max_pain"]), dist(res["max_pain"]), delta_color="off")
            r2[3].metric("Put/call ratio", "n/a" if res["pcr_oi"] is None else f"{res['pcr_oi']:.2f} OI",
                         None if res["pcr_vol"] is None else f"{res['pcr_vol']:.2f} today's volume", delta_color="off")

            if res["total_gex"] >= 0:
                st.success("**Positive gamma.** Dealers tend to sell rallies and buy dips to stay hedged, which usually "
                           "dampens moves. Price is often pulled toward big strikes; the call wall acts as resistance.")
            else:
                st.error("**Negative gamma.** Dealers tend to sell into drops and buy into rallies, which can amplify moves. "
                         "Expect bigger swings, and whale flow is more likely to push price.")

            by = res["by_strike"]
            band = by[(by["Strike"] >= spot * 0.93) & (by["Strike"] <= spot * 1.07)].copy()
            band["Side"] = np.where(band["Net"] >= 0, "Positive", "Negative")
            bars = alt.Chart(band).mark_bar().encode(
                x=alt.X("Strike:Q", title="Strike", scale=alt.Scale(zero=False)),
                y=alt.Y("Net:Q", title="Net dealer gamma ($ per 1% move)", axis=alt.Axis(format="~s")),
                color=alt.Color("Side:N", scale=alt.Scale(domain=["Positive", "Negative"], range=["#13a36b", "#d9483b"]), legend=None),
                tooltip=[alt.Tooltip("Strike:Q"), alt.Tooltip("Net:Q", format="$,.0f"),
                         alt.Tooltip("C:Q", title="Calls", format="$,.0f"), alt.Tooltip("P:Q", title="Puts", format="$,.0f")],
            )
            marks = [{"Level": "Spot", "x": spot}] + ([{"Level": "Gamma flip", "x": res["flip"]}] if res["flip"] else [])
            rules = alt.Chart(pd.DataFrame(marks)).mark_rule(strokeDash=[5, 4], strokeWidth=2).encode(
                x="x:Q", color=alt.Color("Level:N", scale=alt.Scale(range=["#3b6cf6", "#a16be0"]), legend=alt.Legend(orient="top")))
            st.altair_chart((bars + rules).properties(height=360), width="stretch")

            curve = res["curve"]
            cchart = alt.Chart(curve).mark_line(strokeWidth=2.5).encode(
                x=alt.X("Price:Q", scale=alt.Scale(zero=False), title=f"If {gsym} were at…"),
                y=alt.Y("GEX:Q", title="Total dealer gamma", axis=alt.Axis(format="~s")))
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9aa3b2").encode(y="y:Q")
            st.altair_chart((cchart + zero).properties(height=220), width="stretch")
            st.caption(f"Based on open interest across {len(res['expirations'])} expirations as of the last close. "
                       "Assumes dealers are long the calls and short the puts customers trade, the common convention. "
                       "Real positioning is unknown, so treat levels as zones, not exact lines.")
    else:
        st.info("Pick an underlying and select **Calculate**. SPX takes 10-30 seconds; its chains are huge.")


# ======================================================================= did it stick
with tab_stick:
    st.markdown("Every scan saves its flags. Open interest updates overnight, so the next day this tab checks "
                "whether each flagged trade was **really a new position** and how it has done since.")
    s1, s2, s3 = st.columns([1, 1, 3])
    look = s1.select_slider("Look back", [2, 5, 10, 20, 30], value=10, format_func=lambda d: f"{d} days")
    smin = s2.slider("Only flags scored at least", 0, 100, 50)
    s3.write("")
    if s3.button("Check earlier flags", type="primary"):
        st.session_state.history = core.merge_history(storage.get_json("history.json", {}) or {}, st.session_state.history)
        today = core.now_et().date().isoformat()
        with st.spinner("Checking open interest and prices…"):
            rep, summ = core.confirm_report(st.session_state.history, today, chain_cached, close_cached, price, look, smin)
        st.session_state.stick = (rep, summ)
    rep_summ = st.session_state.get("stick")
    if rep_summ is None:
        n_keys = len(st.session_state.history)
        st.info(f"{n_keys:,} contracts in history. Flags need at least one day to be checked; scans from earlier days are graded here.")
    else:
        rep, summ = rep_summ
        if rep.empty:
            st.info("No flags from earlier days yet. Run scans today and check back tomorrow, or turn on the alert job so it scans for you.")
        else:
            k = st.columns(4)
            k[0].metric("Flags checked", f"{summ['checked']}/{summ['flags']}")
            k[1].metric("Confirmed new positions", "n/a" if summ["opened_pct"] is None else f"{summ['opened_pct']:.0f}%")
            k[2].metric("Directional bets: avg follow return", "n/a" if summ["dir_avg_return"] is None else f"{summ['dir_avg_return']:+.1f}%")
            k[3].metric("Directional bets: win rate", "n/a" if summ["dir_win_rate"] is None else f"{summ['dir_win_rate']:.0f}%",
                        f"{summ['dir_count']} trades", delta_color="off")
            st.dataframe(rep.drop(columns=["Opened"]), hide_index=True, height=480, column_config={
                "Follow return %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Price then": st.column_config.NumberColumn(format="$%.2f"),
                "Price now": st.column_config.NumberColumn(format="$%.2f"),
            })
            st.caption("Follow return = the option's price change from the flag to now, from the whale's side "
                       "(inverted for seller-driven flow). Uses mid prices when quotes exist. Ignores fees and slippage.")


# ======================================================================= my trades
@st.fragment(run_every=refresh)
def trades_panel():
    df = st.session_state.get("trades_current", st.session_state.trades)
    now_ = core.now_et()
    rows = []
    for _, r in df.iterrows():
        try:
            sym = core.base_symbol(str(r["symbol"]))
            if not sym or sym in ("NAN", "NONE") or pd.isna(r["expiration"]):
                continue
            k, px, n, typ = float(r["strike"]), float(r["price"]), int(r["contracts"]), "P" if str(r["type"]).upper().startswith("P") else "C"
            iv = float(r["iv"]) if pd.notna(r["iv"]) else 60.0
            sgn = -1 if str(r["side"]).lower().startswith("s") else 1
        except Exception:
            continue
        exp_d = core.to_date(r["expiration"])
        expired = exp_d < now_.date()
        if expired:
            spot = close_cached(sym, exp_d.isoformat())
            status = f"Expired, settled at {spot:,.2f}" if spot else "Expired, settlement unavailable"
            T = 0.0
        else:
            spot = price(sym)
            status = "Expires today" if exp_d == now_.date() else "Open"
            T = core.years_left(r["expiration"], now_)
        note = "" if pd.isna(r["note"]) else str(r["note"])
        scanner = note.lower().startswith("scanner")
        cost = px * 100 * n
        if spot:
            gk = core.bs_greeks(spot, k, T, iv, typ)
            value = gk["price"] * 100 * n
            pnl = sgn * (value - cost)
            delta_sh = 0.0 if expired else sgn * gk["delta"] * 100 * n
            theta = 0.0 if expired else sgn * gk["theta"] * 100 * n
        else:
            value = pnl = delta_sh = theta = float("nan")
        rows.append({"Contract": core.contract_label(sym, r["expiration"], k, typ),
                     "Source": "Scanner: all traders" if scanner else "Logged trade",
                     "Status": status,
                     "Side": ("Mostly sold" if sgn < 0 else "Mostly bought") if scanner else ("Sold" if sgn < 0 else "Bought"),
                     "Size": f"{n:,} @ ${px:.2f}",
                     "Spot": spot, "Breakeven": k + px if typ == "C" else k - px,
                     "DTE": (core.to_date(r["expiration"]) - now_.date()).days,
                     "Premium": cost, "Value": value, "P/L": pnl,
                     "P/L %": pnl / cost * 100 if cost and not math.isnan(pnl) else float("nan"),
                     "Delta (shares)": delta_sh, "Theta / day": theta, "_sym": sym, "_rows": (k, px, n, typ, iv, sgn, r["expiration"])})
    if not rows:
        st.info("No trades yet. Track contracts from the scanner, or add rows below.")
        return
    pos = pd.DataFrame(rows)
    logged = pos[pos["Source"] == "Logged trade"]
    scan = pos[pos["Source"] != "Logged trade"]

    def total(df, col):
        return df[col].sum(skipna=True) if len(df) and df[col].notna().any() else float("nan")

    m = st.columns(4)
    m[0].metric("Logged trades P/L", core.fmt_money(total(logged, "P/L"), signed=True),
                help="Only trades you logged as a single trader's position. Scanner rows are excluded.")
    m[1].metric("Logged premium at work", core.fmt_money(total(logged, "Premium")))
    sc_ret = (total(scan, "P/L") / total(scan, "Premium") * 100) if len(scan) and total(scan, "Premium") else float("nan")
    m[2].metric("Scanner contracts: avg return", "n/a" if math.isnan(sc_ret) else f"{sc_ret:+.0f}%",
                help="Return on the contracts themselves since you tracked them. Dollar totals are left out on purpose.")
    m[3].metric("Positions", len(pos))
    if len(scan):
        st.info("**Scanner rows show the whole day's volume across every trader**, not one whale's position. "
                "Every contract had a buyer and a seller, so their dollar P/L is the market's, not one person's. "
                "Read the **P/L %** column for how the contract itself did.")
    shown = pos.drop(columns=["_sym", "_rows"])
    st.dataframe(
        shown.style.format({"Spot": lambda v: "n/a" if pd.isna(v) else f"${v:,.2f}", "Breakeven": "${:,.2f}",
                            "Premium": "${:,.0f}", "Value": lambda v: "n/a" if pd.isna(v) else f"${v:,.0f}",
                            "P/L": lambda v: core.fmt_money(v, signed=True), "P/L %": lambda v: "n/a" if pd.isna(v) else f"{v:+.0f}%",
                            "Delta (shares)": lambda v: "n/a" if pd.isna(v) else f"{v:+,.0f}",
                            "Theta / day": lambda v: core.fmt_money(v, signed=True)})
        .map(lambda v: "color:#0f8a54;font-weight:600" if isinstance(v, float) and v > 0 else
             ("color:#c2352b;font-weight:600" if isinstance(v, float) and v < 0 else ""), subset=["P/L", "P/L %"]),
        hide_index=True)
    st.caption(f"Updated {now_:%I:%M:%S %p} ET every {refresh}s. Before expiration, values are Black-Scholes estimates from each trade's IV.")

    syms = list(dict.fromkeys(pos["_sym"]))
    csym = st.selectbox("Payoff chart", syms, key="payoff_sym")
    legs = [x for s, x in zip(pos["_sym"], pos["_rows"]) if s == csym]
    spot = price(csym) or legs[0][0]
    ks = [leg[0] for leg in legs] + [spot]
    xs = np.linspace(min(ks) * 0.85, max(ks) * 1.15, 140)
    data = []
    for x in xs:
        a = b = 0.0
        for k, px, n, typ, iv, sgn, exp in legs:
            a += sgn * (core.option_value(x, k, core.years_left(exp, now_), iv, typ) - px) * 100 * n
            b += sgn * (core.option_value(x, k, 0, iv, typ) - px) * 100 * n
        data += [{"Price": x, "P/L": a, "Curve": "Now"}, {"Price": x, "P/L": b, "Curve": "At expiration"}]
    line = alt.Chart(pd.DataFrame(data)).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Price:Q", scale=alt.Scale(zero=False), title=f"{csym} price"),
        y=alt.Y("P/L:Q", axis=alt.Axis(format="~s"), title="P/L ($)"),
        color=alt.Color("Curve:N", scale=alt.Scale(range=["#3b6cf6", "#9aa3b2"]), legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("Curve:N", scale=alt.Scale(range=[[1, 0], [6, 4]]), legend=None))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9aa3b2").encode(y="y:Q")
    now_rule = alt.Chart(pd.DataFrame({"x": [spot]})).mark_rule(strokeDash=[2, 3]).encode(x="x:Q")
    st.altair_chart((line + zero + now_rule).properties(height=300), width="stretch")


with tab_trades:
    box = st.container()
    with st.expander("Edit trades", expanded=False):
        st.caption("Side: buy if it filled at the ask, sell if at the bid. Changes save automatically.")
        edited = st.data_editor(
            st.session_state.trades, key=f"trades_editor_{st.session_state.trades_ver}", num_rows="dynamic", hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", required=True),
                "expiration": st.column_config.DateColumn("Expiration", format="MM/DD/YY", required=True),
                "strike": st.column_config.NumberColumn("Strike", min_value=0.0, step=0.5),
                "type": st.column_config.SelectboxColumn("Call/Put", options=["C", "P"], default="C"),
                "contracts": st.column_config.NumberColumn("Contracts", min_value=1, step=1),
                "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01, format="$%.3f"),
                "side": st.column_config.SelectboxColumn("Side", options=["buy", "sell"], default="buy"),
                "iv": st.column_config.NumberColumn("IV %", min_value=1.0, step=1.0, default=60.0),
                "note": st.column_config.TextColumn("Note"),
            })
        st.session_state.trades_current = edited
        save_trades(edited)
    with box:
        trades_panel()


# ======================================================================= alerts
with tab_alerts:
    status = storage.get_json("status.json", None)
    a1, a2 = st.columns([1.2, 1])
    with a1:
        st.markdown("#### Alert rules")
        st.caption("The scheduled job scans your saved watchlist every 15 minutes during market hours "
                   "and messages you when something crosses these rules.")
        amin = st.slider("Alert when score is at least", 40, 100, int(S["alert_min_score"]))
        acats = st.multiselect("Categories to alert on", core.CATEGORIES, default=S["alert_categories"])
        if st.button("Save alert rules", type="primary"):
            S.update({"alert_min_score": amin, "alert_categories": acats})
            storage.put_json("settings.json", S, message="Update alert rules")
            st.toast("Alert rules saved.")
        st.caption(f"Watchlist used by the job: {', '.join(S['watchlist'])}. Change it in the sidebar, then "
                   "select **Save as alert watchlist** on the scanner tab.")
    with a2:
        st.markdown("#### Job status")
        if storage.kind != "github":
            st.warning("Cloud sync is off, so the app can't see the alert job. Add GITHUB_TOKEN and GITHUB_REPO "
                       "to the app's secrets (see the Guide tab).")
        elif not status:
            st.info("The alert job hasn't reported yet. Add the workflow file and wait for the next market-hours run, "
                    "or start it by hand from GitHub → Actions → Whale Watch scanner → Run workflow.")
        else:
            st.metric("Last run", status.get("last_run", "?").replace("T", " "), status.get("market", ""), delta_color="off")
            st.write(f"Flagged **{status.get('flagged', 0)}** contracts and sent **{status.get('alerts', 0)}** alert(s) on the last run. "
                     f"Signals: {status.get('signals_opened', 0)} opened, {status.get('signals_closed', 0)} closed, "
                     f"{status.get('signals_open', 0)} open.")
            if not status.get("alert_channel"):
                st.warning("No Telegram or Discord set up in GitHub secrets, so alerts are logged but not sent.")
            for e in status.get("errors", []):
                st.caption(f"⚠️ {e}")
        tg, chat, dc = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID"), secret("DISCORD_WEBHOOK_URL")
        if (tg and chat) or dc:
            if st.button("Send a test alert"):
                errs = core.send_alert("🐋 Whale Watch test alert. You're all set.", tg, chat, dc)
                if errs:
                    st.error("; ".join(errs))
                else:
                    st.success("Sent!")
    st.markdown("#### Recent alerts")
    log = storage.get_json("alerts.json", {}) or {}
    rows = [{"Date": d, **v} for d in sorted(log, reverse=True) for v in (log[d].values() if isinstance(log[d], dict) else [])]
    if rows:
        al = pd.DataFrame(rows)
        al["premium"] = al["premium"].map(core.fmt_money)
        st.dataframe(al, hide_index=True)
    else:
        st.caption("None yet.")


# ======================================================================= guide
with tab_guide:
    st.markdown("""
#### How to read the scanner
**Score (0-100)** adds up the evidence that a trade is a real, deliberate bet: size versus your minimum,
volume versus open interest (new money), paying the ask (urgency), short expiry and out-of-the-money strikes,
hedging flow as a share of the stock's daily volume, being flagged on several days, and a confirmed open-interest
increase. It subtracts points for flow that is probably not directional.

**Categories**
- **Directional bet**: bought at or near the ask, more volume than open interest, no matching legs. The cleanest signal.
- **Seller-driven**: filled at or near the bid. Someone is selling premium: sold calls lean bearish, sold puts lean bullish.
- **Spread / roll / combo**: another contract traded almost the same volume, so this is probably one leg of a bigger, often capped, trade.
- **Likely hedge**: out-of-the-money index puts a week or more out, or volatility calls. Usually portfolio insurance, not a crash bet.
- **May be closing**: volume below open interest and not confirmed new. Could be someone exiting.

**Key columns**
- **Fill**: where the last trade printed against the bid and ask. It describes the last print only, not every contract that day.
- **Hedge $**: the stock dealers must buy or sell to hedge (volume × delta × 100 × price). This, not premium, is what moves markets.
- **% ADV**: Hedge $ as a percentage of the stock's normal daily dollar volume. Above about 5% can move a single stock. Blank for indexes.
- **Confirmation**: filled in once a contract has been flagged on an earlier day and open interest has updated.

#### Signals (paper trading)
When a scan finds a **directional bet** scoring at or above your threshold, a paper position opens: you "buy" the
same contract at the current ask. It closes on whichever comes first: **profit target**, **stop loss**,
**breakeven stop** (the stop moves to your entry once you're up by the set amount), a **time exit** on expiration
day so 0DTE trades never ride into the close, or the **max hold**. Exits assume you sell at the bid.
Rules to keep in mind: exits are checked every minute while the app is open and every 15 minutes by the alert job,
so real stops would trigger sooner and fill differently. Judge the rules by the profit factor over 30+ trades, not
a handful of wins.

#### Whale moves
Each directional bet scoring 70+ is followed after the print. **Confirmed holding**: open interest rose overnight
by at least half the whale's size. **Didn't hold overnight**: it was a day trade or a closing trade. **Whale out**:
open interest fell by 20%+ of the position, so the whale most likely sold that much the day before; the exit price
is that day's price. **Possible exit** is a same-day warning from heavy selling at the bid, confirmed or ruled out the
next morning. Turn on **Exit when the whale exits** in Signal rules to have your paper trades follow the whale out.

#### Dealer positioning
- **Positive gamma**: moves tend to get dampened. **Negative gamma**: moves tend to get amplified.
- **Gamma flip**: the price where the regime changes. Below it, markets usually get jumpier.
- **Call wall / put wall**: strikes with the most dealer gamma; they often act as resistance and support.
- **Max pain**: the price where option holders lose the most at expiration. A weak magnet, strongest near expiry.

#### Honest limits
Yahoo gives each contract's total for the day, not each sweep or block, and is about 15 minutes delayed. Fill side
is a guess from the last print. Dealer positioning is a model. Even confirmed whale trades are often hedges or legs of
positions you can't see. Use this to find trades worth a closer look, never as a reason on its own to buy or sell.
This tool is for education and is not financial advice.

#### Turning on cloud sync and phone alerts
See **README.md** in your GitHub repo for the step-by-step setup.
""")
