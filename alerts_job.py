"""
Scheduled scanner. GitHub Actions runs this every 15 minutes on weekdays
(see .github/workflows/scan.yml). It:

  1. reads your watchlist and alert settings from the data branch
  2. scans each ticker for unusual flow
  3. scores it against the saved history (next-day open-interest checks)
  4. saves the flags to history so the app's "Did it stick?" tab can grade them
  5. sends a Telegram and/or Discord alert for new high-score flow

Run locally with:  python alerts_job.py --force
"""
from __future__ import annotations

import os
import sys

import pandas as pd

import core

MAX_ALERTS_PER_RUN = 8


def main() -> int:
    force = "--force" in sys.argv or os.environ.get("FORCE_RUN", "").lower() == "true"
    now = core.now_et()
    status, detail = core.market_status(now)
    if status != "open" and not force:
        print(f"Market closed ({detail}). Nothing to do.")
        return 0

    storage = core.make_storage(os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPO"),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    settings = core.load_settings(storage)
    today = now.date().isoformat()
    history = storage.get_json("history.json", {}) or {}

    frames, errors = [], []
    for sym in settings["watchlist"]:
        df, err = core.scan_symbol(sym, int(settings["max_expirations"]), core.min_premium_for(sym, settings), now)
        if err:
            errors.append(err)
            print(err)
        if df is not None and not df.empty:
            frames.append(df)
    flow = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    flow = core.score_flow(flow, history, today, settings) if not flow.empty else flow
    print(f"{len(flow)} contracts above the premium thresholds.")

    if not flow.empty:
        history = core.record_history(history, flow, today)
        storage.put_json("history.json", history, merge=core.merge_history, message=f"Scan {now:%Y-%m-%d %H:%M} ET")

    sent_log = storage.get_json("alerts.json", {}) or {}
    sent_today = set(sent_log.get(today, {}).keys()) if isinstance(sent_log.get(today), dict) else set()
    tg_token, tg_chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    discord = os.environ.get("DISCORD_WEBHOOK_URL")
    channel = bool((tg_token and tg_chat) or discord)

    new_alerts = []
    if not flow.empty:
        hits = flow[(flow["Score"] >= int(settings["alert_min_score"]))
                    & flow["Category"].isin(settings["alert_categories"])
                    & ~flow["Key"].isin(sent_today)]
        for _, r in hits.head(MAX_ALERTS_PER_RUN).iterrows():
            text = core.alert_text(r)
            errs = core.send_alert(text, tg_token, tg_chat, discord) if channel else ["No alert channel configured."]
            errors.extend(e for e in errs if channel)
            new_alerts.append(r)
            sent_log.setdefault(today, {})[r["Key"]] = {
                "time": now.strftime("%H:%M"), "contract": r["Contract"], "score": int(r["Score"]),
                "category": r["Category"], "bias": r["Bias"], "premium": float(r["Premium"]),
                "sent": channel and not errs,
            }
    keep = sorted(sent_log)[-14:]
    sent_log = {d: sent_log[d] for d in keep}
    if new_alerts:
        storage.put_json("alerts.json", sent_log, message=f"Alerts {now:%Y-%m-%d %H:%M} ET")

    storage.put_json("status.json", {
        "last_run": now.isoformat(timespec="minutes"), "market": detail,
        "scanned": settings["watchlist"], "flagged": int(len(flow)),
        "alerts": len(new_alerts), "alert_channel": channel, "errors": errors[:10],
    }, message="Job status")
    print(f"Sent {len(new_alerts)} alert(s). Channel configured: {channel}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
