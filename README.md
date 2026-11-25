# 🐋 Whale Watch

Options-flow intelligence for SPX, stocks and ETFs: an unusual-flow scanner with a conviction
score, next-day open-interest confirmation, dealer gamma levels, live trade tracking, and phone
alerts from a free scheduled scanner.

*For education only. Not financial advice. Data is from Yahoo Finance: free, unofficial, usually
about 15 minutes delayed.*

## What's inside

| File | What it does |
|---|---|
| `whale_watch.py` | The app (Streamlit). Point Streamlit Cloud at this file. |
| `core.py` | The engine: data, Greeks, scoring, dealer gamma, storage, alerts. |
| `alerts_job.py` | The scheduled scanner that runs on GitHub Actions and sends alerts. |
| `.github/workflows/scan.yml` | Runs `alerts_job.py` every 15 minutes on weekdays. |
| `requirements.txt` | Python packages. |

## Setup

### 1. Upload the files (replacing the old ones)
In your repo on GitHub: **Add file → Upload files**, drag in `whale_watch.py`, `core.py`,
`alerts_job.py`, `requirements.txt` and `README.md`, then **Commit changes**.

The workflow file sits in a hidden folder, so create it by hand:
**Add file → Create new file**, type the name `.github/workflows/scan.yml` (the slashes make
the folders), paste in the contents of `scan.yml`, then **Commit changes**.

Streamlit redeploys the app automatically within a minute or two.

### 2. Turn on cloud sync (so history, trades and settings never reset)
1. On GitHub: profile picture → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. Name it `whale-watch`, expiration up to 1 year, **Only select repositories → whale-watch**.
3. Under **Repository permissions**, set **Contents** to **Read and write**. Generate, and copy the token.
4. On share.streamlit.io: your app's **⋮ menu → Settings → Secrets**, paste the following, and save:
   ```toml
   GITHUB_TOKEN = "github_pat_paste_yours_here"
   GITHUB_REPO = "LG-CDW/whale-watch"
   ```
The sidebar now shows ☁️ *Synced to github.com/…*. Data is stored on a separate `data` branch,
so it never redeploys the app.

### 3. Turn on phone alerts (Telegram recommended, free)
1. In Telegram, message **@BotFather**, send `/newbot`, follow the prompts, and copy the **bot token**.
2. Send any message to your new bot. Then open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and copy the number after `"chat":{"id":`.
   That is your **chat ID**.
3. On GitHub, open your repo's **Settings → Secrets and variables → Actions → New repository secret**.
   Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
   (For Discord instead: Server settings → Integrations → Webhooks → New webhook → copy URL,
   and save it as `DISCORD_WEBHOOK_URL`.)
4. Optional: add the same Telegram values to the Streamlit secrets so the app's **Send a test alert** button works.
5. Test it: repo **Actions** tab → **Whale Watch scanner** → **Run workflow**. A manual run scans
   even when the market is closed. The app's **Alerts** tab shows the job's last run.

### 4. Choose what the job watches
In the app, set the watchlist and thresholds in the sidebar, then select **Save as alert watchlist**.
Set the alert score and categories on the **Alerts** tab.

## Good to know
- GitHub's scheduler can run a few minutes late, and it pauses scheduled jobs in repos with no
  activity for 60 days. If alerts stop, open the Actions tab and re-enable the workflow.
- Yahoo sometimes rate-limits or changes things without notice. If a scan returns errors, wait a
  few minutes. `pip install -U yfinance` (or redeploying) usually fixes breakages.
- Yahoo gives each contract's daily totals, not individual sweeps or blocks. To go further, swap
  `fetch_chain` and `fetch_price` in `core.py` for a paid feed (Polygon, Tradier, Unusual Whales
  API). Everything else keeps working.

## Run on your own computer
```bash
pip install -r requirements.txt
streamlit run whale_watch.py
python alerts_job.py --force    # one scan plus alerts, using env vars for secrets
```
