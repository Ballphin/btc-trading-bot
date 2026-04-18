# Deploying on Render.com — Free Tier

A step-by-step guide to running TradingAgents (Quant Pulse + Dashboard) on
Render's free tier, surviving the two things Render's free plan does to your
service:

1. **15-minute idle sleep** — kills the in-process scheduler
2. **Ephemeral filesystem** — wipes `pulse.jsonl` on every restart

Both are solved here without any paid upgrade.

---

## TL;DR

1. Push this repo to GitHub
2. Create a blank **GitHub Gist** (private or public, both work) with a placeholder file
3. Create a **GitHub Personal Access Token** with the `gist` scope
4. Deploy via Render Blueprint (`render.yaml` is already in the repo)
5. Set `GITHUB_TOKEN` and `PULSE_GIST_ID` as environment variables in Render
6. Set `VITE_API_BASE_URL` on the frontend to point at the backend
7. Configure **UptimeRobot** (free) to ping `/api/health?tick=1` every 5 minutes

Result: pulses fire every 5 min, the service never idles, and pulse history
survives filesystem wipes.

---

## 1. How the free-tier survival mode works

Two pieces of code, both already in the repo:

### Self-scheduling health endpoint — `server.py` → `/api/health?tick=1`

- When queried with `?tick=1`, the endpoint checks if a pulse is due
  (based on `last_run` + `interval_minutes`) and, if so, runs one
  synchronously.
- An external uptime pinger calling this URL every 5 min therefore:
  - Keeps the service awake (resets Render's 15-min idle timer)
  - **Acts as the scheduler itself** — no reliance on the in-process
    `asyncio` task which dies during idle
- The pulse scheduler UI toggle still works: if it's OFF, tick=1 does
  nothing. Turn it ON once from the dashboard.

### Gist-backed persistence — `tradingagents/pulse/gist_sync.py`

- On server startup: downloads every `pulse_<TICKER>.jsonl` file from the
  configured Gist into `eval_results/pulse/<TICKER>/pulse.jsonl`.
- After every new pulse entry: pushes the updated file back to the same
  Gist in a background thread (no latency added to the pulse itself).
- If either `GITHUB_TOKEN` or `PULSE_GIST_ID` is missing, both become
  no-ops. Safe to run without them on paid plans with a real disk.

Rate budget: GitHub allows 5000 authenticated calls/hour. At 5-min
cadence across 3 tickers that's 864 calls/day. Comfortably under limit.

---

## 2. One-time setup outside Render

### Step A — Create the Gist

1. Go to <https://gist.github.com/> (logged in)
2. Create a new Gist with any content (e.g. filename `README.md`, content `TradingAgents pulse archive`)
3. Choose "Create secret gist" (unlisted — only people with the URL can see it)
4. Copy the Gist ID from the URL:
   `https://gist.github.com/<user>/`**`abc123def456...`** ← that's `PULSE_GIST_ID`

### Step B — Create a Personal Access Token

1. Go to <https://github.com/settings/tokens?type=beta> (fine-grained tokens)
2. Click **"Generate new token"**
3. Name: `TradingAgents-pulse`
4. Expiration: 1 year (max)
5. Repository access: **No repositories**
6. Account permissions → **Gists: Read and write**
7. Click Generate and copy the `github_pat_...` token. **This is `GITHUB_TOKEN`.**

*(Classic PATs also work — just need the `gist` scope. Fine-grained is
stricter and therefore safer.)*

---

## 3. Deploy on Render

### Option A — Blueprint (recommended)

The repo ships `render.yaml`. On Render:

1. Dashboard → **New** → **Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and offers to create two services:
   - `tradingagents-api` (web service, Python, free plan)
   - `tradingagents-frontend` (static site, free)
4. Click **Apply**

### Option B — Manual (matches the screenshots in the IDE)

Backend web service:

- **Environment**: Python
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free
- **Health Check Path**: `/api/health`

Frontend static site (**create separately**):

- **Build Command**: `cd frontend && npm ci && npm run build`
- **Publish Directory**: `frontend/dist`
- **Rewrite rule**: `/* → /index.html` (rewrite, 200)

---

## 4. Environment variables

Set these in the Render dashboard (Environment tab) for the **backend service**:

| Variable | Required? | Value | Purpose |
|---|---|---|---|
| `PYTHONUNBUFFERED` | Yes | `1` | Makes logs flush to Render immediately |
| `GITHUB_TOKEN` | **Yes for free tier** | `github_pat_...` from step B | Gist persistence |
| `PULSE_GIST_ID` | **Yes for free tier** | Gist ID from step A | Which gist holds the history |
| `TRADINGAGENTS_RESULTS_DIR` | No | `/opt/render/project/src/eval_results` | Default is fine; leave unset |
| `OPENAI_API_KEY` | Only if using LLM analyst | `sk-...` | LLM calls |
| `ANTHROPIC_API_KEY` | Optional | `sk-ant-...` | Alt LLM |
| `PULSE_DISCORD_WEBHOOK_URL` | Optional | webhook URL | Discord alerts |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Optional | - | Telegram alerts |

For the **frontend static site**:

| Variable | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://tradingagents-api.onrender.com` (replace with your backend URL) |

---

## 5. Set up the uptime pinger (critical)

Without this the service still idles after 15 min. Pick one:

### UptimeRobot (recommended — free, easy)

1. Sign up at <https://uptimerobot.com> (free tier: 50 monitors, 5-min interval)
2. **Add New Monitor**:
   - Monitor Type: **HTTPS**
   - Friendly Name: `TradingAgents pulse tick`
   - URL: `https://tradingagents-api.onrender.com/api/health?tick=1`
   - Monitoring Interval: **5 minutes**
3. Save. Done.

### cron-job.org (alternative)

- Free, reliable, supports minute granularity
- URL: `https://tradingagents-api.onrender.com/api/health?tick=1`
- Schedule: `*/5 * * * *` (every 5 min)

### GitHub Actions (alternative, zero sign-ups)

Add `.github/workflows/ping.yml`:

```yaml
name: Keep-alive ping
on:
  schedule:
    - cron: "*/5 * * * *"
  workflow_dispatch:
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS "https://tradingagents-api.onrender.com/api/health?tick=1"
```

Note: GitHub Actions cron is best-effort (can run late by 5-15 min under
load). UptimeRobot is more reliable for 5-min cadence.

---

## 6. First-run checklist

After the first deploy succeeds:

1. Open the frontend URL → should load the dashboard
2. Open the backend URL → `/api/health` → confirm JSON with `"status": "ok"`
3. Call `https://<backend>.onrender.com/api/health?tick=1` manually once
   → response should show `"ticked": true` and (if pulse scheduler enabled)
     a `ran` array with signal results
4. Visit the **Pulse** tab in the frontend and toggle the scheduler **ON**
5. Wait 5 min, reload. New pulse entry should appear.
6. Redeploy the service (force a restart). Reload dashboard. History should
   still be there — proves Gist persistence is working.

---

## 7. Known limits on free tier

- **750 instance-hours/month** across all free web services. One always-on
  service uses ~730 hours. If you run two (backend + frontend), note:
  *static sites don't count toward the 750h*. You're fine.
- **Cold start after ~15 min no traffic**: avoided by the UptimeRobot ping.
  If the ping ever fails for >15 min the next request pays ~30-60s cold start.
- **Render free services are slow to build** (~3-5 min). CPU during runtime
  is fine for the Pulse engine (all heavy computation is under 1 s).
- **Egress**: Hyperliquid API is free and reliable. No surprise bills.
- **Gist size cap**: we auto-rotate at ~900 KB (~5000 lines). You'll never
  hit a Gist limit in practice.

---

## 8. Upgrading later

When you want bullet-proof persistence:

- **Starter plan ($7/mo)** → no idle sleep, no 750h cap
- **Persistent Disk ($0.25/GB/mo)** → mount at `TRADINGAGENTS_RESULTS_DIR`,
  set `GITHUB_TOKEN=""` to disable Gist sync. Nothing else changes.

---

## 9. Troubleshooting

**Symptom**: dashboard loads but Pulse history is empty after restart.
**Fix**: Check backend logs for `[GistSync]`. If you see `skipped=True`
the env vars aren't set. If you see `Push failed: 401` the token is wrong.

**Symptom**: pulses stop firing while nobody's looking.
**Fix**: UptimeRobot monitor paused, or the URL is wrong. It must be
`/api/health?tick=1` (not `/api/health`).

**Symptom**: cold start every time you load the dashboard.
**Fix**: UptimeRobot interval is >15 min, or pings are failing. Check the
monitor's response-log tab.

**Symptom**: `ran: [{"ok": false, "error": "..."}]` in `/api/health?tick=1`.
**Fix**: Backend can't reach Hyperliquid, or report build is failing.
Check backend logs around the timestamp.
