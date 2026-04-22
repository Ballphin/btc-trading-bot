# Signal History Persistence Guide

By default, Signal History and Pulse data are stored in `eval_results/` inside the git repo. This means:
- ❌ Data is lost on fresh clones
- ❌ Data is wiped on Render free tier restarts
- ❌ Data is lost when switching branches with clean working directory

This guide shows how to **persist your data** across all these scenarios.

---

## Quick Start

Run the interactive setup:

```bash
python scripts/setup_persistence.py
```

Then restart your server.

---

## Persistence Methods

### Method 1: GitHub Gist Sync (Recommended for Cloud)

**Best for:** Render free tier, Railway, Heroku, or any ephemeral filesystem

**Pros:**
- ✅ Free, zero infrastructure
- ✅ Survives filesystem wipes
- ✅ Automatic sync (push on write, pull on startup)
- ✅ Works on any platform

**Cons:**
- ⚠️ Limited to ~100 most recent analyses per ticker (rolling window)
- ⚠️ Requires GitHub account

**Setup:**

1. **Create GitHub Personal Access Token**
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scope: **gist** (create and edit gists)
   - Generate and copy token

2. **Create Two Empty Gists**
   - https://gist.github.com (create two, any content)
   - Copy gist IDs from URLs:
     - `https://gist.github.com/username/XXXXXXXXXX` → ID is `XXXXXXXXXX`

3. **Set Environment Variables**

   Local development (`.env`):
   ```bash
   GITHUB_TOKEN=ghp_your_token_here
   PULSE_GIST_ID=xxxxxxxxxx
   HISTORY_GIST_ID=yyyyyyyyyy
   ```

   Render dashboard:
   - Go to Dashboard → Your Service → Environment
   - Add three secrets above

4. **Restart Server**
   
   Check logs for:
   ```
   📦 PERSISTENCE STATUS
   ==================================================
   ✓ External eval_results dir: False (eval_results)
   ✓ GitHub Token set: True
   ✓ Pulse Gist ID set: True
   ✓ History Gist ID set: True
   ✓ Pulse sync enabled: True
   ✓ History sync enabled: True
   ✅ Data will persist across restarts
   ==================================================
   ```

---

### Method 2: External Data Directory (Recommended for Local)

**Best for:** Local development, Render paid tier with Disk

**Pros:**
- ✅ Fast local access
- ✅ Unlimited storage
- ✅ No external dependencies
- ✅ Survives git operations

**Cons:**
- ⚠️ Doesn't sync across machines
- ⚠️ Still needs backup for true persistence

**Setup:**

1. **Set Environment Variable**

   ```bash
   export EVAL_RESULTS_DIR=$HOME/TradingAgentsData/eval_results
   ```

   Or in `.env`:
   ```bash
   EVAL_RESULTS_DIR=/Users/yourname/TradingAgentsData/eval_results
   ```

2. **Create Directory**
   
   Server will auto-create on startup, or manually:
   ```bash
   mkdir -p $EVAL_RESULTS_DIR
   ```

3. **Migrate Existing Data** (optional)

   If you have data in `eval_results/`:
   ```bash
   python scripts/migrate_history.py
   ```

4. **Restart Server**

   Check logs:
   ```
   📦 PERSISTENCE STATUS
   ==================================================
   ✓ External eval_results dir: True (/Users/.../TradingAgentsData/eval_results)
   ✓ GitHub Token set: False
   ...
   ✅ Data will persist across restarts
   ==================================================
   ```

**Render Paid Tier with Disk:**

1. Create Disk in Render Dashboard:
   - Name: `tradingagents-data`
   - Size: 1GB (or more)
   - Mount Path: `/data`

2. Set Environment Variable:
   ```
   EVAL_RESULTS_DIR=/data/eval_results
   ```

3. That's it! Data persists across restarts.

---

### Method 3: Both (Maximum Safety)

Use external directory + GitHub Gist for redundancy.

**Setup:**

```bash
# .env
EVAL_RESULTS_DIR=/Users/yourname/TradingAgentsData/eval_results
GITHUB_TOKEN=ghp_xxx
PULSE_GIST_ID=xxx
HISTORY_GIST_ID=yyy
```

**Benefits:**
- Fast local access via external directory
- Cloud backup via Gist for Render free tier
- Can recover if one method fails

---

## Migration Guide

### Moving Existing Data

If you already have signal history in `eval_results/`:

```bash
# 1. Configure external directory
export EVAL_RESULTS_DIR=$HOME/TradingAgentsData/eval_results

# 2. Run migration
python scripts/migrate_history.py

# 3. Restart server
python server.py
```

The migration script will:
- Copy all log files, pulse data, shadow decisions
- Preserve timestamps
- Backup original to `eval_results_backup_YYYYMMDD_HHMMSS/`
- Create symlink (optional)

### Switching Between Methods

**From Gist to External:**
```bash
# 1. Setup external directory
python scripts/setup_persistence.py  # Choose option 2

# 2. Pull data from Gist (happens automatically on startup)
export EVAL_RESULTS_DIR=$HOME/TradingAgentsData/eval_results
python server.py
# [Server will pull from Gist on startup]

# 3. Optionally remove Gist env vars
```

**From External to Gist:**
```bash
# 1. Setup Gist
python scripts/setup_persistence.py  # Choose option 1

# 2. Restart - data will push to Gist
python server.py
```

---

## Troubleshooting

### "Signal history will NOT persist across restarts!"

**Cause:** No persistence method configured

**Fix:**
```bash
python scripts/setup_persistence.py
# Restart server
```

### "GistSync pull failed"

**Causes:**
- Token doesn't have `gist` scope
- Token expired
- Gist ID is wrong

**Fix:**
1. Verify token at https://github.com/settings/tokens
2. Check gist IDs match URLs
3. Check logs for specific error

### Data missing after restart

**Checklist:**
1. Are env vars loaded? Print them:
   ```bash
   python -c "import os; print(os.environ.get('EVAL_RESULTS_DIR'))"
   ```
2. Check server startup logs for persistence status
3. Verify Gist has files:
   ```bash
   curl -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/gists/$HISTORY_GIST_ID
   ```

### "No space left on device" (Gist)

**Cause:** Gist has 10MB file limit per file

**Behavior:** System automatically rolls off old entries, keeping last 100 analyses per ticker.

**Fix:** If you need more history, use external directory method instead.

---

## Environment Variables Reference

| Variable | Required For | Description |
|----------|--------------|-------------|
| `EVAL_RESULTS_DIR` | External dir | Path to store data outside repo |
| `PULSE_DIR` | External dir | Pulse subdirectory (auto-derived) |
| `SHADOW_DIR` | External dir | Shadow subdirectory (auto-derived) |
| `GITHUB_TOKEN` | Gist sync | GitHub Personal Access Token with `gist` scope |
| `PULSE_GIST_ID` | Gist sync | ID of pulse data gist |
| `HISTORY_GIST_ID` | Gist sync | ID of history data gist |

---

## How It Works

### Gist Sync Flow

```
Startup:
  Server starts
    ↓
  Pull from Gist (if configured)
    ↓
  Restore files to EVAL_RESULTS_DIR

Analysis Complete:
  Result saved locally
    ↓
  Fire-and-forget push to Gist
    ↓
  Data persists to GitHub

Next Startup (fresh deploy):
  Server starts
    ↓
  Pull from Gist
    ↓
  Data restored from GitHub
```

### External Directory Flow

```
All operations → EVAL_RESULTS_DIR (outside repo)

Git operations don't affect it
Fresh clones don't wipe it
Render Disk persists across restarts
```

---

## FAQ

**Q: Can I use both methods?**  
A: Yes! Use external directory for speed + Gist for cloud backup.

**Q: What happens if I don't configure persistence?**  
A: Data stays in `eval_results/` and will be lost on fresh clones or Render restarts.

**Q: Is my data secure on GitHub Gist?**  
A: Gists are private by default (unless you share the URL). However, they're still on GitHub's servers - don't store sensitive financial data.

**Q: How much data can I store?**  
- Gist: ~10MB per file, rolling window of 100 analyses per ticker
- External dir: Unlimited (disk space limited)

**Q: Can I share history across multiple machines?**  
A: Yes, use Gist method - all machines sync to the same Gist.

**Q: What about the Pulse scheduler state?**  
A: Scheduler state also syncs to Gist (if configured) via `scheduler_state.json`.

---

## Quick Commands

```bash
# Check persistence status
python -c "import os; print('✓' if os.environ.get('GITHUB_TOKEN') else '✗', 'GitHub Token')"

# View current data directory
python -c "from pathlib import Path; print(Path('eval_results').absolute())"

# Count analyses
find eval_results -name "full_states_log_*.json" | wc -l

# Backup manually
cp -r eval_results eval_results_backup_$(date +%Y%m%d)
```

---

## Need Help?

Run the setup script for interactive guidance:

```bash
python scripts/setup_persistence.py
```

Or check the server logs on startup - they show exactly which persistence methods are active.
