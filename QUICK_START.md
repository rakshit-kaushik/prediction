# Quick Start: URL-Based Market Selection

## The Problem
Finding market slugs manually is tedious:
- Visit Polymarket page
- Inspect network requests
- Find the right market among many outcomes
- Copy the slug
- Paste into config.py

## The Solution
**Just paste the Polymarket URL!**

```bash
python data_pipeline/00_select_market_from_url.py
```

---

## Usage Examples

### Example 1: Event with Multiple Markets (Interactive)

```bash
$ python data_pipeline/00_select_market_from_url.py

Enter Polymarket URL: https://polymarket.com/event/2024-us-presidential-election

Found 50 markets for: 2024-us-presidential-election

────────────────────────────────────────────────────────────────────────
1. Trump wins 2024 US Presidential Election?
   Prices: Yes $0.48 | No $0.52
   Volume: $2,150,000,000
   Slug: trump-wins-2024-presidential-election
────────────────────────────────────────────────────────────────────────
2. Biden wins 2024 US Presidential Election?
   Prices: Yes $0.52 | No $0.48
   Volume: $1,800,000,000
   Slug: biden-wins-2024-presidential-election
────────────────────────────────────────────────────────────────────────
...

Select market (1-50): 1

✅ Selected: Trump wins 2024 US Presidential Election?
✅ Updated config.py

Run data pipeline now? (y/n): y

Running pipeline...
  ✅ Step 1: Find market token ID
  ✅ Step 2: Download orderbooks (12,543 snapshots)
  ✅ Step 3: Process orderbooks

✅ PIPELINE COMPLETE
📂 Data saved to: data/orderbook_processed.csv
```

### Example 2: Direct Market URL (Skip Selection)

If you already know which specific market you want, paste the full URL:

```bash
$ python data_pipeline/00_select_market_from_url.py "https://polymarket.com/event/2024-election/trump-wins-2024"

✅ Detected market URL with slug
   Market slug: trump-wins-2024

✅ Updated config.py

Run data pipeline now? (y/n): y
```

### Example 3: Non-Interactive (Automation)

For scripts or batch processing:

```bash
# Select market #3 automatically and run pipeline
python data_pipeline/00_select_market_from_url.py \
    "https://polymarket.com/event/2024-election" \
    --select 3 \
    --auto-run

# Or just update config without running
python data_pipeline/00_select_market_from_url.py \
    "https://polymarket.com/event/2024-election" \
    --select 1
```

### Example 4: Current Market (Fed Rates)

```bash
python data_pipeline/00_select_market_from_url.py \
    "https://polymarket.com/event/anything/fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting" \
    --auto-run
```

---

## How It Works

### Step 1: Parse URL
Extracts event slug or market slug from Polymarket URL:
- `/event/EVENT-SLUG` → Query API for all markets
- `/event/EVENT/MARKET-SLUG` → Use market slug directly

### Step 2: Query Gamma API (if needed)
If only event URL, fetches all markets in that event:
```
https://gamma-api.polymarket.com/markets?limit=1000
```
Filters by event slug to find relevant markets.

### Step 3: Interactive Selection
Shows formatted list with:
- Question text
- Current Yes/No prices
- Total volume
- Market slug

### Step 4: Update config.py
Automatically updates `MARKET_SLUG = "..."` in config file.

### Step 5: Run Pipeline (optional)
Optionally runs:
1. `01_find_market.py` - Get token ID
2. `02_download_orderbooks.py` - Download snapshots
3. `03_process_orderbooks.py` - Process data

---

## Command Line Options

```bash
# Basic usage (interactive)
python 00_select_market_from_url.py

# Provide URL as argument
python 00_select_market_from_url.py "URL"

# Auto-select specific market
python 00_select_market_from_url.py "URL" --select 2

# Run pipeline automatically after selection
python 00_select_market_from_url.py "URL" --auto-run

# Combine both (full automation)
python 00_select_market_from_url.py "URL" --select 1 --auto-run
```

---

## URL Formats Supported

✅ **Event URLs (multiple markets):**
```
https://polymarket.com/event/2024-election
https://polymarket.com/event/fed-decision-december
https://polymarket.com/event/nba-finals-2024?tid=123456
```

✅ **Market URLs (single specific market):**
```
https://polymarket.com/event/2024-election/trump-wins
https://polymarket.com/market/trump-wins-2024
```

✅ **With query parameters:**
```
https://polymarket.com/event/2024-election?tid=1234&utm_source=twitter
(Query params are ignored)
```

---

## Troubleshooting

### "No markets found for event"

**Causes:**
1. Event slug is misspelled
2. Market is closed/resolved
3. Market doesn't exist yet

**Solution:**
Try pasting the full market URL instead of just event URL.

### "Could not parse Polymarket URL"

**Cause:** URL is not in a recognized format

**Solution:** Copy URL directly from Polymarket.com browser address bar.

### Rate Limiting

If you get API errors, wait a few seconds and try again. The Gamma API has rate limits.

---

## Integration with Existing Workflow

### Old Workflow (Manual):
```bash
# 1. Find slug manually (tedious!)
# 2. Edit config.py
nano data_pipeline/config.py

# 3. Run pipeline
python data_pipeline/01_find_market.py
python data_pipeline/02_download_orderbooks.py
python data_pipeline/03_process_orderbooks.py
```

### New Workflow (URL-based):
```bash
# Just paste URL and go!
python data_pipeline/00_select_market_from_url.py "URL" --auto-run
```

**Still works:** You can still manually edit `config.py` if preferred. The new script is optional but highly recommended!

---

## Tips

1. **Copy URL from browser**: Right-click address bar → Copy
2. **Use quotes**: Wrap URLs in quotes to avoid shell issues
3. **Check prices**: The script shows current Yes/No prices to verify you selected the right market
4. **Save time**: Use `--auto-run` to go from URL to data in one command

---

## What's Next?

After running the pipeline, you have clean orderbook data ready for:
- **OFI calculation**: `python old_scripts/calculate_ofi.py`
- **Visualization**: `python visualize_ofi_analysis.py`
- **Analysis**: Your own custom scripts!

---

## Full Example: Zero to Data

```bash
# 1. Install dependencies (first time only)
pip install requests pandas tqdm python-dotenv

# 2. Set API key (first time only)
echo "DOME_API_KEY=your_key_here" > .env

# 3. Get data from URL (repeat for any market)
python data_pipeline/00_select_market_from_url.py \
    "https://polymarket.com/event/2024-election" \
    --select 1 \
    --auto-run

# Done! You now have:
# - data/market_info.json (market metadata)
# - data/orderbook_raw.json (raw snapshots)
# - data/orderbook_processed.csv (clean data)
```

---

## Need Help?

**Check the logs:** Each pipeline step prints detailed progress

**Verify config:** `cat data_pipeline/config.py | grep MARKET_SLUG`

**Test manually:** Run each step individually to debug:
```bash
python data_pipeline/01_find_market.py
python data_pipeline/02_download_orderbooks.py
python data_pipeline/03_process_orderbooks.py
```
