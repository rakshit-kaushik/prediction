# Polymarket Orderbook Data Pipeline

**General-purpose data collection system for Polymarket prediction markets**

Download and process historical orderbook data for ANY Polymarket market with just a market slug and date range.

---

## Overview

This pipeline downloads historical orderbook snapshots from the Dome API for any Polymarket market. It handles:

- ✅ Automatic token ID discovery from market slug
- ✅ Pagination for days with high activity (>200 snapshots)
- ✅ Correct best bid/ask extraction (highest bid, lowest ask)
- ✅ Rate limiting and error handling
- ✅ Clean CSV output ready for analysis

---

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install requests pandas tqdm python-dotenv

# Create .env file with your Dome API key
echo "DOME_API_KEY=your_key_here" > .env
```

### 2. Configure Market and Dates

Edit `config.py`:

```python
# Market slug from Polymarket URL
MARKET_SLUG = "fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting"

# Date range (YYYY-MM-DD)
START_DATE = "2025-10-15"
END_DATE = "2025-10-31"
```

### 3. Run Pipeline

```bash
# Step 1: Find token ID from market slug
python 01_find_market.py

# Step 2: Download orderbook snapshots
python 02_download_orderbooks.py

# Step 3: Process into clean CSV
python 03_process_orderbooks.py
```

---

## Pipeline Steps

### Step 1: Find Market Token ID

**Script:** `01_find_market.py`

**What it does:**
- Queries Polymarket Gamma API with your market slug
- Extracts the token ID needed for orderbook queries
- Saves market metadata for reference

**Input:**
- Market slug (from `config.py` or command line)

**Output:**
- `data/market_info.json` - Contains token IDs and market metadata

**Usage:**
```bash
# Use slug from config.py
python 01_find_market.py

# Or override with command line
python 01_find_market.py "your-market-slug-here"
```

**Example Output:**
```
✅ Found market!
   Title: Fed decreases interest rates by 25 bps after December 2025 meeting?
   Market ID: 0xabc123...
   Volume: $13,674,904.03

🔑 Token IDs:
   Yes: 87769991026114894163580777793845523168226980076553814689875238288185044414090
   No:  13411284055273560855537595688801764123705139415061660246624128667183605973730
```

---

### Step 2: Download Orderbook Snapshots

**Script:** `02_download_orderbooks.py`

**What it does:**
- Downloads ALL orderbook snapshots for your date range
- Automatically handles pagination (fetches multiple pages per day if needed)
- Includes full orderbook depth (all bid/ask levels)

**Input:**
- Token ID (from Step 1)
- Date range (from `config.py` or command line)

**Output:**
- `data/orderbook_raw.json` - Array of raw snapshots

**Usage:**
```bash
# Use dates from config.py
python 02_download_orderbooks.py

# Or override with command line
python 02_download_orderbooks.py --start-date 2025-10-15 --end-date 2025-10-31
```

**How Pagination Works:**

The API returns max 200 snapshots per request. If a day has more than 200 snapshots, the script automatically fetches multiple pages:

```
Day with 600 snapshots:
  Request 1: Gets snapshots 1-200   (pagination_key = "abc")
  Request 2: Gets snapshots 201-400 (pagination_key = "def")
  Request 3: Gets snapshots 401-600 (pagination_key = None, done!)
```

**Example Output:**
```
Downloading orderbook snapshots...
  ✅ 2025-10-15: 89 snapshots
  ✅ 2025-10-16: 127 snapshots
  ✅ 2025-10-30: 1,181 snapshots
  Page 5: 1000 snapshots so far...  (for high-activity days)

📊 Download Summary:
   Total snapshots: 3,972
   Days: 17
   Avg per day: 233.6
   File size: 15.23 MB
```

**What Each Snapshot Contains:**
```json
{
  "timestamp": 1729018714140,
  "bids": [
    {"price": "0.01", "size": "1010640.4"},
    {"price": "0.78", "size": "35311.66"},
    ...
  ],
  "asks": [
    {"price": "0.79", "size": "15283.94"},
    {"price": "0.99", "size": "1020730.88"},
    ...
  ]
}
```

---

### Step 3: Process Orderbooks

**Script:** `03_process_orderbooks.py`

**What it does:**
- Extracts **CORRECT** best bid/ask from each snapshot
- Calculates mid-price, spread, depth metrics
- Outputs clean CSV ready for analysis

**Input:**
- `data/orderbook_raw.json` (from Step 2)

**Output:**
- `data/orderbook_processed.csv`

**Best Bid/Ask Logic:**
```python
# CORRECT approach (used in this pipeline):
best_bid_price = max(bid_prices)  # Highest bid
best_ask_price = min(ask_prices)  # Lowest ask

# This is standard market microstructure:
# - Best Bid = maximum buyers will pay
# - Best Ask = minimum sellers will accept
# - Spread = best_ask - best_bid
```

**Usage:**
```bash
python 03_process_orderbooks.py
```

**Example Output:**
```
📊 Price Statistics:
   Best Bid range: $0.2300 to $0.9000
   Best Ask range: $0.6400 to $0.9100
   Mid-Price mean: $0.6850
   Spread mean: $0.0126 (1.84%)

📊 Price Movement:
   Unique best bid prices: 58
   Unique best ask prices: 54
   Unique mid prices: 58
   ✅ Excellent price movement for analysis!

✅ Saved 3,972 rows × 14 columns to data/orderbook_processed.csv
```

**Output Columns:**
- `timestamp`, `timestamp_ms`
- `best_bid_price`, `best_bid_size`
- `best_ask_price`, `best_ask_size`
- `mid_price`, `spread`, `spread_pct`
- `total_bid_size`, `total_ask_size`, `total_depth`
- `imbalance`, `bid_levels`, `ask_levels`

---

## Configuration Reference

**File:** `config.py`

```python
# ============================================================================
# MARKET CONFIGURATION
# ============================================================================

# Market slug from Polymarket URL
# Example: https://polymarket.com/event/fed-decreases-interest-rates...
#          → slug = "fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting"
MARKET_SLUG = "your-market-slug-here"

# Optional: If you already know the token ID (otherwise auto-discovered)
TOKEN_ID = None

# ============================================================================
# DATE RANGE
# ============================================================================

# Format: "YYYY-MM-DD"
START_DATE = "2025-10-15"
END_DATE = "2025-10-31"

# ============================================================================
# API CONFIGURATION
# ============================================================================

DOME_API_BASE_URL = "https://api.domeapi.io/v1"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"

# ============================================================================
# OUTPUT PATHS
# ============================================================================

MARKET_INFO_FILE = "data/market_info.json"
RAW_ORDERBOOK_FILE = "data/orderbook_raw.json"
PROCESSED_ORDERBOOK_FILE = "data/orderbook_processed.csv"

# ============================================================================
# DOWNLOAD SETTINGS
# ============================================================================

PAGINATION_LIMIT = 200  # Max snapshots per API call
DELAY_BETWEEN_PAGES = 0.3  # Seconds between pagination requests
DELAY_BETWEEN_DAYS = 1.0  # Seconds between day downloads
```

---

## Finding Market Slugs

### Method 1: From Polymarket URL

Visit the market on polymarket.com and copy the slug from the URL:

```
https://polymarket.com/event/fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting
                                ↓
Market slug: fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting
```

### Method 2: Search on Polymarket

1. Go to https://polymarket.com
2. Search for your market
3. Click on the market
4. Copy the slug from the URL

---

## Data Characteristics

### Snapshot Timing

**IMPORTANT:** Orderbook snapshots are **event-driven**, not time-driven.

- Snapshots occur when the orderbook changes (new orders, cancellations, trades)
- Time intervals between snapshots are **irregular**
- High activity days may have 1000+ snapshots
- Low activity days may have <10 snapshots

**Example:**
```
Oct 15: 89 snapshots   (quiet day)
Oct 30: 1,181 snapshots (high activity!)
Oct 31: 1,189 snapshots (market closing?)
```

### Price Discovery

Markets show varying levels of price movement:

- **Excellent:** 50+ unique prices (good for OFI analysis)
- **Moderate:** 10-50 unique prices
- **Limited:** <10 unique prices (may not be suitable for analysis)

The pipeline reports price movement statistics to help you assess data quality.

---

## Common Issues

### Issue 1: Market Not Found

**Error:** `❌ No market found with slug: your-slug`

**Fix:**
1. Verify slug is spelled correctly
2. Check the slug on polymarket.com
3. Market may have been renamed or removed

### Issue 2: No Data for Date Range

**Error:** `❌ No data downloaded!`

**Reasons:**
1. Date range is before market creation
2. Market had no activity during this period
3. Token ID is incorrect

**Fix:** Try a different date range or market

### Issue 3: API Rate Limiting

**Error:** `Error 429: Too Many Requests`

**Fix:**
1. Increase `DELAY_BETWEEN_PAGES` in config
2. Increase `DELAY_BETWEEN_DAYS` in config
3. Wait a few minutes and try again

---

## Output Files

After running the full pipeline, you'll have:

```
data/
├── market_info.json              # Market metadata
├── orderbook_raw.json            # Raw snapshots (15+ MB)
└── orderbook_processed.csv       # Clean CSV ready for analysis
```

The processed CSV is ready to use for:
- Order Flow Imbalance (OFI) calculation
- Price prediction models
- Liquidity analysis
- Market microstructure research

---

## Example: Complete Workflow

```bash
# 1. Edit config.py
nano config.py
# Set MARKET_SLUG = "trump-wins-2024-election"
# Set START_DATE = "2024-11-01"
# Set END_DATE = "2024-11-05"

# 2. Run pipeline
python 01_find_market.py
# ✅ Found market! Token ID: 71321045679252212594626385532706912750332728571942532289631379312455583992563

python 02_download_orderbooks.py
# Downloading... 100%|██████████| 5/5 [00:15<00:00]
# ✅ Downloaded 872 snapshots (3.2 MB)

python 03_process_orderbooks.py
# Processing... 100%|██████████| 872/872 [00:02<00:00]
# ✅ Processed 872 snapshots
# 📊 Unique mid prices: 127
# ✅ Excellent price movement!

# 3. Analyze!
python your_analysis_script.py
```

---

## Technical Details

### API Endpoints

**Polymarket Gamma API:**
- Base: `https://gamma-api.polymarket.com`
- Market lookup: `GET /markets?slug={slug}`
- No authentication required

**Dome API:**
- Base: `https://api.domeapi.io/v1`
- Orderbooks: `GET /polymarket/orderbooks`
- Requires authentication (Bearer token)

### Rate Limiting

The pipeline includes automatic rate limiting:
- 0.3s delay between pagination requests
- 1.0s delay between day downloads
- Adjust in `config.py` if needed

### Data Format

**Raw orderbooks** are saved as JSON array:
```json
[
  {
    "timestamp": 1729018714140,
    "bids": [{"price": "0.78", "size": "35311.66"}, ...],
    "asks": [{"price": "0.79", "size": "15283.94"}, ...]
  },
  ...
]
```

**Processed CSV** has one row per snapshot with best bid/ask extracted.

---

## Next Steps

After running the pipeline, you can:

1. **Calculate OFI** - Use the Cont et al. (2011) formula with the processed data
2. **Build Models** - Train prediction models on price changes vs orderbook features
3. **Analyze Liquidity** - Study bid-ask spreads, depth, imbalance over time
4. **Compare Markets** - Run pipeline on multiple markets and compare dynamics

---

## Support

If you encounter issues:

1. Check the error message and "Common Issues" section above
2. Verify your Dome API key is valid in `.env`
3. Ensure the market slug and dates are correct
4. Check that dependencies are installed

---

## Credits

This pipeline was built to replicate the methodology from:

**Cont, R., Kukanov, A., & Stoikov, S. (2011)**
*The Price Impact of Order Book Events*
Journal of Financial Econometrics

For Order Flow Imbalance (OFI) analysis on prediction markets.
