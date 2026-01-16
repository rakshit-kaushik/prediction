# Data Pipeline Documentation

## Overview

This document describes how data flows from DOME API into the prediction market analysis pipeline.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DOME API                                       │
│                    (api.domeapi.io/v1)                                  │
└─────────────────────┬───────────────────────────┬───────────────────────┘
                      │                           │
                      ▼                           ▼
         ┌────────────────────┐      ┌────────────────────────┐
         │  /polymarket/      │      │  /polymarket/orders    │
         │  orderbooks        │      │  (Trade History)       │
         └─────────┬──────────┘      └───────────┬────────────┘
                   │                              │
                   ▼                              ▼
         ┌────────────────────┐      ┌────────────────────────┐
         │ 02_download_       │      │ 02b_download_trades.py │
         │ orderbooks.py      │      │                        │
         └─────────┬──────────┘      └───────────┬────────────┘
                   │                              │
                   ▼                              ▼
         ┌────────────────────┐      ┌────────────────────────┐
         │ raw.json           │      │ trades_raw.json        │
         │ (1.5 GB)           │      │ (4 MB)                 │
         │ Full order book    │      │ Executed trades        │
         └─────────┬──────────┘      └───────────┬────────────┘
                   │                              │
                   ▼                              ▼
         ┌────────────────────┐      ┌────────────────────────┐
         │ 03_process_        │      │ 03b_process_trades.py  │
         │ orderbooks.py      │      │                        │
         └─────────┬──────────┘      └───────────┬────────────┘
                   │                              │
                   ▼                              ▼
         ┌────────────────────┐      ┌────────────────────────┐
         │ processed.csv      │      │ trades_processed.csv   │
         │ (21 MB)            │      │ (1.3 MB)               │
         └─────────┬──────────┘      └────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│ L1 OFI       │    │ MLOFI (Multi-    │
│ Analysis     │    │ Level)           │
│ 04_calculate │    │ mlofi/01_extract │
│ _ofi.py      │    │ _multilevel.py   │
└──────────────┘    └──────────────────┘
```

---

## Two Data Types from DOME

### 1. Orderbook Snapshots (Event-Driven)

**API Endpoint:** `GET /polymarket/orderbooks`

**Parameters:**
- `token_id`: The market token ID
- `start_time`: Start timestamp (milliseconds)
- `end_time`: End timestamp (milliseconds)
- `limit`: Max snapshots per request (default 200)
- `pagination_key`: For fetching next page

**Response Structure:**
```json
{
  "timestamp": 1760486408685,
  "bids": [
    {"price": "0.881", "size": "19.96"},
    {"price": "0.880", "size": "43923.45"}
  ],
  "asks": [
    {"price": "0.883", "size": "4616.65"},
    {"price": "0.884", "size": "14082.54"}
  ]
}
```

**Key Characteristics:**
- **Event-driven, not time-driven** - Snapshots arrive when the book changes
- **Full depth** - All price levels (100+ bids, 70+ asks typically)
- **Irregular intervals** - High activity = many snapshots; low activity = few
- **Large files** - 1.5 GB for 21 days of data
- **Timestamp in milliseconds**

### 2. Trade History (Executed Orders)

**API Endpoint:** `GET /polymarket/orders`

**Parameters:**
- `token_id`: The market token ID
- `start_time`: Start timestamp (seconds, not milliseconds!)
- `end_time`: End timestamp (seconds)
- `limit`: Max trades per request (default 100)
- `offset`: Pagination offset

**Response Structure:**
```json
{
  "token_id": "33945469...",
  "side": "BUY",
  "price": 0.882,
  "shares_normalized": 292.0,
  "timestamp": 1729018714,
  "maker_address": "0xc2a3...",
  "taker_address": "0x7f4a...",
  "order_hash": "0x363820c...",
  "block_no": 77703451
}
```

**Key Characteristics:**
- Actual trade executions (not just book updates)
- Includes maker/taker addresses (useful for trader analysis)
- Timestamp in **seconds** (not milliseconds like orderbooks)
- Smaller dataset compared to orderbook snapshots

---

## Pipeline Scripts

### Step 0: Select Market
**Script:** `data_pipeline/00_select_market_from_url.py`

Extracts market slug from Polymarket URL and finds the token ID.

### Step 1: Find Market
**Script:** `data_pipeline/01_find_market.py`

Queries Polymarket Gamma API to get market metadata and token ID.
Saves to `data/market_info.json`.

### Step 2: Download Orderbooks
**Script:** `data_pipeline/02_download_orderbooks.py`

Downloads orderbook snapshots day by day with pagination handling.
Saves to `data/*_raw.json`.

### Step 2b: Download Trades
**Script:** `data_pipeline/02b_download_trades.py`

Downloads trade history with retry logic for rate limiting.
Saves to `data/*_trades_raw.json`.

### Step 3: Process Orderbooks
**Script:** `data_pipeline/03_process_orderbooks.py`

Extracts best bid/ask from each snapshot:
- Best Bid = HIGHEST bid price
- Best Ask = LOWEST ask price
- Calculates mid-price, spread, depth metrics

Saves to `data/*_processed.csv`.

### Step 3b: Process Trades
**Script:** `data_pipeline/03b_process_trades.py`

Cleans and normalizes trade data.
Saves to `data/*_trades_processed.csv`.

### Step 4: Calculate OFI
**Script:** `data_pipeline/04_calculate_ofi.py`

Calculates Order Flow Imbalance using the Cont, Kukanov & Stoikov (2011) formula.

---

## Processed Data Structures

### L1 Orderbook (`*_processed.csv`)

| Column | Description |
|--------|-------------|
| `timestamp` | ISO format timestamp |
| `timestamp_ms` | Unix timestamp in milliseconds |
| `best_bid_price` | Highest bid price |
| `best_bid_size` | Size at best bid |
| `best_ask_price` | Lowest ask price |
| `best_ask_size` | Size at best ask |
| `mid_price` | (best_bid + best_ask) / 2 |
| `spread` | best_ask - best_bid |
| `spread_pct` | Spread as percentage of mid |
| `bid_levels` | Number of bid price levels |
| `ask_levels` | Number of ask price levels |
| `total_bid_size` | Sum of all bid sizes |
| `total_ask_size` | Sum of all ask sizes |
| `total_depth` | total_bid + total_ask |
| `imbalance` | (bid - ask) / total |

### Multi-Level Orderbook (`multilevel_processed.csv`)

Contains 25 levels of depth:
```
timestamp, timestamp_ms, n_bid_levels, n_ask_levels,
bid_price_l1, bid_size_l1, ask_price_l1, ask_size_l1,
bid_price_l2, bid_size_l2, ask_price_l2, ask_size_l2,
...
bid_price_l25, bid_size_l25, ask_price_l25, ask_size_l25,
mid_price
```

### Trades Processed (`*_trades_processed.csv`)

| Column | Description |
|--------|-------------|
| `timestamp` | Trade execution time |
| `price` | Execution price |
| `size` | Trade size (normalized) |
| `side` | BUY or SELL |
| `maker_address` | Liquidity provider |
| `taker_address` | Order taker |

---

## Configuration

All pipeline configuration is in `data_pipeline/config.py`:

```python
# Market
MARKET_SLUG = "will-zohran-mamdani-win-the-2025-nyc-mayoral-election"
TOKEN_ID = "33945469..."

# Date Range
START_DATE = "2025-10-15"
END_DATE = "2025-11-04"

# API
DOME_API_BASE_URL = "https://api.domeapi.io/v1"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"

# Rate Limiting
PAGINATION_LIMIT = 200
DELAY_BETWEEN_PAGES = 0.3
DELAY_BETWEEN_DAYS = 1.0
```

API key is loaded from `.env` file:
```
DOME_API_KEY=your_key_here
```

---

## Data Quality Notes

1. **Event-driven snapshots** - Not regular time intervals. Aggregation into time windows (1, 5, 10... 90 min) handles this.

2. **Timestamp units differ** - Orderbooks use milliseconds, trades use seconds. Processing scripts handle conversion.

3. **Best bid/ask convention**:
   - Best Bid = HIGHEST bid (max buyers will pay)
   - Best Ask = LOWEST ask (min sellers will accept)

4. **Large file sizes** - Raw orderbook JSON can be 1+ GB. Consider filtering during download for specific analysis.

---

## Current Dataset Summary

| File | Records | Size | Content |
|------|---------|------|---------|
| `nyc_mayor_oct15_nov04_raw.json` | ~100K snapshots | 1.5 GB | Full order book depth |
| `nyc_mayor_oct15_nov04_processed.csv` | ~100K rows | 21 MB | L1 best bid/ask |
| `multilevel_processed.csv` | ~100K rows | 148 MB | 25 levels of depth |
| `trades_raw.json` | ~10K trades | 4 MB | Via API |
| `trades_processed.csv` | ~10K trades | 1.3 MB | Cleaned trades |

---

## Usage Example

```bash
# 1. Set up market (edit config.py with your market)
python data_pipeline/01_find_market.py

# 2. Download orderbook data
python data_pipeline/02_download_orderbooks.py

# 3. Download trade data
python data_pipeline/02b_download_trades.py

# 4. Process orderbooks
python data_pipeline/03_process_orderbooks.py

# 5. Process trades
python data_pipeline/03b_process_trades.py

# 6. Calculate OFI
python data_pipeline/04_calculate_ofi.py
```
