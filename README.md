# Polymarket Order Flow Imbalance (OFI) Analysis

**Replicating Cont, Kukanov & Stoikov (2011) methodology on prediction market data**

This project downloads and analyzes orderbook data from Polymarket prediction markets to study Order Flow Imbalance (OFI) and its relationship to price changes.

---

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install requests pandas tqdm python-dotenv

# Create .env file with Dome API key
echo "DOME_API_KEY=your_key_here" > .env
```

### 2. Download Data

```bash
cd data_pipeline

# Edit config.py to set your market and dates
nano config.py

# Run the 3-step pipeline
python 01_find_market.py
python 02_download_orderbooks.py
python 03_process_orderbooks.py
```

### 3. Analyze

The processed CSV (`data/orderbook_processed.csv`) contains clean best bid/ask data ready for:
- Order Flow Imbalance calculation
- Price prediction models
- Liquidity analysis
- Market microstructure research

---

## Project Structure

```
prediction-1/
├── data_pipeline/          # ⭐ Main data collection system
│   ├── README.md          # Detailed pipeline documentation
│   ├── config.py          # Configuration (market, dates, API settings)
│   ├── 01_find_market.py  # Step 1: Find token ID from market slug
│   ├── 02_download_orderbooks.py  # Step 2: Download snapshots
│   └── 03_process_orderbooks.py   # Step 3: Extract best bid/ask
│
├── data/                  # Output directory (created automatically)
│   ├── market_info.json   # Market metadata
│   ├── orderbook_raw.json # Raw orderbook snapshots
│   └── orderbook_processed.csv  # Clean CSV for analysis
│
├── old_scripts/           # Legacy scripts (kept for reference)
│   └── README.md          # Explains old scripts
│
└── README.md              # This file
```

---

## Data Pipeline

The pipeline is designed to work with **ANY Polymarket market** - just provide a market slug and date range.

### Step 1: Find Market Token ID

```bash
python data_pipeline/01_find_market.py
```

Queries Polymarket Gamma API to find the token ID for your market slug.

**Input:** Market slug (from Polymarket URL)
**Output:** `data/market_info.json` with token IDs

### Step 2: Download Orderbook Snapshots

```bash
python data_pipeline/02_download_orderbooks.py
```

Downloads ALL orderbook snapshots for your date range with automatic pagination.

**Input:** Token ID (from Step 1) + Date range
**Output:** `data/orderbook_raw.json` with full orderbook depth

**Key Features:**
- Automatic pagination (handles days with 1000+ snapshots)
- Rate limiting (respects API limits)
- Progress tracking

### Step 3: Process Orderbooks

```bash
python data_pipeline/03_process_orderbooks.py
```

Extracts **correct** best bid/ask from raw snapshots.

**Input:** `data/orderbook_raw.json`
**Output:** `data/orderbook_processed.csv`

**Important:**
- Best Bid = HIGHEST bid price (max buyers will pay)
- Best Ask = LOWEST ask price (min sellers will accept)

See `data_pipeline/README.md` for complete documentation.

---

## Configuration

Edit `data_pipeline/config.py` to change market and dates:

```python
# Market slug from Polymarket URL
MARKET_SLUG = "fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting"

# Date range
START_DATE = "2025-10-15"
END_DATE = "2025-10-31"
```

That's it! The pipeline handles everything else automatically.

---

## Finding Market Slugs

Visit the market on polymarket.com and copy the slug from the URL:

```
https://polymarket.com/event/fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting
                                ↓
Slug: fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting
```

---

## Data Characteristics

### Snapshot Timing

Orderbook snapshots are **event-driven** (not time-driven):
- Snapshots occur when orderbook changes
- Time intervals are **irregular**
- High activity days: 1000+ snapshots
- Low activity days: <10 snapshots

### Example Output

From Fed rates market (Oct 15-31, 2025):
- **3,972 snapshots** across 17 days
- **58 unique mid prices** (good price movement)
- **1.84% average spread** (tight, liquid market)
- **Correlation OFI vs ΔPrice: 0.16** (positive relationship)

---

## Order Flow Imbalance (OFI)

This project implements the methodology from:

**Cont, R., Kukanov, A., & Stoikov, S. (2011)**
*The Price Impact of Order Book Events*
Journal of Financial Econometrics

### Formula

```
OFI_n = I{P^B_n > P^B_{n-1}} × q^B_n - I{P^B_n < P^B_{n-1}} × q^B_{n-1}
      - I{P^A_n < P^A_{n-1}} × q^A_n + I{P^A_n > P^A_{n-1}} × q^A_{n-1}
```

Where:
- `P^B_n` = best bid price at snapshot n
- `P^A_n` = best ask price at snapshot n
- `q^B_n` = size at best bid
- `q^A_n` = size at best ask
- `I{}` = indicator function

### Interpretation

- **Positive OFI:** Buy pressure (bid side strengthening)
- **Negative OFI:** Sell pressure (ask side strengthening)
- **OFI predicts price changes:** Higher OFI → prices tend to rise

---

## Requirements

```
requests
pandas
tqdm
python-dotenv
```

Install with:
```bash
pip install requests pandas tqdm python-dotenv
```

---

## API Access

This project uses two APIs:

1. **Polymarket Gamma API** (public, no key needed)
   - Market metadata and token IDs
   - `https://gamma-api.polymarket.com`

2. **Dome API** (requires API key)
   - Historical orderbook data
   - Get your key from domeapi.io
   - Add to `.env` file: `DOME_API_KEY=your_key_here`

---

## Example Workflow

```bash
# 1. Configure
cd data_pipeline
nano config.py  # Set market slug and dates

# 2. Download data (3 commands)
python 01_find_market.py          # Find token ID
python 02_download_orderbooks.py  # Download snapshots
python 03_process_orderbooks.py   # Extract best bid/ask

# 3. Analyze
# Now you have clean CSV data ready for:
# - OFI calculation
# - Price prediction models
# - Liquidity analysis
# - Research
```

---

## Output Data

The final CSV (`data/orderbook_processed.csv`) contains:

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp |
| `timestamp_ms` | Unix timestamp (milliseconds) |
| `best_bid_price` | Highest bid price |
| `best_bid_size` | Size at best bid |
| `best_ask_price` | Lowest ask price |
| `best_ask_size` | Size at best ask |
| `mid_price` | (best_bid + best_ask) / 2 |
| `spread` | best_ask - best_bid |
| `spread_pct` | Spread as % of mid price |
| `total_bid_size` | Total liquidity on bid side |
| `total_ask_size` | Total liquidity on ask side |
| `total_depth` | Total liquidity (both sides) |
| `imbalance` | (bid_size - ask_size) / total_depth |
| `bid_levels` | Number of bid price levels |
| `ask_levels` | Number of ask price levels |

---

## Documentation

- **Pipeline Documentation:** `data_pipeline/README.md` (detailed guide)
- **Configuration:** `data_pipeline/config.py` (all settings)
- **Old Scripts:** `old_scripts/README.md` (legacy code reference)

---

## Common Issues

### Market not found
- Verify slug spelling from polymarket.com URL
- Check market still exists

### No data downloaded
- Date range may be before market creation
- Market may have had no activity
- Try a different date range

### API errors
- Check your Dome API key in `.env`
- Increase rate limiting delays in `config.py`

---

## Research Applications

This data pipeline enables research on:

1. **Order Flow Dynamics**
   - How order flow predicts price changes
   - Information content of orderbook events

2. **Price Discovery**
   - How prediction markets incorporate information
   - Speed of price adjustment to news

3. **Market Microstructure**
   - Bid-ask spread dynamics
   - Liquidity provision patterns
   - Order book depth analysis

4. **Prediction & Trading**
   - OFI-based price prediction models
   - Market making strategies
   - Optimal execution

---

## License

This project is for research and educational purposes.

---

## Credits

Pipeline developed to replicate the OFI methodology from Cont et al. (2011) on Polymarket prediction market data.
