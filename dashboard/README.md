# OFI Analysis Dashboard

Simple dashboard for exploring Polymarket Order Flow Imbalance (OFI) data.

## Quick Start

```bash
streamlit run dashboard/dashboard_simple.py
```

Dashboard opens at **http://localhost:8501**

## What It Does

- Explore pre-downloaded OFI data for prediction markets
- Interactive charts showing price, OFI, and order book depth
- Filter by date/time ranges
- View regression statistics and patterns

## Dashboard Features

### Tabs:
1. **Price & Depth** - Price evolution, spreads, order book depth
2. **OFI Analysis** - OFI vs price change scatter plots with regression
3. **3-Phase Analysis** - NYC market beta evolution across trading phases
4. **Summary Stats** - Comprehensive statistics and raw data

### Controls (Sidebar):
- Market selection dropdown
- Date/time filters
- Data info and statistics

## Input Methods (For Adding New Markets)

When downloading new market data, use one of these input methods:

### Method 1: Polymarket URL (Easiest)
Copy full URL from browser:
```
https://polymarket.com/event/fed-decision/fed-decreases-interest-rates-by-25-bps
```
Dashboard auto-extracts the market slug.

### Method 2: Market Slug
Direct input of market identifier:
```
fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting
```
Found in the URL after `/event/[event-name]/`

### Method 3: Token ID (Advanced)
For when slug lookup fails:
```
21742633143463906290569050155826241533067272736897614950488156847949938836455
```

## How to Use

1. **Run dashboard**: `streamlit run dashboard/dashboard_simple.py`
2. **Select market** from dropdown
3. **Adjust date filters** to focus on time periods
4. **Explore tabs** for different analyses
5. **Interact with charts**: hover, zoom, pan

## Adding New Markets

1. Download data using data pipeline:
   ```bash
   python data_pipeline/02_download_orderbooks.py
   python data_pipeline/03_process_orderbooks.py
   ```

2. Add market to `MARKETS` config in `dashboard_simple.py`:
   ```python
   MARKETS = {
       "Your Market Name": {
           "ofi_file": "your_ofi_results.csv",
           "orderbook_file": "your_orderbook_processed.csv",
           "market_info_file": "your_market_info.json",
           "description": "Market description",
           "date_range": ("2025-01-01", "2025-12-31")
       }
   }
   ```

3. Restart dashboard

## Troubleshooting

### "No data found"
- Check CSV files exist in `data/` folder
- Verify file names in config match actual files

### "No data in selected date range"
- Check "Available" date range in sidebar
- Adjust filters to include that range

### Charts not updating
- Refresh browser (Cmd+R / Ctrl+R)
- Click "Rerun" in Streamlit (top-right)

### Module errors
```bash
pip install streamlit pandas numpy plotly scipy
```

## Data Requirements

Each market needs these files in `data/`:

- **ofi_results.csv** - OFI calculations with columns: timestamp, ofi, delta_mid_price, mid_price, spread, etc.
- **orderbook_processed.csv** - Processed order book data (optional)
- **market_info.json** - Market metadata (optional)

## Project Structure

```
prediction-1/
├── dashboard/
│   ├── dashboard_simple.py      # Main dashboard
│   ├── test_dashboard_markets.py
│   └── README.md                 # This file
├── data/                         # Market data files
├── data_pipeline/                # Data collection scripts
└── scripts/                      # Analysis scripts
```

## Tips

- **Save token IDs**: After first successful run, save for faster future access
- **Date filtering**: All charts update automatically when you adjust dates
- **Export data**: Use "View Raw Data" expander in Summary Stats tab
- **UTC timezone**: All timestamps are in UTC

---

**Last Updated:** November 2025
