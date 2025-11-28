# OFI Analysis Dashboard

Comprehensive dashboard for exploring Polymarket Order Flow Imbalance (OFI) data with multi-time-window analysis.

## Quick Start

```bash
streamlit run dashboard/dashboard_simple.py
```

Dashboard opens at **http://localhost:8501**

## What It Does

- **243 regression analyses**: 9 time windows × 9 outlier methods × 3 phases
- Explore pre-downloaded OFI data for prediction markets
- Interactive scatter plots with regression statistics
- Filter by date/time ranges
- Configurable time aggregation (1, 5, 10, 15, 20, 30, 45, 60, 90 minutes)

## Dashboard Features

### Tabs:
1. **Home** - Price evolution, spreads, order book depth, and master R² summary heatmaps
2. **Last Day** - 81 scatter plots analyzing the final 24 hours before market expiry
3. **1 min - 90 min** - 9 tabs showing OFI analysis for each time aggregation window

### Analysis Features:
- **9 Outlier Handling Methods**: Raw, IQR, Percentile (1%-99%), Z-Score, Winsorized, Absolute Threshold (±200k, ±100k), MAD, Percentile (5%-95%)
- **3-Phase Analysis**: Early, Middle, Near Expiry market phases
- **Phase Split Options**: By observation count or by calendar time

### Controls (Sidebar):
- Market selection dropdown (NYC Mayor, Fed Rate Decision)
- Phase split method toggle
- Date/time filters (UTC)
- Data info metrics

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
