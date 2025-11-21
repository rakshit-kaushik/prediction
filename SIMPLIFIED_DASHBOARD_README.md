# Simplified OFI Analysis Dashboard

## Overview

This is a **simplified, fast, and reliable** dashboard for exploring pre-existing Polymarket OFI data. Unlike the full pipeline dashboard, this focuses on **data exploration** rather than data collection.

## Key Features

✅ **No API Calls** - Uses pre-downloaded CSV data
✅ **Fast Loading** - Instant startup, no waiting
✅ **Interactive Date Filtering** - Explore different time periods
✅ **Live Visualizations** - Charts update as you adjust date ranges
✅ **Pre-configured Markets** - Simple dropdown selection
✅ **Plotly Charts** - Smooth, interactive, zoomable plots

## Quick Start

```bash
streamlit run dashboard_simple.py
```

Dashboard opens at: **http://localhost:8501**

## Dashboard Layout

### Sidebar (Left):
1. **Market Selection Dropdown**
   - Choose from pre-configured markets
   - Currently available: "Fed Rate Decision (Dec 2025)"

2. **Date/Time Filters**
   - Start Date & Time
   - End Date & Time
   - UTC timezone
   - Real-time filtering

3. **Data Info**
   - Total snapshots available
   - Full date range

### Main Area (Tabs):

#### Tab 1: 📈 Price & Depth
- **Price Evolution**: Mid-price, best bid, best ask over time
- **Spread Analysis**: Absolute and percentage spreads
- **Order Book Depth**: Bid/ask depth evolution

#### Tab 2: 📊 OFI Analysis
- **OFI vs Price Change**: Scatter plot with regression line
- **OFI Distribution**: Histogram of OFI values
- R² and p-value statistics

#### Tab 3: ⏰ Intraday Patterns
- **Hourly OFI Patterns**: Average OFI by hour of day
- **Hourly Price Changes**: Average price changes by hour
- Error bars showing standard deviation

#### Tab 4: 📋 Summary Stats
- Comprehensive statistics for selected period
- Total snapshots, date range, duration
- Price metrics (range, average, std dev)
- OFI metrics (range, mean, correlation)
- Spread statistics
- Depth statistics
- Raw data preview

## How to Use

### Basic Exploration:

1. **Start the dashboard**
   ```bash
   streamlit run dashboard_simple.py
   ```

2. **Select a market** from the dropdown (sidebar)

3. **Adjust date/time filters** to focus on specific periods
   - All charts update automatically
   - No need to click "Apply" or "Refresh"

4. **Explore the tabs** to see different analyses

5. **Interact with charts**:
   - Hover for detailed values
   - Zoom in/out
   - Pan around
   - Double-click to reset view

### Example Workflows:

#### Workflow 1: Compare Different Time Periods

```
1. Set date range: Oct 15-20, 2025
2. Note the average OFI and price range
3. Set date range: Oct 25-31, 2025
4. Compare statistics and patterns
```

#### Workflow 2: Analyze Specific Events

```
1. Know an important date (e.g., Oct 28)
2. Set narrow date range: Oct 28, 00:00 - Oct 28, 23:59
3. Check OFI patterns and price movements
4. Look at hourly patterns in Tab 3
```

#### Workflow 3: Identify Trading Hours

```
1. Select full date range
2. Go to Tab 3 (Intraday Patterns)
3. Identify hours with highest OFI activity
4. Filter to those specific hours in sidebar
5. Analyze detailed patterns in other tabs
```

## Adding New Markets

To add a new market to the dashboard:

### Step 1: Download Market Data

Use the full pipeline dashboard (`dashboard_app.py`) or manual scripts to download:
- `ofi_results.csv`
- `orderbook_processed.csv`
- `market_info.json`

Save them in the `data/` folder with descriptive names.

### Step 2: Add Market to Configuration

Edit `dashboard_simple.py`, find the `MARKETS` dictionary (around line 43):

```python
MARKETS = {
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "ofi_results.csv",
        "orderbook_file": "orderbook_fed_oct15_31_processed.csv",
        "market_info_file": "market_info.json",
        "description": "Fed decreases interest rates by 25 bps after December 2025 meeting?",
        "date_range": ("2025-10-15", "2025-10-31")
    },

    # Add your new market here:
    "Trump 2024 Election": {
        "ofi_file": "trump_ofi_results.csv",
        "orderbook_file": "trump_orderbook_processed.csv",
        "market_info_file": "trump_market_info.json",
        "description": "Trump wins 2024 US Presidential Election?",
        "date_range": ("2024-10-01", "2024-11-05")
    }
}
```

### Step 3: Restart Dashboard

```bash
# Stop current dashboard (Ctrl+C if running in terminal)
# Or kill the process

# Restart
streamlit run dashboard_simple.py
```

Your new market will appear in the dropdown!

## Data Requirements

For each market, you need these files in `data/`:

### Required Files:

1. **`ofi_results.csv`** - OFI analysis results
   - Must have columns: `timestamp`, `ofi`, `delta_mid_price`, `mid_price`, `spread`, `spread_pct`, `total_bid_size`, `total_ask_size`, `total_depth`, `best_bid_price`, `best_ask_price`

2. **`orderbook_processed.csv`** - Processed orderbook (optional but recommended)
   - Same columns as OFI results (without OFI-specific columns)

3. **`market_info.json`** - Market metadata (optional)
   - Contains market name, description, dates

### Data Format:

```csv
timestamp,mid_price,best_bid_price,best_ask_price,...,ofi,delta_mid_price
2025-10-15 00:18:34+00:00,0.7200,0.7100,0.7300,...,0.0,0.0
2025-10-15 00:24:12+00:00,0.7205,0.7105,0.7305,...,123.45,0.0005
...
```

**Important:**
- `timestamp` must be in ISO 8601 format with timezone
- All numeric columns must be valid numbers (no NaN or inf)
- Data should be sorted by timestamp

## Troubleshooting

### "No data found for this market"

**Cause:** CSV files don't exist or are in wrong location

**Solution:**
1. Check files exist: `ls data/ofi_results.csv`
2. Verify file names in `MARKETS` configuration match actual files
3. Check file permissions: `ls -l data/`

### "No data in selected date range"

**Cause:** Date filters exclude all data

**Solution:**
1. Check the "Available" date range in sidebar
2. Adjust filters to include some of that range
3. Reset to full range: match start/end dates with available range

### Charts not updating

**Cause:** Browser cache or Streamlit issue

**Solution:**
1. Refresh browser (Cmd+R or Ctrl+R)
2. Click "Rerun" in Streamlit (top-right)
3. Restart dashboard

### "Module not found" errors

**Cause:** Missing Python packages

**Solution:**
```bash
pip install streamlit pandas numpy plotly scipy
```

## Comparison: Simple vs Full Dashboard

| Feature | Simple Dashboard | Full Dashboard |
|---------|-----------------|----------------|
| **Purpose** | Data exploration | Data collection + analysis |
| **Data Source** | Pre-existing CSV | Live API downloads |
| **Speed** | Instant | Slow (downloads data) |
| **API Calls** | None | Many (Gamma, Dome) |
| **Reliability** | Very high | Depends on APIs |
| **Use Case** | Analyze existing data | Get new market data |
| **Date Filtering** | ✅ Yes, interactive | ❌ Fixed at download |
| **Market Selection** | Dropdown | URL/Slug/Token input |

**Recommendation:**
- Use **Simple Dashboard** for daily analysis and exploration
- Use **Full Dashboard** when you need to download new market data

## Advanced Features

### Custom Date Ranges

You can combine date and time filters for precise periods:
- **Single Day**: Start = End = same date, adjust times
- **Business Hours**: 9am-5pm across multiple days
- **Weekend Only**: Saturday-Sunday date range
- **Specific Hours**: e.g., 2pm-4pm daily

### Export Filtered Data

The "View Raw Data" expander in Tab 4 shows the filtered dataframe. You can:
1. Expand it
2. Copy visible data
3. Use in external analysis tools

### Statistical Analysis

All statistics in Tab 4 update based on your selected date range:
- Compare different periods by noting stats before/after
- Identify outlier periods with extreme values
- Validate OFI-price relationship across time segments

## Performance Tips

1. **Cache is Enabled**: Data loads once, then cached. Changing date filters is instant.

2. **Large Date Ranges**: All data loads quickly, but rendering 1000s of points may slow down charts slightly. Use reasonable ranges for smooth interaction.

3. **Multiple Markets**: Each market's data is cached separately. Switching between markets is fast after first load.

## Future Enhancements

Possible additions (not yet implemented):
- Correlation heatmaps
- Volume-weighted statistics
- Event markers (news, announcements)
- Comparison mode (overlay multiple periods)
- Export to CSV/Excel
- Custom metric calculations
- Alert threshold visualization

## File Structure

```
prediction-1/
├── dashboard_simple.py          # Simplified dashboard (THIS FILE)
├── dashboard_app.py             # Full pipeline dashboard
├── data/
│   ├── ofi_results.csv         # Fed market OFI data
│   ├── orderbook_fed_oct15_31_processed.csv
│   ├── market_info.json
│   └── [other market files...]
└── SIMPLIFIED_DASHBOARD_README.md
```

## Support

### Common Questions:

**Q: Can I analyze multiple markets at once?**
A: Not currently. Select one market at a time from dropdown. Future versions may support comparison.

**Q: How do I download new data?**
A: Use the full dashboard (`dashboard_app.py`) or manual pipeline scripts in `data_pipeline/`.

**Q: Can I share my filtered views?**
A: Not yet. You can screenshot charts or note the date range for others to reproduce.

**Q: What timezone are timestamps in?**
A: UTC. All times shown are in UTC timezone.

**Q: How often should I refresh data?**
A: Depends on market activity. For active markets, daily. For slower markets, weekly.

---

**Dashboard Version:** 1.0 (Simplified)
**Last Updated:** November 2025
**Recommended for:** Daily data exploration and analysis
