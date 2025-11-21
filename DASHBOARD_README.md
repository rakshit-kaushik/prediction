# Interactive OFI Analysis Dashboard

## Overview

This interactive web dashboard provides a complete end-to-end workflow for analyzing Order Flow Imbalance (OFI) in Polymarket prediction markets. Simply input a market slug, select dates, and click a button to run the entire analysis pipeline.

## Quick Start

### 1. Start the Dashboard

```bash
streamlit run dashboard_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 2. Use the Dashboard

#### Sidebar Controls:

1. **Market Slug**: Enter the market slug from a Polymarket URL
   - Example: `fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting`

2. **Verify Market** button: Checks if the market exists via Gamma API
   - Shows market question and current prices
   - Not required for newer markets (can proceed anyway)

3. **Date Range**: Select start and end dates for data collection

4. **Run Analysis Pipeline** button: Executes the complete pipeline:
   - Finds market token ID
   - Downloads orderbook snapshots from Dome API
   - Processes raw data
   - Calculates OFI metrics
   - Generates visualizations

5. **Download All Data (ZIP)**: Downloads complete dataset including:
   - Raw orderbook JSON
   - Processed CSV files
   - OFI results
   - All visualization images
   - Summary statistics

#### Main Display Area:

- **Pipeline Progress** tab: Real-time progress tracking for each step
- **Results** tabs: Display all 5 visualization phases + summary stats

## Features

### All Data in One Place
- Everything stored in single `data/` folder
- No scattered files across multiple directories
- Easy to backup, share, or delete

### Interactive & User-Friendly
- No need to manually edit config files
- No need to run multiple scripts
- Real-time progress tracking
- In-browser visualization display

### Reusable & General
- Works for ANY Polymarket market
- Auto-detects market metadata
- Dynamic date range selection
- No hardcoded values

## Data Files

After running the pipeline, the `data/` folder contains:

```
data/
├── market_info.json              # Market metadata
├── orderbook_raw.json            # Raw snapshots (5-15 MB)
├── orderbook_processed.csv       # Cleaned orderbook data
├── ofi_results.csv               # OFI calculations
├── ofi_analysis_phase1_quality.png
├── ofi_analysis_phase2_overview.png
├── ofi_analysis_phase3_core.png  # KEY RESULTS (OFI vs ΔP)
├── ofi_analysis_phase4_orderbook.png
├── ofi_analysis_phase5_intraday.png
└── summary_statistics.txt        # Numerical results
```

## Visualization Phases

### Phase 1: Data Quality Check
- Missing data analysis
- Timestamp distribution
- Data completeness metrics

### Phase 2: Market Overview
- Price evolution over time
- Trading volume patterns
- Market activity heatmap

### Phase 3: Core OFI Analysis (KEY RESULTS)
- **OFI vs Price Change scatter plot** (replicates Cont et al. 2011)
- Linear regression statistics (R², correlation, p-value)
- OFI distribution
- Price impact analysis

### Phase 4: Detailed Orderbook View
- Bid-ask spread analysis
- Market depth evolution
- Liquidity metrics

### Phase 5: Intraday Patterns
- Hour-of-day analysis
- Day-of-week patterns
- Time-based OFI clustering

## Requirements

### Python Packages
```bash
pip install streamlit pandas numpy requests python-dotenv matplotlib seaborn scipy
```

### API Keys
Create a `.env` file in the project root:
```
DOME_API_KEY=your_dome_api_key_here
```

## Troubleshooting

### "No markets found" after verification
- This is normal for newer markets (Gamma API only has historical data)
- Click "Run Analysis Pipeline" anyway - token ID lookup will work via Dome API

### "DOME_API_KEY not found"
- Make sure you have a `.env` file with `DOME_API_KEY=...`
- The `.env` file must be in the same directory as `dashboard_app.py`

### Dashboard won't start
- Check if port 8501 is already in use: `lsof -i :8501`
- Try a different port: `streamlit run dashboard_app.py --server.port 8502`
- Kill old processes: `pkill -f streamlit`

### Visualizations not showing
- Make sure you clicked "Run Analysis Pipeline" first
- Check that files exist in `data/` folder: `ls data/*.png`
- Try refreshing the browser

## Comparison with Manual Workflow

### Old Manual Workflow:
```bash
# 1. Edit config file
nano data_pipeline/config.py

# 2. Run 4 separate scripts
python data_pipeline/01_find_market.py
python data_pipeline/02_download_orderbooks.py
python data_pipeline/03_process_orderbooks.py
python old_scripts/calculate_ofi.py

# 3. Generate visualizations
python visualize_ofi_analysis.py

# 4. Check results in multiple folders
ls data/
ls results/
```

### New Dashboard Workflow:
```bash
# 1. Start dashboard
streamlit run dashboard_app.py

# 2. Use web interface:
#    - Enter market slug
#    - Select dates
#    - Click "Run Analysis Pipeline"

# 3. View all results in browser
# 4. Download ZIP with everything
```

**Time saved: ~5 minutes per analysis**

## Advanced Usage

### Running Multiple Markets
1. Run analysis for first market
2. Download ZIP
3. Clear `data/` folder (or keep for comparison)
4. Run analysis for next market
5. Repeat

### Batch Processing
While the dashboard is interactive, you can still use the command-line scripts for automation:
```bash
# Use URL-based market selector for batch processing
python data_pipeline/00_select_market_from_url.py "URL" --select 1 --auto-run
```

### Custom Analysis
All data files are in standard formats (JSON, CSV) for custom analysis:
```python
import pandas as pd

# Load OFI results
df = pd.read_csv('data/ofi_results.csv')

# Your custom analysis here
...
```

## Architecture

The dashboard integrates all existing pipeline components:
- `data_pipeline/config.py` - Configuration
- Market lookup via Gamma API
- Orderbook download via Dome API
- Data processing functions
- OFI calculation (Cont et al. 2011 methodology)
- `visualize_ofi_analysis.py` - Visualization generation

Everything runs in a single unified interface with progress tracking and error handling.

## Support

For issues or questions:
1. Check existing data files: `ls -lh data/`
2. Review Streamlit logs in terminal
3. Try manual pipeline: `python data_pipeline/01_find_market.py`
4. Check API key: `echo $DOME_API_KEY` or `cat .env`

## Next Steps

After running the analysis:
1. Review summary statistics (Results tab)
2. Focus on Phase 3 for key OFI findings (R², correlation)
3. Download ZIP for record-keeping
4. Compare across different markets or time periods
5. Use raw data for custom analysis

---

**Dashboard Version**: 1.0
**Last Updated**: November 2025
**Based on**: Cont, Kukanov & Stoikov (2011) "The Price Impact of Order Book Events"
