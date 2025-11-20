# Old Scripts

These scripts were created during the initial development and testing phase. They are kept for reference but are **NOT part of the main data pipeline**.

## Files in this folder:

### Early Data Exploration
- `01_explore_data.py` - Initial data exploration script
- `explore.py` - General exploration utilities

### Download Experiments
- `02_download_one_day.py` - Prototype for downloading one day
- `02_download_orderbook.py` - Early orderbook download attempt
- `05_download_orderbook_with_depth.py` - Testing full depth downloads
- `download_fed_rates_data.py` - Fed rates market specific download
- `find_fed_market.py` - Finding token ID for Fed market

### Data Processing
- `reprocess_orderbook_data.py` - **IMPORTANT**: This script fixed the critical bug in best bid/ask extraction. The logic from this script was incorporated into the main pipeline (`data_pipeline/03_process_orderbooks.py`).

### Analysis Scripts
- `analyze_orderbook_structure.py` - Analyzed where liquidity sits in orderbook
- `visualize_orderbook_basics.py` - Created visualizations of orderbook patterns
- `calculate_ofi.py` - OFI calculation using Cont et al. (2011) formula

## Important Note

The **correct** best bid/ask logic discovered during development:
```python
# Best Bid = HIGHEST bid price (max buyers will pay)
best_bid_price = max(bid_prices)

# Best Ask = LOWEST ask price (min sellers will accept)
best_ask_price = min(ask_prices)
```

This is now properly implemented in `data_pipeline/03_process_orderbooks.py`.

## Using the Main Pipeline

Instead of these scripts, use the main data pipeline:
```bash
cd data_pipeline
python 01_find_market.py
python 02_download_orderbooks.py
python 03_process_orderbooks.py
```

See `data_pipeline/README.md` for full documentation.
