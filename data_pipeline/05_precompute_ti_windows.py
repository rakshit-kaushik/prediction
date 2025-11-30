"""
05_precompute_ti_windows.py
============================
Pre-compute Trade Imbalance aggregated by time window for dashboard use.
This creates a small CSV file so the dashboard doesn't need the 402MB DOME file.

Usage:
    python data_pipeline/05_precompute_ti_windows.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
DOME_FILE = Path(__file__).parent.parent / "DOME_zohran-oct-15_2025-11-29.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "ti_aggregated_45min.csv"

TIME_WINDOW = 45  # minutes


def main():
    print("=" * 60)
    print("PRE-COMPUTING TI AGGREGATED DATA")
    print("=" * 60)

    # Load DOME trades
    print(f"\nLoading DOME trades from {DOME_FILE}...")
    trades_df = pd.read_csv(DOME_FILE)
    print(f"  Loaded {len(trades_df):,} trades")

    # Parse timestamp
    trades_df['timestamp'] = pd.to_datetime(trades_df['block_timestamp'], utc=True)

    # Normalize shares (divide by 1e6)
    trades_df['shares_normalized'] = trades_df['shares'] / 1e6

    # Create direction: +1 for BUY, -1 for SELL
    trades_df['direction'] = np.where(trades_df['side'] == 'BUY', 1, -1)

    # Create signed volume
    trades_df['signed_volume'] = trades_df['direction'] * trades_df['shares_normalized']

    # Sort by timestamp
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)

    # Create time bins
    trades_df['time_bin'] = trades_df['timestamp'].dt.floor(f'{TIME_WINDOW}min')

    # Aggregate by time bin
    print(f"\nAggregating to {TIME_WINDOW}-min windows...")
    ti_agg = trades_df.groupby('time_bin').agg({
        'signed_volume': 'sum',        # Trade Imbalance (TI)
        'shares_normalized': 'sum',    # Total Volume
        'direction': 'count'           # Number of trades
    }).reset_index()

    ti_agg.columns = ['timestamp', 'trade_imbalance', 'total_volume', 'num_trades']

    print(f"  Created {len(ti_agg)} time windows")
    print(f"  Date range: {ti_agg['timestamp'].min()} to {ti_agg['timestamp'].max()}")

    # Summary stats
    print(f"\nSummary Statistics:")
    print(f"  Trade Imbalance: mean={ti_agg['trade_imbalance'].mean():.2f}, std={ti_agg['trade_imbalance'].std():.2f}")
    print(f"  Total Volume: mean={ti_agg['total_volume'].mean():.2f}, std={ti_agg['total_volume'].std():.2f}")
    print(f"  Num Trades: mean={ti_agg['num_trades'].mean():.1f}, std={ti_agg['num_trades'].std():.1f}")

    # Save to CSV
    print(f"\nSaving to {OUTPUT_FILE}...")
    ti_agg.to_csv(OUTPUT_FILE, index=False)

    # Check file size
    file_size = OUTPUT_FILE.stat().st_size
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("DONE - Pre-computed TI data saved")
    print("=" * 60)

    return ti_agg


if __name__ == "__main__":
    main()
