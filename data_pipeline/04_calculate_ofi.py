"""
04_calculate_ofi.py
===================
STEP 4: Calculate Order Flow Imbalance (OFI) using Cont, Kukanov & Stoikov (2011)

WHAT IT DOES:
-------------
Input:  Processed orderbook snapshots (from step 3)
Output: OFI values with price changes and event indicators

HOW IT WORKS:
------------
1. Loads processed orderbook data
2. Creates lagged values (previous snapshot)
3. Calculates indicator functions for price movements
4. Computes OFI using the paper's formula
5. Calculates price changes (both dollar and tick-normalized)
6. Saves results to CSV

OFI FORMULA (from Cont et al. 2011):
-----------------------------------
e_n = I{P^B_n ≥ P^B_{n-1}} × q^B_n - I{P^B_n ≤ P^B_{n-1}} × q^B_{n-1}
      - I{P^A_n ≤ P^A_{n-1}} × q^A_n + I{P^A_n ≥ P^A_{n-1}} × q^A_{n-1}

Where:
- P^B_n = best bid price at time n
- P^A_n = best ask price at time n
- q^B_n = depth at best bid (best_bid_size)
- q^A_n = depth at best ask (best_ask_size)
- I{} = indicator function (1 if condition true, 0 otherwise)

IMPORTANT FIXES vs OLD VERSION:
------------------------------
1. Uses INCLUSIVE inequalities (>= and <=) as per paper, not strict (> and <)
2. Adds tick-normalized price changes (delta_mid_price_ticks)
3. Uses best bid/ask sizes (depth at level 1) - already correct

USAGE:
------
    python data_pipeline/04_calculate_ofi.py

OUTPUT:
-------
- data/market_name_ofi.csv with columns:
  • timestamp, timestamp_ms
  • best_bid_price, best_bid_size, best_ask_price, best_ask_size
  • mid_price, spread, spread_pct
  • ofi (Order Flow Imbalance)
  • delta_mid_price (price change in $)
  • delta_mid_price_ticks (price change in ticks)
  • delta_best_bid, delta_best_ask
  • bid_up, bid_down, ask_up, ask_down (event indicators)
  • total_bid_size, total_ask_size, total_depth
  • time_diff
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.config import *
from scripts.config_analysis import TICK_SIZE


def calculate_ofi(df):
    """
    Calculate Order Flow Imbalance from orderbook snapshots

    Uses the Cont, Kukanov & Stoikov (2011) formula with INCLUSIVE inequalities

    Args:
        df: DataFrame with best bid/ask prices and sizes

    Returns:
        DataFrame with OFI values, price changes, and event indicators
    """
    print("\n" + "=" * 80)
    print("CALCULATING ORDER FLOW IMBALANCE (OFI)")
    print("=" * 80)
    print("\nUsing Cont, Kukanov & Stoikov (2011) formula")
    print("✓ Inclusive inequalities (>= and <=)")
    print(f"✓ Tick size: ${TICK_SIZE}")

    # Make a copy and sort by timestamp
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\n📊 Input data:")
    print(f"   Snapshots: {len(df)}")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Create lagged values (previous snapshot)
    df['prev_best_bid_price'] = df['best_bid_price'].shift(1)
    df['prev_best_bid_size'] = df['best_bid_size'].shift(1)
    df['prev_best_ask_price'] = df['best_ask_price'].shift(1)
    df['prev_best_ask_size'] = df['best_ask_size'].shift(1)
    df['prev_mid_price'] = df['mid_price'].shift(1)

    # Calculate indicator functions using INCLUSIVE inequalities (>= and <=)
    # This follows the paper's formula exactly

    # I{P^B_n >= P^B_{n-1}} - bid price increased or stayed same
    df['bid_up'] = (df['best_bid_price'] >= df['prev_best_bid_price']).astype(int)

    # I{P^B_n <= P^B_{n-1}} - bid price decreased or stayed same
    df['bid_down'] = (df['best_bid_price'] <= df['prev_best_bid_price']).astype(int)

    # I{P^A_n <= P^A_{n-1}} - ask price decreased or stayed same
    df['ask_down'] = (df['best_ask_price'] <= df['prev_best_ask_price']).astype(int)

    # I{P^A_n >= P^A_{n-1}} - ask price increased or stayed same
    df['ask_up'] = (df['best_ask_price'] >= df['prev_best_ask_price']).astype(int)

    # Calculate OFI using the Cont et al. (2011) formula
    # Note: q^B_n and q^A_n are depth at best bid/ask (best_bid_size, best_ask_size)
    df['ofi'] = (
        df['bid_up'] * df['best_bid_size'] -
        df['bid_down'] * df['prev_best_bid_size'] -
        df['ask_down'] * df['best_ask_size'] +
        df['ask_up'] * df['prev_best_ask_size']
    )

    # Calculate price changes in dollars
    df['delta_mid_price'] = df['mid_price'] - df['prev_mid_price']
    df['delta_best_bid'] = df['best_bid_price'] - df['prev_best_bid_price']
    df['delta_best_ask'] = df['best_ask_price'] - df['prev_best_ask_price']

    # Calculate price changes in TICKS (critical for analysis)
    df['delta_mid_price_ticks'] = df['delta_mid_price'] / TICK_SIZE

    # Calculate time differences (in seconds)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    # Drop first row (no previous values)
    df = df.iloc[1:].reset_index(drop=True)

    # Print OFI statistics
    print("\n" + "-" * 80)
    print("OFI STATISTICS")
    print("-" * 80)

    print(f"\n📈 Order Flow Imbalance:")
    print(f"   Mean: {df['ofi'].mean():,.2f}")
    print(f"   Std:  {df['ofi'].std():,.2f}")
    print(f"   Min:  {df['ofi'].min():,.2f}")
    print(f"   Max:  {df['ofi'].max():,.2f}")
    print(f"   Median: {df['ofi'].median():,.2f}")

    # Non-zero OFI
    non_zero_ofi = (df['ofi'] != 0).sum()
    print(f"\n   Non-zero OFI: {non_zero_ofi} / {len(df)} ({100*non_zero_ofi/len(df):.1f}%)")

    # Positive vs negative OFI
    positive_ofi = (df['ofi'] > 0).sum()
    negative_ofi = (df['ofi'] < 0).sum()
    print(f"   Positive OFI: {positive_ofi} ({100*positive_ofi/len(df):.1f}%)")
    print(f"   Negative OFI: {negative_ofi} ({100*negative_ofi/len(df):.1f}%)")

    # Price movement statistics
    print("\n" + "-" * 80)
    print("PRICE CHANGE STATISTICS")
    print("-" * 80)

    print(f"\n📊 Mid-price changes (dollars):")
    print(f"   Mean: {df['delta_mid_price'].mean():.6f}")
    print(f"   Std:  {df['delta_mid_price'].std():.6f}")
    print(f"   Non-zero: {(df['delta_mid_price'] != 0).sum()} / {len(df)} ({100*(df['delta_mid_price'] != 0).sum()/len(df):.1f}%)")

    print(f"\n📊 Mid-price changes (ticks):")
    print(f"   Mean: {df['delta_mid_price_ticks'].mean():.4f}")
    print(f"   Std:  {df['delta_mid_price_ticks'].std():.4f}")
    print(f"   Non-zero: {(df['delta_mid_price_ticks'] != 0).sum()} / {len(df)} ({100*(df['delta_mid_price_ticks'] != 0).sum()/len(df):.1f}%)")

    print(f"\n📊 Best bid changes:")
    print(f"   Mean: {df['delta_best_bid'].mean():.6f}")
    print(f"   Std:  {df['delta_best_bid'].std():.6f}")
    print(f"   Non-zero: {(df['delta_best_bid'] != 0).sum()} / {len(df)} ({100*(df['delta_best_bid'] != 0).sum()/len(df):.1f}%)")

    print(f"\n📊 Best ask changes:")
    print(f"   Mean: {df['delta_best_ask'].mean():.6f}")
    print(f"   Std:  {df['delta_best_ask'].std():.6f}")
    print(f"   Non-zero: {(df['delta_best_ask'] != 0).sum()} / {len(df)} ({100*(df['delta_best_ask'] != 0).sum()/len(df):.1f}%)")

    # Correlation between OFI and price changes
    print("\n" + "-" * 80)
    print("OFI vs PRICE CHANGE CORRELATION")
    print("-" * 80)

    corr_mid = df['ofi'].corr(df['delta_mid_price'])
    corr_mid_ticks = df['ofi'].corr(df['delta_mid_price_ticks'])
    corr_bid = df['ofi'].corr(df['delta_best_bid'])
    corr_ask = df['ofi'].corr(df['delta_best_ask'])

    print(f"\n   OFI vs ΔMid-Price ($):    {corr_mid:.4f}")
    print(f"   OFI vs ΔMid-Price (ticks): {corr_mid_ticks:.4f}")
    print(f"   OFI vs ΔBest Bid:          {corr_bid:.4f}")
    print(f"   OFI vs ΔBest Ask:          {corr_ask:.4f}")

    if abs(corr_mid_ticks) > 0.1:
        print(f"\n   ✅ Correlation detected! OFI may predict price changes")
    else:
        print(f"\n   ⚠️  Weak correlation - OFI may not strongly predict price changes")

    return df


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "OFI CALCULATION (UPDATED)" + " " * 32 + "║")
    print("║" + " " * 15 + "Cont, Kukanov & Stoikov (2011)" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check if processed orderbook file exists
    import os
    if not os.path.exists(PROCESSED_ORDERBOOK_FILE):
        print(f"\n❌ Processed orderbook file not found: {PROCESSED_ORDERBOOK_FILE}")
        print(f"\n💡 Run step 3 first: python data_pipeline/03_process_orderbooks.py")
        sys.exit(1)

    # Load processed orderbook data
    print("\nLoading orderbook data...")
    df = pd.read_csv(PROCESSED_ORDERBOOK_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    print(f"✓ Loaded {len(df)} snapshots")

    # Calculate OFI
    df_ofi = calculate_ofi(df)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Determine output file name (replace _processed with _ofi)
    output_file = PROCESSED_ORDERBOOK_FILE.replace('_processed.csv', '_ofi.csv')

    # Select columns to save
    output_cols = [
        'timestamp', 'timestamp_ms',
        'best_bid_price', 'best_bid_size',
        'best_ask_price', 'best_ask_size',
        'mid_price', 'spread', 'spread_pct',
        'ofi',
        'delta_mid_price',
        'delta_mid_price_ticks',  # NEW: tick-normalized price change
        'delta_best_bid', 'delta_best_ask',
        'bid_up', 'bid_down', 'ask_up', 'ask_down',
        'total_bid_size', 'total_ask_size', 'total_depth',
        'time_diff'
    ]

    df_output = df_ofi[output_cols]
    df_output.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    print(f"   Rows: {len(df_output)}")
    print(f"   Columns: {len(output_cols)}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ OFI CALCULATION COMPLETE")
    print("=" * 80)

    print(f"\n📋 Key Statistics:")
    print(f"   • {len(df_output)} snapshots analyzed")
    print(f"   • {(df_output['ofi'] != 0).sum()} snapshots with non-zero OFI")
    print(f"   • {(df_output['delta_mid_price'] != 0).sum()} snapshots with price changes")
    print(f"   • Correlation (OFI vs ΔPrice $): {df_output['ofi'].corr(df_output['delta_mid_price']):.4f}")
    print(f"   • Correlation (OFI vs ΔPrice ticks): {df_output['ofi'].corr(df_output['delta_mid_price_ticks']):.4f}")

    print(f"\n📂 Output file: {output_file}")
    print(f"   Contains: OFI values, price changes, indicator functions")

    print(f"\n💡 Key improvements vs old version:")
    print(f"   ✓ Uses inclusive inequalities (>= and <=) as per paper")
    print(f"   ✓ Includes tick-normalized price changes")
    print(f"   ✓ Ready for regression analysis with configurable TIME_WINDOW")

    print("\n" + "=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
