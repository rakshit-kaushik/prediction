"""
calculate_ofi.py
================
Calculate Order Flow Imbalance (OFI) using Cont, Kukanov & Stoikov (2011)

Uses CORRECT best bid/ask:
- Best Bid = Highest bid price
- Best Ask = Lowest ask price

OFI Formula from the paper:
e_n = I{P^B_n > P^B_{n-1}} × q^B_n - I{P^B_n < P^B_{n-1}} × q^B_{n-1}
      - I{P^A_n < P^A_{n-1}} × q^A_n + I{P^A_n > P^A_{n-1}} × q^A_{n-1}

Where:
- P^B_n = best bid price at time n
- P^A_n = best ask price at time n
- q^B_n = queue size at best bid
- q^A_n = queue size at best ask
- I{} = indicator function (1 if condition true, 0 otherwise)

Note: Using strict inequalities (> and <) to avoid double-counting

Usage:
    python calculate_ofi.py

Input:
    data/orderbook_fed_oct15_31_processed_CORRECT.csv

Output:
    data/ofi_results.csv - OFI values and price changes
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_ofi(df):
    """
    Calculate Order Flow Imbalance from orderbook snapshots

    Args:
        df: DataFrame with best bid/ask prices and sizes

    Returns:
        DataFrame with OFI values
    """
    print("\n" + "=" * 80)
    print("CALCULATING ORDER FLOW IMBALANCE (OFI)")
    print("=" * 80)
    print("\nUsing Cont, Kukanov & Stoikov (2011) formula")

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

    # Calculate indicator functions using STRICT inequalities
    # This avoids double-counting when prices are unchanged

    # I{P^B_n > P^B_{n-1}} - bid price increased
    df['bid_up'] = (df['best_bid_price'] > df['prev_best_bid_price']).astype(int)

    # I{P^B_n < P^B_{n-1}} - bid price decreased
    df['bid_down'] = (df['best_bid_price'] < df['prev_best_bid_price']).astype(int)

    # I{P^A_n < P^A_{n-1}} - ask price decreased
    df['ask_down'] = (df['best_ask_price'] < df['prev_best_ask_price']).astype(int)

    # I{P^A_n > P^A_{n-1}} - ask price increased
    df['ask_up'] = (df['best_ask_price'] > df['prev_best_ask_price']).astype(int)

    # Calculate OFI using the Cont et al. (2011) formula
    df['ofi'] = (
        df['bid_up'] * df['best_bid_size'] -
        df['bid_down'] * df['prev_best_bid_size'] -
        df['ask_down'] * df['best_ask_size'] +
        df['ask_up'] * df['prev_best_ask_size']
    )

    # Calculate price changes
    df['delta_mid_price'] = df['mid_price'] - df['prev_mid_price']
    df['delta_best_bid'] = df['best_bid_price'] - df['prev_best_bid_price']
    df['delta_best_ask'] = df['best_ask_price'] - df['prev_best_ask_price']

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

    print(f"\n📊 Mid-price changes:")
    print(f"   Mean: {df['delta_mid_price'].mean():.6f}")
    print(f"   Std:  {df['delta_mid_price'].std():.6f}")
    print(f"   Non-zero: {(df['delta_mid_price'] != 0).sum()} / {len(df)} ({100*(df['delta_mid_price'] != 0).sum()/len(df):.1f}%)")

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
    corr_bid = df['ofi'].corr(df['delta_best_bid'])
    corr_ask = df['ofi'].corr(df['delta_best_ask'])

    print(f"\n   OFI vs ΔMid-Price: {corr_mid:.4f}")
    print(f"   OFI vs ΔBest Bid:  {corr_bid:.4f}")
    print(f"   OFI vs ΔBest Ask:  {corr_ask:.4f}")

    if abs(corr_mid) > 0.1:
        print(f"\n   ✅ Correlation detected! OFI may predict price changes")
    else:
        print(f"\n   ⚠️  Weak correlation - OFI may not strongly predict price changes")

    return df


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "OFI CALCULATION FROM PAPER" + " " * 31 + "║")
    print("║" + " " * 15 + "Cont, Kukanov & Stoikov (2011)" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")

    # Load corrected orderbook data
    print("\nLoading orderbook data...")
    df = pd.read_csv('data/orderbook_nov01_20_processed.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    print(f"Loaded {len(df)} snapshots")

    # Calculate OFI
    df_ofi = calculate_ofi(df)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Select columns to save
    output_cols = [
        'timestamp', 'timestamp_ms',
        'best_bid_price', 'best_bid_size',
        'best_ask_price', 'best_ask_size',
        'mid_price', 'spread', 'spread_pct',
        'ofi',
        'delta_mid_price', 'delta_best_bid', 'delta_best_ask',
        'bid_up', 'bid_down', 'ask_up', 'ask_down',
        'total_bid_size', 'total_ask_size', 'total_depth',
        'time_diff'
    ]

    df_output = df_ofi[output_cols]
    output_file = 'data/ofi_nov01_20_results.csv'
    df_output.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
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
    print(f"   • Correlation (OFI vs ΔPrice): {df_output['ofi'].corr(df_output['delta_mid_price']):.4f}")

    print(f"\n📂 Output file: {output_file}")
    print(f"   Contains: OFI values, price changes, indicator functions")

    print("\n" + "=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
