"""
02_calculate_mlofi.py
=====================
STEP 2: Calculate Multi-Level Order Flow Imbalance (MLOFI)

WHAT IT DOES:
-------------
Input:  Multi-level processed data (from step 1)
Output: OFI calculated at each level, plus cumulative OFI

HOW IT WORKS:
-------------
For each level L, apply the Cont, Kukanov & Stoikov (2011) formula:

OFI_L = I{P^B_L >= P^B_L,prev} × q^B_L - I{P^B_L <= P^B_L,prev} × q^B_L,prev
      - I{P^A_L <= P^A_L,prev} × q^A_L + I{P^A_L >= P^A_L,prev} × q^A_L,prev

Where:
- P^B_L = Bid price at level L
- P^A_L = Ask price at level L
- q^B_L = Size at bid level L
- q^A_L = Size at ask level L
- I{} = Indicator function (1 if true, 0 otherwise)

OUTPUT COLUMNS:
--------------
timestamp, timestamp_ms,
ofi_l1, ofi_l2, ..., ofi_lN,                    # Per-level OFI
ofi_cumulative_l1, ofi_cumulative_l2, ...,      # Cumulative through level N
delta_mid_price, delta_mid_price_ticks,         # Price changes
mid_price, spread

USAGE:
------
    python mlofi/02_calculate_mlofi.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MULTILEVEL_PROCESSED_FILE,
    MLOFI_CALCULATED_FILE,
    EXISTING_OFI_FILE,
    TICK_SIZE,
    MLOFI_OUTPUT_DIR
)


def calculate_ofi_for_level(df, level):
    """
    Calculate OFI for a specific level using Cont et al. formula.

    Args:
        df: DataFrame with multi-level data
        level: Level number (1, 2, 3, etc.)

    Returns:
        Series with OFI values for this level
    """
    bid_price_col = f'bid_price_l{level}'
    bid_size_col = f'bid_size_l{level}'
    ask_price_col = f'ask_price_l{level}'
    ask_size_col = f'ask_size_l{level}'

    # Check if columns exist
    if bid_price_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Current values
    bid_price = df[bid_price_col]
    bid_size = df[bid_size_col]
    ask_price = df[ask_price_col]
    ask_size = df[ask_size_col]

    # Previous values
    prev_bid_price = bid_price.shift(1)
    prev_bid_size = bid_size.shift(1)
    prev_ask_price = ask_price.shift(1)
    prev_ask_size = ask_size.shift(1)

    # Indicator functions (using inclusive inequalities as per paper)
    bid_up = (bid_price >= prev_bid_price).astype(float)
    bid_down = (bid_price <= prev_bid_price).astype(float)
    ask_up = (ask_price >= prev_ask_price).astype(float)
    ask_down = (ask_price <= prev_ask_price).astype(float)

    # Handle NaN in price comparisons (treat as no change)
    bid_up = bid_up.fillna(0)
    bid_down = bid_down.fillna(0)
    ask_up = ask_up.fillna(0)
    ask_down = ask_down.fillna(0)

    # OFI formula
    ofi = (
        bid_up * bid_size.fillna(0) -
        bid_down * prev_bid_size.fillna(0) -
        ask_down * ask_size.fillna(0) +
        ask_up * prev_ask_size.fillna(0)
    )

    return ofi


def verify_against_existing_ofi(df_mlofi, existing_file):
    """
    Verify that ofi_l1 matches existing OFI calculation.

    Args:
        df_mlofi: DataFrame with MLOFI data
        existing_file: Path to existing OFI CSV

    Returns:
        bool: True if verification passes
    """
    print("\n" + "=" * 80)
    print("VERIFICATION: Comparing OFI_L1 with existing OFI")
    print("=" * 80)

    if not Path(existing_file).exists():
        print(f"⚠️  Existing OFI file not found: {existing_file}")
        print("   Skipping verification")
        return True

    # Load existing OFI
    df_existing = pd.read_csv(existing_file)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'], format='mixed')

    print(f"\n📊 Existing OFI data: {len(df_existing)} rows")
    print(f"📊 MLOFI data: {len(df_mlofi)} rows")

    # Compare by timestamp_ms
    common_ts = set(df_mlofi['timestamp_ms']) & set(df_existing['timestamp_ms'])
    print(f"📊 Common timestamps: {len(common_ts)}")

    if len(common_ts) == 0:
        print("⚠️  No common timestamps found!")
        return False

    # Merge for comparison
    df_compare = df_mlofi[['timestamp_ms', 'ofi_l1']].merge(
        df_existing[['timestamp_ms', 'ofi']],
        on='timestamp_ms',
        how='inner'
    )

    # Calculate correlation
    correlation = df_compare['ofi_l1'].corr(df_compare['ofi'])
    print(f"\n📊 Correlation between ofi_l1 and existing ofi: {correlation:.6f}")

    # Check for exact matches
    df_compare['diff'] = abs(df_compare['ofi_l1'] - df_compare['ofi'])
    exact_matches = (df_compare['diff'] < 0.01).sum()
    print(f"📊 Exact matches (diff < 0.01): {exact_matches}/{len(df_compare)} ({100*exact_matches/len(df_compare):.1f}%)")

    # Sample comparison
    print(f"\n📊 Sample comparison (first 5):")
    print(df_compare[['timestamp_ms', 'ofi_l1', 'ofi', 'diff']].head())

    if correlation > 0.99:
        print(f"\n✅ Verification PASSED! Correlation > 0.99")
        return True
    elif correlation > 0.95:
        print(f"\n⚠️  Verification WARNING: Correlation between 0.95-0.99")
        print("   Minor differences may be due to float precision or edge cases")
        return True
    else:
        print(f"\n❌ Verification FAILED! Correlation < 0.95")
        return False


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "MULTI-LEVEL OFI CALCULATION" + " " * 35 + "║")
    print("║" + " " * 12 + "Cont, Kukanov & Stoikov (2011) at Each Level" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not MULTILEVEL_PROCESSED_FILE.exists():
        print(f"\n❌ Multi-level processed file not found: {MULTILEVEL_PROCESSED_FILE}")
        print("   Please run step 1 first: python mlofi/01_process_multilevel.py")
        sys.exit(1)

    # Create output directory
    MLOFI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load multi-level data
    print(f"\n📂 Loading multi-level data...")
    df = pd.read_csv(MULTILEVEL_PROCESSED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   Loaded {len(df)} snapshots")

    # Determine number of levels available
    level_cols = [c for c in df.columns if c.startswith('bid_price_l')]
    max_level = max([int(c.replace('bid_price_l', '')) for c in level_cols])
    print(f"   Max levels available: {max_level}")

    # Calculate OFI for each level
    print("\n" + "-" * 80)
    print("CALCULATING OFI AT EACH LEVEL")
    print("-" * 80)

    ofi_columns = []
    for level in range(1, max_level + 1):
        col_name = f'ofi_l{level}'
        df[col_name] = calculate_ofi_for_level(df, level)
        ofi_columns.append(col_name)

        if level <= 10 or level == max_level:
            non_zero = (df[col_name].abs() > 0).sum()
            mean_val = df[col_name].mean()
            print(f"   Level {level}: {non_zero} non-zero values, mean = {mean_val:.2f}")

    # Calculate cumulative OFI
    print("\n" + "-" * 80)
    print("CALCULATING CUMULATIVE OFI")
    print("-" * 80)

    cumulative_columns = []
    for level in range(1, max_level + 1):
        col_name = f'ofi_cumulative_l{level}'
        # Sum OFI from level 1 through current level
        ofi_cols_to_sum = [f'ofi_l{l}' for l in range(1, level + 1)]
        df[col_name] = df[ofi_cols_to_sum].sum(axis=1)
        cumulative_columns.append(col_name)

        if level in [1, 5, 10, max_level]:
            non_zero = (df[col_name].abs() > 0).sum()
            mean_val = df[col_name].mean()
            print(f"   Cumulative L1-L{level}: {non_zero} non-zero, mean = {mean_val:.2f}")

    # Calculate price changes
    print("\n" + "-" * 80)
    print("CALCULATING PRICE CHANGES")
    print("-" * 80)

    df['prev_mid_price'] = df['mid_price'].shift(1)
    df['delta_mid_price'] = df['mid_price'] - df['prev_mid_price']
    df['delta_mid_price_ticks'] = df['delta_mid_price'] / TICK_SIZE

    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    print(f"\n📊 Price changes:")
    print(f"   Mean (dollars): {df['delta_mid_price'].mean():.6f}")
    print(f"   Std (dollars): {df['delta_mid_price'].std():.6f}")
    print(f"   Mean (ticks): {df['delta_mid_price_ticks'].mean():.4f}")
    print(f"   Std (ticks): {df['delta_mid_price_ticks'].std():.4f}")

    # Remove first row (no lagged values)
    df = df.iloc[1:].reset_index(drop=True)

    # Print correlation analysis
    print("\n" + "-" * 80)
    print("OFI-PRICE CORRELATION BY LEVEL")
    print("-" * 80)

    print(f"\n📊 Correlation of OFI at each level with price change:")
    for level in [1, 2, 3, 4, 5, 10, max_level]:
        if level <= max_level:
            corr = df[f'ofi_l{level}'].corr(df['delta_mid_price_ticks'])
            print(f"   Level {level}: {corr:.4f}")

    print(f"\n📊 Correlation of cumulative OFI with price change:")
    for level in [1, 5, 10, max_level]:
        if level <= max_level:
            corr = df[f'ofi_cumulative_l{level}'].corr(df['delta_mid_price_ticks'])
            print(f"   Cumulative L1-L{level}: {corr:.4f}")

    # Cross-level correlation matrix (for first 5 levels)
    print("\n" + "-" * 80)
    print("CROSS-LEVEL OFI CORRELATION (L1-L5)")
    print("-" * 80)

    ofi_l1_l5 = [f'ofi_l{l}' for l in range(1, min(6, max_level + 1))]
    corr_matrix = df[ofi_l1_l5].corr()

    print(f"\n📊 Correlation matrix:")
    print(corr_matrix.round(3).to_string())

    # Check for multicollinearity warning
    high_corr = []
    for i in range(len(ofi_l1_l5)):
        for j in range(i + 1, len(ofi_l1_l5)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append((ofi_l1_l5[i], ofi_l1_l5[j], corr_val))

    if high_corr:
        print(f"\n⚠️  High correlations detected (>0.7):")
        for c1, c2, val in high_corr[:5]:
            print(f"   {c1} vs {c2}: {val:.3f}")
        print("   → Ridge/Lasso/ElasticNet regression recommended!")

    # Verify against existing OFI
    verify_against_existing_ofi(df, EXISTING_OFI_FILE)

    # Select columns to save
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    output_cols = (
        ['timestamp', 'timestamp_ms'] +
        ofi_columns +
        cumulative_columns +
        ['delta_mid_price', 'delta_mid_price_ticks', 'mid_price', 'spread', 'time_diff']
    )

    df_output = df[output_cols]
    df_output.to_csv(MLOFI_CALCULATED_FILE, index=False)

    print(f"\n✓ Saved to: {MLOFI_CALCULATED_FILE}")
    print(f"   Rows: {len(df_output)}")
    print(f"   Columns: {len(output_cols)}")

    print(f"\n📋 Columns saved:")
    print(f"   - timestamp, timestamp_ms")
    print(f"   - ofi_l1 to ofi_l{max_level} (per-level OFI)")
    print(f"   - ofi_cumulative_l1 to ofi_cumulative_l{max_level}")
    print(f"   - delta_mid_price, delta_mid_price_ticks")
    print(f"   - mid_price, spread, time_diff")

    print("\n" + "=" * 80)
    print("✅ MLOFI CALCULATION COMPLETE")
    print("=" * 80)
    print(f"\n💡 Next step: python mlofi/03_regression_analysis.py")
    print("\n")

    return df_output


if __name__ == "__main__":
    main()
