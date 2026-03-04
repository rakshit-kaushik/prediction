"""
04_process_trades_ti.py
========================
Process DOME trade data and calculate Trade Flow Imbalance (TFI) following
Silantyev (2019) / Cont et al. (2014).

Improvements over v1:
  1. Filter to YES token only (removes NO token contamination)
  2. Use DOME trade prices for mid-price (not sparse OFI snapshots)
  3. Two mid-price methods: last-trade price and VWAP
  4. Extended time windows up to 360 min

Trade Imbalance = Sigma(signed_volume) where:
  - BUY trades -> positive volume
  - SELL trades -> negative volume

Price Impact Model: Delta_P = alpha + beta * TFI + epsilon

Usage:
    python data_pipeline/04_process_trades_ti.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# File paths
DOME_TRADES_FILE = Path(__file__).parent.parent / "DOME_zohran-oct-15_2025-11-29.csv"
OUTPUT_81_FILE = Path(__file__).parent.parent / "data" / "ti_81_configs.csv"

TICK_SIZE = 0.01

# ============================================================================
# CONFIGURATION
# ============================================================================

TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360]  # minutes

OUTLIER_METHODS = [
    'Raw',
    'IQR (1.5x)',
    'Pctl (1%-99%)',
    'Z-Score (3)',
    'Winsorized',
    'Abs (200k)',
    'Abs (100k)',
    'MAD (3)',
    'Pctl (5%-95%)'
]

# Mid-price methods to try
MID_PRICE_METHODS = ['last_trade', 'vwap']

# ============================================================================
# OUTLIER FILTERING FUNCTIONS (Same as OFI dashboard)
# ============================================================================

def filter_outliers_iqr(df, column, multiplier=1.5):
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_percentile(df, column, lower_pct=0.01, upper_pct=0.99):
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_zscore(df, column, threshold=3):
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return df
    z_scores = (df[column] - mean) / std
    return df[z_scores.abs() <= threshold]

def winsorize_data(df, column, limits=(0.01, 0.01)):
    df = df.copy()
    lower = df[column].quantile(limits[0])
    upper = df[column].quantile(1 - limits[1])
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df

def filter_absolute_threshold(df, column, lower=-200000, upper=200000):
    df = df.copy()
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_mad(df, column, threshold=3):
    df = df.copy()
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    if mad == 0:
        return df
    modified_z = 0.6745 * (df[column] - median) / mad
    return df[np.abs(modified_z) <= threshold]

def apply_outlier_method(df, method_idx, column='trade_imbalance'):
    if method_idx == 0:
        return df.copy()
    elif method_idx == 1:
        return filter_outliers_iqr(df, column)
    elif method_idx == 2:
        return filter_outliers_percentile(df, column)
    elif method_idx == 3:
        return filter_outliers_zscore(df, column)
    elif method_idx == 4:
        return winsorize_data(df, column)
    elif method_idx == 5:
        return filter_absolute_threshold(df, column, -200000, 200000)
    elif method_idx == 6:
        return filter_absolute_threshold(df, column, -100000, 100000)
    elif method_idx == 7:
        return filter_outliers_mad(df, column, 3)
    elif method_idx == 8:
        return filter_outliers_percentile(df, column, 0.05, 0.95)
    return df.copy()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dome_trades():
    """Load DOME trade data, filtered to YES token only."""
    print("\n Loading DOME trade data...")

    trades = pd.read_csv(DOME_TRADES_FILE)
    print(f"   Total records (both tokens): {len(trades):,}")

    # Filter to YES token only
    primary_token = str(trades['primary_token_id'].iloc[0])
    trades = trades[trades['token_id'].astype(str) == primary_token].copy()
    print(f"   YES token records: {len(trades):,}")

    # Parse timestamp
    trades['timestamp'] = pd.to_datetime(trades['block_timestamp'], utc=True)

    # Normalize shares (divide by 1e6)
    trades['shares_normalized'] = trades['shares'] / 1e6

    # Signed volume: BUY -> +, SELL -> -
    trades['signed_volume'] = np.where(
        trades['side'] == 'BUY',
        trades['shares_normalized'],
        -trades['shares_normalized']
    )

    # Dollar volume for VWAP: price * shares
    trades['dollar_volume'] = trades['price'] * trades['shares_normalized']

    # Sort by timestamp
    trades = trades.sort_values('timestamp').reset_index(drop=True)

    buys = (trades['side'] == 'BUY').sum()
    sells = (trades['side'] == 'SELL').sum()
    print(f"   BUY trades:  {buys:,} ({buys/len(trades)*100:.1f}%)")
    print(f"   SELL trades: {sells:,} ({sells/len(trades)*100:.1f}%)")
    print(f"   Price range: ${trades['price'].min():.4f} - ${trades['price'].max():.4f}")
    print(f"   Unique prices: {trades['price'].nunique()}")
    print(f"   Date range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")

    return trades


# ============================================================================
# TI + PRICE CALCULATION (all from trades — no external merge needed)
# ============================================================================

def calculate_ti_and_price_per_window(trades, time_window_minutes, mid_price_method='last_trade'):
    """
    Calculate Trade Imbalance AND mid-price change per window, all from trade data.

    Mid-price methods:
      - 'last_trade': use last trade price in each window
      - 'vwap': use volume-weighted average price in each window
    """
    trades = trades.copy()
    trades['window'] = trades['timestamp'].dt.floor(f'{time_window_minutes}min')

    if mid_price_method == 'last_trade':
        # Aggregate: TFI + first/last trade price
        grouped = trades.groupby('window').agg(
            trade_imbalance=('signed_volume', 'sum'),
            total_volume=('shares_normalized', 'sum'),
            trade_count=('side', 'count'),
            price_first=('price', 'first'),
            price_last=('price', 'last'),
        ).reset_index()

        grouped['delta_mid_price'] = grouped['price_last'] - grouped['price_first']

    elif mid_price_method == 'vwap':
        # VWAP = sum(price * volume) / sum(volume)
        grouped = trades.groupby('window').agg(
            trade_imbalance=('signed_volume', 'sum'),
            total_volume=('shares_normalized', 'sum'),
            trade_count=('side', 'count'),
            dollar_volume_sum=('dollar_volume', 'sum'),
        ).reset_index()

        grouped['vwap'] = grouped['dollar_volume_sum'] / grouped['total_volume']
        # Price change = VWAP(k) - VWAP(k-1)
        grouped = grouped.sort_values('window').reset_index(drop=True)
        grouped['delta_mid_price'] = grouped['vwap'].diff()

    grouped['delta_mid_price_ticks'] = grouped['delta_mid_price'] / TICK_SIZE

    return grouped


# ============================================================================
# REGRESSION
# ============================================================================

def run_ti_regression(df):
    """Run TFI regression: Delta_P = alpha + beta * TFI + epsilon"""
    df_clean = df.dropna(subset=['trade_imbalance', 'delta_mid_price_ticks'])

    if len(df_clean) < 10:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_clean['trade_imbalance'],
        df_clean['delta_mid_price_ticks']
    )

    return {
        'beta': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'n_windows': len(df_clean)
    }


# ============================================================================
# FULL ANALYSIS
# ============================================================================

def run_full_analysis(trades):
    """
    Run TFI regression for all configurations:
      - 13 time windows x 9 outlier methods x 2 mid-price methods
    """
    total = len(TIME_WINDOWS) * len(OUTLIER_METHODS) * len(MID_PRICE_METHODS)
    print(f"\n Running full TFI analysis...")
    print(f"   Time Windows: {TIME_WINDOWS}")
    print(f"   Outlier Methods: {len(OUTLIER_METHODS)}")
    print(f"   Mid-price Methods: {MID_PRICE_METHODS}")
    print(f"   Total configs: {total}")

    results = []

    for mp_method in MID_PRICE_METHODS:
        for tw in TIME_WINDOWS:
            # Calculate TI + price from trades directly
            data = calculate_ti_and_price_per_window(trades, tw, mp_method)

            if len(data.dropna(subset=['delta_mid_price_ticks'])) < 10:
                continue

            for method_idx, method_name in enumerate(OUTLIER_METHODS):
                filtered = apply_outlier_method(data, method_idx, 'trade_imbalance')

                if len(filtered.dropna(subset=['delta_mid_price_ticks'])) < 10:
                    continue

                reg_result = run_ti_regression(filtered)

                if reg_result is not None:
                    results.append({
                        'mid_price_method': mp_method,
                        'time_window': tw,
                        'outlier_method': method_name,
                        'r_squared': reg_result['r_squared'],
                        'beta': reg_result['beta'],
                        'p_value': reg_result['p_value'],
                        'n_windows': reg_result['n_windows'],
                        'std_err': reg_result['std_err']
                    })

    results_df = pd.DataFrame(results)
    print(f"\n   Computed {len(results_df)} configurations")
    return results_df


def print_summary(results_df):
    """Print summary by mid-price method."""

    for mp_method in MID_PRICE_METHODS:
        subset = results_df[results_df['mid_price_method'] == mp_method]
        if len(subset) == 0:
            continue

        print("\n" + "=" * 80)
        print(f"TFI RESULTS — Mid-price method: {mp_method.upper()}")
        print("=" * 80)

        best = subset.loc[subset['r_squared'].idxmax()]
        print(f"\n BEST CONFIG:")
        print(f"   Window:  {int(best['time_window'])} min")
        print(f"   Method:  {best['outlier_method']}")
        print(f"   R^2:     {best['r_squared']:.4f} ({best['r_squared']*100:.2f}%)")
        print(f"   Beta:    {best['beta']:.6f}")
        print(f"   p-value: {best['p_value']:.2e}")
        print(f"   N:       {int(best['n_windows'])}")

        # By time window
        print(f"\n R^2 by Time Window:")
        tw_stats = subset.groupby('time_window')['r_squared'].agg(['mean', 'max'])
        for tw in TIME_WINDOWS:
            if tw in tw_stats.index:
                print(f"   {tw:4d} min: avg={tw_stats.loc[tw, 'mean']*100:6.2f}%, max={tw_stats.loc[tw, 'max']*100:6.2f}%")

        # By outlier method
        print(f"\n R^2 by Outlier Method:")
        method_stats = subset.groupby('outlier_method')['r_squared'].agg(['mean', 'max'])
        for method in OUTLIER_METHODS:
            if method in method_stats.index:
                print(f"   {method:15s}: avg={method_stats.loc[method, 'mean']*100:6.2f}%, max={method_stats.loc[method, 'max']*100:6.2f}%")

        print(f"\n Overall: mean={subset['r_squared'].mean()*100:.2f}%, median={subset['r_squared'].median()*100:.2f}%, max={subset['r_squared'].max()*100:.2f}%")


def print_heatmap(results_df, mp_method):
    """Print ASCII heatmap for one mid-price method."""
    subset = results_df[results_df['mid_price_method'] == mp_method]
    if len(subset) == 0:
        return

    print(f"\n{'=' * 120}")
    print(f"R^2 HEATMAP — {mp_method.upper()} (Time Window x Outlier Method)")
    print(f"{'=' * 120}")

    pivot = subset.pivot(index='time_window', columns='outlier_method', values='r_squared')
    pivot = pivot.reindex(columns=OUTLIER_METHODS)

    print("\n" + " " * 10, end="")
    for method in OUTLIER_METHODS:
        print(f"{method[:8]:>10}", end="")
    print()

    for tw in TIME_WINDOWS:
        if tw in pivot.index:
            print(f"{tw:5d}min:", end="")
            for method in OUTLIER_METHODS:
                if method in pivot.columns and pd.notna(pivot.loc[tw, method]):
                    val = pivot.loc[tw, method] * 100
                    print(f"{val:9.2f}%", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print()


def main():
    print("\n" + "=" * 80)
    print("TRADE FLOW IMBALANCE (TFI) ANALYSIS — IMPROVED")
    print("  - YES token only (no NO token contamination)")
    print("  - Mid-price from trades (last_trade + VWAP)")
    print("  - Extended windows up to 360 min")
    print("=" * 80)

    trades = load_dome_trades()
    results_df = run_full_analysis(trades)

    if len(results_df) == 0:
        print("\n No results computed!")
        return None

    print_summary(results_df)
    for mp in MID_PRICE_METHODS:
        print_heatmap(results_df, mp)

    # Save
    print(f"\n Saving results to {OUTPUT_81_FILE}...")
    results_df.to_csv(OUTPUT_81_FILE, index=False)
    print(f"   Saved {len(results_df)} configurations")

    # Top 10 overall
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGS (across all mid-price methods)")
    print("=" * 80)
    top10 = results_df.nlargest(10, 'r_squared')
    for i, row in top10.iterrows():
        print(f"   {row['mid_price_method']:12s} | {int(row['time_window']):4d}min | {row['outlier_method']:15s} | R²={row['r_squared']*100:6.2f}% | β={row['beta']:.6f} | p={row['p_value']:.2e} | N={int(row['n_windows'])}")

    print("\n" + "=" * 80)
    print(" TFI ANALYSIS COMPLETE")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    main()
