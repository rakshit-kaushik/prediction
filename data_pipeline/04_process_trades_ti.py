"""
04_process_trades_ti.py
========================
Process DOME trade data and calculate Trade Imbalance (TI) following Cont et al. (2014).

81-Configuration Analysis (same as OFI):
- 9 time windows x 9 outlier methods = 81 R^2 scores
- Enables direct comparison with OFI results

Trade Imbalance = Sigma(signed_volume) where:
  - BUY trades -> positive volume
  - SELL trades -> negative volume

Price Impact Model: Delta_P = alpha + beta * TI + epsilon

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
OFI_RESULTS_FILE = Path(__file__).parent.parent / "data" / "ofi_results.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "ti_analysis_results.csv"
OUTPUT_81_FILE = Path(__file__).parent.parent / "data" / "ti_81_configs.csv"

# ============================================================================
# CONFIGURATION (Same as OFI dashboard)
# ============================================================================

TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]  # minutes

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

TICK_SIZE = 0.01

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
    """Apply outlier method by index (0-8) to TI column"""
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
    """Load and process DOME trade data"""
    print("\n Loading DOME trade data...")

    trades = pd.read_csv(DOME_TRADES_FILE)
    print(f"   Total records: {len(trades):,}")

    # Parse timestamp and make UTC-aware
    trades['timestamp'] = pd.to_datetime(trades['block_timestamp'], utc=True)

    # Normalize shares (divide by 1e6)
    trades['shares_normalized'] = trades['shares'] / 1e6

    # Create signed volume (positive for BUY, negative for SELL)
    trades['signed_volume'] = np.where(
        trades['side'] == 'BUY',
        trades['shares_normalized'],
        -trades['shares_normalized']
    )

    # Sort by timestamp
    trades = trades.sort_values('timestamp').reset_index(drop=True)

    # Stats
    buys = (trades['side'] == 'BUY').sum()
    sells = (trades['side'] == 'SELL').sum()
    print(f"   BUY trades:  {buys:,} ({buys/len(trades)*100:.1f}%)")
    print(f"   SELL trades: {sells:,} ({sells/len(trades)*100:.1f}%)")
    print(f"   Date range: {trades['timestamp'].min()} to {trades['timestamp'].max()}")

    return trades


def load_ofi_data():
    """Load OFI data for mid-price changes"""
    print("\n Loading OFI data (for order book mid-prices)...")

    ofi_df = pd.read_csv(OFI_RESULTS_FILE)
    ofi_df['timestamp'] = pd.to_datetime(ofi_df['timestamp'], format='mixed', utc=True)

    print(f"   Total snapshots: {len(ofi_df):,}")
    print(f"   Date range: {ofi_df['timestamp'].min()} to {ofi_df['timestamp'].max()}")

    return ofi_df


# ============================================================================
# TI CALCULATION
# ============================================================================

def calculate_ti_per_window(trades, time_window_minutes):
    """
    Calculate Trade Imbalance per time window.

    Args:
        trades: DataFrame with timestamp, signed_volume, shares_normalized
        time_window_minutes: Window size in minutes

    Returns:
        DataFrame with TI per window
    """
    # Floor to window boundaries
    trades = trades.copy()
    trades['window'] = trades['timestamp'].dt.floor(f'{time_window_minutes}min')

    # Aggregate per window
    ti_data = trades.groupby('window').agg({
        'signed_volume': 'sum',       # Trade Imbalance
        'shares_normalized': 'sum',   # Total Volume
        'side': 'count',              # Trade Count
    })

    # Flatten column names
    ti_data.columns = ['trade_imbalance', 'total_volume', 'trade_count']
    ti_data = ti_data.reset_index()

    return ti_data


def get_mid_price_changes(ofi_df, time_window_minutes):
    """
    Get mid-price changes from order book data per window.

    Args:
        ofi_df: OFI DataFrame with timestamp, mid_price
        time_window_minutes: Window size in minutes

    Returns:
        DataFrame with mid-price changes per window
    """
    ofi_df = ofi_df.copy()
    ofi_df['window'] = ofi_df['timestamp'].dt.floor(f'{time_window_minutes}min')

    # Get first and last mid-price per window
    price_data = ofi_df.groupby('window').agg({
        'mid_price': ['first', 'last', 'count'],
    })

    price_data.columns = ['mid_first', 'mid_last', 'snapshot_count']
    price_data = price_data.reset_index()

    # Calculate price change in ticks
    price_data['delta_mid_price'] = price_data['mid_last'] - price_data['mid_first']
    price_data['delta_mid_price_ticks'] = price_data['delta_mid_price'] / TICK_SIZE

    return price_data


# ============================================================================
# REGRESSION
# ============================================================================

def run_ti_regression(df):
    """
    Run Trade Imbalance regression: Delta_P = alpha + beta * TI + epsilon

    Args:
        df: DataFrame with trade_imbalance and delta_mid_price_ticks

    Returns:
        Dict with regression results or None if insufficient data
    """
    df_clean = df.dropna(subset=['trade_imbalance', 'delta_mid_price_ticks'])

    if len(df_clean) < 10:
        return None

    # Run regression
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
# 81-CONFIG ANALYSIS
# ============================================================================

def run_81_config_analysis(trades, ofi_df):
    """
    Run TI regression for all 81 configurations (9 windows x 9 outlier methods).

    Returns:
        DataFrame with all results
    """
    print("\n Running 81-configuration TI analysis...")
    print(f"   Time Windows: {TIME_WINDOWS}")
    print(f"   Outlier Methods: {len(OUTLIER_METHODS)}")
    print(f"   Total configs: {len(TIME_WINDOWS) * len(OUTLIER_METHODS)}")

    results = []

    for tw in TIME_WINDOWS:
        # Calculate TI for this time window
        ti_data = calculate_ti_per_window(trades, tw)

        # Get mid-price changes from order book
        price_data = get_mid_price_changes(ofi_df, tw)

        # Merge on window
        merged = pd.merge(ti_data, price_data, on='window', how='inner')

        if len(merged) < 10:
            print(f"   {tw}min: Only {len(merged)} windows, skipping")
            continue

        for method_idx, method_name in enumerate(OUTLIER_METHODS):
            # Apply outlier filtering
            filtered = apply_outlier_method(merged, method_idx, 'trade_imbalance')

            if len(filtered) < 10:
                continue

            # Run regression
            reg_result = run_ti_regression(filtered)

            if reg_result is not None:
                results.append({
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
    """Print summary of 81-config results"""

    print("\n" + "=" * 80)
    print("TI ANALYSIS SUMMARY (81 Configurations)")
    print("=" * 80)

    # Best config
    best = results_df.loc[results_df['r_squared'].idxmax()]
    print(f"\n BEST CONFIG:")
    print(f"   Window:  {best['time_window']} min")
    print(f"   Method:  {best['outlier_method']}")
    print(f"   R^2:     {best['r_squared']:.4f} ({best['r_squared']*100:.2f}%)")
    print(f"   Beta:    {best['beta']:.6f}")
    print(f"   p-value: {best['p_value']:.2e}")

    # Stats by time window
    print("\n R^2 by Time Window:")
    tw_stats = results_df.groupby('time_window')['r_squared'].agg(['mean', 'max'])
    for tw in TIME_WINDOWS:
        if tw in tw_stats.index:
            print(f"   {tw:3d} min: avg={tw_stats.loc[tw, 'mean']:.4f}, max={tw_stats.loc[tw, 'max']:.4f}")

    # Stats by outlier method
    print("\n R^2 by Outlier Method:")
    method_stats = results_df.groupby('outlier_method')['r_squared'].agg(['mean', 'max'])
    for method in OUTLIER_METHODS:
        if method in method_stats.index:
            print(f"   {method:15s}: avg={method_stats.loc[method, 'mean']:.4f}, max={method_stats.loc[method, 'max']:.4f}")

    # Overall stats
    print(f"\n Overall Statistics:")
    print(f"   Mean R^2:   {results_df['r_squared'].mean():.4f} ({results_df['r_squared'].mean()*100:.2f}%)")
    print(f"   Median R^2: {results_df['r_squared'].median():.4f}")
    print(f"   Std R^2:    {results_df['r_squared'].std():.4f}")


def print_heatmap(results_df):
    """Print ASCII heatmap of R^2 values"""

    print("\n" + "=" * 80)
    print("R^2 HEATMAP (Time Window x Outlier Method)")
    print("=" * 80)

    # Create pivot table
    pivot = results_df.pivot(index='time_window', columns='outlier_method', values='r_squared')

    # Reorder columns to match OUTLIER_METHODS order
    pivot = pivot.reindex(columns=OUTLIER_METHODS)

    # Print header
    print("\n" + " " * 8, end="")
    for method in OUTLIER_METHODS:
        print(f"{method[:6]:>8}", end="")
    print()

    # Print rows
    for tw in TIME_WINDOWS:
        if tw in pivot.index:
            print(f"{tw:4d}min:", end="")
            for method in OUTLIER_METHODS:
                if method in pivot.columns and pd.notna(pivot.loc[tw, method]):
                    val = pivot.loc[tw, method] * 100
                    print(f"{val:7.2f}%", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            print()


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("TRADE IMBALANCE ANALYSIS - 81 CONFIGURATIONS")
    print("(Same framework as OFI analysis for direct comparison)")
    print("=" * 80)

    # Load data
    trades = load_dome_trades()
    ofi_df = load_ofi_data()

    # Run 81-config analysis
    results_df = run_81_config_analysis(trades, ofi_df)

    if len(results_df) == 0:
        print("\n No results computed!")
        return None

    # Print summary
    print_summary(results_df)
    print_heatmap(results_df)

    # Save results
    print(f"\n Saving results to {OUTPUT_81_FILE}...")
    results_df.to_csv(OUTPUT_81_FILE, index=False)
    print(f"   Saved {len(results_df)} configurations")

    print("\n" + "=" * 80)
    print(" TI ANALYSIS COMPLETE")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    main()
