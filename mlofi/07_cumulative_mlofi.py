"""
07_cumulative_mlofi.py
======================
Cumulative MLOFI Calculation - Alternative Approach

DIFFERENCE FROM STANDARD MLOFI:
-------------------------------
Standard MLOFI:
    cumulative_ofi_L2 = ofi_l1 + ofi_l2  (sum OFI at each level)

Cumulative MLOFI (this script):
    cumulative_ofi_L2 = OFI(cumulative_bid_size_L2, cumulative_ask_size_L2)
    where cumulative_bid_size_L2 = bid_size_l1 + bid_size_l2

This approach first aggregates the liquidity through level N, then calculates
OFI on the aggregated quantities. This captures the total order flow imbalance
across the top N levels as a single measure.

CONFIGURATION:
--------------
- Levels: L2, L5, L10
- Exponent: a = 0.3 (size transformation)
- Time Windows: 45, 60, 90 minutes (best 3)
- Outlier Methods: Raw, Z-Score, Winsorized (best 3)

OUTPUT:
-------
- data/mlofi/cumulative_mlofi_results.csv

USAGE:
------
    python mlofi/07_cumulative_mlofi.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MULTILEVEL_PROCESSED_FILE,
    MLOFI_OUTPUT_DIR,
    TICK_SIZE
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Levels to analyze (cumulative through this level)
LEVELS = [2, 5, 10]

# Size exponent (q^a transformation)
EXPONENT = 0.3

# Time windows (minutes)
TIME_WINDOWS = [45, 60, 90]

# Outlier methods
OUTLIER_METHODS = ['Raw', 'Z-Score', 'Winsorized']

# Output file
OUTPUT_FILE = MLOFI_OUTPUT_DIR / "cumulative_mlofi_results.csv"


# ============================================================================
# OUTLIER FILTERING FUNCTIONS
# ============================================================================

def filter_outliers_zscore(df, column='ofi', threshold=3):
    """Remove outliers using Z-score method"""
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return df
    z_scores = (df[column] - mean) / std
    return df[z_scores.abs() <= threshold]


def winsorize_data(df, column='ofi', limits=(0.01, 0.01)):
    """Cap extreme values at percentile limits"""
    df = df.copy()
    lower = df[column].quantile(limits[0])
    upper = df[column].quantile(1 - limits[1])
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def apply_outlier_method(df, method, ofi_column):
    """Apply outlier filtering method"""
    if method == 'Raw':
        return df.copy()
    elif method == 'Z-Score':
        return filter_outliers_zscore(df, ofi_column, threshold=3)
    elif method == 'Winsorized':
        return winsorize_data(df, ofi_column)
    else:
        return df.copy()


# ============================================================================
# CUMULATIVE MLOFI CALCULATION
# ============================================================================

def calculate_cumulative_sizes(df, max_level):
    """
    Calculate cumulative bid/ask sizes through level N.

    Args:
        df: DataFrame with bid_size_l1, bid_size_l2, etc.
        max_level: Maximum level to include (e.g., 2, 5, 10)

    Returns:
        DataFrame with cumulative_bid_size and cumulative_ask_size columns
    """
    df = df.copy()

    # Sum bid sizes through level N
    bid_cols = [f'bid_size_l{l}' for l in range(1, max_level + 1)]
    available_bid_cols = [c for c in bid_cols if c in df.columns]
    df['cumulative_bid_size'] = df[available_bid_cols].fillna(0).sum(axis=1)

    # Sum ask sizes through level N
    ask_cols = [f'ask_size_l{l}' for l in range(1, max_level + 1)]
    available_ask_cols = [c for c in ask_cols if c in df.columns]
    df['cumulative_ask_size'] = df[available_ask_cols].fillna(0).sum(axis=1)

    return df


def calculate_cumulative_ofi(df, max_level, exponent=0.3):
    """
    Calculate OFI using cumulative sizes through level N.

    This is different from summing OFI at each level.
    Here we first sum the sizes, then apply OFI formula.

    Args:
        df: DataFrame with multi-level data
        max_level: Level to cumulate through (2, 5, or 10)
        exponent: Size transformation exponent (default 0.3)

    Returns:
        Series with cumulative OFI values
    """
    # Get cumulative sizes
    df = calculate_cumulative_sizes(df, max_level)

    # Apply size transformation: q^a
    df['transformed_bid'] = np.power(df['cumulative_bid_size'] + 1e-10, exponent)
    df['transformed_ask'] = np.power(df['cumulative_ask_size'] + 1e-10, exponent)

    # Use best bid/ask prices from level 1 for indicator functions
    bid_price = df['bid_price_l1']
    ask_price = df['ask_price_l1']

    # Previous values
    prev_bid_price = bid_price.shift(1)
    prev_ask_price = ask_price.shift(1)
    prev_transformed_bid = df['transformed_bid'].shift(1)
    prev_transformed_ask = df['transformed_ask'].shift(1)

    # Indicator functions (using inclusive inequalities as per Cont et al.)
    bid_up = (bid_price >= prev_bid_price).astype(float).fillna(0)
    bid_down = (bid_price <= prev_bid_price).astype(float).fillna(0)
    ask_up = (ask_price >= prev_ask_price).astype(float).fillna(0)
    ask_down = (ask_price <= prev_ask_price).astype(float).fillna(0)

    # OFI formula using cumulative transformed sizes
    ofi = (
        bid_up * df['transformed_bid'] -
        bid_down * prev_transformed_bid.fillna(0) -
        ask_down * df['transformed_ask'] +
        ask_up * prev_transformed_ask.fillna(0)
    )

    return ofi


def aggregate_to_time_window(df, time_window_minutes):
    """
    Aggregate data to specified time window.

    Args:
        df: DataFrame with timestamp and OFI columns
        time_window_minutes: Window size in minutes

    Returns:
        Aggregated DataFrame
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

    # Create time bins
    df['time_bin'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')

    # Get OFI columns to aggregate
    ofi_cols = [c for c in df.columns if c.startswith('cumulative_ofi_')]

    # Aggregation rules
    agg_dict = {
        'mid_price': ['first', 'last'],
    }
    for col in ofi_cols:
        agg_dict[col] = 'sum'

    # Aggregate
    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten column names
    new_cols = ['timestamp']
    for col in aggregated.columns[1:]:
        if isinstance(col, tuple):
            if col[0] == 'mid_price':
                new_cols.append(f'mid_price_{col[1]}')
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Calculate price change
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    return aggregated


def run_regression(df, ofi_column, dep_var='delta_mid_price_ticks'):
    """
    Run linear regression: ΔP = α + β × OFI + ε

    Returns:
        dict with regression results
    """
    # Remove NaN
    valid = df[[ofi_column, dep_var]].dropna()

    if len(valid) < 10:
        return None

    x = valid[ofi_column].values
    y = valid[dep_var].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        'r2': r_value ** 2,
        'beta': slope,
        'p_value': p_value,
        'n_obs': len(valid),
        'std_err': std_err
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "CUMULATIVE MLOFI ANALYSIS" + " " * 37 + "║")
    print("║" + " " * 10 + "Aggregate Sizes First, Then Calculate OFI" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not MULTILEVEL_PROCESSED_FILE.exists():
        print(f"\n❌ Multi-level processed file not found: {MULTILEVEL_PROCESSED_FILE}")
        print("   Please run: python mlofi/01_process_multilevel.py first")
        sys.exit(1)

    # Create output directory
    MLOFI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n📂 Loading multi-level data...")
    df = pd.read_csv(MULTILEVEL_PROCESSED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   Loaded {len(df)} snapshots")

    # Calculate cumulative OFI for each level configuration
    print(f"\n" + "-" * 80)
    print(f"CALCULATING CUMULATIVE OFI (a = {EXPONENT})")
    print("-" * 80)

    for level in LEVELS:
        col_name = f'cumulative_ofi_L{level}'
        df[col_name] = calculate_cumulative_ofi(df, level, EXPONENT)
        non_zero = (df[col_name].abs() > 0).sum()
        print(f"   L{level}: {non_zero} non-zero values, mean = {df[col_name].mean():.4f}")

    # Remove first row (no lagged values)
    df = df.iloc[1:].reset_index(drop=True)

    # Run analysis for all configurations
    print(f"\n" + "-" * 80)
    print("RUNNING REGRESSION ANALYSIS")
    print(f"  Levels: {LEVELS}")
    print(f"  Time Windows: {TIME_WINDOWS}")
    print(f"  Outlier Methods: {OUTLIER_METHODS}")
    print("-" * 80)

    results = []
    total_configs = len(LEVELS) * len(TIME_WINDOWS) * len(OUTLIER_METHODS)
    config_count = 0

    for level in LEVELS:
        ofi_col = f'cumulative_ofi_L{level}'

        for tw in TIME_WINDOWS:
            # Aggregate to time window
            agg_df = aggregate_to_time_window(df, tw)

            for method in OUTLIER_METHODS:
                config_count += 1

                # Apply outlier filtering
                filtered_df = apply_outlier_method(agg_df, method, ofi_col)

                # Run regression
                reg_result = run_regression(filtered_df, ofi_col)

                if reg_result:
                    results.append({
                        'level': f'L{level}',
                        'time_window': tw,
                        'outlier_method': method,
                        'exponent': EXPONENT,
                        'r2': reg_result['r2'],
                        'beta': reg_result['beta'],
                        'p_value': reg_result['p_value'],
                        'n_obs': reg_result['n_obs'],
                        'std_err': reg_result['std_err']
                    })

                    if config_count <= 9 or config_count == total_configs:
                        print(f"   [{config_count}/{total_configs}] L{level} | {tw}min | {method}: R² = {reg_result['r2']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2', ascending=False).reset_index(drop=True)

    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved results to: {OUTPUT_FILE}")

    # Print summary
    print(f"\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Total configurations tested: {len(results_df)}")
    print(f"📊 Best R²: {results_df['r2'].max():.4f}")
    print(f"📊 Mean R²: {results_df['r2'].mean():.4f}")

    print(f"\n📊 Top 10 Configurations:")
    print("-" * 80)
    for i, row in results_df.head(10).iterrows():
        print(f"   {i+1}. {row['level']} | {row['time_window']}min | {row['outlier_method']}: R² = {row['r2']:.4f}")

    # Summary by level
    print(f"\n📊 Best R² by Level:")
    for level in LEVELS:
        level_data = results_df[results_df['level'] == f'L{level}']
        if len(level_data) > 0:
            best = level_data.iloc[0]
            print(f"   L{level}: R² = {best['r2']:.4f} ({best['time_window']}min, {best['outlier_method']})")

    # Summary by time window
    print(f"\n📊 Best R² by Time Window:")
    for tw in TIME_WINDOWS:
        tw_data = results_df[results_df['time_window'] == tw]
        if len(tw_data) > 0:
            best = tw_data.iloc[0]
            print(f"   {tw}min: R² = {best['r2']:.4f} ({best['level']}, {best['outlier_method']})")

    print(f"\n" + "=" * 80)
    print("✅ CUMULATIVE MLOFI ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n💡 Results saved to: {OUTPUT_FILE}")
    print(f"💡 Add to dashboard: streamlit run dashboard/dashboard_simple.py")
    print("\n")

    return results_df


if __name__ == "__main__":
    main()
