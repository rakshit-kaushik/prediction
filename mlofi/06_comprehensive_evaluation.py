"""
06_comprehensive_evaluation.py
==============================
Comprehensive MLOFI evaluation testing 1,944 configurations:
- 6 Level Configs: L1, L5, L10, L25, L50pct, ALL
- 9 Time Windows: 1, 5, 10, 15, 20, 30, 45, 60, 90 min
- 9 Outlier Methods: Raw, IQR, Percentile, Z-Score, etc.
- 4 q^a Exponents: 0.3, 0.5, 0.7, 1.0

Output:
- Full CSV with all 1,944 results
- Console shows Top 20 and Worst 10

USAGE:
------
    python mlofi/06_comprehensive_evaluation.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MULTILEVEL_PROCESSED_FILE,
    MLOFI_OUTPUT_DIR,
    TICK_SIZE,
)

# Output file
RESULTS_FILE = MLOFI_OUTPUT_DIR / "mlofi_all_configs.csv"

# Configuration dimensions
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]
LEVEL_CONFIGS = ['L2', 'L3', 'L5', 'L10', 'L20', 'L50pct', 'ALL']
SIZE_EXPONENTS = [0.3, 0.5, 0.7, 1.0]

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


# =============================================================================
# OUTLIER FILTERING FUNCTIONS (from L1 code)
# =============================================================================

def filter_outliers_iqr(df, column='ofi', multiplier=1.5):
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


def filter_outliers_percentile(df, column='ofi', lower_pct=0.01, upper_pct=0.99):
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]


def filter_outliers_zscore(df, column='ofi', threshold=3):
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return df
    z_scores = (df[column] - mean) / std
    return df[z_scores.abs() <= threshold]


def winsorize_data(df, column='ofi', limits=(0.01, 0.01)):
    df = df.copy()
    lower = df[column].quantile(limits[0])
    upper = df[column].quantile(1 - limits[1])
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def filter_absolute_threshold(df, column='ofi', lower=-200000, upper=200000):
    df = df.copy()
    return df[(df[column] >= lower) & (df[column] <= upper)]


def filter_outliers_mad(df, column='ofi', threshold=3):
    df = df.copy()
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    if mad == 0:
        return df
    modified_z = 0.6745 * (df[column] - median) / mad
    return df[np.abs(modified_z) <= threshold]


def filter_outliers_percentile_aggressive(df, column='ofi', lower_pct=0.05, upper_pct=0.95):
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]


def apply_outlier_method(df, method_idx, column='ofi'):
    """Apply outlier method by index (0-8)"""
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
        return filter_outliers_percentile_aggressive(df, column)
    return df.copy()


# =============================================================================
# OFI CALCULATION
# =============================================================================

def calculate_ofi_with_exponent(df, level, exponent=1.0):
    """
    Calculate OFI for a level with q^a size weighting.

    Following Cont et al. formula:
    e_n = I{P^B_n >= P^B_{n-1}} * q^B_n
        - I{P^B_n <= P^B_{n-1}} * q^B_{n-1}
        - I{P^A_n <= P^A_{n-1}} * q^A_n
        + I{P^A_n >= P^A_{n-1}} * q^A_{n-1}
    """
    bid_price_col = f'bid_price_l{level}'
    bid_size_col = f'bid_size_l{level}'
    ask_price_col = f'ask_price_l{level}'
    ask_size_col = f'ask_size_l{level}'

    if bid_price_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Current values with q^a transformation
    bid_price = df[bid_price_col]
    bid_size = df[bid_size_col].fillna(0).clip(lower=0) ** exponent
    ask_price = df[ask_price_col]
    ask_size = df[ask_size_col].fillna(0).clip(lower=0) ** exponent

    # Previous values
    prev_bid_price = bid_price.shift(1)
    prev_bid_size = df[bid_size_col].shift(1).fillna(0).clip(lower=0) ** exponent
    prev_ask_price = ask_price.shift(1)
    prev_ask_size = df[ask_size_col].shift(1).fillna(0).clip(lower=0) ** exponent

    # Indicator functions (following paper exactly)
    bid_up = (bid_price >= prev_bid_price).astype(float).fillna(0)
    bid_down = (bid_price <= prev_bid_price).astype(float).fillna(0)
    ask_up = (ask_price >= prev_ask_price).astype(float).fillna(0)
    ask_down = (ask_price <= prev_ask_price).astype(float).fillna(0)

    # OFI formula from paper
    ofi = (bid_up * bid_size
           - bid_down * prev_bid_size
           - ask_down * ask_size
           + ask_up * prev_ask_size)

    return ofi


def calculate_cumulative_ofi(df, max_level, exponent=1.0):
    """Calculate OFI at each level and return cumulative sums for different configs."""
    # Calculate OFI at each level
    ofi_by_level = {}
    for level in range(1, max_level + 1):
        ofi_by_level[level] = calculate_ofi_with_exponent(df, level, exponent)

    # Create cumulative OFI for different configs
    cumulative = {}

    # L2 (levels 1-2)
    cumulative['L2'] = sum(ofi_by_level[i] for i in range(1, min(3, max_level + 1)))

    # L3 (levels 1-3)
    cumulative['L3'] = sum(ofi_by_level[i] for i in range(1, min(4, max_level + 1)))

    # L5 (levels 1-5)
    cumulative['L5'] = sum(ofi_by_level[i] for i in range(1, min(6, max_level + 1)))

    # L10 (levels 1-10)
    cumulative['L10'] = sum(ofi_by_level[i] for i in range(1, min(11, max_level + 1)))

    # L20 (levels 1-20)
    cumulative['L20'] = sum(ofi_by_level[i] for i in range(1, min(21, max_level + 1)))

    # L50pct (50% of available levels)
    l50pct = max(1, max_level // 2)
    cumulative['L50pct'] = sum(ofi_by_level[i] for i in range(1, l50pct + 1))

    # ALL
    cumulative['ALL'] = sum(ofi_by_level[i] for i in range(1, max_level + 1))

    return cumulative


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_by_time_window(df, time_window_minutes, ofi_col='ofi'):
    """Aggregate data by time window."""
    df = df.copy()
    df['time_bin'] = df['timestamp'].dt.floor(f'{time_window_minutes}T')

    agg_dict = {
        'mid_price': ['first', 'last'],
        ofi_col: 'sum'
    }

    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten columns
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1]:
                new_cols.append(f'{col[0]}_{col[1]}')
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Rename
    aggregated = aggregated.rename(columns={
        'time_bin': 'timestamp',
        'mid_price_first': 'mid_price_start',
        'mid_price_last': 'mid_price_end',
        f'{ofi_col}_sum': 'ofi'
    })

    # Price change within the SAME window (contemporaneous - like the paper)
    aggregated['delta_mid_price'] = aggregated['mid_price_end'] - aggregated['mid_price_start']
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    # Remove first row
    aggregated = aggregated.iloc[1:].reset_index(drop=True)

    return aggregated


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_config(df_agg, min_obs=30):
    """Run regression and return R² for a single configuration."""
    df_clean = df_agg.dropna(subset=['ofi', 'delta_mid_price_ticks'])

    if len(df_clean) < min_obs:
        return None

    # Simple OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_clean['ofi'], df_clean['delta_mid_price_ticks']
    )

    return {
        'r2': r_value ** 2,
        'beta': slope,
        'p_value': p_value,
        'n_obs': len(df_clean),
        'std_err': std_err
    }


def split_into_phases(df):
    """Split data into 3 equal phases: Early, Middle, Late."""
    n = len(df)
    phase_size = n // 3
    return {
        'Early': df.iloc[:phase_size].copy(),
        'Middle': df.iloc[phase_size:2*phase_size].copy(),
        'Late': df.iloc[2*phase_size:].copy()
    }


def evaluate_config_by_phase(df_agg, min_obs=10):
    """Run regression for each phase and return R² breakdown."""
    df_clean = df_agg.dropna(subset=['ofi', 'delta_mid_price_ticks'])

    if len(df_clean) < 30:
        return None

    # Overall
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_clean['ofi'], df_clean['delta_mid_price_ticks']
    )

    result = {
        'overall_r2': r_value ** 2,
        'overall_n': len(df_clean),
        'overall_beta': slope,
    }

    # By phase
    phases = split_into_phases(df_clean)

    for phase_name, phase_df in phases.items():
        if len(phase_df) >= min_obs:
            p_slope, _, p_r, p_pval, _ = stats.linregress(
                phase_df['ofi'], phase_df['delta_mid_price_ticks']
            )
            result[f'{phase_name.lower()}_r2'] = p_r ** 2
            result[f'{phase_name.lower()}_n'] = len(phase_df)
            result[f'{phase_name.lower()}_beta'] = p_slope
        else:
            result[f'{phase_name.lower()}_r2'] = None
            result[f'{phase_name.lower()}_n'] = len(phase_df)
            result[f'{phase_name.lower()}_beta'] = None

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n")
    print("=" * 70)
    print("  MLOFI COMPREHENSIVE EVALUATION")
    print("  Testing 1,944 configurations")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {MULTILEVEL_PROCESSED_FILE}...")
    df = pd.read_csv(MULTILEVEL_PROCESSED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Determine max levels
    level_cols = [c for c in df.columns if c.startswith('bid_price_l')]
    max_level = max([int(c.replace('bid_price_l', '')) for c in level_cols])

    print(f"  Loaded {len(df):,} snapshots")
    print(f"  Max level available: {max_level}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calculate total configs
    total_configs = len(LEVEL_CONFIGS) * len(TIME_WINDOWS) * len(OUTLIER_METHODS) * len(SIZE_EXPONENTS)
    print(f"\n  Total configurations to test: {total_configs}")
    print(f"    - {len(LEVEL_CONFIGS)} level configs: {LEVEL_CONFIGS}")
    print(f"    - {len(TIME_WINDOWS)} time windows: {TIME_WINDOWS}")
    print(f"    - {len(OUTLIER_METHODS)} outlier methods")
    print(f"    - {len(SIZE_EXPONENTS)} q^a exponents: {SIZE_EXPONENTS}")

    # Run evaluation
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    all_results = []
    config_count = 0

    for exponent in SIZE_EXPONENTS:
        print(f"\n  q^a = {exponent}...")

        # Calculate cumulative OFI for all level configs with this exponent
        cumulative_ofi = calculate_cumulative_ofi(df, max_level, exponent)

        for level_config in LEVEL_CONFIGS:
            # Create dataframe with this level's OFI
            df_level = df[['timestamp', 'mid_price']].copy()
            df_level['ofi'] = cumulative_ofi[level_config]

            for time_window in TIME_WINDOWS:
                # Aggregate by time window
                df_agg = aggregate_by_time_window(df_level, time_window, 'ofi')

                for method_idx, method_name in enumerate(OUTLIER_METHODS):
                    config_count += 1

                    # Apply outlier filter
                    df_filtered = apply_outlier_method(df_agg, method_idx, 'ofi')

                    # Evaluate
                    result = evaluate_config(df_filtered)

                    if result:
                        all_results.append({
                            'level_config': level_config,
                            'time_window': time_window,
                            'outlier_method': method_name,
                            'q_exponent': exponent,
                            'r2': result['r2'],
                            'beta': result['beta'],
                            'p_value': result['p_value'],
                            'n_obs': result['n_obs'],
                            'std_err': result['std_err']
                        })

                    # Progress
                    if config_count % 200 == 0:
                        print(f"    Processed {config_count}/{total_configs} configs...")

    print(f"\n  Completed {config_count} configurations")
    print(f"  Valid results: {len(all_results)}")

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('r2', ascending=False).reset_index(drop=True)

    # Display Top 20
    print("\n" + "=" * 70)
    print("TOP 20 CONFIGURATIONS")
    print("=" * 70)

    print(f"\n{'Rank':<6} {'Level':<8} {'Window':<8} {'Outlier':<15} {'q^a':<6} {'R²':<10} {'N obs':<8}")
    print("-" * 70)

    for i, row in df_results.head(20).iterrows():
        print(f"{i+1:<6} {row['level_config']:<8} {row['time_window']:<8} {row['outlier_method']:<15} {row['q_exponent']:<6} {row['r2']:<10.4f} {row['n_obs']:<8}")

    # Display Worst 10
    print("\n" + "=" * 70)
    print("WORST 10 CONFIGURATIONS")
    print("=" * 70)

    print(f"\n{'Rank':<6} {'Level':<8} {'Window':<8} {'Outlier':<15} {'q^a':<6} {'R²':<10} {'N obs':<8}")
    print("-" * 70)

    worst_10 = df_results.tail(10).iloc[::-1]
    for i, (_, row) in enumerate(worst_10.iterrows()):
        rank = len(df_results) - 9 + i
        print(f"{rank:<6} {row['level_config']:<8} {row['time_window']:<8} {row['outlier_method']:<15} {row['q_exponent']:<6} {row['r2']:<10.4f} {row['n_obs']:<8}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\n  Overall R² statistics:")
    print(f"    Mean:   {df_results['r2'].mean():.4f}")
    print(f"    Median: {df_results['r2'].median():.4f}")
    print(f"    Max:    {df_results['r2'].max():.4f}")
    print(f"    Min:    {df_results['r2'].min():.4f}")

    # Best by dimension
    print(f"\n  Best R² by q^a exponent:")
    for exp in SIZE_EXPONENTS:
        exp_data = df_results[df_results['q_exponent'] == exp]
        if len(exp_data) > 0:
            print(f"    a={exp}: max R² = {exp_data['r2'].max():.4f}")

    print(f"\n  Best R² by level config:")
    for level in LEVEL_CONFIGS:
        level_data = df_results[df_results['level_config'] == level]
        if len(level_data) > 0:
            print(f"    {level}: max R² = {level_data['r2'].max():.4f}")

    print(f"\n  Best R² by time window:")
    for tw in TIME_WINDOWS:
        tw_data = df_results[df_results['time_window'] == tw]
        if len(tw_data) > 0:
            print(f"    {tw} min: max R² = {tw_data['r2'].max():.4f}")

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Saved all {len(df_results)} results to: {RESULTS_FILE}")

    # Best overall
    best = df_results.iloc[0]
    print(f"\n  BEST CONFIGURATION:")
    print(f"    Level:   {best['level_config']}")
    print(f"    Window:  {best['time_window']} min")
    print(f"    Outlier: {best['outlier_method']}")
    print(f"    q^a:     {best['q_exponent']}")
    print(f"    R²:      {best['r2']:.4f} ({best['r2']*100:.2f}%)")
    print(f"    N obs:   {best['n_obs']}")

    # =========================================================================
    # PHASE ANALYSIS FOR TOP 20
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE ANALYSIS (Early / Middle / Late) - TOP 20 CONFIGS")
    print("=" * 70)

    phase_results = []
    top_20 = df_results.head(20)

    for _, row in top_20.iterrows():
        level_config = row['level_config']
        time_window = row['time_window']
        method_idx = OUTLIER_METHODS.index(row['outlier_method'])
        exponent = row['q_exponent']

        # Recalculate for this config
        cumulative_ofi = calculate_cumulative_ofi(df, max_level, exponent)
        df_level = df[['timestamp', 'mid_price']].copy()
        df_level['ofi'] = cumulative_ofi[level_config]

        df_agg = aggregate_by_time_window(df_level, time_window, 'ofi')
        df_filtered = apply_outlier_method(df_agg, method_idx, 'ofi')

        phase_result = evaluate_config_by_phase(df_filtered)

        if phase_result:
            phase_results.append({
                'level_config': level_config,
                'time_window': time_window,
                'outlier_method': row['outlier_method'],
                'q_exponent': exponent,
                **phase_result
            })

    # Display phase results
    print(f"\n{'Level':<8} {'Window':<8} {'q^a':<6} {'Overall':<10} {'Early':<10} {'Middle':<10} {'Late':<10}")
    print("-" * 75)

    for pr in phase_results:
        overall = f"{pr['overall_r2']:.4f}" if pr['overall_r2'] else "N/A"
        early = f"{pr['early_r2']:.4f}" if pr.get('early_r2') else "N/A"
        middle = f"{pr['middle_r2']:.4f}" if pr.get('middle_r2') else "N/A"
        late = f"{pr['late_r2']:.4f}" if pr.get('late_r2') else "N/A"

        print(f"{pr['level_config']:<8} {pr['time_window']:<8} {pr['q_exponent']:<6} {overall:<10} {early:<10} {middle:<10} {late:<10}")

    # Summary by phase
    print("\n" + "-" * 75)
    print("PHASE SUMMARY (Average R² across top 20 configs)")
    print("-" * 75)

    overall_avg = np.mean([p['overall_r2'] for p in phase_results if p.get('overall_r2')])
    early_avg = np.mean([p['early_r2'] for p in phase_results if p.get('early_r2')])
    middle_avg = np.mean([p['middle_r2'] for p in phase_results if p.get('middle_r2')])
    late_avg = np.mean([p['late_r2'] for p in phase_results if p.get('late_r2')])

    print(f"\n  Overall:  {overall_avg:.4f} ({overall_avg*100:.2f}%)")
    print(f"  Early:    {early_avg:.4f} ({early_avg*100:.2f}%)")
    print(f"  Middle:   {middle_avg:.4f} ({middle_avg*100:.2f}%)")
    print(f"  Late:     {late_avg:.4f} ({late_avg*100:.2f}%)")

    # Which phase is strongest?
    phase_avgs = {'Early': early_avg, 'Middle': middle_avg, 'Late': late_avg}
    best_phase = max(phase_avgs, key=phase_avgs.get)
    worst_phase = min(phase_avgs, key=phase_avgs.get)

    print(f"\n  Best phase:  {best_phase} (R² = {phase_avgs[best_phase]:.4f})")
    print(f"  Worst phase: {worst_phase} (R² = {phase_avgs[worst_phase]:.4f})")

    # Save phase results
    phase_df = pd.DataFrame(phase_results)
    phase_file = MLOFI_OUTPUT_DIR / "mlofi_top20_phase_analysis.csv"
    phase_df.to_csv(phase_file, index=False)
    print(f"\n  Phase analysis saved to: {phase_file}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70 + "\n")

    return df_results


if __name__ == "__main__":
    main()
