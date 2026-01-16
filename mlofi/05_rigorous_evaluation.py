"""
05_rigorous_evaluation.py
=========================
OFI Evaluation following Cont, Kukanov & Stoikov (2011) methodology

METHODOLOGY:
-----------
1. Calculate OFI and price change in the SAME time window (contemporaneous)
2. Run regression on FULL data (no train/test split, like the paper)
3. Report in-sample R²

TESTS:
------
1. q^a SIZE WEIGHTING - Test different exponents on order size
2. TIME WINDOWS - Test different aggregation windows
3. LEVEL COMPARISON - L1 only vs Cumulative MLOFI

USAGE:
------
    python mlofi/05_rigorous_evaluation.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MULTILEVEL_PROCESSED_FILE,
    MLOFI_OUTPUT_DIR,
    TICK_SIZE,
    TIME_WINDOWS,
)

# Output file
RESULTS_FILE = MLOFI_OUTPUT_DIR / "evaluation_results.csv"

# q^a exponents to test (non-linear size weighting)
SIZE_EXPONENTS = [0.3, 0.5, 0.7, 1.0]


def calculate_ofi_with_exponent(df, level, exponent=1.0):
    """
    Calculate OFI for a level with q^a size weighting.

    Following Cont et al. formula:
    e_n = I{P^B_n >= P^B_{n-1}} * q^B_n
        - I{P^B_n <= P^B_{n-1}} * q^B_{n-1}
        - I{P^A_n <= P^A_{n-1}} * q^A_n
        + I{P^A_n >= P^A_{n-1}} * q^A_{n-1}

    With q^a transformation where a is the exponent.
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


def aggregate_by_time_window(df, time_window_minutes, feature_cols):
    """Aggregate data by time window."""
    df = df.copy()
    df['time_bin'] = df['timestamp'].dt.floor(f'{time_window_minutes}T')

    agg_dict = {'mid_price': ['first', 'last']}
    for col in feature_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'

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
    })

    for col in feature_cols:
        if f'{col}_sum' in aggregated.columns:
            aggregated = aggregated.rename(columns={f'{col}_sum': col})

    # Price change within the SAME window (contemporaneous - like the paper)
    aggregated['delta_mid_price'] = aggregated['mid_price_end'] - aggregated['mid_price_start']
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    # Remove first row (no prior data for OFI calculation)
    aggregated = aggregated.iloc[1:].reset_index(drop=True)

    return aggregated


def full_data_evaluation(df, feature_cols):
    """
    Evaluate on full data like the original paper.
    No train/test split - simple OLS regression.
    """
    X_cols = [c for c in feature_cols if c in df.columns]
    if len(X_cols) == 0:
        return None

    X = df[X_cols].fillna(0).values
    y = df['delta_mid_price_ticks'].fillna(0).values

    # Remove NaN
    valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[valid_mask], y[valid_mask]

    if len(y) < 30:
        return None

    # Simple OLS regression (like the paper)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return {
        'r2': r2,
        'rmse': rmse,
        'n_obs': len(y),
        'coef': model.coef_[0] if len(model.coef_) == 1 else model.coef_,
    }


def main():
    print("\n")
    print("=" * 70)
    print("  OFI EVALUATION - Following Cont, Kukanov & Stoikov (2011)")
    print("=" * 70)
    print("\nMethodology:")
    print("  - OFI and price change in SAME time window (contemporaneous)")
    print("  - Regression on FULL data (no train/test split)")
    print("  - In-sample R² reported")

    # Load multi-level data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(MULTILEVEL_PROCESSED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Loaded {len(df):,} snapshots")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Determine max levels
    level_cols = [c for c in df.columns if c.startswith('bid_price_l')]
    max_level = min(25, max([int(c.replace('bid_price_l', '')) for c in level_cols]))
    print(f"  Using levels 1 to {max_level}")

    all_results = []

    # =========================================================================
    # TEST 1: q^a Exponent Test
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: q^a SIZE WEIGHTING")
    print("=" * 70)
    print("\n  a < 1: reduces impact of large orders")
    print("  a = 1: raw size (original paper)")
    print("  a > 1: increases impact of large orders")

    time_window = 10  # 10-minute window

    print(f"\n  {'Exponent':<12} {'R²':<12} {'RMSE':<12} {'N obs':<10}")
    print("  " + "-" * 50)

    for exponent in SIZE_EXPONENTS:
        # Calculate OFI with this exponent
        df_exp = df.copy()
        for level in range(1, max_level + 1):
            df_exp[f'ofi_l{level}'] = calculate_ofi_with_exponent(df_exp, level, exponent)

        # Cumulative OFI through all levels
        ofi_cols = [f'ofi_l{l}' for l in range(1, max_level + 1)]
        df_exp['ofi_cumulative'] = df_exp[ofi_cols].sum(axis=1)

        # Aggregate and evaluate
        df_agg = aggregate_by_time_window(df_exp, time_window, ['ofi_cumulative'])
        result = full_data_evaluation(df_agg, ['ofi_cumulative'])

        if result:
            print(f"  a={exponent:<10} {result['r2']:<12.4f} {result['rmse']:<12.4f} {result['n_obs']:<10}")
            all_results.append({
                'test_type': 'q^a_exponent',
                'config': f'a={exponent}',
                'time_window': time_window,
                'r2': result['r2'],
                'rmse': result['rmse'],
                'n_obs': result['n_obs'],
            })

    # =========================================================================
    # TEST 2: Time Window Comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 2: TIME WINDOW COMPARISON")
    print("=" * 70)

    # Use raw OFI (a=1.0)
    df_tw = df.copy()
    for level in range(1, max_level + 1):
        df_tw[f'ofi_l{level}'] = calculate_ofi_with_exponent(df_tw, level, exponent=1.0)

    ofi_cols = [f'ofi_l{l}' for l in range(1, max_level + 1)]
    df_tw['ofi_cumulative'] = df_tw[ofi_cols].sum(axis=1)
    df_tw['ofi_l1'] = df_tw['ofi_l1']  # Level 1 only

    print(f"\n  Cumulative OFI (L1-L{max_level}):")
    print(f"\n  {'Window':<12} {'R²':<12} {'RMSE':<12} {'N obs':<10}")
    print("  " + "-" * 50)

    for time_window in TIME_WINDOWS:
        df_agg = aggregate_by_time_window(df_tw, time_window, ['ofi_cumulative'])
        result = full_data_evaluation(df_agg, ['ofi_cumulative'])

        if result:
            print(f"  {time_window} min{'':<6} {result['r2']:<12.4f} {result['rmse']:<12.4f} {result['n_obs']:<10}")
            all_results.append({
                'test_type': 'time_window',
                'config': f'Cumulative_L{max_level}',
                'time_window': time_window,
                'r2': result['r2'],
                'rmse': result['rmse'],
                'n_obs': result['n_obs'],
            })

    # =========================================================================
    # TEST 3: Level Comparison (L1 vs Cumulative)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 3: LEVEL COMPARISON (L1 vs Cumulative)")
    print("=" * 70)

    print(f"\n  {'Config':<20} {'Window':<10} {'R²':<12} {'RMSE':<12}")
    print("  " + "-" * 55)

    for time_window in [1, 5, 10, 30, 60]:
        # Level 1 only
        df_agg = aggregate_by_time_window(df_tw, time_window, ['ofi_l1'])
        result_l1 = full_data_evaluation(df_agg, ['ofi_l1'])

        # Cumulative
        df_agg = aggregate_by_time_window(df_tw, time_window, ['ofi_cumulative'])
        result_cum = full_data_evaluation(df_agg, ['ofi_cumulative'])

        if result_l1:
            print(f"  {'L1 Only':<20} {time_window} min{'':<4} {result_l1['r2']:<12.4f} {result_l1['rmse']:<12.4f}")
            all_results.append({
                'test_type': 'level_comparison',
                'config': 'L1_only',
                'time_window': time_window,
                'r2': result_l1['r2'],
                'rmse': result_l1['rmse'],
                'n_obs': result_l1['n_obs'],
            })

        if result_cum:
            print(f"  {'Cumulative L1-L25':<20} {time_window} min{'':<4} {result_cum['r2']:<12.4f} {result_cum['rmse']:<12.4f}")
            all_results.append({
                'test_type': 'level_comparison',
                'config': f'Cumulative_L{max_level}',
                'time_window': time_window,
                'r2': result_cum['r2'],
                'rmse': result_cum['rmse'],
                'n_obs': result_cum['n_obs'],
            })

        if result_l1 and result_cum:
            improvement = (result_cum['r2'] - result_l1['r2']) / result_l1['r2'] * 100
            print(f"  {'  -> Improvement':<20} {'':<10} {improvement:>+10.1f}%")
        print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df_results = pd.DataFrame(all_results)

    # Best q^a
    qa_results = df_results[df_results['test_type'] == 'q^a_exponent']
    if len(qa_results) > 0:
        best_qa = qa_results.loc[qa_results['r2'].idxmax()]
        print(f"\n  Best q^a exponent: {best_qa['config']} (R² = {best_qa['r2']:.4f})")

    # Best time window
    tw_results = df_results[(df_results['test_type'] == 'time_window')]
    if len(tw_results) > 0:
        best_tw = tw_results.loc[tw_results['r2'].idxmax()]
        print(f"  Best time window: {int(best_tw['time_window'])} min (R² = {best_tw['r2']:.4f})")

    # L1 vs Cumulative at 10min
    level_10 = df_results[(df_results['test_type'] == 'level_comparison') &
                          (df_results['time_window'] == 10)]
    if len(level_10) >= 2:
        l1_r2 = level_10[level_10['config'] == 'L1_only']['r2'].values[0]
        cum_r2 = level_10[level_10['config'].str.startswith('Cumulative')]['r2'].values[0]
        print(f"\n  At 10-min window:")
        print(f"    L1 only:    R² = {l1_r2:.4f}")
        print(f"    Cumulative: R² = {cum_r2:.4f}")
        print(f"    Improvement: {(cum_r2-l1_r2)/l1_r2*100:+.1f}%")

    # Save results
    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results saved to: {RESULTS_FILE}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70 + "\n")

    return df_results


if __name__ == "__main__":
    main()
