"""
03b_regression_enhanced.py
==========================
Enhanced regression analysis comparing all MLOFI variants

VARIANTS TESTED:
---------------
1. Raw OFI (Level 1 only) - Baseline
2. Raw OFI (Multi-level with Ridge)
3. Cumulative OFI
4. Depth-Normalized OFI
5. Exponentially Weighted OFI
6. PCA-Compressed OFI

This allows direct comparison of the paper's enhancements.

USAGE:
------
    python mlofi/03b_regression_enhanced.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MLOFI_OUTPUT_DIR,
    TIME_WINDOWS,
    RIDGE_ALPHAS,
    CV_FOLDS,
    MIN_OBS_FOR_REGRESSION
)

# Enhanced file paths
MLOFI_ENHANCED_FILE = MLOFI_OUTPUT_DIR / "mlofi_enhanced.csv"
REGRESSION_ENHANCED_FILE = MLOFI_OUTPUT_DIR / "regression_enhanced_results.csv"


def aggregate_by_time_window(df, time_window_minutes, feature_cols):
    """Aggregate data by time window."""
    df = df.copy()
    df['time_bin'] = df['timestamp'].dt.floor(f'{time_window_minutes}T')

    # Aggregation rules
    agg_dict = {
        'mid_price': ['first', 'last'],
        'spread': 'mean',
    }

    # Sum OFI-related columns within window
    for col in feature_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'

    # Aggregate
    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten column names
    aggregated.columns = [
        f'{col[0]}_{col[1]}' if isinstance(col, tuple) and col[1] else col[0]
        for col in aggregated.columns
    ]

    # Rename
    aggregated = aggregated.rename(columns={
        'time_bin': 'timestamp',
        'mid_price_first': 'mid_price_start',
        'mid_price_last': 'mid_price_end',
        'spread_mean': 'spread',
    })

    # Rename feature columns (remove _sum suffix)
    for col in feature_cols:
        if f'{col}_sum' in aggregated.columns:
            aggregated = aggregated.rename(columns={f'{col}_sum': col})

    # Calculate price change
    aggregated['delta_mid_price'] = aggregated['mid_price_end'].diff()
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / 0.01

    # Remove first row
    aggregated = aggregated.iloc[1:].reset_index(drop=True)

    return aggregated


def run_regression(X, y, method='OLS'):
    """Run a single regression and return results."""
    if len(y) < MIN_OBS_FOR_REGRESSION:
        return None

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1) if X.ndim == 1 else X)

    try:
        if method == 'OLS':
            model = LinearRegression()
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            best_alpha = np.nan
        elif method == 'Ridge':
            model = RidgeCV(alphas=RIDGE_ALPHAS, cv=CV_FOLDS)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            best_alpha = model.alpha_
        elif method == 'Lasso':
            model = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1], cv=CV_FOLDS, max_iter=10000)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            best_alpha = model.alpha_
        else:
            return None

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        return {
            'r_squared': r2,
            'rmse': rmse,
            'best_alpha': best_alpha,
            'n_obs': len(y),
        }
    except Exception as e:
        print(f"      Error: {e}")
        return None


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 12 + "ENHANCED MLOFI REGRESSION ANALYSIS" + " " * 30 + "║")
    print("║" + " " * 8 + "Comparing: Raw vs Normalized vs Weighted vs PCA" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not MLOFI_ENHANCED_FILE.exists():
        print(f"\n❌ Enhanced MLOFI file not found: {MLOFI_ENHANCED_FILE}")
        print("   Please run: python mlofi/02b_calculate_mlofi_enhanced.py")
        sys.exit(1)

    # Load data
    print(f"\n📂 Loading enhanced MLOFI data...")
    df = pd.read_csv(MLOFI_ENHANCED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Define feature groups to test
    feature_groups = {}

    # 1. Raw Level 1 only (baseline)
    if 'ofi_l1' in df.columns:
        feature_groups['Raw_L1'] = ['ofi_l1']

    # 2. Raw Multi-level (L1-L10)
    raw_cols = [f'ofi_l{i}' for i in range(1, 11) if f'ofi_l{i}' in df.columns]
    if len(raw_cols) > 0:
        feature_groups['Raw_L10'] = raw_cols

    # 3. Cumulative OFI
    for level in [5, 10, 25, 50]:
        col = f'ofi_cumulative_l{level}'
        if col in df.columns:
            feature_groups[f'Cumulative_L{level}'] = [col]

    # 4. Depth-Normalized L1
    if 'ofi_norm_l1' in df.columns:
        feature_groups['Normalized_L1'] = ['ofi_norm_l1']

    # 5. Depth-Normalized Multi-level
    norm_cols = [f'ofi_norm_l{i}' for i in range(1, 11) if f'ofi_norm_l{i}' in df.columns]
    if len(norm_cols) > 0:
        feature_groups['Normalized_L10'] = norm_cols

    # 6. Exponentially Weighted
    if 'ofi_weighted' in df.columns:
        feature_groups['Exp_Weighted'] = ['ofi_weighted']
    if 'ofi_norm_weighted' in df.columns:
        feature_groups['Exp_Weighted_Norm'] = ['ofi_norm_weighted']

    # 7. PCA components
    pca_cols = [f'ofi_pca_{i}' for i in range(1, 6) if f'ofi_pca_{i}' in df.columns]
    if len(pca_cols) > 0:
        feature_groups['PCA_Raw'] = pca_cols

    pca_norm_cols = [f'ofi_norm_pca_{i}' for i in range(1, 6) if f'ofi_norm_pca_{i}' in df.columns]
    if len(pca_norm_cols) > 0:
        feature_groups['PCA_Normalized'] = pca_norm_cols

    print(f"\n📊 Feature groups to test: {len(feature_groups)}")
    for name, cols in feature_groups.items():
        print(f"   {name}: {len(cols)} features")

    # Run regressions
    print("\n" + "=" * 80)
    print("RUNNING REGRESSIONS")
    print("=" * 80)

    all_results = []
    total_configs = len(feature_groups) * len(TIME_WINDOWS)
    config_count = 0

    for group_name, feature_cols in feature_groups.items():
        print(f"\n📊 Testing: {group_name}")

        for time_window in TIME_WINDOWS:
            config_count += 1

            # Get all needed columns for aggregation
            all_cols = feature_cols.copy()

            # Aggregate
            df_agg = aggregate_by_time_window(df, time_window, all_cols)

            # Prepare features
            X_cols = [c for c in feature_cols if c in df_agg.columns]
            if len(X_cols) == 0:
                continue

            X = df_agg[X_cols].fillna(0).values
            y = df_agg['delta_mid_price_ticks'].fillna(0).values

            # Remove NaN
            valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
            X = X[valid_mask]
            y = y[valid_mask]

            if len(y) < MIN_OBS_FOR_REGRESSION:
                continue

            # Determine regression method
            if X.shape[1] == 1:
                # Single feature - use OLS
                result = run_regression(X, y, 'OLS')
                method = 'OLS'
            else:
                # Multiple features - use Ridge
                result = run_regression(X, y, 'Ridge')
                method = 'Ridge'

            if result:
                result['feature_group'] = group_name
                result['time_window'] = time_window
                result['method'] = method
                result['n_features'] = X.shape[1]
                all_results.append(result)

                if time_window == 10:  # Print 10-min results
                    print(f"      {time_window}min: R²={result['r_squared']:.4f}")

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Total regressions: {len(df_results)}")

    # Best R² by feature group
    print(f"\n📊 Best R² by feature group:")
    best_by_group = df_results.loc[df_results.groupby('feature_group')['r_squared'].idxmax()]
    best_by_group = best_by_group.sort_values('r_squared', ascending=False)

    for _, row in best_by_group.iterrows():
        print(f"   {row['feature_group']:25s}: R²={row['r_squared']:.4f} @ {row['time_window']}min")

    # Compare at fixed time window (10 min)
    print(f"\n📊 Comparison at 10-minute window:")
    df_10min = df_results[df_results['time_window'] == 10].sort_values('r_squared', ascending=False)

    print(f"\n   {'Feature Group':<25} {'R²':>10} {'RMSE':>10} {'Features':>10}")
    print("   " + "-" * 60)
    for _, row in df_10min.iterrows():
        print(f"   {row['feature_group']:<25} {row['r_squared']:>10.4f} {row['rmse']:>10.4f} {row['n_features']:>10}")

    # Calculate improvement over baseline
    print("\n" + "-" * 80)
    print("IMPROVEMENT OVER RAW L1 BASELINE")
    print("-" * 80)

    baseline = df_results[df_results['feature_group'] == 'Raw_L1']

    print(f"\n   {'Feature Group':<25} {'Best R²':>10} {'Baseline R²':>12} {'Improvement':>12}")
    print("   " + "-" * 65)

    for group_name in best_by_group['feature_group'].unique():
        group_best = df_results[df_results['feature_group'] == group_name]['r_squared'].max()
        group_best_tw = df_results[df_results['feature_group'] == group_name].loc[
            df_results[df_results['feature_group'] == group_name]['r_squared'].idxmax(), 'time_window']

        # Get baseline at same time window
        baseline_at_tw = baseline[baseline['time_window'] == group_best_tw]['r_squared'].values
        if len(baseline_at_tw) > 0:
            baseline_r2 = baseline_at_tw[0]
            improvement = ((group_best - baseline_r2) / baseline_r2 * 100) if baseline_r2 > 0 else 0
            print(f"   {group_name:<25} {group_best:>10.4f} {baseline_r2:>12.4f} {improvement:>+11.1f}%")

    # Best overall
    best_idx = df_results['r_squared'].idxmax()
    best = df_results.loc[best_idx]

    print(f"\n🏆 BEST OVERALL:")
    print(f"   Feature Group: {best['feature_group']}")
    print(f"   Time Window: {best['time_window']} min")
    print(f"   R²: {best['r_squared']:.4f}")
    print(f"   RMSE: {best['rmse']:.4f}")

    # Save results
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    df_results.to_csv(REGRESSION_ENHANCED_FILE, index=False)
    print(f"\n✓ Saved to: {REGRESSION_ENHANCED_FILE}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check which enhancement helps most
    raw_l10_best = df_results[df_results['feature_group'] == 'Raw_L10']['r_squared'].max() if 'Raw_L10' in df_results['feature_group'].values else 0
    norm_l10_best = df_results[df_results['feature_group'] == 'Normalized_L10']['r_squared'].max() if 'Normalized_L10' in df_results['feature_group'].values else 0
    weighted_best = df_results[df_results['feature_group'] == 'Exp_Weighted']['r_squared'].max() if 'Exp_Weighted' in df_results['feature_group'].values else 0
    pca_best = df_results[df_results['feature_group'] == 'PCA_Raw']['r_squared'].max() if 'PCA_Raw' in df_results['feature_group'].values else 0

    print(f"\n📊 Enhancement comparison (best R² achieved):")
    print(f"   Raw Multi-level:        {raw_l10_best:.4f}")
    print(f"   Depth Normalized:       {norm_l10_best:.4f}")
    print(f"   Exponential Weighted:   {weighted_best:.4f}")
    print(f"   PCA Compressed:         {pca_best:.4f}")

    if norm_l10_best > raw_l10_best:
        print(f"\n   ✅ Depth normalization IMPROVES results (+{(norm_l10_best-raw_l10_best)/raw_l10_best*100:.1f}%)")
    else:
        print(f"\n   ⚠️  Depth normalization does not improve over raw")

    if weighted_best > raw_l10_best * 0.9:
        print(f"   ✅ Exponential weighting is effective (single feature!)")

    if pca_best > raw_l10_best * 0.9:
        print(f"   ✅ PCA compression retains most information")

    print("\n" + "=" * 80)
    print("✅ ENHANCED REGRESSION ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n")

    return df_results


if __name__ == "__main__":
    main()
