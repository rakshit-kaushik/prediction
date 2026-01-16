"""
03_regression_analysis.py
=========================
STEP 3: Run regularized regression analysis on MLOFI data

WHAT IT DOES:
-------------
Input:  MLOFI calculated data (from step 2)
Output: Regression results for all configurations

REGRESSION METHODS:
------------------
1. OLS (Level 1 only) - Baseline
2. OLS (Cumulative) - Sum of OFI through level N
3. Ridge - L2 regularization (shrinks coefficients)
4. Lasso - L1 regularization (feature selection)
5. ElasticNet - L1 + L2 (best of both)

CONFIGURATIONS TESTED:
---------------------
- 4 level configs: L5, L10, L50%, ALL
- 9 time windows: 1, 5, 10, 15, 20, 30, 45, 60, 90 minutes
- 5 regression methods

Total: 4 × 9 × 5 = 180 regressions

USAGE:
------
    python mlofi/03_regression_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MLOFI_CALCULATED_FILE,
    REGRESSION_RESULTS_FILE,
    LEVEL_IMPORTANCE_FILE,
    LEVEL_CONFIGS,
    TIME_WINDOWS,
    RIDGE_ALPHAS,
    LASSO_ALPHAS,
    ELASTICNET_ALPHAS,
    ELASTICNET_L1_RATIOS,
    CV_FOLDS,
    MIN_OBS_FOR_REGRESSION,
    STANDARDIZE_FEATURES,
    MLOFI_OUTPUT_DIR
)


def aggregate_by_time_window(df, time_window_minutes):
    """
    Aggregate MLOFI data by time window.

    Args:
        df: DataFrame with MLOFI data
        time_window_minutes: Aggregation window in minutes

    Returns:
        Aggregated DataFrame
    """
    df = df.copy()
    df['time_bin'] = df['timestamp'].dt.floor(f'{time_window_minutes}T')

    # Get OFI columns
    ofi_cols = [c for c in df.columns if c.startswith('ofi_l') and not c.startswith('ofi_cumulative')]
    cumulative_cols = [c for c in df.columns if c.startswith('ofi_cumulative')]

    # Aggregation rules
    agg_dict = {
        'mid_price': ['first', 'last'],
        'spread': 'mean',
        'time_diff': 'sum',
    }

    # Sum OFI values within window
    for col in ofi_cols + cumulative_cols:
        agg_dict[col] = 'sum'

    # Aggregate
    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten column names
    aggregated.columns = [
        f'{col[0]}_{col[1]}' if isinstance(col, tuple) and col[1] else col[0]
        for col in aggregated.columns
    ]

    # Rename specific columns
    aggregated = aggregated.rename(columns={
        'time_bin': 'timestamp',
        'mid_price_first': 'mid_price_start',
        'mid_price_last': 'mid_price_end',
        'spread_mean': 'spread',
        'time_diff_sum': 'time_diff',
    })

    # Rename OFI columns (remove _sum suffix)
    for col in ofi_cols + cumulative_cols:
        if f'{col}_sum' in aggregated.columns:
            aggregated = aggregated.rename(columns={f'{col}_sum': col})

    # Calculate price change
    aggregated['delta_mid_price'] = aggregated['mid_price_end'].diff()
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / 0.01

    # Remove first row
    aggregated = aggregated.iloc[1:].reset_index(drop=True)

    return aggregated


def get_levels_for_config(config_name, max_level, n_bid_levels_median=None):
    """
    Get the number of levels to use for a given configuration.

    Args:
        config_name: 'L5', 'L10', 'L50pct', 'ALL'
        max_level: Maximum level available in data
        n_bid_levels_median: Median number of bid levels (for 50%)

    Returns:
        int: Number of levels to use
    """
    config_value = LEVEL_CONFIGS[config_name]

    if isinstance(config_value, int):
        return min(config_value, max_level)
    elif config_value == '50%':
        if n_bid_levels_median:
            return max(1, int(n_bid_levels_median * 0.5))
        else:
            return max(1, max_level // 2)
    elif config_value == 'all':
        return max_level
    else:
        return max_level


def run_regression_suite(X, y, level_config, time_window, n_levels):
    """
    Run all regression methods for a given configuration.

    Args:
        X: Feature matrix (OFI values)
        y: Target vector (price changes)
        level_config: Configuration name
        time_window: Time window in minutes
        n_levels: Number of levels used

    Returns:
        List of result dictionaries
    """
    results = []

    # Standardize features if configured
    if STANDARDIZE_FEATURES:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # 1. OLS Level 1 only (baseline)
    if X.shape[1] >= 1:
        try:
            ols_l1 = LinearRegression()
            ols_l1.fit(X_scaled[:, 0:1], y)
            y_pred = ols_l1.predict(X_scaled[:, 0:1])
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            results.append({
                'level_config': level_config,
                'time_window': time_window,
                'method': 'OLS_L1',
                'n_levels': 1,
                'r_squared': r2,
                'rmse': rmse,
                'best_alpha': np.nan,
                'l1_ratio': np.nan,
                'n_nonzero_coefs': 1,
                'coefficients': str(ols_l1.coef_.tolist()),
            })
        except Exception as e:
            print(f"      OLS_L1 failed: {e}")

    # 2. OLS Cumulative (sum of all levels)
    try:
        X_cumulative = X.sum(axis=1).reshape(-1, 1)
        if STANDARDIZE_FEATURES:
            X_cumulative = StandardScaler().fit_transform(X_cumulative)

        ols_cumul = LinearRegression()
        ols_cumul.fit(X_cumulative, y)
        y_pred = ols_cumul.predict(X_cumulative)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        results.append({
            'level_config': level_config,
            'time_window': time_window,
            'method': 'OLS_Cumulative',
            'n_levels': n_levels,
            'r_squared': r2,
            'rmse': rmse,
            'best_alpha': np.nan,
            'l1_ratio': np.nan,
            'n_nonzero_coefs': 1,
            'coefficients': str(ols_cumul.coef_.tolist()),
        })
    except Exception as e:
        print(f"      OLS_Cumulative failed: {e}")

    # 3. Ridge Regression
    try:
        ridge_cv = RidgeCV(alphas=RIDGE_ALPHAS, cv=CV_FOLDS)
        ridge_cv.fit(X_scaled, y)
        y_pred = ridge_cv.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        n_nonzero = np.sum(np.abs(ridge_cv.coef_) > 1e-10)

        results.append({
            'level_config': level_config,
            'time_window': time_window,
            'method': 'Ridge',
            'n_levels': n_levels,
            'r_squared': r2,
            'rmse': rmse,
            'best_alpha': ridge_cv.alpha_,
            'l1_ratio': np.nan,
            'n_nonzero_coefs': n_nonzero,
            'coefficients': str(ridge_cv.coef_.tolist()),
        })
    except Exception as e:
        print(f"      Ridge failed: {e}")

    # 4. Lasso Regression
    try:
        lasso_cv = LassoCV(alphas=LASSO_ALPHAS, cv=CV_FOLDS, max_iter=10000)
        lasso_cv.fit(X_scaled, y)
        y_pred = lasso_cv.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        n_nonzero = np.sum(np.abs(lasso_cv.coef_) > 1e-10)

        results.append({
            'level_config': level_config,
            'time_window': time_window,
            'method': 'Lasso',
            'n_levels': n_levels,
            'r_squared': r2,
            'rmse': rmse,
            'best_alpha': lasso_cv.alpha_,
            'l1_ratio': 1.0,
            'n_nonzero_coefs': n_nonzero,
            'coefficients': str(lasso_cv.coef_.tolist()),
        })
    except Exception as e:
        print(f"      Lasso failed: {e}")

    # 5. ElasticNet Regression
    try:
        elasticnet_cv = ElasticNetCV(
            alphas=ELASTICNET_ALPHAS,
            l1_ratio=ELASTICNET_L1_RATIOS,
            cv=CV_FOLDS,
            max_iter=10000
        )
        elasticnet_cv.fit(X_scaled, y)
        y_pred = elasticnet_cv.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        n_nonzero = np.sum(np.abs(elasticnet_cv.coef_) > 1e-10)

        results.append({
            'level_config': level_config,
            'time_window': time_window,
            'method': 'ElasticNet',
            'n_levels': n_levels,
            'r_squared': r2,
            'rmse': rmse,
            'best_alpha': elasticnet_cv.alpha_,
            'l1_ratio': elasticnet_cv.l1_ratio_,
            'n_nonzero_coefs': n_nonzero,
            'coefficients': str(elasticnet_cv.coef_.tolist()),
        })
    except Exception as e:
        print(f"      ElasticNet failed: {e}")

    return results


def extract_level_importance(df_results):
    """
    Extract coefficient importance for each level from regression results.

    Args:
        df_results: DataFrame with regression results

    Returns:
        DataFrame with level importance analysis
    """
    importance_data = []

    # Focus on L10 config with 10-minute window for coefficient analysis
    for method in ['Ridge', 'Lasso', 'ElasticNet']:
        subset = df_results[
            (df_results['level_config'] == 'L10') &
            (df_results['time_window'] == 10) &
            (df_results['method'] == method)
        ]

        if len(subset) == 0:
            continue

        row = subset.iloc[0]
        try:
            coefs = eval(row['coefficients'])
            for i, coef in enumerate(coefs):
                importance_data.append({
                    'level': i + 1,
                    'method': method,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef),
                })
        except:
            pass

    if not importance_data:
        return pd.DataFrame()

    df_importance = pd.DataFrame(importance_data)

    # Pivot to get coefficients by method
    df_pivot = df_importance.pivot(index='level', columns='method', values='coefficient').reset_index()

    # Add importance rank based on absolute Lasso coefficient
    if 'Lasso' in df_pivot.columns:
        df_pivot['abs_lasso'] = df_pivot['Lasso'].abs()
        df_pivot['importance_rank'] = df_pivot['abs_lasso'].rank(ascending=False)
        df_pivot = df_pivot.drop(columns=['abs_lasso'])

    return df_pivot


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "MLOFI REGRESSION ANALYSIS" + " " * 37 + "║")
    print("║" + " " * 12 + "Ridge / Lasso / ElasticNet Comparison" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not MLOFI_CALCULATED_FILE.exists():
        print(f"\n❌ MLOFI calculated file not found: {MLOFI_CALCULATED_FILE}")
        print("   Please run step 2 first: python mlofi/02_calculate_mlofi.py")
        sys.exit(1)

    # Create output directory
    MLOFI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load MLOFI data
    print(f"\n📂 Loading MLOFI data...")
    df = pd.read_csv(MLOFI_CALCULATED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   Loaded {len(df)} rows")

    # Determine max levels
    ofi_cols = [c for c in df.columns if c.startswith('ofi_l') and not c.startswith('ofi_cumulative')]
    max_level = len(ofi_cols)
    print(f"   Max levels available: {max_level}")

    # Run regression for all configurations
    print("\n" + "=" * 80)
    print("RUNNING REGRESSIONS")
    print("=" * 80)

    all_results = []
    total_configs = len(LEVEL_CONFIGS) * len(TIME_WINDOWS)
    config_count = 0

    for level_config in LEVEL_CONFIGS.keys():
        n_levels = get_levels_for_config(level_config, max_level)
        print(f"\n📊 Level config: {level_config} ({n_levels} levels)")

        for time_window in TIME_WINDOWS:
            config_count += 1
            print(f"   [{config_count}/{total_configs}] Time window: {time_window} min...")

            # Aggregate data
            df_agg = aggregate_by_time_window(df, time_window)

            # Prepare features (OFI at levels 1 to n_levels)
            ofi_cols_subset = [f'ofi_l{l}' for l in range(1, n_levels + 1)]
            X = df_agg[ofi_cols_subset].values
            y = df_agg['delta_mid_price_ticks'].values

            # Remove NaN
            valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
            X = X[valid_mask]
            y = y[valid_mask]

            if len(y) < MIN_OBS_FOR_REGRESSION:
                print(f"      Skipping: only {len(y)} observations (min: {MIN_OBS_FOR_REGRESSION})")
                continue

            # Run regression suite
            results = run_regression_suite(X, y, level_config, time_window, n_levels)
            all_results.extend(results)

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Total regressions run: {len(df_results)}")

    # Best R² by method
    print(f"\n📊 Best R² by method:")
    method_best = df_results.groupby('method')['r_squared'].max()
    for method, r2 in method_best.items():
        print(f"   {method}: {r2:.4f}")

    # Best R² by level config
    print(f"\n📊 Best R² by level config:")
    config_best = df_results.groupby('level_config')['r_squared'].max()
    for config, r2 in config_best.items():
        print(f"   {config}: {r2:.4f}")

    # Best overall configuration
    best_idx = df_results['r_squared'].idxmax()
    best_row = df_results.loc[best_idx]
    print(f"\n🏆 Best overall configuration:")
    print(f"   Config: {best_row['level_config']}")
    print(f"   Time Window: {best_row['time_window']} min")
    print(f"   Method: {best_row['method']}")
    print(f"   R²: {best_row['r_squared']:.4f}")
    print(f"   RMSE: {best_row['rmse']:.6f}")
    print(f"   Non-zero coefficients: {best_row['n_nonzero_coefs']}")

    # Extract level importance
    print("\n" + "-" * 80)
    print("LEVEL IMPORTANCE ANALYSIS")
    print("-" * 80)

    df_importance = extract_level_importance(df_results)
    if len(df_importance) > 0:
        print(f"\n📊 Coefficient values by level (L10 config, 10-min window):")
        print(df_importance.to_string(index=False))

        # Save level importance
        df_importance.to_csv(LEVEL_IMPORTANCE_FILE, index=False)
        print(f"\n✓ Level importance saved to: {LEVEL_IMPORTANCE_FILE}")

    # Save results
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    df_results.to_csv(REGRESSION_RESULTS_FILE, index=False)
    print(f"\n✓ Regression results saved to: {REGRESSION_RESULTS_FILE}")
    print(f"   Total rows: {len(df_results)}")

    print("\n" + "=" * 80)
    print("✅ REGRESSION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n💡 Next step: python mlofi/04_compare_results.py")
    print("\n")

    return df_results


if __name__ == "__main__":
    main()
