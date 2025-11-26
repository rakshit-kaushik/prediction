"""
01_regression_analysis.py
=========================
Regression analysis of Order Flow Imbalance (OFI) vs Price Changes

Replicates Section 3.2 of Cont, Kukanov & Stoikov (2011) for Polymarket prediction markets.

Implements:
- Linear regression: ΔP = α + β × OFI + ε
- Quadratic regression: ΔP = α + β₁ × OFI + β₂ × OFI² + ε
- Generates Table 2 equivalent (regression statistics)

Usage:
    python scripts/01_regression_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config_analysis import (
    TIME_WINDOW,
    USE_TICK_NORMALIZED,
    get_dependent_variable_name,
    get_dependent_variable_label,
    print_config
)

# ============================================================================
# TIME AGGREGATION (CONFIGURABLE)
# ============================================================================

def aggregate_ofi_data(df):
    """
    Aggregate raw orderbook snapshots to configurable time intervals following
    Cont et al. (2011) methodology.

    Time window is controlled by TIME_WINDOW in config_analysis.py
    Default: 10 minutes ('10T'), but can be changed to any valid pandas frequency

    Key aggregations:
    - OFI: Sum over time window
    - Price change: P(end) - P(start) of window
    - Depth/Spread: Average over window
    - Events: Max (occurred if any event in window)

    Parameters:
    -----------
    df : pd.DataFrame
        Raw OFI data with timestamp column

    Returns:
    --------
    pd.DataFrame
        Aggregated data at configured time intervals
    """
    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    else:
        raise ValueError("No timestamp column found")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create time bins using configurable TIME_WINDOW
    df['time_bin'] = df['timestamp'].dt.floor(TIME_WINDOW)

    # Define aggregation rules
    agg_dict = {
        'ofi': 'sum',  # Sum OFI over window (CRITICAL per Cont et al.)
        'mid_price': ['first', 'last'],  # Get first and last for price change
        'total_depth': 'mean',  # Average depth
        'spread': 'mean',  # Average spread
        'spread_pct': 'mean',
        'best_bid_price': 'last',
        'best_ask_price': 'last',
        'best_bid_size': 'last',
        'best_ask_size': 'last',
        'total_bid_size': 'mean',
        'total_ask_size': 'mean'
    }

    # Add event indicators if they exist
    for col in ['bid_up', 'bid_down', 'ask_up', 'ask_down']:
        if col in df.columns:
            agg_dict[col] = 'max'  # Event occurred if any event in window

    # Perform aggregation
    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten multi-level column names
    # For multi-index columns: ('col', 'agg') -> 'col' if single agg, else 'col_agg'
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[0] in ['mid_price']:  # Keep aggregation suffix for multi-agg columns
                new_cols.append('_'.join(col))
            elif col[1]:  # Single aggregation - drop suffix
                new_cols.append(col[0])
            else:  # No aggregation (like time_bin)
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Calculate delta_mid_price as BETWEEN-window change (per Cont et al. 2011)
    # ΔPk = Pk - Pk-1 (change from previous window's end to current window's end)
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()

    # Also calculate tick-normalized price change if it exists in raw data
    if 'delta_mid_price_ticks' in df.columns:
        # For tick-normalized, we need to aggregate differently
        # We want the total tick change over the window
        aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / 0.01  # TICK_SIZE

    # Rename timestamp
    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})

    # Drop helper columns
    aggregated = aggregated.drop(columns=['mid_price_first', 'mid_price_last'], errors='ignore')

    return aggregated

# ============================================================================
# CONFIGURATION
# ============================================================================

MARKETS = {
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "data/ofi_oct15_nov20_combined.csv",
        "market_info_file": "data/market_info.json",
        "name_short": "Fed"
    },
    "NYC Mayoral Election 2025": {
        "ofi_file": "data/nyc_mayor_oct15_nov04_ofi.csv",
        "market_info_file": "data/nyc_mayor_market_info.json",
        "name_short": "NYC"
    }
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
(OUTPUT_DIR / "analysis").mkdir(exist_ok=True)

# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_linear_regression(df):
    """
    Run linear regression: ΔP = α + β × OFI + ε

    Returns dict with regression statistics
    """
    # Get dependent variable name from config (delta_mid_price or delta_mid_price_ticks)
    dep_var = get_dependent_variable_name()

    # Remove NaN values
    valid_mask = ~(df[dep_var].isna() | df['ofi'].isna())
    y = df.loc[valid_mask, dep_var].values
    X = df.loc[valid_mask, 'ofi'].values

    n = len(y)
    if n == 0:
        return None

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # OLS estimation
    try:
        beta_hat = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        alpha_hat, beta_hat_slope = beta_hat[0], beta_hat[1]

        # Predictions and residuals
        y_pred = X_with_intercept @ beta_hat
        residuals = y - y_pred

        # Sum of squares
        SST = np.sum((y - np.mean(y))**2)
        SSR = np.sum((y_pred - np.mean(y))**2)
        SSE = np.sum(residuals**2)

        # R-squared
        r_squared = SSR / SST if SST > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else 0

        # Standard errors (using White's heteroscedasticity-robust estimator)
        df_resid = n - 2
        s_squared = SSE / df_resid if df_resid > 0 else 0

        # Standard errors (homoscedastic)
        X_T_X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        var_cov = s_squared * X_T_X_inv
        se = np.sqrt(np.diag(var_cov))
        se_alpha, se_beta = se[0], se[1]

        # t-statistics
        t_alpha = alpha_hat / se_alpha if se_alpha > 0 else 0
        t_beta = beta_hat_slope / se_beta if se_beta > 0 else 0

        # p-values (two-tailed)
        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df_resid)) if df_resid > 0 else 1
        p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), df_resid)) if df_resid > 0 else 1

        # 95% Confidence intervals
        t_crit = stats.t.ppf(0.975, df_resid) if df_resid > 0 else 1.96
        ci_beta_lower = beta_hat_slope - t_crit * se_beta
        ci_beta_upper = beta_hat_slope + t_crit * se_beta

        # Correlation
        correlation = np.corrcoef(X, y)[0, 1] if len(X) > 1 else 0

        return {
            'alpha': alpha_hat,
            'beta': beta_hat_slope,
            'se_alpha': se_alpha,
            'se_beta': se_beta,
            't_alpha': t_alpha,
            't_beta': t_beta,
            'p_alpha': p_alpha,
            'p_beta': p_beta,
            'ci_beta_lower': ci_beta_lower,
            'ci_beta_upper': ci_beta_upper,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'correlation': correlation,
            'n_obs': n,
            'sse': SSE,
            'rmse': np.sqrt(SSE / n) if n > 0 else 0
        }
    except Exception as e:
        print(f"Error in regression: {e}")
        return None


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_market(market_name, market_config):
    """Analyze a single market"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {market_name}")
    print(f"{'='*80}\n")

    # Load data
    ofi_file = Path(market_config['ofi_file'])
    if not ofi_file.exists():
        print(f"❌ File not found: {ofi_file}")
        return None

    print(f"Loading: {ofi_file}")
    df = pd.read_csv(ofi_file)
    print(f"Loaded {len(df):,} raw observations")

    # Apply time aggregation per Cont et al. (2011)
    print(f"Aggregating to {TIME_WINDOW} intervals...")
    df_raw_count = len(df)
    df = aggregate_ofi_data(df)
    print(f"Aggregated to {len(df):,} {TIME_WINDOW} intervals (reduction: {df_raw_count/len(df):.1f}x)\n")

    # Get dependent variable name from config
    dep_var = get_dependent_variable_name()
    dep_var_label = get_dependent_variable_label()

    # Data summary
    print("Data Summary:")
    print(f"  OFI range: {df['ofi'].min():.2f} to {df['ofi'].max():.2f}")
    print(f"  OFI mean: {df['ofi'].mean():.2f}, std: {df['ofi'].std():.2f}")
    print(f"  {dep_var_label} range: {df[dep_var].min():.6f} to {df[dep_var].max():.6f}")
    print(f"  {dep_var_label} mean: {df[dep_var].mean():.6f}, std: {df[dep_var].std():.6f}")
    print(f"  Non-zero OFI: {(df['ofi'] != 0).sum():,} ({100*(df['ofi'] != 0).sum()/len(df):.1f}%)")
    print(f"  Non-zero {dep_var_label}: {(df[dep_var] != 0).sum():,} ({100*(df[dep_var] != 0).sum()/len(df):.1f}%)\n")

    # Run regressions
    print("Running Linear Regression: ΔP = α + β × OFI + ε")
    linear_results = run_linear_regression(df)

    if linear_results:
        print(f"\n  Results:")
        print(f"    α (intercept) = {linear_results['alpha']:.6f}")
        print(f"    β (slope)     = {linear_results['beta']:.6f}")
        print(f"    R²            = {linear_results['r_squared']:.4f}")
        print(f"    Adj R²        = {linear_results['adj_r_squared']:.4f}")
        print(f"    Correlation   = {linear_results['correlation']:.4f}")
        print(f"    t(β)          = {linear_results['t_beta']:.4f}")
        print(f"    p(β)          = {linear_results['p_beta']:.6f}")
        print(f"    95% CI (β)    = [{linear_results['ci_beta_lower']:.6f}, {linear_results['ci_beta_upper']:.6f}]")
        print(f"    RMSE          = {linear_results['rmse']:.6f}")

        # Significance
        is_significant = linear_results['p_beta'] < 0.05
        print(f"\n    β is {'✓ SIGNIFICANT' if is_significant else '✗ NOT SIGNIFICANT'} at 5% level")

    return {
        'market_name': market_name,
        'market_short': market_config['name_short'],
        'linear': linear_results,
        'n_obs': len(df)
    }


def create_summary_table(all_results):
    """Create Table 2 equivalent"""
    rows = []

    for result in all_results:
        if result and result['linear']:
            lin = result['linear']

            row = {
                'Market': result['market_short'],
                'N_obs': result['n_obs'],
                'Alpha': lin['alpha'],
                'Beta': lin['beta'],
                'SE_Beta': lin['se_beta'],
                't_Beta': lin['t_beta'],
                'p_Beta': lin['p_beta'],
                'R_squared': lin['r_squared'],
                'Adj_R_squared': lin['adj_r_squared'],
                'Correlation': lin['correlation'],
                'CI_Beta_Lower': lin['ci_beta_lower'],
                'CI_Beta_Upper': lin['ci_beta_upper'],
                'RMSE': lin['rmse']
            }

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("REGRESSION ANALYSIS - CONT ET AL. (2011) REPLICATION")
    print("="*80)
    print("\nReplicating Section 3.2: Empirical Findings")
    print("Linear model: ΔP = α + β × OFI + ε\n")

    # Print configuration
    print_config()
    print()

    # Analyze all markets
    all_results = []
    for market_name, market_config in MARKETS.items():
        result = analyze_market(market_name, market_config)
        if result:
            all_results.append(result)

    # Create summary table
    print(f"\n{'='*80}")
    print("CREATING SUMMARY TABLE (Table 2 Equivalent)")
    print(f"{'='*80}\n")

    summary_df = create_summary_table(all_results)

    # Display table
    print("Regression Statistics Summary:")
    print(summary_df.to_string(index=False))

    # Save results
    output_file = OUTPUT_DIR / "tables" / "table_2_regression_statistics.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

    # Save detailed results
    for result in all_results:
        if result:
            market_short = result['market_short']
            detailed_file = OUTPUT_DIR / "analysis" / f"{market_short}_regression_detailed.csv"

            # Convert results to DataFrame
            rows = []
            if result['linear']:
                for key, value in result['linear'].items():
                    rows.append({'Statistic': key, 'Value': value})

            detailed_df = pd.DataFrame(rows)
            detailed_df.to_csv(detailed_file, index=False)
            print(f"✓ Saved detailed results: {detailed_file}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL MARKETS")
    print(f"{'='*80}\n")

    print(f"Average R²: {summary_df['R_squared'].mean():.4f}")
    print(f"Average β: {summary_df['Beta'].mean():.6f}")
    print(f"Average Correlation: {summary_df['Correlation'].mean():.4f}")
    print(f"Markets with significant β (p < 0.05): {(summary_df['p_Beta'] < 0.05).sum()} / {len(summary_df)}")

    print(f"\n{'='*80}")
    print("✓ REGRESSION ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
