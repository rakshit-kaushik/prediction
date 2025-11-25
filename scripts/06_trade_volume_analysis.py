"""
06_trade_volume_analysis.py
============================
Trade Imbalance and Volume Analysis

Replicates Section 4 of Cont, Kukanov & Stoikov (2011):
- Trade Imbalance (TI) proxy calculation
- OFI vs TI comparison for price prediction
- Volume-price relationship analysis
- Horse-race regressions: ΔP ~ OFI, ΔP ~ TI, ΔP ~ OFI + TI

Note: Since Polymarket doesn't provide trade data directly, we use
orderbook events as a proxy for trade imbalance.

Usage:
    python scripts/06_trade_volume_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 10-SECOND TIME AGGREGATION
# ============================================================================

def aggregate_to_10_seconds(df):
    """
    Aggregate raw orderbook snapshots to 10-minute intervals following
    Cont et al. (2011) methodology.
    """
    if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    else:
        raise ValueError("No timestamp column found")

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_bin'] = df['timestamp'].dt.floor('10T')

    agg_dict = {
        'ofi': 'sum',
        'mid_price': ['first', 'last'],
        'total_depth': 'mean',
        'spread': 'mean',
        'spread_pct': 'mean',
        'best_bid_price': ['first', 'last'],
        'best_ask_price': ['first', 'last'],
        'best_bid_size': 'last',
        'best_ask_size': 'last',
        'total_bid_size': 'mean',
        'total_ask_size': 'mean'
    }

    for col in ['bid_up', 'bid_down', 'ask_up', 'ask_down']:
        if col in df.columns:
            agg_dict[col] = 'max'

    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()
    # Flatten multi-level column names
    # For multi-index columns: ('col', 'agg') -> 'col' if single agg, else 'col_agg'
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[0] in ['mid_price', 'best_bid_price', 'best_ask_price']:  # Keep aggregation suffix for multi-agg columns
                new_cols.append('_'.join(col))
            elif col[1]:  # Single aggregation - drop suffix
                new_cols.append(col[0])
            else:  # No aggregation (like time_bin)
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Use last prices as the price for the interval
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['best_bid_price'] = aggregated['best_bid_price_last']
    aggregated['best_ask_price'] = aggregated['best_ask_price_last']

    # Calculate BETWEEN-window price changes (per Cont et al. 2011)
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()
    aggregated['delta_best_bid'] = aggregated['best_bid_price'].diff()
    aggregated['delta_best_ask'] = aggregated['best_ask_price'].diff()
    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})
    aggregated = aggregated.drop(columns=['mid_price_first', 'mid_price_last',
                                          'best_bid_price_first', 'best_bid_price_last',
                                          'best_ask_price_first', 'best_ask_price_last'], errors='ignore')

    return aggregated

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Configuration
MARKETS = {
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "data/ofi_oct15_nov20_combined.csv",
        "name_short": "Fed"
    },
    "NYC Mayoral Election 2025": {
        "ofi_file": "data/nyc_mayor_oct15_nov04_ofi.csv",
        "name_short": "NYC"
    }
}

OUTPUT_DIR = Path("results")
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "analysis").mkdir(parents=True, exist_ok=True)


def calculate_trade_imbalance_proxy(df):
    """
    Calculate Trade Imbalance (TI) proxy from orderbook events

    For prediction markets without direct trade data, we use:
    TI = (bid_up - bid_down) + (ask_down - ask_up)

    Rationale:
    - bid_up: buying pressure (bullish)
    - bid_down: selling pressure (bearish)
    - ask_down: buying pressure (ask removed by buy)
    - ask_up: selling pressure (new ask added)
    """

    df = df.copy()

    # Simple TI proxy based on event indicators
    df['ti_proxy'] = (
        df['bid_up'].astype(int) -
        df['bid_down'].astype(int) +
        df['ask_down'].astype(int) -
        df['ask_up'].astype(int)
    )

    # Alternative: use delta in best bid/ask as proxy for executed trades
    # Positive delta suggests buying pressure
    df['ti_price_based'] = df['delta_best_bid'] - df['delta_best_ask']

    # Scaled by depth to normalize
    df['ti_depth_scaled'] = df['ti_proxy'] / (df['total_depth'] + 1e-10)

    return df


def run_horse_race_regressions(df):
    """
    Run horse-race regressions to compare OFI vs TI

    Models:
    1. ΔP = α + β_OFI × OFI
    2. ΔP = α + β_TI × TI
    3. ΔP = α + β_OFI × OFI + β_TI × TI
    """

    # Remove NaN
    valid = ~(df['delta_mid_price'].isna() | df['ofi'].isna() | df['ti_proxy'].isna())
    y = df.loc[valid, 'delta_mid_price'].values
    X_ofi = df.loc[valid, 'ofi'].values
    X_ti = df.loc[valid, 'ti_proxy'].values

    n = len(y)

    # Model 1: OFI only
    X1 = np.column_stack([np.ones(n), X_ofi])
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    y_pred1 = X1 @ beta1

    SST = np.sum((y - np.mean(y))**2)
    SSR1 = np.sum((y_pred1 - np.mean(y))**2)
    r2_ofi = SSR1 / SST if SST > 0 else 0

    # Model 2: TI only
    X2 = np.column_stack([np.ones(n), X_ti])
    beta2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    y_pred2 = X2 @ beta2

    SSR2 = np.sum((y_pred2 - np.mean(y))**2)
    r2_ti = SSR2 / SST if SST > 0 else 0

    # Model 3: Both OFI and TI
    X3 = np.column_stack([np.ones(n), X_ofi, X_ti])
    beta3 = np.linalg.lstsq(X3, y, rcond=None)[0]
    y_pred3 = X3 @ beta3

    SSR3 = np.sum((y_pred3 - np.mean(y))**2)
    r2_both = SSR3 / SST if SST > 0 else 0

    # Standard errors for Model 3
    residuals3 = y - y_pred3
    s_squared = np.sum(residuals3**2) / (n - 3) if n > 3 else 0
    X3_T_X3_inv = np.linalg.inv(X3.T @ X3)
    var_cov = s_squared * X3_T_X3_inv
    se = np.sqrt(np.diag(var_cov))

    # t-statistics
    t_stats = beta3 / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - 3))

    # Correlation between OFI and TI
    corr_ofi_ti = np.corrcoef(X_ofi, X_ti)[0, 1] if len(X_ofi) > 1 else 0

    return {
        'model_ofi': {
            'beta': beta1[1],
            'r_squared': r2_ofi
        },
        'model_ti': {
            'beta': beta2[1],
            'r_squared': r2_ti
        },
        'model_both': {
            'beta_ofi': beta3[1],
            'beta_ti': beta3[2],
            'se_ofi': se[1],
            'se_ti': se[2],
            't_ofi': t_stats[1],
            't_ti': t_stats[2],
            'p_ofi': p_values[1],
            'p_ti': p_values[2],
            'r_squared': r2_both
        },
        'corr_ofi_ti': corr_ofi_ti,
        'n_obs': n
    }


def volume_price_analysis(df):
    """
    Analyze relationship between volume (depth) and price changes

    Examines:
    - Correlation between depth changes and price changes
    - Price impact conditional on volume levels
    """

    # Remove NaN
    valid = ~(df['delta_mid_price'].isna() | df['total_depth'].isna())
    price_changes = df.loc[valid, 'delta_mid_price'].values
    depth = df.loc[valid, 'total_depth'].values

    # Depth changes
    depth_changes = np.diff(depth, prepend=depth[0])

    # Correlation
    corr_depth_price = np.corrcoef(depth_changes[1:], price_changes[1:])[0, 1] if len(depth) > 1 else 0

    # Divide into volume quintiles
    depth_quintiles = pd.qcut(depth, q=5, labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'], duplicates='drop')

    quintile_stats = []
    for q in depth_quintiles.unique():
        mask = (depth_quintiles == q)
        if mask.sum() > 0:
            quintile_stats.append({
                'quintile': q,
                'count': mask.sum(),
                'mean_depth': depth[mask].mean(),
                'mean_price_change': price_changes[mask].mean(),
                'std_price_change': price_changes[mask].std()
            })

    return {
        'corr_depth_price': corr_depth_price,
        'quintile_stats': pd.DataFrame(quintile_stats) if quintile_stats else None
    }


def analyze_market(market_name, config):
    """Full trade imbalance and volume analysis for a market"""

    print(f"\n{'='*80}")
    print(f"TRADE IMBALANCE & VOLUME ANALYSIS: {market_name}")
    print(f"{'='*80}\n")

    # Load data
    df = pd.read_csv(config['ofi_file'])
    df = aggregate_to_10_seconds(df)
    print(f"Loaded {len(df):,} observations\n")

    # Calculate TI proxy
    df = calculate_trade_imbalance_proxy(df)

    # Basic statistics
    print("Trade Imbalance Proxy Statistics:")
    print(f"  Mean TI: {df['ti_proxy'].mean():.4f}")
    print(f"  Std TI: {df['ti_proxy'].std():.4f}")
    print(f"  Correlation(OFI, TI): {df[['ofi', 'ti_proxy']].corr().iloc[0, 1]:.4f}")
    print()

    # Horse-race regressions
    print("Running Horse-Race Regressions...")
    print(f"{'-'*80}")
    hr_results = run_horse_race_regressions(df)

    print("\nModel 1: ΔP = α + β_OFI × OFI")
    print(f"  β_OFI = {hr_results['model_ofi']['beta']:.6e}")
    print(f"  R² = {hr_results['model_ofi']['r_squared']:.4f}")

    print("\nModel 2: ΔP = α + β_TI × TI")
    print(f"  β_TI = {hr_results['model_ti']['beta']:.6e}")
    print(f"  R² = {hr_results['model_ti']['r_squared']:.4f}")

    print("\nModel 3: ΔP = α + β_OFI × OFI + β_TI × TI")
    print(f"  β_OFI = {hr_results['model_both']['beta_ofi']:.6e} (t = {hr_results['model_both']['t_ofi']:.2f}, p = {hr_results['model_both']['p_ofi']:.4f})")
    print(f"  β_TI = {hr_results['model_both']['beta_ti']:.6e} (t = {hr_results['model_both']['t_ti']:.2f}, p = {hr_results['model_both']['p_ti']:.4f})")
    print(f"  R² = {hr_results['model_both']['r_squared']:.4f}")
    print(f"  OFI-TI Correlation = {hr_results['corr_ofi_ti']:.4f}")

    # Interpretation
    r2_improvement = hr_results['model_both']['r_squared'] - hr_results['model_ofi']['r_squared']
    print(f"\nR² improvement from adding TI: {r2_improvement:.6f} ({100*r2_improvement:.3f}%)")

    # Volume analysis
    print(f"\n{'-'*80}")
    print("Volume-Price Analysis:")
    vol_results = volume_price_analysis(df)
    print(f"  Correlation(Depth Change, Price Change) = {vol_results['corr_depth_price']:.4f}")

    if vol_results['quintile_stats'] is not None:
        print("\n  Price Volatility by Depth Quintile:")
        print(vol_results['quintile_stats'].to_string(index=False))

    # Save results
    output_file = OUTPUT_DIR / "analysis" / f"{config['name_short']}_ti_comparison.csv"
    ti_df = pd.DataFrame([{
        'Market': config['name_short'],
        'N_obs': hr_results['n_obs'],
        'Corr_OFI_TI': hr_results['corr_ofi_ti'],
        'R2_OFI_only': hr_results['model_ofi']['r_squared'],
        'R2_TI_only': hr_results['model_ti']['r_squared'],
        'R2_Both': hr_results['model_both']['r_squared'],
        'Beta_OFI_solo': hr_results['model_ofi']['beta'],
        'Beta_TI_solo': hr_results['model_ti']['beta'],
        'Beta_OFI_joint': hr_results['model_both']['beta_ofi'],
        'Beta_TI_joint': hr_results['model_both']['beta_ti'],
        'p_OFI_joint': hr_results['model_both']['p_ofi'],
        'p_TI_joint': hr_results['model_both']['p_ti']
    }])
    ti_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results: {output_file}")

    return {
        'market_name': market_name,
        'market_short': config['name_short'],
        'horse_race': hr_results,
        'volume': vol_results
    }


def create_comparison_plots(all_results):
    """Create visualization comparing OFI and TI"""

    n_markets = len(all_results)
    fig, axes = plt.subplots(2, n_markets, figsize=(7*n_markets, 12))

    if n_markets == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(all_results):
        market_short = result['market_short']
        hr = result['horse_race']

        # ----------------------------------------------------------------
        # Plot 1: R² Comparison
        # ----------------------------------------------------------------
        ax1 = axes[0, idx]

        models = ['OFI Only', 'TI Only', 'OFI + TI']
        r2_values = [
            hr['model_ofi']['r_squared'],
            hr['model_ti']['r_squared'],
            hr['model_both']['r_squared']
        ]

        colors = ['steelblue', 'coral', 'green']
        bars = ax1.bar(models, r2_values, color=colors, alpha=0.7)

        ax1.set_ylabel('R² (Explained Variance)', fontweight='bold')
        ax1.set_title(f'{market_short} Market\nModel Comparison: Price Prediction', fontweight='bold')
        ax1.set_ylim([0, max(r2_values) * 1.2])
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

        # ----------------------------------------------------------------
        # Plot 2: Coefficient Comparison
        # ----------------------------------------------------------------
        ax2 = axes[1, idx]

        coef_names = ['β_OFI\n(solo)', 'β_TI\n(solo)', 'β_OFI\n(joint)', 'β_TI\n(joint)']
        coef_values = [
            hr['model_ofi']['beta'],
            hr['model_ti']['beta'],
            hr['model_both']['beta_ofi'],
            hr['model_both']['beta_ti']
        ]

        colors_coef = ['steelblue', 'coral', 'darkblue', 'darkred']
        bars = ax2.bar(coef_names, coef_values, color=colors_coef, alpha=0.7)

        ax2.set_ylabel('Coefficient Value', fontweight='bold')
        ax2.set_title(f'{market_short} Market\nRegression Coefficients', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, coef_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=8, fontweight='bold')

    plt.suptitle('OFI vs Trade Imbalance Comparison\nHorse-Race Regression Results',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    png_file = OUTPUT_DIR / "figures" / "figure_6_ofi_ti_comparison.png"
    pdf_file = OUTPUT_DIR / "figures" / "figure_6_ofi_ti_comparison.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"\n✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")

    plt.close()


def main():
    print("\n" + "="*80)
    print("TRADE IMBALANCE AND VOLUME ANALYSIS")
    print("="*80)
    print("\nComparing OFI vs Trade Imbalance for price prediction")
    print("Volume-price relationship analysis\n")

    all_results = []

    for market_name, config in MARKETS.items():
        result = analyze_market(market_name, config)
        if result:
            all_results.append(result)

    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*80}")
    create_comparison_plots(all_results)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")

    summary_rows = []
    for result in all_results:
        hr = result['horse_race']
        summary_rows.append({
            'Market': result['market_short'],
            'R2_OFI': hr['model_ofi']['r_squared'],
            'R2_TI': hr['model_ti']['r_squared'],
            'R2_Both': hr['model_both']['r_squared'],
            'Corr_OFI_TI': hr['corr_ofi_ti'],
            'R2_Improvement': hr['model_both']['r_squared'] - hr['model_ofi']['r_squared']
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    output_file = OUTPUT_DIR / "tables" / "table_5_ofi_ti_comparison.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    print(f"\n{'='*80}")
    print("✓ TRADE IMBALANCE AND VOLUME ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
