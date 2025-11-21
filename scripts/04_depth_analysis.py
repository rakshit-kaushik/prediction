"""
04_depth_analysis.py
====================
Depth Analysis - Relationship between Price Impact and Market Depth

Replicates Section 3.2 of Cont, Kukanov & Stoikov (2011):
"The price impact coefficient β is inversely proportional to average depth"

Implements:
- Rolling window regression to estimate time-varying β
- Average depth (AD) calculation over time windows
- Power law fit: β ~ 1/AD^λ
- Generate Table 3 equivalent and depth statistics

Usage:
    python scripts/04_depth_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
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

# Rolling window parameters
WINDOW_SIZE = 500  # Number of observations per window
MIN_OBS = 100  # Minimum observations for regression


def rolling_regression(df, window_size=500, min_obs=100):
    """
    Perform rolling window regression to estimate time-varying β

    Returns DataFrame with:
    - window_start, window_end: time bounds
    - beta: estimated coefficient
    - avg_depth: average depth in window
    - r_squared: fit quality
    - n_obs: number of observations
    """

    results = []

    # Sort by timestamp
    df = df.sort_values('timestamp_ms').reset_index(drop=True)

    n = len(df)
    step = window_size // 2  # 50% overlap for smoother estimates

    for start_idx in range(0, n - window_size + 1, step):
        end_idx = start_idx + window_size

        window_df = df.iloc[start_idx:end_idx]

        # Remove NaN
        valid = ~(window_df['delta_mid_price'].isna() | window_df['ofi'].isna() | window_df['total_depth'].isna())

        if valid.sum() < min_obs:
            continue

        y = window_df.loc[valid, 'delta_mid_price'].values
        X = window_df.loc[valid, 'ofi'].values
        depth = window_df.loc[valid, 'total_depth'].values

        # Linear regression
        try:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            beta_hat = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            alpha_hat, beta_slope = beta_hat[0], beta_hat[1]

            # Predictions
            y_pred = X_with_intercept @ beta_hat

            # R-squared
            SST = np.sum((y - np.mean(y))**2)
            SSR = np.sum((y_pred - np.mean(y))**2)
            r_squared = SSR / SST if SST > 0 else 0

            # Average depth in window
            avg_depth = np.mean(depth)

            # Standard error of beta
            residuals = y - y_pred
            s_squared = np.sum(residuals**2) / (len(y) - 2)
            X_T_X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(s_squared * X_T_X_inv[1, 1])

            results.append({
                'window_start': window_df['timestamp'].iloc[0],
                'window_end': window_df['timestamp'].iloc[-1],
                'window_start_ms': window_df['timestamp_ms'].iloc[0],
                'window_end_ms': window_df['timestamp_ms'].iloc[-1],
                'beta': beta_slope,
                'alpha': alpha_hat,
                'se_beta': se_beta,
                'avg_depth': avg_depth,
                'avg_mid_price': np.mean(window_df.loc[valid, 'mid_price']),
                'r_squared': r_squared,
                'n_obs': len(y)
            })
        except:
            continue

    return pd.DataFrame(results)


def fit_power_law(avg_depth, beta):
    """
    Fit power law: β = a / AD^λ

    Equivalently: log(β) = log(a) - λ * log(AD)
    """

    # Remove invalid values
    valid = (beta > 0) & (avg_depth > 0) & np.isfinite(beta) & np.isfinite(avg_depth)

    if valid.sum() < 10:
        return None

    log_depth = np.log(avg_depth[valid])
    log_beta = np.log(beta[valid])

    # Linear regression in log-log space
    X_log = np.column_stack([np.ones(len(log_depth)), log_depth])

    try:
        coeffs = np.linalg.lstsq(X_log, log_beta, rcond=None)[0]
        log_a, lambda_hat = coeffs[0], -coeffs[1]  # Note: λ = -slope
        a = np.exp(log_a)

        # R-squared
        log_beta_pred = X_log @ coeffs
        SST = np.sum((log_beta - np.mean(log_beta))**2)
        SSR = np.sum((log_beta_pred - np.mean(log_beta))**2)
        r_squared = SSR / SST if SST > 0 else 0

        # Standard errors
        residuals = log_beta - log_beta_pred
        s_squared = np.sum(residuals**2) / (len(log_beta) - 2)
        X_T_X_inv = np.linalg.inv(X_log.T @ X_log)
        se_lambda = np.sqrt(s_squared * X_T_X_inv[1, 1])

        return {
            'a': a,
            'lambda': lambda_hat,
            'se_lambda': se_lambda,
            'r_squared': r_squared,
            'log_a': log_a,
            'n_windows': len(log_beta)
        }
    except:
        return None


def analyze_market_depth(market_name, config):
    """Analyze depth relationship for a single market"""

    print(f"\n{'='*80}")
    print(f"DEPTH ANALYSIS: {market_name}")
    print(f"{'='*80}\n")

    # Load data
    ofi_file = Path(config['ofi_file'])
    print(f"Loading: {ofi_file}")
    df = pd.read_csv(ofi_file)

    # Convert timestamp
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

    print(f"Loaded {len(df):,} observations\n")

    # Overall depth statistics
    print("Overall Depth Statistics:")
    print(f"  Mean depth: {df['total_depth'].mean():,.2f}")
    print(f"  Median depth: {df['total_depth'].median():,.2f}")
    print(f"  Std depth: {df['total_depth'].std():,.2f}")
    print(f"  Min depth: {df['total_depth'].min():,.2f}")
    print(f"  Max depth: {df['total_depth'].max():,.2f}\n")

    # Rolling window regression
    print(f"Running rolling window regression (window={WINDOW_SIZE}, min_obs={MIN_OBS})...")
    rolling_results = rolling_regression(df, window_size=WINDOW_SIZE, min_obs=MIN_OBS)

    if len(rolling_results) == 0:
        print(f"❌ Not enough data for rolling regression")
        return None

    print(f"Generated {len(rolling_results)} windows\n")

    # Remove outlier betas (keep reasonable values)
    beta_q01 = rolling_results['beta'].quantile(0.01)
    beta_q99 = rolling_results['beta'].quantile(0.99)
    filtered_results = rolling_results[
        (rolling_results['beta'] >= beta_q01) &
        (rolling_results['beta'] <= beta_q99) &
        (rolling_results['beta'] > 0)
    ].copy()

    print(f"After filtering outliers: {len(filtered_results)} windows")
    print(f"Beta range: [{filtered_results['beta'].min():.6e}, {filtered_results['beta'].max():.6e}]")
    print(f"Avg depth range: [{filtered_results['avg_depth'].min():,.0f}, {filtered_results['avg_depth'].max():,.0f}]\n")

    # Fit power law
    print("Fitting power law: β = a / AD^λ")
    power_law_fit = fit_power_law(
        filtered_results['avg_depth'].values,
        filtered_results['beta'].values
    )

    if power_law_fit:
        print(f"\nPower Law Results:")
        print(f"  a = {power_law_fit['a']:.6e}")
        print(f"  λ = {power_law_fit['lambda']:.4f} ± {power_law_fit['se_lambda']:.4f}")
        print(f"  R² = {power_law_fit['r_squared']:.4f}")
        print(f"  n_windows = {power_law_fit['n_windows']}")

        # Compare to paper's finding (λ ≈ 1)
        lambda_val = power_law_fit['lambda']
        if abs(lambda_val - 1.0) < 0.3:
            print(f"  ✓ λ is close to 1 (paper's finding)")
        else:
            print(f"  Note: λ differs from paper's finding (λ ≈ 1)")

    # Save results
    output_file = OUTPUT_DIR / "analysis" / f"{config['name_short']}_rolling_regression.csv"
    rolling_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved rolling regression results: {output_file}")

    return {
        'market_name': market_name,
        'market_short': config['name_short'],
        'overall_depth_mean': df['total_depth'].mean(),
        'overall_depth_std': df['total_depth'].std(),
        'rolling_results': filtered_results,
        'power_law_fit': power_law_fit
    }


def create_depth_plots(all_results):
    """Create depth analysis visualization"""

    n_markets = len(all_results)
    fig, axes = plt.subplots(2, n_markets, figsize=(7*n_markets, 12))

    if n_markets == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(all_results):
        if result is None:
            continue

        market_short = result['market_short']
        market_name = result['market_name']
        rolling_df = result['rolling_results']
        power_law = result['power_law_fit']

        # ----------------------------------------------------------------
        # Plot 1: Time series of β and average depth
        # ----------------------------------------------------------------
        ax1 = axes[0, idx]
        ax1_twin = ax1.twinx()

        # Plot beta
        ax1.plot(rolling_df.index, rolling_df['beta'], 'b-', alpha=0.6, linewidth=1.5, label='β (Price Impact)')
        ax1.set_xlabel('Window Index', fontweight='bold')
        ax1.set_ylabel('β (Price Impact Coefficient)', fontweight='bold', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)

        # Plot depth
        ax1_twin.plot(rolling_df.index, rolling_df['avg_depth'], 'r-', alpha=0.6, linewidth=1.5, label='Avg Depth')
        ax1_twin.set_ylabel('Average Depth', fontweight='bold', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')

        ax1.set_title(f'{market_short} Market\nTime Evolution of β and Depth', fontweight='bold')

        # ----------------------------------------------------------------
        # Plot 2: β vs Average Depth (Power Law)
        # ----------------------------------------------------------------
        ax2 = axes[1, idx]

        # Scatter plot in log-log space
        log_depth = np.log(rolling_df['avg_depth'])
        log_beta = np.log(rolling_df['beta'])

        ax2.scatter(log_depth, log_beta, alpha=0.4, s=30, color='steelblue', edgecolors='none')

        # Power law fit line
        if power_law:
            log_depth_range = np.linspace(log_depth.min(), log_depth.max(), 100)
            log_beta_pred = power_law['log_a'] - power_law['lambda'] * log_depth_range
            ax2.plot(log_depth_range, log_beta_pred, 'r-', linewidth=2.5,
                    label=f'β = {power_law["a"]:.2e} / AD^{power_law["lambda"]:.2f}')

        ax2.set_xlabel('log(Average Depth)', fontweight='bold')
        ax2.set_ylabel('log(β)', fontweight='bold')
        ax2.set_title(f'{market_short} Market\nPower Law: β ∝ 1/AD^λ', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add statistics box
        if power_law:
            textstr = f'λ = {power_law["lambda"]:.3f} ± {power_law["se_lambda"]:.3f}\n'
            textstr += f'R² = {power_law["r_squared"]:.4f}\n'
            textstr += f'n = {power_law["n_windows"]}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

    plt.suptitle('Depth Analysis: Price Impact vs Market Depth\nCont et al. (2011) Replication',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    png_file = OUTPUT_DIR / "figures" / "figure_4_depth_analysis.png"
    pdf_file = OUTPUT_DIR / "figures" / "figure_4_depth_analysis.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"\n✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")

    plt.close()


def create_summary_table(all_results):
    """Create Table 3 equivalent - Depth Analysis Summary"""

    rows = []

    for result in all_results:
        if result and result['power_law_fit']:
            pl = result['power_law_fit']

            row = {
                'Market': result['market_short'],
                'Avg_Depth_Mean': result['overall_depth_mean'],
                'Avg_Depth_Std': result['overall_depth_std'],
                'Power_Law_a': pl['a'],
                'Power_Law_lambda': pl['lambda'],
                'SE_lambda': pl['se_lambda'],
                'R_squared': pl['r_squared'],
                'N_windows': pl['n_windows']
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("\n" + "="*80)
    print("DEPTH ANALYSIS - PRICE IMPACT vs MARKET DEPTH")
    print("="*80)
    print("\nReplicating Section 3.2: β inversely proportional to average depth")
    print("Power law model: β = a / AD^λ\n")

    all_results = []

    for market_name, config in MARKETS.items():
        result = analyze_market_depth(market_name, config)
        if result:
            all_results.append(result)

    if len(all_results) == 0:
        print("\n❌ No results to analyze")
        return

    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING DEPTH ANALYSIS PLOTS")
    print(f"{'='*80}")
    create_depth_plots(all_results)

    # Create summary table
    print(f"\n{'='*80}")
    print("CREATING SUMMARY TABLE (Table 3 Equivalent)")
    print(f"{'='*80}\n")

    summary_df = create_summary_table(all_results)
    print("Depth Analysis Summary:")
    print(summary_df.to_string(index=False))

    # Save table
    output_file = OUTPUT_DIR / "tables" / "table_3_depth_analysis.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    print(f"\n{'='*80}")
    print("✓ DEPTH ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
