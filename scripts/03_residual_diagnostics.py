"""
03_residual_diagnostics.py
===========================
Residual Diagnostics for OFI Regression Models

Validates regression assumptions through:
- Residual plots (heteroscedasticity check)
- Q-Q plots (normality check)
- Autocorrelation analysis
- Residual statistics

Usage:
    python scripts/03_residual_diagnostics.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

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

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_regression_get_residuals(df):
    """Run regression and return residuals"""
    # Remove NaN values
    valid_mask = ~(df['delta_mid_price'].isna() | df['ofi'].isna())
    y = df.loc[valid_mask, 'delta_mid_price'].values
    X = df.loc[valid_mask, 'ofi'].values

    n = len(y)
    if n == 0:
        return None

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # OLS estimation
    beta_hat = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

    # Predictions and residuals
    y_pred = X_with_intercept @ beta_hat
    residuals = y - y_pred

    # Standardized residuals
    residual_std = np.std(residuals)
    standardized_residuals = residuals / residual_std if residual_std > 0 else residuals

    return {
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'std_residuals': standardized_residuals,
        'beta': beta_hat
    }

def create_diagnostic_plots(df, market_name, market_short):
    """Create comprehensive residual diagnostic plots"""

    # Get regression results
    results = run_regression_get_residuals(df)
    if results is None:
        print(f"❌ Could not compute residuals for {market_name}")
        return

    X = results['X']
    y = results['y']
    y_pred = results['y_pred']
    residuals = results['residuals']
    std_residuals = results['std_residuals']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -------------------------------------------------------------------------
    # Plot 1: Residuals vs Fitted Values
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.3, s=5, color='steelblue', edgecolors='none')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted Values', fontweight='bold')
    ax.set_ylabel('Residuals', fontweight='bold')
    ax.set_title('Residuals vs Fitted Values\n(Check for Heteroscedasticity)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add LOWESS smooth line to detect patterns
    from scipy.signal import savgol_filter
    if len(y_pred) > 100:
        # Sort by fitted values for smooth line
        sorted_idx = np.argsort(y_pred)
        sorted_pred = y_pred[sorted_idx]
        sorted_resid = residuals[sorted_idx]

        # Use every Nth point to avoid memory issues
        step = max(1, len(sorted_pred) // 1000)
        try:
            window = min(51, len(sorted_resid[::step]) // 2 * 2 + 1)  # Must be odd
            if window >= 5:
                smooth = savgol_filter(sorted_resid[::step], window, 3)
                ax.plot(sorted_pred[::step], smooth, 'g-', linewidth=2, alpha=0.7, label='Trend')
                ax.legend()
        except:
            pass

    # -------------------------------------------------------------------------
    # Plot 2: Q-Q Plot (Normality Check)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    stats.probplot(std_residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot\n(Check for Normality of Residuals)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Plot 3: Histogram of Residuals
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.hist(residuals, bins=100, alpha=0.7, color='steelblue', edgecolor='black')

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats.norm.pdf(x_range, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 100
    ax.plot(x_range, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.2e}, σ={sigma:.2e})')

    ax.set_xlabel('Residuals', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Residuals\n(Check for Normality)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    textstr = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # -------------------------------------------------------------------------
    # Plot 4: Residuals vs OFI
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.scatter(X, residuals, alpha=0.3, s=5, color='steelblue', edgecolors='none')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Order Flow Imbalance (OFI)', fontweight='bold')
    ax.set_ylabel('Residuals', fontweight='bold')
    ax.set_title('Residuals vs OFI\n(Check for Patterns)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Overall title
    plt.suptitle(f'{market_name}\nResidual Diagnostics',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    png_file = OUTPUT_DIR / f"figure_3_{market_short}_residual_diagnostics.png"
    pdf_file = OUTPUT_DIR / f"figure_3_{market_short}_residual_diagnostics.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")

    plt.close()

    return {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'n_obs': len(residuals)
    }

def run_diagnostic_tests(df, market_name):
    """Run statistical tests on residuals"""

    results = run_regression_get_residuals(df)
    if results is None:
        return None

    residuals = results['residuals']

    print(f"\n{'-'*80}")
    print(f"DIAGNOSTIC TESTS: {market_name}")
    print(f"{'-'*80}\n")

    # 1. Normality tests
    print("1. NORMALITY TESTS:")

    # Shapiro-Wilk test (sample if too large)
    sample_size = min(5000, len(residuals))
    sample_idx = np.random.choice(len(residuals), sample_size, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(residuals[sample_idx])
    print(f"   Shapiro-Wilk test (n={sample_size}):")
    print(f"   - Statistic: {shapiro_stat:.6f}")
    print(f"   - p-value: {shapiro_p:.6f}")
    print(f"   - Result: {'✗ Reject normality' if shapiro_p < 0.05 else '✓ Cannot reject normality'}")

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(residuals)
    print(f"\n   Jarque-Bera test:")
    print(f"   - Statistic: {jb_stat:.6f}")
    print(f"   - p-value: {jb_p:.6f}")
    print(f"   - Result: {'✗ Reject normality' if jb_p < 0.05 else '✓ Cannot reject normality'}")

    # 2. Autocorrelation (Durbin-Watson)
    print(f"\n2. AUTOCORRELATION TEST:")
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"   Durbin-Watson statistic: {dw_stat:.6f}")
    print(f"   - Interpretation: {'Positive autocorr.' if dw_stat < 1.5 else 'No autocorr.' if dw_stat < 2.5 else 'Negative autocorr.'}")

    # 3. Heteroscedasticity (Breusch-Pagan-style)
    print(f"\n3. HETEROSCEDASTICITY:")
    X = results['X']
    squared_residuals = residuals ** 2

    # Regress squared residuals on X
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    try:
        beta_het = np.linalg.lstsq(X_with_intercept, squared_residuals, rcond=None)[0]
        y_pred_het = X_with_intercept @ beta_het

        # R² for heteroscedasticity regression
        SST_het = np.sum((squared_residuals - np.mean(squared_residuals))**2)
        SSR_het = np.sum((y_pred_het - np.mean(squared_residuals))**2)
        r2_het = SSR_het / SST_het if SST_het > 0 else 0

        # LM statistic
        lm_stat = len(X) * r2_het
        lm_p = 1 - stats.chi2.cdf(lm_stat, 1)

        print(f"   Breusch-Pagan-style test:")
        print(f"   - LM Statistic: {lm_stat:.6f}")
        print(f"   - p-value: {lm_p:.6f}")
        print(f"   - Result: {'✗ Heteroscedasticity detected' if lm_p < 0.05 else '✓ Homoscedastic'}")
    except:
        print(f"   Could not compute heteroscedasticity test")

    # 4. Residual statistics
    print(f"\n4. RESIDUAL STATISTICS:")
    print(f"   Mean: {residuals.mean():.6e}")
    print(f"   Std Dev: {residuals.std():.6e}")
    print(f"   Min: {residuals.min():.6e}")
    print(f"   Max: {residuals.max():.6e}")
    print(f"   Skewness: {stats.skew(residuals):.6f}")
    print(f"   Kurtosis: {stats.kurtosis(residuals):.6f}")

    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'dw_stat': dw_stat
    }

def main():
    print("\n" + "="*80)
    print("RESIDUAL DIAGNOSTICS FOR OFI REGRESSIONS")
    print("="*80 + "\n")

    all_diagnostics = []

    for market_name, config in MARKETS.items():
        print(f"Processing: {market_name}")
        print(f"Loading: {config['ofi_file']}")

        df = pd.read_csv(config['ofi_file'])

        # Create diagnostic plots
        diag_stats = create_diagnostic_plots(df, market_name, config['name_short'])

        # Run diagnostic tests
        test_results = run_diagnostic_tests(df, market_name)

        if diag_stats and test_results:
            all_diagnostics.append({
                'market': market_name,
                'market_short': config['name_short'],
                **diag_stats,
                **test_results
            })

        print()

    # Summary
    print(f"\n{'='*80}")
    print("DIAGNOSTICS SUMMARY")
    print(f"{'='*80}\n")

    summary_df = pd.DataFrame(all_diagnostics)
    print(summary_df.to_string(index=False))

    # Save summary
    output_file = OUTPUT_DIR.parent / "analysis" / "residual_diagnostics_summary.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved summary: {output_file}")

    print(f"\n{'='*80}")
    print("✓ RESIDUAL DIAGNOSTICS COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
