"""
07_time_to_expiry_analysis.py
==============================
Analyze how OFI-price relationship changes as market approaches expiry

Research Question:
How does the slope (β) of the OFI regression change in the final hours
before market resolution compared to earlier periods?

Hypothesis:
- Price impact might be higher/lower near expiry
- Liquidity dynamics change as resolution approaches
- Information incorporation may be different

Analysis:
1. Segment data by time-to-expiry (e.g., last 2h, 6h, 24h, etc.)
2. Run OFI regression for each segment
3. Compare β, R², and other metrics across segments
4. Visualize how relationship evolves

Usage:
    python scripts/07_time_to_expiry_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Create directories
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Market data files and expiry times
MARKET_FILES = {
    'NYC': 'nyc_mayor_oct15_nov04_ofi.csv',
    'Fed': 'ofi_oct15_nov20_combined.csv'
}

# Market expiry times (only for markets that have actually closed)
EXPIRY_TIMES = {
    'NYC': pd.Timestamp('2025-11-05 00:00:00', tz='UTC')   # Election night (Nov 4 11:59pm)
    # Note: Fed market not included as it hasn't closed yet
}

# Time bins (hours before expiry)
TIME_BINS = [
    (0, 2, "Last 2 hours"),
    (2, 6, "2-6 hours before"),
    (6, 24, "6-24 hours before"),
    (24, 72, "1-3 days before"),
    (72, 168, "3-7 days before"),
    (168, float('inf'), "More than 7 days before")
]


def load_ofi_data(market_name):
    """Load OFI data for a market"""
    if market_name not in MARKET_FILES:
        raise ValueError(f"Unknown market: {market_name}")

    ofi_file = DATA_DIR / MARKET_FILES[market_name]

    if not ofi_file.exists():
        raise FileNotFoundError(f"OFI data not found: {ofi_file}")

    df = pd.read_csv(ofi_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

    # Rename columns if needed for consistency
    if 'delta_mid_price' in df.columns and 'price_change' not in df.columns:
        df['price_change'] = df['delta_mid_price']

    return df


def calculate_time_to_expiry(df, expiry_time):
    """Calculate time to expiry in hours for each observation"""
    df = df.copy()
    df['time_to_expiry_hours'] = (expiry_time - df['timestamp']).dt.total_seconds() / 3600
    return df


def segment_by_time_to_expiry(df):
    """Segment data into time bins"""
    segments = {}

    for start_h, end_h, label in TIME_BINS:
        mask = (df['time_to_expiry_hours'] >= start_h) & (df['time_to_expiry_hours'] < end_h)
        segment_df = df[mask].copy()

        if len(segment_df) > 30:  # Minimum observations for regression
            segments[label] = {
                'df': segment_df,
                'start_hours': start_h,
                'end_hours': end_h,
                'n_obs': len(segment_df)
            }

    return segments


def run_segment_regression(segment_df):
    """Run OFI regression for a segment"""
    # Prepare data
    df = segment_df.dropna(subset=['ofi', 'price_change'])

    if len(df) < 30:
        return None

    X = df['ofi'].values.reshape(-1, 1)
    y = df['price_change'].values

    # Add constant
    X_with_const = sm.add_constant(X)

    # Run regression with robust standard errors
    model = sm.OLS(y, X_with_const)
    results = model.fit(cov_type='HC0')  # White's robust standard errors

    # Calculate average depth (handle different column names)
    if 'bid_size' in df.columns:
        avg_depth = (df['bid_size'] + df['ask_size']).mean() / 2
    elif 'best_bid_size' in df.columns:
        avg_depth = (df['best_bid_size'] + df['best_ask_size']).mean() / 2
    else:
        avg_depth = df['total_depth'].mean() / 2 if 'total_depth' in df.columns else 0

    # Extract statistics
    stats_dict = {
        'n_obs': len(df),
        'beta': results.params[1],
        'alpha': results.params[0],
        'r_squared': results.rsquared,
        'r_squared_adj': results.rsquared_adj,
        'beta_se': results.bse[1],
        'beta_tstat': results.tvalues[1],
        'beta_pvalue': results.pvalues[1],
        'alpha_pvalue': results.pvalues[0],
        'correlation': np.corrcoef(df['ofi'], df['price_change'])[0, 1],
        'avg_price': df['mid_price'].mean(),
        'avg_spread': df['spread'].mean(),
        'avg_depth': avg_depth,
        'price_volatility': df['price_change'].std(),
        'ofi_volatility': df['ofi'].std()
    }

    return stats_dict


def analyze_market(market_name, expiry_time):
    """Complete time-to-expiry analysis for a market"""
    print(f"\n{'='*80}")
    print(f"TIME-TO-EXPIRY ANALYSIS: {market_name}")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading data for {market_name}...")
    df = load_ofi_data(market_name)
    print(f"  Total observations: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Market expiry: {expiry_time}")

    # Calculate time to expiry
    df = calculate_time_to_expiry(df, expiry_time)
    print(f"  Time to expiry range: {df['time_to_expiry_hours'].min():.1f}h to {df['time_to_expiry_hours'].max():.1f}h")

    # Segment by time to expiry
    print(f"\nSegmenting by time to expiry...")
    segments = segment_by_time_to_expiry(df)
    print(f"  Found {len(segments)} valid time segments")

    # Run regression for each segment
    results = []
    for label, segment_info in segments.items():
        print(f"\n  Analyzing: {label}")
        print(f"    Observations: {segment_info['n_obs']:,}")
        print(f"    Time range: {segment_info['start_hours']:.0f}h - {segment_info['end_hours']:.0f}h before expiry")

        stats = run_segment_regression(segment_info['df'])

        if stats:
            stats['segment'] = label
            stats['start_hours'] = segment_info['start_hours']
            stats['end_hours'] = segment_info['end_hours']
            results.append(stats)

            print(f"    β = {stats['beta']:.2e} (p={stats['beta_pvalue']:.4f})")
            print(f"    R² = {stats['r_squared']:.4f}")
            print(f"    Correlation = {stats['correlation']:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by time to expiry (closest to expiry first)
    results_df = results_df.sort_values('start_hours')

    return results_df, df


def create_visualization(results_df, market_name):
    """Create comprehensive visualization of results"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Beta evolution
    ax1 = fig.add_subplot(gs[0, :])
    x = results_df['start_hours'].values
    y = results_df['beta'].values
    yerr = results_df['beta_se'].values

    ax1.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Hours Before Expiry', fontsize=12, fontweight='bold')
    ax1.set_ylabel('β (Price Impact Coefficient)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{market_name}: How OFI Price Impact Changes Near Expiry', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Most recent on left

    # Add segment labels
    for idx, row in results_df.iterrows():
        ax1.annotate(f"n={row['n_obs']:,}",
                    (row['start_hours'], row['beta']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Panel 2: R² evolution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results_df['start_hours'], results_df['r_squared'], marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax2.set_title('Explanatory Power Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Panel 3: Correlation evolution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(results_df['start_hours'], results_df['correlation'], marker='o', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Correlation (OFI, ΔP)', fontsize=11, fontweight='bold')
    ax3.set_title('OFI-Price Correlation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Panel 4: Average depth evolution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(results_df['start_hours'], results_df['avg_depth'], marker='o', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Depth', fontsize=11, fontweight='bold')
    ax4.set_title('Liquidity Near Expiry', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()

    # Panel 5: Price volatility
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(results_df['start_hours'], results_df['price_volatility'], marker='o', linewidth=2, markersize=8, color='red')
    ax5.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Price Volatility (std)', fontsize=11, fontweight='bold')
    ax5.set_title('Price Volatility Near Expiry', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.invert_xaxis()

    # Panel 6: OFI volatility
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(results_df['start_hours'], results_df['ofi_volatility'], marker='o', linewidth=2, markersize=8, color='orange')
    ax6.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
    ax6.set_ylabel('OFI Volatility (std)', fontsize=11, fontweight='bold')
    ax6.set_title('Order Flow Volatility', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.invert_xaxis()

    # Panel 7: Spread evolution
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(results_df['start_hours'], results_df['avg_spread'], marker='o', linewidth=2, markersize=8, color='brown')
    ax7.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Average Spread', fontsize=11, fontweight='bold')
    ax7.set_title('Bid-Ask Spread Near Expiry', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.invert_xaxis()

    plt.suptitle(f'Time-to-Expiry Analysis: {market_name}', fontsize=16, fontweight='bold', y=0.995)

    return fig


def create_comparison_plot(all_results):
    """Create side-by-side comparison of both markets"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for market_name, results_df in all_results.items():
        # Beta comparison
        ax = axes[0, 0] if market_name == 'Fed' else axes[0, 1]
        x = results_df['start_hours'].values
        y = results_df['beta'].values
        yerr = results_df['beta_se'].values

        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
        ax.set_ylabel('β (Price Impact)', fontsize=11, fontweight='bold')
        ax.set_title(f'{market_name}: β Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # R² comparison
        ax = axes[1, 0] if market_name == 'Fed' else axes[1, 1]
        ax.plot(results_df['start_hours'], results_df['r_squared'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Hours Before Expiry', fontsize=11, fontweight='bold')
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title(f'{market_name}: R² Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.suptitle('OFI Price Impact Near Market Expiry: Multi-Market Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def print_summary_stats(results_df, market_name):
    """Print summary statistics comparing early vs late periods"""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {market_name}")
    print(f"{'='*80}\n")

    # Compare last 2 hours vs rest
    last_2h = results_df[results_df['segment'] == 'Last 2 hours']
    rest = results_df[results_df['segment'] != 'Last 2 hours']

    if len(last_2h) > 0 and len(rest) > 0:
        print("LAST 2 HOURS vs REST OF TIME:")
        print(f"  β (last 2h):  {last_2h['beta'].values[0]:.2e}")
        print(f"  β (rest):     {rest['beta'].mean():.2e}")
        print(f"  β change:     {(last_2h['beta'].values[0] / rest['beta'].mean() - 1) * 100:.1f}%")
        print()
        print(f"  R² (last 2h): {last_2h['r_squared'].values[0]:.4f}")
        print(f"  R² (rest):    {rest['r_squared'].mean():.4f}")
        print()
        print(f"  Depth (last 2h): {last_2h['avg_depth'].values[0]:.2f}")
        print(f"  Depth (rest):    {rest['avg_depth'].mean():.2f}")
        print(f"  Depth change:    {(last_2h['avg_depth'].values[0] / rest['avg_depth'].mean() - 1) * 100:.1f}%")
        print()
        print(f"  Vol (last 2h):   {last_2h['price_volatility'].values[0]:.2e}")
        print(f"  Vol (rest):      {rest['price_volatility'].mean():.2e}")
        print(f"  Vol change:      {(last_2h['price_volatility'].values[0] / rest['price_volatility'].mean() - 1) * 100:.1f}%")

    print(f"\n{'-'*80}\n")
    print(results_df[['segment', 'n_obs', 'beta', 'r_squared', 'correlation', 'avg_depth']].to_string(index=False))


def main():
    print("\n" + "="*80)
    print("TIME-TO-EXPIRY ANALYSIS: OFI PRICE IMPACT NEAR MARKET RESOLUTION")
    print("="*80)
    print("\nResearch Question:")
    print("How does the OFI-price relationship change in the final hours before expiry?")
    print("\n" + "="*80 + "\n")

    all_results = {}
    all_data = {}

    # Analyze each market
    for market_name, expiry_time in EXPIRY_TIMES.items():
        try:
            results_df, df = analyze_market(market_name, expiry_time)
            all_results[market_name] = results_df
            all_data[market_name] = df

            # Save results
            output_file = ANALYSIS_DIR / f"{market_name}_time_to_expiry_analysis.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved: {output_file}")

            # Create visualization
            fig = create_visualization(results_df, market_name)

            # Save figure
            for ext in ['png', 'pdf']:
                fig_file = FIGURES_DIR / f"figure_7_{market_name}_time_to_expiry.{ext}"
                fig.savefig(fig_file, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {fig_file}")

            plt.close(fig)

            # Print summary
            print_summary_stats(results_df, market_name)

        except FileNotFoundError as e:
            print(f"⚠ Skipping {market_name}: {e}")
            continue

    # Create comparison plot if we have multiple markets
    if len(all_results) > 1:
        print("\nCreating multi-market comparison...")
        fig = create_comparison_plot(all_results)

        for ext in ['png', 'pdf']:
            fig_file = FIGURES_DIR / f"figure_7_time_to_expiry_comparison.{ext}"
            fig.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {fig_file}")

        plt.close(fig)

    # Create summary table
    summary_rows = []
    for market_name, results_df in all_results.items():
        last_2h = results_df[results_df['segment'] == 'Last 2 hours']
        rest = results_df[results_df['segment'] != 'Last 2 hours']

        if len(last_2h) > 0 and len(rest) > 0:
            summary_rows.append({
                'Market': market_name,
                'Beta_Last2h': last_2h['beta'].values[0],
                'Beta_Rest': rest['beta'].mean(),
                'Beta_Change_Pct': (last_2h['beta'].values[0] / rest['beta'].mean() - 1) * 100,
                'R2_Last2h': last_2h['r_squared'].values[0],
                'R2_Rest': rest['r_squared'].mean(),
                'Depth_Last2h': last_2h['avg_depth'].values[0],
                'Depth_Rest': rest['avg_depth'].mean(),
                'Depth_Change_Pct': (last_2h['avg_depth'].values[0] / rest['avg_depth'].mean() - 1) * 100,
                'Vol_Last2h': last_2h['price_volatility'].values[0],
                'Vol_Rest': rest['price_volatility'].mean(),
                'Vol_Change_Pct': (last_2h['price_volatility'].values[0] / rest['price_volatility'].mean() - 1) * 100
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = TABLES_DIR / "table_6_time_to_expiry_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")

        print("\n" + "="*80)
        print("SUMMARY TABLE: Last 2 Hours vs Rest")
        print("="*80 + "\n")
        print(summary_df.to_string(index=False))

    print("\n" + "="*80)
    print("✓ TIME-TO-EXPIRY ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
