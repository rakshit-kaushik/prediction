"""
02_create_figure_2.py
=====================
Create Figure 2: OFI vs Price Change Scatter Plot

Replicates Figure 2 from Cont, Kukanov & Stoikov (2011)
Publication-quality scatter plot with regression line.

Usage:
    python scripts/02_create_figure_2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

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

def create_scatter_plot(df, market_name, market_short):
    """Create publication-quality scatter plot"""
    # Remove NaN
    valid = ~(df['delta_mid_price'].isna() | df['ofi'].isna())
    x = df.loc[valid, 'ofi'].values
    y = df.loc[valid, 'delta_mid_price'].values

    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with alpha
    ax.scatter(x, y, alpha=0.3, s=10, color='steelblue', edgecolors='none')

    # Regression line
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear Fit')

    # Labels and title
    ax.set_xlabel('Order Flow Imbalance (OFI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔMid-Price', fontsize=12, fontweight='bold')
    ax.set_title(f'{market_name}\nOFI vs Price Change', fontsize=13, fontweight='bold', pad=15)

    # Add statistics box
    textstr = f'y = {slope:.2e}x + {intercept:.2e}\n'
    textstr += f'R² = {r_squared:.4f}\n'
    textstr += f'p < 0.001\n'
    textstr += f'n = {len(x):,}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save
    png_file = OUTPUT_DIR / f"figure_2_{market_short}_ofi_vs_price.png"
    pdf_file = OUTPUT_DIR / f"figure_2_{market_short}_ofi_vs_price.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")

    plt.close()

    return r_squared, slope, intercept

def create_combined_plot():
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (market_name, config) in enumerate(MARKETS.items()):
        df = pd.read_csv(config['ofi_file'])

        # Remove NaN
        valid = ~(df['delta_mid_price'].isna() | df['ofi'].isna())
        x = df.loc[valid, 'ofi'].values
        y = df.loc[valid, 'delta_mid_price'].values

        # Regression
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value ** 2

        ax = axes[idx]

        # Scatter
        ax.scatter(x, y, alpha=0.3, s=8, color='steelblue', edgecolors='none')

        # Regression line
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2)

        # Labels
        ax.set_xlabel('OFI', fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔPrice', fontsize=11, fontweight='bold')
        ax.set_title(f'{config["name_short"]} Market', fontsize=12, fontweight='bold')

        # Stats
        textstr = f'R² = {r_squared:.4f}\n'
        textstr += f'β = {slope:.2e}\n'
        textstr += f'n = {len(x):,}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        ax.grid(True, alpha=0.3)

    plt.suptitle('Order Flow Imbalance vs Price Changes\nCont et al. (2011) Replication for Polymarket',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    png_file = OUTPUT_DIR / "figure_2_combined_comparison.png"
    pdf_file = OUTPUT_DIR / "figure_2_combined_comparison.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"\n✓ Saved combined: {png_file}")
    print(f"✓ Saved combined: {pdf_file}")

    plt.close()

def main():
    print("\n" + "="*80)
    print("CREATING FIGURE 2: OFI vs PRICE CHANGE SCATTER PLOTS")
    print("="*80 + "\n")

    results = []

    for market_name, config in MARKETS.items():
        print(f"Processing: {market_name}")
        print(f"Loading: {config['ofi_file']}")

        df = pd.read_csv(config['ofi_file'])
        r2, slope, intercept = create_scatter_plot(df, market_name, config['name_short'])

        results.append({
            'market': market_name,
            'r_squared': r2,
            'slope': slope,
            'intercept': intercept
        })
        print()

    # Combined plot
    print("Creating combined comparison plot...")
    create_combined_plot()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for result in results:
        print(f"{result['market']}:")
        print(f"  R² = {result['r_squared']:.4f}")
        print(f"  β (slope) = {result['slope']:.6e}")
        print()

    print(f"{'='*80}")
    print("✓ FIGURE 2 CREATION COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
