"""
07_three_phase_analysis.py
===========================
Three-Phase Beta Analysis for NYC Mayoral Election Market

Divides the NYC market timeline into 3 phases and analyzes how the OFI-price
relationship (β coefficient) changes across phases:
- Phase 1: Early period (first 1/3 of timeline)
- Phase 2: Middle period (middle 1/3)
- Phase 3: Near expiry (final 1/3)

This answers: Does the price impact of order flow change as the market
approaches resolution?

Usage:
    python scripts/07_three_phase_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Import configuration
from config_analysis import (
    TIME_WINDOW,
    USE_TICK_NORMALIZED,
    N_PHASES,
    MIN_OBS_PER_PHASE,
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

    Time window controlled by TIME_WINDOW in config_analysis.py
    """
    if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    else:
        raise ValueError("No timestamp column found")

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_bin'] = df['timestamp'].dt.floor(TIME_WINDOW)

    agg_dict = {
        'ofi': 'sum',
        'mid_price': ['first', 'last'],
        'total_depth': 'mean',
        'spread': 'mean',
        'spread_pct': 'mean',
        'best_bid_price': 'last',
        'best_ask_price': 'last',
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
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[0] in ['mid_price']:
                new_cols.append('_'.join(col))
            elif col[1]:
                new_cols.append(col[0])
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()

    # Calculate tick-normalized price change if needed
    if 'delta_mid_price_ticks' in df.columns:
        aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / 0.01  # TICK_SIZE

    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})
    aggregated = aggregated.drop(columns=['mid_price_first', 'mid_price_last'], errors='ignore')

    return aggregated


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

# NYC Market data
NYC_OFI_FILE = DATA_DIR / 'nyc_mayor_oct15_nov04_ofi.csv'


def load_nyc_data():
    """Load and aggregate NYC market data"""
    if not NYC_OFI_FILE.exists():
        raise FileNotFoundError(f"NYC OFI data not found: {NYC_OFI_FILE}")

    df = pd.read_csv(NYC_OFI_FILE)
    df = aggregate_ofi_data(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

    return df


def divide_into_three_phases(df):
    """
    Divide data into 3 equal time phases:
    - Phase 1: Early period (first 1/3)
    - Phase 2: Middle period (middle 1/3)
    - Phase 3: Near expiry (final 1/3)
    """
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    total_duration = (max_time - min_time).total_seconds()

    # Calculate phase boundaries
    phase_duration = total_duration / 3
    phase1_end = min_time + pd.Timedelta(seconds=phase_duration)
    phase2_end = min_time + pd.Timedelta(seconds=2 * phase_duration)

    # Create phase labels
    df = df.copy()
    df['phase'] = 'Phase 3: Near Expiry'
    df.loc[df['timestamp'] < phase2_end, 'phase'] = 'Phase 2: Middle'
    df.loc[df['timestamp'] < phase1_end, 'phase'] = 'Phase 1: Early'

    phases = {
        'Phase 1: Early': {
            'df': df[df['phase'] == 'Phase 1: Early'].copy(),
            'start': min_time,
            'end': phase1_end,
            'description': f"{min_time.strftime('%b %d')} - {phase1_end.strftime('%b %d')}"
        },
        'Phase 2: Middle': {
            'df': df[df['phase'] == 'Phase 2: Middle'].copy(),
            'start': phase1_end,
            'end': phase2_end,
            'description': f"{phase1_end.strftime('%b %d')} - {phase2_end.strftime('%b %d')}"
        },
        'Phase 3: Near Expiry': {
            'df': df[df['phase'] == 'Phase 3: Near Expiry'].copy(),
            'start': phase2_end,
            'end': max_time,
            'description': f"{phase2_end.strftime('%b %d')} - {max_time.strftime('%b %d')}"
        }
    }

    return phases


def run_phase_regression(phase_df):
    """Run OFI regression for a phase"""
    # Get dependent variable name from config
    dep_var = get_dependent_variable_name()

    # Prepare data
    df = phase_df.dropna(subset=['ofi', dep_var])

    if len(df) < MIN_OBS_PER_PHASE:
        return None

    X = df['ofi'].values.reshape(-1, 1)
    y = df[dep_var].values

    # Add constant
    X_with_const = sm.add_constant(X)

    # Run regression with robust standard errors
    model = sm.OLS(y, X_with_const)
    results = model.fit(cov_type='HC0')  # White's robust standard errors

    # Calculate statistics
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
        'correlation': np.corrcoef(df['ofi'], df[dep_var])[0, 1],
        'avg_price': df['mid_price'].mean(),
        'avg_spread': df['spread'].mean() if 'spread' in df.columns else 0,
        'price_volatility': df[dep_var].std(),
        'ofi_volatility': df['ofi'].std(),
        'ci_lower': results.conf_int()[1, 0],  # 95% CI lower bound for beta
        'ci_upper': results.conf_int()[1, 1]   # 95% CI upper bound for beta
    }

    return stats_dict


def create_visualization(results_df):
    """Create visualization showing beta evolution across 3 phases"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Set style
    sns.set_style("whitegrid")
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

    # -------------------------------------------------------------------------
    # Panel 1: Beta Coefficient by Phase
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    phases = results_df['phase'].values
    betas = results_df['beta'].values
    beta_se = results_df['beta_se'].values

    x_pos = np.arange(len(phases))
    bars = ax.bar(x_pos, betas, yerr=beta_se, capsize=10, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=0, ha='center')
    ax.set_ylabel('β (Price Impact Coefficient)', fontweight='bold', fontsize=11)
    ax.set_title('OFI Price Impact Across Market Phases', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, beta, se) in enumerate(zip(bars, betas, beta_se)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + se,
                f'{beta:.2e}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel 2: R² by Phase
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    r_squared = results_df['r_squared'].values * 100  # Convert to percentage

    bars = ax.bar(x_pos, r_squared, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=0, ha='center')
    ax.set_ylabel('R² (%)', fontweight='bold', fontsize=11)
    ax.set_title('Explanatory Power Across Phases', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, r2 in zip(bars, r_squared):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel 3: Sample Size by Phase
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    n_obs = results_df['n_obs'].values

    bars = ax.bar(x_pos, n_obs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=0, ha='center')
    ax.set_ylabel('Number of 10-Minute Intervals', fontweight='bold', fontsize=11)
    ax.set_title('Sample Size by Phase', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, n in zip(bars, n_obs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(n):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel 4: Summary Statistics Table
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary table
    table_data = []
    for idx, row in results_df.iterrows():
        table_data.append([
            row['phase'],
            f"{row['beta']:.2e}",
            f"{row['beta_pvalue']:.4f}",
            f"{row['r_squared']:.4f}",
            f"{int(row['n_obs']):,}"
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Phase', 'β', 'p-value', 'R²', 'N'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows with phase colors
    for i, color in enumerate(colors, start=1):
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_text_props(weight='bold', color='white')

    ax.set_title('Regression Statistics', fontweight='bold', fontsize=12, pad=20)

    # Overall title
    plt.suptitle('NYC Mayoral Election 2025: Three-Phase OFI Analysis',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    png_file = FIGURES_DIR / "figure_7_three_phase_analysis.png"
    pdf_file = FIGURES_DIR / "figure_7_three_phase_analysis.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")

    plt.close()


def main():
    print("\n" + "="*80)
    print("THREE-PHASE BETA ANALYSIS: NYC MAYORAL ELECTION 2025")
    print("="*80 + "\n")

    # Print configuration
    print_config()
    print()

    # Load data
    print("Loading NYC market data...")
    df = load_nyc_data()
    print(f"  Total {TIME_WINDOW} intervals: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    # Divide into phases
    print("\nDividing into 3 phases...")
    phases = divide_into_three_phases(df)

    # Run regression for each phase
    results = []
    for phase_name, phase_info in phases.items():
        print(f"\n{phase_name}:")
        print(f"  Period: {phase_info['description']}")
        print(f"  Observations: {len(phase_info['df']):,}")

        stats = run_phase_regression(phase_info['df'])

        if stats:
            stats['phase'] = phase_name
            stats['date_range'] = phase_info['description']
            results.append(stats)

            print(f"  β = {stats['beta']:.2e} (SE={stats['beta_se']:.2e})")
            print(f"  p-value = {stats['beta_pvalue']:.4f} {'***' if stats['beta_pvalue'] < 0.001 else '**' if stats['beta_pvalue'] < 0.01 else '*' if stats['beta_pvalue'] < 0.05 else ''}")
            print(f"  R² = {stats['r_squared']:.4f} ({stats['r_squared']*100:.2f}%)")
            print(f"  Correlation = {stats['correlation']:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = TABLES_DIR / "table_7_three_phase_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(results_df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nBeta Evolution:")
    for idx, row in results_df.iterrows():
        phase_num = idx + 1
        print(f"  Phase {phase_num}: β = {row['beta']:.2e}, R² = {row['r_squared']:.4f}")

    # Calculate beta change
    beta_change_pct = ((results_df.iloc[-1]['beta'] - results_df.iloc[0]['beta']) /
                       abs(results_df.iloc[0]['beta']) * 100)
    print(f"\nBeta change from Phase 1 to Phase 3: {beta_change_pct:+.1f}%")

    print("\n" + "="*80)
    print("✓ THREE-PHASE ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
