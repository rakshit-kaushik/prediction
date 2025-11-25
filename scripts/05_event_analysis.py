"""
05_event_analysis.py
====================
Event Pattern Analysis and Variance Decomposition

Replicates Section 3 of Cont, Kukanov & Stoikov (2011):
- Analyzes OFI patterns for different orderbook event types
- Variance decomposition: how much price variance is explained by OFI
- Event-specific price impact analysis

Orderbook Events:
- bid_up: Increase in bid side depth
- bid_down: Decrease in bid side depth
- ask_up: Increase in ask side depth
- ask_down: Decrease in ask side depth

Usage:
    python scripts/05_event_analysis.py
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

    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()
    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})
    aggregated = aggregated.drop(columns=['mid_price_first', 'mid_price_last'], errors='ignore')

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


def classify_events(df):
    """
    Classify each observation by event type
    Returns DataFrame with event type columns
    """

    df = df.copy()

    # Create event indicators (already present in data)
    # bid_up, bid_down, ask_up, ask_down are binary columns

    # Classify combined event types
    df['event_type'] = 'no_change'

    df.loc[(df['bid_up'] == 1) & (df['ask_down'] == 0) & (df['ask_up'] == 0), 'event_type'] = 'bid_up_only'
    df.loc[(df['bid_down'] == 1) & (df['ask_down'] == 0) & (df['ask_up'] == 0), 'event_type'] = 'bid_down_only'
    df.loc[(df['ask_up'] == 1) & (df['bid_up'] == 0) & (df['bid_down'] == 0), 'event_type'] = 'ask_up_only'
    df.loc[(df['ask_down'] == 1) & (df['bid_up'] == 0) & (df['bid_down'] == 0), 'event_type'] = 'ask_down_only'

    # Combined events
    df.loc[(df['bid_up'] == 1) & (df['ask_down'] == 1), 'event_type'] = 'bid_up_ask_down'
    df.loc[(df['bid_down'] == 1) & (df['ask_up'] == 1), 'event_type'] = 'bid_down_ask_up'
    df.loc[(df['bid_up'] == 1) & (df['ask_up'] == 1), 'event_type'] = 'both_up'
    df.loc[(df['bid_down'] == 1) & (df['ask_down'] == 1), 'event_type'] = 'both_down'

    return df


def analyze_event_patterns(df, market_name):
    """Analyze OFI and price changes by event type"""

    print(f"\n{'='*80}")
    print(f"EVENT PATTERN ANALYSIS: {market_name}")
    print(f"{'='*80}\n")

    # Classify events
    df = classify_events(df)

    # Event type statistics
    print("Event Type Distribution:")
    event_counts = df['event_type'].value_counts()
    print(event_counts)
    print()

    # Analyze each event type
    event_stats = []

    for event_type in df['event_type'].unique():
        event_df = df[df['event_type'] == event_type]

        if len(event_df) < 10:
            continue

        # Statistics
        mean_ofi = event_df['ofi'].mean()
        std_ofi = event_df['ofi'].std()
        mean_price_change = event_df['delta_mid_price'].mean()
        std_price_change = event_df['delta_mid_price'].std()

        # Regression for this event type
        valid = ~(event_df['delta_mid_price'].isna() | event_df['ofi'].isna())
        if valid.sum() < 10:
            continue

        y = event_df.loc[valid, 'delta_mid_price'].values
        X = event_df.loc[valid, 'ofi'].values

        if len(X) > 0 and np.std(X) > 0:
            # Simple linear regression
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            try:
                beta_hat = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                beta = beta_hat[1]

                # R-squared
                y_pred = X_with_intercept @ beta_hat
                SST = np.sum((y - np.mean(y))**2)
                SSR = np.sum((y_pred - np.mean(y))**2)
                r_squared = SSR / SST if SST > 0 else 0
            except:
                beta = np.nan
                r_squared = np.nan
        else:
            beta = np.nan
            r_squared = np.nan

        event_stats.append({
            'event_type': event_type,
            'count': len(event_df),
            'pct': 100 * len(event_df) / len(df),
            'mean_ofi': mean_ofi,
            'std_ofi': std_ofi,
            'mean_price_change': mean_price_change,
            'std_price_change': std_price_change,
            'beta': beta,
            'r_squared': r_squared
        })

    event_stats_df = pd.DataFrame(event_stats).sort_values('count', ascending=False)

    print("\nEvent Type Statistics:")
    print(event_stats_df.to_string(index=False))
    print()

    return event_stats_df


def variance_decomposition(df, market_name):
    """
    Variance decomposition: how much of price variance is explained by OFI?

    Decomposes price changes into:
    1. Component explained by OFI (fitted values from regression)
    2. Residual component (unexplained variance)
    """

    print(f"\n{'-'*80}")
    print(f"VARIANCE DECOMPOSITION: {market_name}")
    print(f"{'-'*80}\n")

    # Remove NaN
    valid = ~(df['delta_mid_price'].isna() | df['ofi'].isna())
    y = df.loc[valid, 'delta_mid_price'].values
    X = df.loc[valid, 'ofi'].values

    # Regression
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    beta_hat = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

    # Predictions
    y_pred = X_with_intercept @ beta_hat
    residuals = y - y_pred

    # Variance decomposition
    var_total = np.var(y)
    var_explained = np.var(y_pred)
    var_residual = np.var(residuals)

    pct_explained = 100 * var_explained / var_total if var_total > 0 else 0
    pct_residual = 100 * var_residual / var_total if var_total > 0 else 0

    print("Price Change Variance Decomposition:")
    print(f"  Total variance: {var_total:.6e}")
    print(f"  Explained by OFI: {var_explained:.6e} ({pct_explained:.2f}%)")
    print(f"  Residual variance: {var_residual:.6e} ({pct_residual:.2f}%)")
    print()

    # By time of day (if applicable)
    if 'timestamp' in df.columns:
        df_valid = df.loc[valid].copy()
        df_valid['hour'] = pd.to_datetime(df_valid['timestamp']).dt.hour

        hourly_variance = []
        for hour in range(24):
            hour_df = df_valid[df_valid['hour'] == hour]
            if len(hour_df) > 10:
                hourly_variance.append({
                    'hour': hour,
                    'count': len(hour_df),
                    'var_price': np.var(hour_df['delta_mid_price']),
                    'var_ofi': np.var(hour_df['ofi'])
                })

        if len(hourly_variance) > 0:
            hourly_df = pd.DataFrame(hourly_variance)
            print("\nVariance by Hour of Day (Top 5 hours by count):")
            print(hourly_df.sort_values('count', ascending=False).head(5).to_string(index=False))
            print()

    return {
        'var_total': var_total,
        'var_explained': var_explained,
        'var_residual': var_residual,
        'pct_explained': pct_explained,
        'pct_residual': pct_residual
    }


def create_event_plots(all_event_stats, all_variance, market_names):
    """Create event analysis visualizations"""

    n_markets = len(all_event_stats)
    fig, axes = plt.subplots(2, n_markets, figsize=(7*n_markets, 12))

    if n_markets == 1:
        axes = axes.reshape(-1, 1)

    for idx, (event_stats_df, variance_result, market_name) in enumerate(zip(all_event_stats, all_variance, market_names)):

        # ----------------------------------------------------------------
        # Plot 1: Event Type Distribution
        # ----------------------------------------------------------------
        ax1 = axes[0, idx]

        # Get top 10 event types by count
        top_events = event_stats_df.head(10).sort_values('count')

        ax1.barh(range(len(top_events)), top_events['count'], color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(top_events)))
        ax1.set_yticklabels(top_events['event_type'], fontsize=9)
        ax1.set_xlabel('Count', fontweight='bold')
        ax1.set_title(f'{market_name}\nEvent Type Distribution (Top 10)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # ----------------------------------------------------------------
        # Plot 2: Price Impact by Event Type
        # ----------------------------------------------------------------
        ax2 = axes[1, idx]

        # Filter events with valid beta
        beta_events = event_stats_df[event_stats_df['beta'].notna()].head(10)

        if len(beta_events) > 0:
            beta_events_sorted = beta_events.sort_values('beta')

            colors = ['red' if b < 0 else 'green' for b in beta_events_sorted['beta']]

            ax2.barh(range(len(beta_events_sorted)), beta_events_sorted['beta'],
                    color=colors, alpha=0.7)
            ax2.set_yticks(range(len(beta_events_sorted)))
            ax2.set_yticklabels(beta_events_sorted['event_type'], fontsize=9)
            ax2.set_xlabel('β (Price Impact Coefficient)', fontweight='bold')
            ax2.set_title(f'{market_name}\nPrice Impact by Event Type', fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='x')

        # Add variance decomposition text
        textstr = f'Variance Explained by OFI: {variance_result["pct_explained"]:.2f}%\n'
        textstr += f'Residual Variance: {variance_result["pct_residual"]:.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle('Event Pattern Analysis\nOrderbook Event Types and Price Impact',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    png_file = OUTPUT_DIR / "figures" / "figure_5_event_analysis.png"
    pdf_file = OUTPUT_DIR / "figures" / "figure_5_event_analysis.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_file, bbox_inches='tight')

    print(f"\n✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")

    plt.close()


def main():
    print("\n" + "="*80)
    print("EVENT PATTERN ANALYSIS AND VARIANCE DECOMPOSITION")
    print("="*80)
    print("\nAnalyzing OFI patterns across different orderbook event types")
    print("Variance decomposition: price variance explained by OFI\n")

    all_event_stats = []
    all_variance = []
    market_names = []

    for market_name, config in MARKETS.items():
        print(f"\nProcessing: {market_name}")
        print(f"Loading: {config['ofi_file']}")

        df = pd.read_csv(config['ofi_file'])
        df = aggregate_to_10_seconds(df)

        # Convert timestamp if needed
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
            except:
                pass

        # Event analysis
        event_stats = analyze_event_patterns(df, market_name)

        # Variance decomposition
        variance_result = variance_decomposition(df, market_name)

        # Save event stats
        output_file = OUTPUT_DIR / "analysis" / f"{config['name_short']}_event_statistics.csv"
        event_stats.to_csv(output_file, index=False)
        print(f"✓ Saved event statistics: {output_file}")

        all_event_stats.append(event_stats)
        all_variance.append(variance_result)
        market_names.append(config['name_short'])

    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING EVENT ANALYSIS PLOTS")
    print(f"{'='*80}")
    create_event_plots(all_event_stats, all_variance, market_names)

    # Summary table
    print(f"\n{'='*80}")
    print("VARIANCE DECOMPOSITION SUMMARY")
    print(f"{'='*80}\n")

    variance_df = pd.DataFrame([
        {
            'Market': name,
            'Variance_Total': var['var_total'],
            'Variance_Explained': var['var_explained'],
            'Variance_Residual': var['var_residual'],
            'Pct_Explained': var['pct_explained'],
            'Pct_Residual': var['pct_residual']
        }
        for name, var in zip(market_names, all_variance)
    ])

    print(variance_df.to_string(index=False))

    output_file = OUTPUT_DIR / "tables" / "table_4_variance_decomposition.csv"
    variance_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    print(f"\n{'='*80}")
    print("✓ EVENT ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
