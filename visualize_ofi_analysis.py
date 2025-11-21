"""
visualize_ofi_analysis.py
=========================
Comprehensive OFI (Order Flow Imbalance) Analysis Visualization

GENERAL & REUSABLE - Works for ANY market and date range
Auto-detects market name, dates, and statistics from data

Based on: Cont, Kukanov & Stoikov (2011) "The Price Impact of Order Book Events"

Usage:
    python visualize_ofi_analysis.py

Input:
    - data/ofi_results.csv (from calculate_ofi.py)
    - data/market_info.json (optional, for market name)

Output:
    - results/ofi_analysis_phase*.png (5 figure sets)
    - results/summary_statistics.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (User can adjust these)
# ============================================================================

# Input files
DATA_FILE = "data/ofi_results.csv"
MARKET_INFO_FILE = "data/market_info.json"

# Output directory
OUTPUT_DIR = "data"

# Plot styling
STYLE = 'whitegrid'
COLOR_BIDS = '#2E86AB'      # Blue for bids
COLOR_ASKS = '#A23B72'      # Red for asks
COLOR_OFI_POS = '#06A77D'   # Green for positive OFI
COLOR_OFI_NEG = '#E63946'   # Red for negative OFI
DPI = 300                    # High resolution
FIGSIZE_LARGE = (16, 12)    # Multi-panel plots
FIGSIZE_MEDIUM = (12, 8)    # Single detailed plot
FIGSIZE_SMALL = (10, 6)     # Simple plots

# Statistical settings
OUTLIER_STD = 3  # Standard deviations for outlier detection
REGRESSION_ALPHA = 0.05  # Significance level


# ============================================================================
# DATA LOADING (Auto-detection)
# ============================================================================

def load_market_name():
    """Load market name from market_info.json if available"""
    try:
        with open(MARKET_INFO_FILE, 'r') as f:
            market_info = json.load(f)
        market_name = market_info.get('question', 'Market')
        # Truncate if too long
        if len(market_name) > 80:
            market_name = market_name[:77] + '...'
        return market_name
    except:
        return "Market"


def load_data():
    """Load OFI results and prepare for analysis"""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load CSV
    print(f"\n📂 Loading: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # Parse timestamps (handle mixed formats)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    # Extract date and hour for grouping
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    # Get market name
    market_name = load_market_name()

    print(f"✅ Loaded {len(df):,} snapshots")
    print(f"   Market: {market_name}")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    return df, market_name


def print_data_summary(df):
    """Print comprehensive data summary"""
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    # Basic stats
    print(f"\n📊 Dataset:")
    print(f"   Total snapshots: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    # OFI stats
    non_zero_ofi = (df['ofi'] != 0).sum()
    print(f"\n📈 Order Flow Imbalance:")
    print(f"   Non-zero OFI: {non_zero_ofi:,} / {len(df):,} ({100*non_zero_ofi/len(df):.1f}%)")
    print(f"   OFI range: {df['ofi'].min():,.2f} to {df['ofi'].max():,.2f}")
    print(f"   Mean OFI: {df['ofi'].mean():,.2f}")
    print(f"   Std OFI: {df['ofi'].std():,.2f}")

    # Price change stats
    non_zero_dp = (df['delta_mid_price'] != 0).sum()
    print(f"\n💰 Price Changes:")
    print(f"   Non-zero ΔP: {non_zero_dp:,} / {len(df):,} ({100*non_zero_dp/len(df):.1f}%)")
    print(f"   ΔP range: {df['delta_mid_price'].min():.6f} to {df['delta_mid_price'].max():.6f}")
    print(f"   Mean ΔP: {df['delta_mid_price'].mean():.6f}")

    # Market microstructure
    print(f"\n📚 Market Microstructure:")
    print(f"   Avg spread: {df['spread_pct'].mean():.2f}%")
    print(f"   Avg depth: {df['total_depth'].mean():,.0f}")
    print(f"   Price range: ${df['mid_price'].min():.4f} to ${df['mid_price'].max():.4f}")


# ============================================================================
# PHASE 1: DATA QUALITY
# ============================================================================

def plot_phase1_quality(df, market_name):
    """Phase 1: Data quality and coverage analysis"""
    print("\n" + "=" * 80)
    print("PHASE 1: DATA QUALITY")
    print("=" * 80)

    fig = plt.figure(figsize=FIGSIZE_LARGE)
    fig.suptitle(f'{market_name}\nPhase 1: Data Quality & Coverage',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Snapshots per day
    ax1 = plt.subplot(2, 2, 1)
    daily_counts = df.groupby('date').size()
    ax1.bar(range(len(daily_counts)), daily_counts.values, color=COLOR_BIDS, alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Snapshots')
    ax1.set_title('Daily Snapshot Coverage')
    ax1.set_xticks(range(len(daily_counts)))
    ax1.set_xticklabels([str(d) for d in daily_counts.index], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(daily_counts.mean(), color='red', linestyle='--',
                label=f'Mean: {daily_counts.mean():.0f}')
    ax1.legend()

    # 2. Hourly distribution
    ax2 = plt.subplot(2, 2, 2)
    hourly_counts = df.groupby('hour').size()
    ax2.bar(hourly_counts.index, hourly_counts.values, color=COLOR_OFI_POS, alpha=0.7)
    ax2.set_xlabel('Hour of Day (UTC)')
    ax2.set_ylabel('Number of Snapshots')
    ax2.set_title('Hourly Snapshot Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))

    # 3. Data statistics table
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')

    stats_data = [
        ['Metric', 'Value'],
        ['Total Snapshots', f'{len(df):,}'],
        ['Date Range', f'{df["timestamp"].min().date()} to {df["timestamp"].max().date()}'],
        ['Duration (days)', f'{(df["timestamp"].max() - df["timestamp"].min()).days}'],
        ['Avg Snapshots/Day', f'{len(df) / len(df["date"].unique()):.1f}'],
        ['Non-zero OFI %', f'{100 * (df["ofi"] != 0).sum() / len(df):.1f}%'],
        ['Non-zero ΔP %', f'{100 * (df["delta_mid_price"] != 0).sum() / len(df):.1f}%'],
        ['Avg Spread', f'{df["spread_pct"].mean():.2f}%'],
    ]

    table = ax3.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title('Dataset Statistics', fontsize=12, fontweight='bold', pad=20)

    # 4. Time between snapshots
    ax4 = plt.subplot(2, 2, 4)
    if 'time_diff' in df.columns:
        time_diffs = df['time_diff'].dropna()
        time_diffs_cleaned = time_diffs[time_diffs < time_diffs.quantile(0.99)]  # Remove outliers
        ax4.hist(time_diffs_cleaned, bins=50, color=COLOR_ASKS, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Time Between Snapshots (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Snapshot Interval Distribution')
        ax4.axvline(time_diffs.median(), color='red', linestyle='--',
                   label=f'Median: {time_diffs.median():.1f}s')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Time diff data not available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')

    plt.tight_layout()

    # Save
    output_file = f'{OUTPUT_DIR}/ofi_analysis_phase1_quality.png'
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


# ============================================================================
# PHASE 2: MARKET OVERVIEW
# ============================================================================

def plot_phase2_overview(df, market_name):
    """Phase 2: Market microstructure overview"""
    print("\n" + "=" * 80)
    print("PHASE 2: MARKET OVERVIEW")
    print("=" * 80)

    fig = plt.figure(figsize=FIGSIZE_LARGE)
    fig.suptitle(f'{market_name}\nPhase 2: Market Overview',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Bid-Ask Spread over time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['timestamp'], df['spread_pct'], color=COLOR_ASKS, alpha=0.6, linewidth=0.5)
    ax1.axhline(df['spread_pct'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["spread_pct"].mean():.2f}%')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Spread (%)')
    ax1.set_title('Bid-Ask Spread Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Market depth time series
    ax2 = plt.subplot(2, 2, 2)
    ax2_twin = ax2.twinx()

    ax2.plot(df['timestamp'], df['total_bid_size'], color=COLOR_BIDS,
             alpha=0.6, linewidth=0.8, label='Bid Depth')
    ax2_twin.plot(df['timestamp'], df['total_ask_size'], color=COLOR_ASKS,
                  alpha=0.6, linewidth=0.8, label='Ask Depth')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Bid Depth', color=COLOR_BIDS)
    ax2_twin.set_ylabel('Ask Depth', color=COLOR_ASKS)
    ax2.set_title('Market Depth Over Time')
    ax2.tick_params(axis='y', labelcolor=COLOR_BIDS)
    ax2_twin.tick_params(axis='y', labelcolor=COLOR_ASKS)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Mid-price evolution with volume
    ax3 = plt.subplot(2, 2, 3)
    ax3_twin = ax3.twinx()

    ax3.plot(df['timestamp'], df['mid_price'], color='black',
             linewidth=1.5, label='Mid-Price')
    ax3_twin.bar(df['timestamp'], df['total_depth'], color=COLOR_BIDS,
                 alpha=0.3, width=0.01, label='Total Depth')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mid-Price ($)', color='black')
    ax3_twin.set_ylabel('Total Depth', color=COLOR_BIDS)
    ax3.set_title('Mid-Price Evolution with Total Depth')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Summary distributions (2x2 mini grid)
    ax4 = plt.subplot(2, 2, 4)

    # Create 2x2 mini subplots within ax4
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=ax4.get_subplotspec(),
                                  hspace=0.4, wspace=0.4)

    # Spread distribution
    ax4_1 = fig.add_subplot(gs[0, 0])
    ax4_1.hist(df['spread_pct'], bins=30, color=COLOR_ASKS, alpha=0.7, edgecolor='black')
    ax4_1.set_xlabel('Spread (%)', fontsize=8)
    ax4_1.set_ylabel('Frequency', fontsize=8)
    ax4_1.set_title('Spread Distribution', fontsize=9)
    ax4_1.tick_params(labelsize=7)

    # Price distribution
    ax4_2 = fig.add_subplot(gs[0, 1])
    ax4_2.hist(df['mid_price'], bins=30, color='gray', alpha=0.7, edgecolor='black')
    ax4_2.set_xlabel('Mid-Price ($)', fontsize=8)
    ax4_2.set_ylabel('Frequency', fontsize=8)
    ax4_2.set_title('Price Distribution', fontsize=9)
    ax4_2.tick_params(labelsize=7)

    # Depth distribution
    ax4_3 = fig.add_subplot(gs[1, 0])
    depth_clean = df['total_depth'][df['total_depth'] < df['total_depth'].quantile(0.95)]
    ax4_3.hist(depth_clean, bins=30, color=COLOR_BIDS, alpha=0.7, edgecolor='black')
    ax4_3.set_xlabel('Total Depth', fontsize=8)
    ax4_3.set_ylabel('Frequency', fontsize=8)
    ax4_3.set_title('Depth Distribution', fontsize=9)
    ax4_3.tick_params(labelsize=7)

    # Summary stats
    ax4_4 = fig.add_subplot(gs[1, 1])
    ax4_4.axis('off')
    summary_text = f"""Summary Stats:

Spread: {df['spread_pct'].mean():.2f}%
Price: ${df['mid_price'].mean():.4f}
Depth: {df['total_depth'].mean():,.0f}
Unique Prices: {df['mid_price'].nunique()}
"""
    ax4_4.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
               family='monospace')

    ax4.axis('off')  # Hide the parent ax4

    plt.tight_layout()

    # Save
    output_file = f'{OUTPUT_DIR}/ofi_analysis_phase2_overview.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


# ============================================================================
# PHASE 3: OFI CORE ANALYSIS (MOST IMPORTANT)
# ============================================================================

def plot_phase3_ofi_analysis(df, market_name):
    """Phase 3: Core OFI analysis - KEY RESULTS"""
    print("\n" + "=" * 80)
    print("PHASE 3: OFI CORE ANALYSIS (KEY RESULTS)")
    print("=" * 80)

    fig = plt.figure(figsize=FIGSIZE_LARGE)
    fig.suptitle(f'{market_name}\nPhase 3: Order Flow Imbalance Analysis (Cont et al. 2011)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. OFI vs ΔP Scatter - THE KEY PLOT
    ax1 = plt.subplot(2, 2, 1)

    # Filter for non-zero OFI for clarity
    df_nonzero = df[(df['ofi'] != 0) | (df['delta_mid_price'] != 0)].copy()

    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_nonzero['ofi'], df_nonzero['delta_mid_price']
    )

    # Scatter plot with color gradient
    scatter = ax1.scatter(df_nonzero['ofi'], df_nonzero['delta_mid_price'],
                         c=np.abs(df_nonzero['ofi']), cmap='viridis',
                         alpha=0.6, s=20, edgecolors='none')

    # Regression line
    x_line = np.array([df_nonzero['ofi'].min(), df_nonzero['ofi'].max()])
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Fit')

    # Add text box with statistics
    textstr = f'ΔP = {slope:.2e} × OFI + {intercept:.6f}\n'
    textstr += f'R² = {r_value**2:.4f}\n'
    textstr += f'p-value = {p_value:.2e}\n'
    textstr += f'β (slope) = {slope:.2e}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    ax1.set_xlabel('Order Flow Imbalance (OFI)')
    ax1.set_ylabel('Mid-Price Change (ΔP)')
    ax1.set_title('OFI vs Price Change (Figure 2 from Cont et al.)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('|OFI| Magnitude', rotation=270, labelpad=15)

    print(f"\n🔑 KEY RESULT - OFI vs ΔP Linear Regression:")
    print(f"   Slope (β): {slope:.6e}")
    print(f"   R²: {r_value**2:.4f}")
    print(f"   p-value: {p_value:.2e}")
    print(f"   Interpretation: {'Significant' if p_value < REGRESSION_ALPHA else 'Not significant'} relationship")

    # 2. OFI and ΔP Time Series (aligned)
    ax2 = plt.subplot(2, 2, 2)
    ax2_twin = ax2.twinx()

    # Only plot non-zero for clarity
    df_plot = df[df['ofi'] != 0].copy()

    ax2.scatter(df_plot['timestamp'], df_plot['ofi'],
               c=df_plot['ofi'], cmap='RdYlGn', s=10, alpha=0.6)
    ax2_twin.scatter(df_plot['timestamp'], df_plot['delta_mid_price'],
                    color='black', s=5, alpha=0.4, marker='x')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('OFI', color=COLOR_OFI_POS)
    ax2_twin.set_ylabel('ΔP', color='black')
    ax2.set_title('OFI and Price Changes Over Time')
    ax2.tick_params(axis='y', labelcolor=COLOR_OFI_POS)
    ax2_twin.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. OFI Distribution
    ax3 = plt.subplot(2, 2, 3)

    # Separate positive and negative
    ofi_pos = df[df['ofi'] > 0]['ofi']
    ofi_neg = df[df['ofi'] < 0]['ofi']

    ax3.hist(ofi_pos, bins=50, color=COLOR_OFI_POS, alpha=0.7,
            label=f'Positive OFI ({len(ofi_pos)})', edgecolor='black')
    ax3.hist(ofi_neg, bins=50, color=COLOR_OFI_NEG, alpha=0.7,
            label=f'Negative OFI ({len(ofi_neg)})', edgecolor='black')

    ax3.axvline(df['ofi'].mean(), color='black', linestyle='--',
               label=f'Mean: {df["ofi"].mean():.2f}')
    ax3.axvline(df['ofi'].median(), color='blue', linestyle='--',
               label=f'Median: {df["ofi"].median():.2f}')

    ax3.set_xlabel('Order Flow Imbalance (OFI)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('OFI Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Correlation matrix
    ax4 = plt.subplot(2, 2, 4)

    # Calculate correlations
    corr_cols = ['ofi', 'delta_mid_price', 'delta_best_bid', 'delta_best_ask',
                 'spread_pct', 'total_depth']
    corr_cols = [col for col in corr_cols if col in df.columns]

    corr_matrix = df[corr_cols].corr()

    # Heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Correlation Matrix')

    # Highlight OFI vs ΔP correlation
    ofi_dp_corr = df['ofi'].corr(df['delta_mid_price'])
    print(f"\n📊 Correlations:")
    print(f"   OFI vs ΔP: {ofi_dp_corr:.4f}")

    plt.tight_layout()

    # Save
    output_file = f'{OUTPUT_DIR}/ofi_analysis_phase3_core.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


# ============================================================================
# PHASE 4: ORDERBOOK DETAIL
# ============================================================================

def plot_phase4_orderbook(df, market_name):
    """Phase 4: Detailed orderbook visualization"""
    print("\n" + "=" * 80)
    print("PHASE 4: ORDERBOOK DETAIL")
    print("=" * 80)

    fig = plt.figure(figsize=FIGSIZE_LARGE)
    fig.suptitle(f'{market_name}\nPhase 4: Orderbook Detail',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Daily opening snapshots - Best bid/ask
    ax1 = plt.subplot(2, 2, (1, 2))

    # Get first snapshot of each day
    daily_open = df.groupby('date').first().reset_index()
    dates = [str(d) for d in daily_open['date']]

    x = np.arange(len(dates))
    width = 0.35

    ax1.bar(x - width/2, daily_open['best_bid_price'], width,
           label='Best Bid', color=COLOR_BIDS, alpha=0.8)
    ax1.bar(x + width/2, daily_open['best_ask_price'], width,
           label='Best Ask', color=COLOR_ASKS, alpha=0.8)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Daily Opening Best Bid/Ask Prices')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add spread annotation
    for i, (bid, ask) in enumerate(zip(daily_open['best_bid_price'],
                                       daily_open['best_ask_price'])):
        spread = ask - bid
        ax1.text(i, max(bid, ask) + 0.01, f'{spread:.3f}',
                ha='center', fontsize=7, color='red')

    # 2. High activity day analysis
    ax2 = plt.subplot(2, 2, 3)

    # Find day with most OFI activity
    daily_ofi_activity = df.groupby('date')['ofi'].apply(
        lambda x: (x != 0).sum()
    )
    high_activity_date = daily_ofi_activity.idxmax()

    df_high_day = df[df['date'] == high_activity_date].copy()

    ax2.plot(df_high_day['timestamp'], df_high_day['mid_price'],
            color='black', linewidth=2, label='Mid-Price')
    ax2.scatter(df_high_day[df_high_day['ofi'] > 0]['timestamp'],
               df_high_day[df_high_day['ofi'] > 0]['mid_price'],
               color=COLOR_OFI_POS, s=30, alpha=0.6, label='Positive OFI')
    ax2.scatter(df_high_day[df_high_day['ofi'] < 0]['timestamp'],
               df_high_day[df_high_day['ofi'] < 0]['mid_price'],
               color=COLOR_OFI_NEG, s=30, alpha=0.6, label='Negative OFI')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mid-Price ($)')
    ax2.set_title(f'High Activity Day: {high_activity_date}\n({len(df_high_day)} snapshots)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    print(f"\n📅 High Activity Day: {high_activity_date}")
    print(f"   Snapshots: {len(df_high_day)}")
    print(f"   Non-zero OFI: {(df_high_day['ofi'] != 0).sum()}")

    # 3. Bid-Ask size evolution
    ax3 = plt.subplot(2, 2, 4)

    ax3.plot(df['timestamp'], df['best_bid_size'],
            color=COLOR_BIDS, alpha=0.6, linewidth=1, label='Best Bid Size')
    ax3.plot(df['timestamp'], df['best_ask_size'],
            color=COLOR_ASKS, alpha=0.6, linewidth=1, label='Best Ask Size')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Size')
    ax3.set_title('Best Bid/Ask Size Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_file = f'{OUTPUT_DIR}/ofi_analysis_phase4_orderbook.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


# ============================================================================
# PHASE 5: INTRADAY PATTERNS
# ============================================================================

def plot_phase5_intraday(df, market_name):
    """Phase 5: Intraday patterns and temporal analysis"""
    print("\n" + "=" * 80)
    print("PHASE 5: INTRADAY PATTERNS")
    print("=" * 80)

    fig = plt.figure(figsize=FIGSIZE_LARGE)
    fig.suptitle(f'{market_name}\nPhase 5: Intraday Patterns',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. OFI by hour of day
    ax1 = plt.subplot(2, 2, 1)

    hourly_ofi = df.groupby('hour')['ofi'].apply(list)
    hours = sorted(hourly_ofi.index)
    ofi_by_hour = [hourly_ofi[h] for h in hours]

    bp = ax1.boxplot(ofi_by_hour, labels=hours, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(COLOR_OFI_POS)
        patch.set_alpha(0.6)

    # Add mean line
    hourly_mean = df.groupby('hour')['ofi'].mean()
    ax1.plot(range(1, len(hours)+1), hourly_mean.values,
            'r-o', linewidth=2, markersize=4, label='Mean')

    ax1.set_xlabel('Hour of Day (UTC)')
    ax1.set_ylabel('OFI')
    ax1.set_title('OFI Distribution by Hour')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Price impact by hour (β coefficient)
    ax2 = plt.subplot(2, 2, 2)

    hourly_beta = []
    hourly_r2 = []
    valid_hours = []

    for hour in hours:
        df_hour = df[df['hour'] == hour]
        df_hour_nz = df_hour[(df_hour['ofi'] != 0) | (df_hour['delta_mid_price'] != 0)]

        if len(df_hour_nz) >= 10:  # Minimum sample size
            slope, _, r_value, _, _ = stats.linregress(
                df_hour_nz['ofi'], df_hour_nz['delta_mid_price']
            )
            hourly_beta.append(slope)
            hourly_r2.append(r_value**2)
            valid_hours.append(hour)

    if len(valid_hours) > 0:
        ax2.bar(valid_hours, hourly_beta, color=COLOR_BIDS, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Hour of Day (UTC)')
        ax2.set_ylabel('β (Price Impact Coefficient)')
        ax2.set_title('Price Impact by Hour')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.grid(True, alpha=0.3)

        # Add R² as text
        for i, (h, r2) in enumerate(zip(valid_hours, hourly_r2)):
            if r2 > 0.1:  # Only show significant R²
                ax2.text(h, hourly_beta[i], f'R²={r2:.2f}',
                        ha='center', fontsize=6, rotation=90)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for hourly analysis',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')

    # 3. Activity heatmap (day × hour)
    ax3 = plt.subplot(2, 2, 3)

    # Create pivot table
    df['day_name'] = df['timestamp'].dt.day_name()
    activity_pivot = df.pivot_table(
        values='timestamp',
        index='day_name',
        columns='hour',
        aggfunc='count',
        fill_value=0
    )

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity_pivot = activity_pivot.reindex([d for d in day_order if d in activity_pivot.index])

    sns.heatmap(activity_pivot, cmap='YlOrRd', annot=False, fmt='d',
                cbar_kws={'label': 'Snapshot Count'}, ax=ax3)
    ax3.set_xlabel('Hour of Day (UTC)')
    ax3.set_ylabel('Day of Week')
    ax3.set_title('Activity Heatmap (Day × Hour)')

    # 4. Cumulative OFI
    ax4 = plt.subplot(2, 2, 4)

    df_sorted = df.sort_values('timestamp')
    cumulative_ofi = df_sorted['ofi'].cumsum()

    ax4.plot(df_sorted['timestamp'], cumulative_ofi,
            color=COLOR_OFI_POS, linewidth=2)
    ax4.fill_between(df_sorted['timestamp'], 0, cumulative_ofi,
                     where=(cumulative_ofi >= 0), color=COLOR_OFI_POS, alpha=0.3)
    ax4.fill_between(df_sorted['timestamp'], 0, cumulative_ofi,
                     where=(cumulative_ofi < 0), color=COLOR_OFI_NEG, alpha=0.3)

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative OFI')
    ax4.set_title('Cumulative Order Flow Imbalance')
    ax4.axhline(0, color='black', linewidth=1)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_file = f'{OUTPUT_DIR}/ofi_analysis_phase5_intraday.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def save_summary_report(df, market_name):
    """Generate text summary report"""
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    # Calculate key statistics
    df_nonzero = df[(df['ofi'] != 0) | (df['delta_mid_price'] != 0)].copy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_nonzero['ofi'], df_nonzero['delta_mid_price']
    )

    ofi_dp_corr = df['ofi'].corr(df['delta_mid_price'])

    report = f"""
{'='*80}
OFI ANALYSIS SUMMARY REPORT
{'='*80}

Market: {market_name}
Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}
Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days
Generated: {pd.Timestamp.now()}

{'='*80}
DATASET OVERVIEW
{'='*80}

Total Snapshots: {len(df):,}
Non-zero OFI Events: {(df['ofi'] != 0).sum():,} ({100*(df['ofi'] != 0).sum()/len(df):.1f}%)
Non-zero Price Changes: {(df['delta_mid_price'] != 0).sum():,} ({100*(df['delta_mid_price'] != 0).sum()/len(df):.1f}%)

Average Snapshots per Day: {len(df) / len(df['date'].unique()):.1f}
Unique Days: {df['date'].nunique()}
Unique Mid-Prices: {df['mid_price'].nunique()}

{'='*80}
ORDER FLOW IMBALANCE (OFI) STATISTICS
{'='*80}

OFI Range: {df['ofi'].min():,.2f} to {df['ofi'].max():,.2f}
Mean OFI: {df['ofi'].mean():,.2f}
Median OFI: {df['ofi'].median():,.2f}
Std Dev: {df['ofi'].std():,.2f}

Positive OFI Events: {(df['ofi'] > 0).sum():,} ({100*(df['ofi'] > 0).sum()/len(df):.1f}%)
Negative OFI Events: {(df['ofi'] < 0).sum():,} ({100*(df['ofi'] < 0).sum()/len(df):.1f}%)
Zero OFI: {(df['ofi'] == 0).sum():,} ({100*(df['ofi'] == 0).sum()/len(df):.1f}%)

{'='*80}
PRICE CHANGE STATISTICS
{'='*80}

ΔP Range: {df['delta_mid_price'].min():.6f} to {df['delta_mid_price'].max():.6f}
Mean ΔP: {df['delta_mid_price'].mean():.6f}
Median ΔP: {df['delta_mid_price'].median():.6f}
Std Dev: {df['delta_mid_price'].std():.6f}

Price Range: ${df['mid_price'].min():.4f} to ${df['mid_price'].max():.4f}
Mean Price: ${df['mid_price'].mean():.4f}

{'='*80}
KEY RESULT: OFI vs PRICE CHANGE RELATIONSHIP
{'='*80}

Linear Regression: ΔP = β × OFI + α

Slope (β): {slope:.6e}
Intercept (α): {intercept:.6f}
R-squared (R²): {r_value**2:.4f}
Correlation: {ofi_dp_corr:.4f}
P-value: {p_value:.2e}
Standard Error: {std_err:.6e}

Interpretation:
- {"SIGNIFICANT" if p_value < REGRESSION_ALPHA else "NOT SIGNIFICANT"} relationship at α={REGRESSION_ALPHA}
- R² of {r_value**2:.4f} means OFI explains {100*r_value**2:.1f}% of price change variance
- {"Positive" if slope > 0 else "Negative"} price impact (β = {slope:.2e})

Comparison to Cont et al. (2011):
- Expected R²: ~50-70% (equity markets)
- Our R²: {100*r_value**2:.1f}%
- {"✓ Within expected range" if 0.5 <= r_value**2 <= 0.7 else "⚠ Outside typical range"}

{'='*80}
MARKET MICROSTRUCTURE
{'='*80}

Bid-Ask Spread:
  Mean: {df['spread_pct'].mean():.2f}%
  Median: {df['spread_pct'].median():.2f}%
  Range: {df['spread_pct'].min():.2f}% to {df['spread_pct'].max():.2f}%

Market Depth:
  Mean Total Depth: {df['total_depth'].mean():,.0f}
  Mean Bid Depth: {df['total_bid_size'].mean():,.0f}
  Mean Ask Depth: {df['total_ask_size'].mean():,.0f}

Best Bid/Ask:
  Bid Range: ${df['best_bid_price'].min():.4f} to ${df['best_bid_price'].max():.4f}
  Ask Range: ${df['best_ask_price'].min():.4f} to ${df['best_ask_price'].max():.4f}

{'='*80}
OUTPUT FILES
{'='*80}

Phase 1: {OUTPUT_DIR}/ofi_analysis_phase1_quality.png
Phase 2: {OUTPUT_DIR}/ofi_analysis_phase2_overview.png
Phase 3: {OUTPUT_DIR}/ofi_analysis_phase3_core.png (KEY RESULTS)
Phase 4: {OUTPUT_DIR}/ofi_analysis_phase4_orderbook.png
Phase 5: {OUTPUT_DIR}/ofi_analysis_phase5_intraday.png

This Report: {OUTPUT_DIR}/summary_statistics.txt

{'='*80}
END OF REPORT
{'='*80}
"""

    # Save report
    output_file = f'{OUTPUT_DIR}/summary_statistics.txt'
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✅ Saved: {output_file}")

    # Print key results to console
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"\n📊 OFI vs ΔP Relationship:")
    print(f"   R² = {r_value**2:.4f} ({100*r_value**2:.1f}% variance explained)")
    print(f"   Correlation = {ofi_dp_corr:.4f}")
    print(f"   β (slope) = {slope:.2e}")
    print(f"   p-value = {p_value:.2e} ({'significant' if p_value < REGRESSION_ALPHA else 'not significant'})")
    print(f"\n💰 Market Summary:")
    print(f"   Average Spread: {df['spread_pct'].mean():.2f}%")
    print(f"   Price Range: ${df['mid_price'].min():.4f} - ${df['mid_price'].max():.4f}")
    print(f"   OFI Activity: {100*(df['ofi'] != 0).sum()/len(df):.1f}% of snapshots")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "OFI VISUALIZATION & ANALYSIS" + " " * 29 + "║")
    print("║" + " " * 18 + "Cont, Kukanov & Stoikov (2011)" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")

    # Set style
    sns.set_style(STYLE)
    plt.rcParams['figure.facecolor'] = 'white'

    # Load data
    df, market_name = load_data()

    # Print summary
    print_data_summary(df)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Generate all visualizations
    plot_phase1_quality(df, market_name)
    plot_phase2_overview(df, market_name)
    plot_phase3_ofi_analysis(df, market_name)  # KEY RESULTS
    plot_phase4_orderbook(df, market_name)
    plot_phase5_intraday(df, market_name)

    # Generate summary report
    save_summary_report(df, market_name)

    # Final summary
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n📂 All outputs saved to: {OUTPUT_DIR}/")
    print(f"\n🔑 Key plot: {OUTPUT_DIR}/ofi_analysis_phase3_core.png")
    print(f"   This contains the main OFI vs ΔP relationship (Figure 2 from paper)")
    print()


if __name__ == "__main__":
    main()
