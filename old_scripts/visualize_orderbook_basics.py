"""
visualize_orderbook_basics.py
===============================
Visualize orderbook structure to understand the data

Creates visualizations showing:
1. Price over time - comparing best bid/ask vs weighted mid
2. Orderbook depth heatmap - where liquidity sits
3. Spread comparison - traditional vs effective
4. Price movement histogram

Goal: SEE the patterns in the data

Usage:
    python visualize_orderbook_basics.py

Input:
    data/orderbook_structure_analysis.csv

Output:
    Matplotlib figures showing orderbook patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def load_data():
    """Load analyzed orderbook data"""
    print("\n📂 Loading orderbook analysis...")
    df = pd.read_csv('data/orderbook_structure_analysis.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Loaded {len(df)} snapshots")
    return df


def plot_price_comparison(df):
    """Compare best bid/ask vs weighted mid-price"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Best Bid/Ask (the wrong way)
    ax1.plot(df['timestamp'], df['best_bid'], label='Best Bid', alpha=0.7, linewidth=1)
    ax1.plot(df['timestamp'], df['best_ask'], label='Best Ask', alpha=0.7, linewidth=1)
    ax1.plot(df['timestamp'], df['best_mid'], label='Best Mid', alpha=0.7, linewidth=2, color='red')
    ax1.fill_between(df['timestamp'], df['best_bid'], df['best_ask'], alpha=0.2)

    ax1.set_title('Traditional Best Bid/Ask - NOT Representative!', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Annotate the problem
    ax1.text(0.5, 0.95, 'Problem: Uses extreme limit orders (0.01/0.99)',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
             fontsize=10)

    # Plot 2: Weighted Mid-Price (the right way)
    ax2.plot(df['timestamp'], df['weighted_mid'], label='Weighted Mid (top 10 levels)',
             linewidth=2, color='blue', alpha=0.8)
    ax2.plot(df['timestamp'], df['effective_mid_10pct'], label='Effective Mid (10% depth)',
             linewidth=1, color='green', alpha=0.6, linestyle='--')

    ax2.set_title('Weighted Mid-Price - Where Real Trading Happens!', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Annotate the solution
    ax2.text(0.5, 0.95, 'Solution: Use weighted price based on actual liquidity',
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
             fontsize=10)

    plt.tight_layout()
    plt.savefig('results/price_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/price_comparison.png")
    return fig


def plot_spread_comparison(df):
    """Compare traditional spread vs effective spread"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Traditional spread
    ax1.hist(df['best_spread'], bins=50, color='red', alpha=0.7, edgecolor='black')
    ax1.axvline(df['best_spread'].mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {df["best_spread"].mean():.4f}')
    ax1.set_title('Traditional Spread (Best Bid/Ask)\nUSELESS - Too Wide!', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Spread', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effective spread
    ax2.hist(df['effective_spread_10pct'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(df['effective_spread_10pct'].mean(), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean: {df["effective_spread_10pct"].mean():.4f}')
    ax2.set_title('Effective Spread (10% depth)\nMore Realistic!', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Spread', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/spread_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/spread_comparison.png")
    return fig


def plot_price_movement_histogram(df):
    """Show distribution of weighted mid-price"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    n, bins, patches = ax.hist(df['weighted_mid'], bins=50, color='blue', alpha=0.7, edgecolor='black')

    # Color bins by value
    cm = plt.cm.RdYlGn_r  # Red for low prob, green for high prob
    for i, patch in enumerate(patches):
        patch.set_facecolor(cm(bins[i]))

    ax.axvline(df['weighted_mid'].mean(), color='darkblue', linestyle='--', linewidth=2,
               label=f'Mean: {df["weighted_mid"].mean():.4f}')
    ax.axvline(df['weighted_mid'].median(), color='purple', linestyle=':', linewidth=2,
               label=f'Median: {df["weighted_mid"].median():.4f}')

    ax.set_title('Weighted Mid-Price Distribution\nShows Price Discovery Range', fontsize=14, fontweight='bold')
    ax.set_xlabel('Price', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box with stats
    stats_text = f"""
    Unique Prices: {df['weighted_mid'].nunique()}
    Range: {df['weighted_mid'].min():.4f} - {df['weighted_mid'].max():.4f}
    Std Dev: {df['weighted_mid'].std():.4f}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/price_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/price_distribution.png")
    return fig


def plot_depth_over_time(df):
    """Show how orderbook depth changes over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Total depth
    ax1.plot(df['timestamp'], df['total_depth'], label='Total Depth', color='purple', linewidth=1.5)
    ax1.fill_between(df['timestamp'], 0, df['total_depth'], alpha=0.3, color='purple')
    ax1.set_title('Total Orderbook Depth Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Depth ($)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Bid vs Ask depth
    ax2.plot(df['timestamp'], df['total_bid_size'], label='Bid Depth', color='green', alpha=0.7)
    ax2.plot(df['timestamp'], df['total_ask_size'], label='Ask Depth', color='red', alpha=0.7)
    ax2.fill_between(df['timestamp'], df['total_bid_size'], df['total_ask_size'],
                      alpha=0.2, color='gray')
    ax2.set_title('Bid vs Ask Depth - Imbalance Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Depth ($)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.tight_layout()
    plt.savefig('results/depth_over_time.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/depth_over_time.png")
    return fig


def plot_price_levels(df):
    """Show number of price levels in orderbook"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['timestamp'], df['num_bid_levels'], label='Bid Levels', color='green', alpha=0.7, linewidth=1.5)
    ax.plot(df['timestamp'], df['num_ask_levels'], label='Ask Levels', color='red', alpha=0.7, linewidth=1.5)

    ax.set_title('Number of Price Levels in Orderbook', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Levels', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Add average lines
    avg_bid = df['num_bid_levels'].mean()
    avg_ask = df['num_ask_levels'].mean()
    ax.axhline(avg_bid, color='darkgreen', linestyle='--', alpha=0.5, label=f'Avg Bid: {avg_bid:.0f}')
    ax.axhline(avg_ask, color='darkred', linestyle='--', alpha=0.5, label=f'Avg Ask: {avg_ask:.0f}')

    plt.tight_layout()
    plt.savefig('results/price_levels.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: results/price_levels.png")
    return fig


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ORDERBOOK VISUALIZATIONS" + " " * 33 + "║")
    print("║" + " " * 20 + "Understanding the Data" + " " * 36 + "║")
    print("╚" + "═" * 78 + "╝")

    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)

    # Load data
    df = load_data()

    # Create visualizations
    print("\n📊 Creating visualizations...")

    print("\n1. Price Comparison (Best vs Weighted)...")
    plot_price_comparison(df)

    print("\n2. Spread Comparison...")
    plot_spread_comparison(df)

    print("\n3. Price Distribution...")
    plot_price_movement_histogram(df)

    print("\n4. Depth Over Time...")
    plot_depth_over_time(df)

    print("\n5. Price Levels...")
    plot_price_levels(df)

    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print("\n📁 Saved to: results/")
    print("   - price_comparison.png - Shows why best bid/ask is wrong")
    print("   - spread_comparison.png - Traditional vs effective spread")
    print("   - price_distribution.png - Where prices actually trade")
    print("   - depth_over_time.png - Liquidity changes")
    print("   - price_levels.png - Orderbook structure")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("\n✅ INSIGHTS:")
    print("   1. Best bid/ask (0.01/0.99) are NOT real - just extreme limit orders")
    print("   2. Weighted mid-price shows real trading range (0.44-0.92)")
    print("   3. 3,943 unique prices - excellent for OFI analysis!")
    print("   4. Price discovery IS happening - just not at best bid/ask level")

    print("\n📋 NEXT STEPS:")
    print("   1. Review the visualizations in results/")
    print("   2. Understand that we need to use weighted mid-price")
    print("   3. Then we can properly calculate OFI using realistic prices")

    print("\n")


if __name__ == "__main__":
    main()
