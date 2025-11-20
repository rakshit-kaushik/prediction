"""
analyze_orderbook_structure.py
================================
Analyze orderbook structure to understand where liquidity actually sits

This script examines:
1. Where is the real liquidity? (not just best bid/ask at 0.01/0.99)
2. What is the effective mid-price based on actual trading levels?
3. How does orderbook depth change over time?
4. Which price levels have meaningful activity?

Goal: Understand the data BEFORE doing any OFI analysis

Usage:
    python analyze_orderbook_structure.py

Input:
    data/orderbook_fed_oct15_31_raw.json - Raw orderbook snapshots

Output:
    Console analysis + saved summary statistics
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_orderbook_data(filename='data/orderbook_fed_oct15_31_raw.json'):
    """Load raw orderbook data"""
    print(f"\n📂 Loading {filename}...")
    with open(filename, 'r') as f:
        snapshots = json.load(f)
    print(f"✅ Loaded {len(snapshots)} orderbook snapshots")
    return snapshots


def analyze_snapshot(snapshot):
    """Analyze a single orderbook snapshot"""
    timestamp_ms = snapshot.get('timestamp', 0)
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

    bids = snapshot.get('bids', [])
    asks = snapshot.get('asks', [])

    if not bids or not asks:
        return None

    # Convert to DataFrames for easier analysis
    bid_df = pd.DataFrame(bids)
    ask_df = pd.DataFrame(asks)

    bid_df['price'] = bid_df['price'].astype(float)
    bid_df['size'] = bid_df['size'].astype(float)
    ask_df['price'] = ask_df['price'].astype(float)
    ask_df['size'] = ask_df['size'].astype(float)

    # Sort
    bid_df = bid_df.sort_values('price', ascending=False).reset_index(drop=True)
    ask_df = ask_df.sort_values('price', ascending=True).reset_index(drop=True)

    # Best bid/ask (traditional)
    best_bid = bid_df.iloc[0]['price']
    best_ask = ask_df.iloc[0]['price']
    best_bid_size = bid_df.iloc[0]['size']
    best_ask_size = ask_df.iloc[0]['size']

    # Total depth
    total_bid_size = bid_df['size'].sum()
    total_ask_size = ask_df['size'].sum()

    # Find where meaningful liquidity sits
    # Define "meaningful" as cumulative 10% of total depth
    bid_df['cumsize'] = bid_df['size'].cumsum()
    ask_df['cumsize'] = ask_df['size'].cumsum()

    # Price at 10% depth
    bid_10pct = bid_df[bid_df['cumsize'] >= total_bid_size * 0.1].iloc[0]['price'] if len(bid_df) > 0 else best_bid
    ask_10pct = ask_df[ask_df['cumsize'] >= total_ask_size * 0.1].iloc[0]['price'] if len(ask_df) > 0 else best_ask

    # Price at 50% depth (median)
    bid_50pct = bid_df[bid_df['cumsize'] >= total_bid_size * 0.5].iloc[0]['price'] if len(bid_df) > 0 else best_bid
    ask_50pct = ask_df[ask_df['cumsize'] >= total_ask_size * 0.5].iloc[0]['price'] if len(ask_df) > 0 else best_ask

    # Weighted mid-price (based on top 10 levels)
    top_n = 10
    if len(bid_df) >= top_n and len(ask_df) >= top_n:
        bid_top = bid_df.head(top_n)
        ask_top = ask_df.head(top_n)

        bid_weighted_price = np.average(bid_top['price'], weights=bid_top['size'])
        ask_weighted_price = np.average(ask_top['price'], weights=ask_top['size'])
        weighted_mid = (bid_weighted_price + ask_weighted_price) / 2
    else:
        weighted_mid = (best_bid + best_ask) / 2

    # Find price range with most liquidity (0.05 width bins)
    all_prices = pd.concat([bid_df[['price', 'size']], ask_df[['price', 'size']]])
    all_prices['price_bin'] = (all_prices['price'] / 0.05).round() * 0.05
    liquidity_by_bin = all_prices.groupby('price_bin')['size'].sum().sort_values(ascending=False)

    most_liquid_price = liquidity_by_bin.index[0] if len(liquidity_by_bin) > 0 else (best_bid + best_ask) / 2

    return {
        'timestamp': timestamp,
        'timestamp_ms': timestamp_ms,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'best_bid_size': best_bid_size,
        'best_ask_size': best_ask_size,
        'best_mid': (best_bid + best_ask) / 2,
        'best_spread': best_ask - best_bid,
        'best_spread_pct': ((best_ask - best_bid) / ((best_bid + best_ask) / 2)) * 100,
        'weighted_mid': weighted_mid,
        'bid_10pct': bid_10pct,
        'ask_10pct': ask_10pct,
        'effective_mid_10pct': (bid_10pct + ask_10pct) / 2,
        'effective_spread_10pct': ask_10pct - bid_10pct,
        'bid_50pct': bid_50pct,
        'ask_50pct': ask_50pct,
        'most_liquid_price': most_liquid_price,
        'total_bid_size': total_bid_size,
        'total_ask_size': total_ask_size,
        'total_depth': total_bid_size + total_ask_size,
        'num_bid_levels': len(bid_df),
        'num_ask_levels': len(ask_df),
    }


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ORDERBOOK STRUCTURE ANALYSIS" + " " * 30 + "║")
    print("║" + " " * 15 + "Understanding the Data Before OFI" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")

    # Load data
    snapshots = load_orderbook_data()

    # Analyze all snapshots
    print("\n⚙️  Analyzing orderbook structure...")
    results = []
    for snapshot in snapshots:
        analysis = analyze_snapshot(snapshot)
        if analysis:
            results.append(analysis)

    df = pd.DataFrame(results)
    print(f"✅ Analyzed {len(df)} snapshots")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("ORDERBOOK STRUCTURE SUMMARY")
    print("=" * 80)

    print(f"\n📅 Time range:")
    print(f"   {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    print(f"\n📊 Traditional Best Bid/Ask:")
    print(f"   Best Bid: {df['best_bid'].min():.4f} to {df['best_bid'].max():.4f}")
    print(f"   Best Ask: {df['best_ask'].min():.4f} to {df['best_ask'].max():.4f}")
    print(f"   Best Mid: {df['best_mid'].mean():.4f}")
    print(f"   Best Spread: {df['best_spread'].mean():.4f} ({df['best_spread_pct'].mean():.2f}%)")

    print(f"\n💡 Where Liquidity Actually Sits (10% depth):")
    print(f"   Bid at 10% depth: {df['bid_10pct'].min():.4f} to {df['bid_10pct'].max():.4f}")
    print(f"   Ask at 10% depth: {df['ask_10pct'].min():.4f} to {df['ask_10pct'].max():.4f}")
    print(f"   Effective Mid (10%): {df['effective_mid_10pct'].mean():.4f}")
    print(f"   Effective Spread (10%): {df['effective_spread_10pct'].mean():.4f}")

    print(f"\n📍 Weighted Mid-Price (top 10 levels):")
    print(f"   Range: {df['weighted_mid'].min():.4f} to {df['weighted_mid'].max():.4f}")
    print(f"   Mean: {df['weighted_mid'].mean():.4f}")
    print(f"   Std: {df['weighted_mid'].std():.4f}")

    print(f"\n🎯 Most Liquid Price Level:")
    print(f"   Range: {df['most_liquid_price'].min():.4f} to {df['most_liquid_price'].max():.4f}")
    print(f"   Mean: {df['most_liquid_price'].mean():.4f}")

    print(f"\n📏 Orderbook Depth:")
    print(f"   Bid levels: {df['num_bid_levels'].mean():.0f} (mean)")
    print(f"   Ask levels: {df['num_ask_levels'].mean():.0f} (mean)")
    print(f"   Total bid size: ${df['total_bid_size'].mean():,.0f} (mean)")
    print(f"   Total ask size: ${df['total_ask_size'].mean():,.0f} (mean)")

    # Check for price movement at different levels
    print("\n" + "=" * 80)
    print("PRICE MOVEMENT ANALYSIS")
    print("=" * 80)

    print(f"\n📈 Best Bid/Ask Movement:")
    unique_best_bid = df['best_bid'].nunique()
    unique_best_ask = df['best_ask'].nunique()
    print(f"   Unique best bid prices: {unique_best_bid}")
    print(f"   Unique best ask prices: {unique_best_ask}")

    if unique_best_bid == 1 and unique_best_ask == 1:
        print(f"   ⚠️  NO movement at best bid/ask level (stuck at {df['best_bid'].iloc[0]:.2f}/{df['best_ask'].iloc[0]:.2f})")
    else:
        print(f"   ✅ Price movement detected at best level")

    print(f"\n📈 Effective Mid-Price Movement (10% depth):")
    unique_eff_mid = df['effective_mid_10pct'].nunique()
    print(f"   Unique effective mid prices: {unique_eff_mid}")

    if unique_eff_mid > 100:
        print(f"   ✅ Significant price movement at effective level")
    elif unique_eff_mid > 10:
        print(f"   ⚠️  Moderate price movement")
    else:
        print(f"   ⚠️  Limited price movement")

    print(f"\n📈 Weighted Mid-Price Movement:")
    unique_weighted = df['weighted_mid'].nunique()
    print(f"   Unique weighted mid prices: {unique_weighted}")

    if unique_weighted > 100:
        print(f"   ✅ Excellent price discovery happening")
    elif unique_weighted > 10:
        print(f"   ⚠️  Some price discovery")
    else:
        print(f"   ⚠️  Minimal price discovery")

    # Price level distribution
    print("\n" + "=" * 80)
    print("PRICE LEVEL DISTRIBUTION")
    print("=" * 80)

    # Show distribution of where weighted mid sits
    print(f"\nWeighted Mid-Price Distribution:")
    print(df['weighted_mid'].describe())

    # Show distribution of most liquid price
    print(f"\nMost Liquid Price Level Distribution:")
    print(df['most_liquid_price'].describe())

    # Save results
    print("\n💾 Saving analysis results...")
    df.to_csv('data/orderbook_structure_analysis.csv', index=False)
    print(f"✅ Saved to: data/orderbook_structure_analysis.csv")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print(f"\n1. BEST BID/ASK:")
    print(f"   - Stuck at {df['best_bid'].iloc[0]:.2f}/{df['best_ask'].iloc[0]:.2f} (extreme wide spread)")
    print(f"   - NOT representative of actual trading")

    print(f"\n2. ACTUAL LIQUIDITY:")
    print(f"   - Concentrated around {df['effective_mid_10pct'].mean():.4f}")
    print(f"   - Effective spread: {df['effective_spread_10pct'].mean():.4f}")
    print(f"   - This is where real price discovery happens")

    print(f"\n3. PRICE MOVEMENT:")
    if unique_weighted > 50:
        print(f"   - ✅ Sufficient price movement for OFI analysis ({unique_weighted} unique prices)")
        print(f"   - Use weighted mid-price, NOT best bid/ask")
    else:
        print(f"   - ⚠️  Limited price movement ({unique_weighted} unique prices)")
        print(f"   - OFI analysis may still be challenging")

    print(f"\n4. RECOMMENDATION:")
    if unique_weighted > 50:
        print(f"   - ✅ This market is suitable for OFI analysis")
        print(f"   - Use 'weighted_mid' or 'effective_mid_10pct' instead of simple mid-price")
        print(f"   - Ignore the 0.01/0.99 extreme orders")
    else:
        print(f"   - Consider finding a different market with more price movement")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📋 Next step: Create visualizations to see these patterns\n")


if __name__ == "__main__":
    main()
