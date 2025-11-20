"""
reprocess_orderbook_data.py
============================
CORRECTLY process orderbook data with proper best bid/ask

IMPORTANT:
- Best Bid = HIGHEST bid price (max someone will pay)
- Best Ask = LOWEST ask price (min someone will sell for)

The original processing took bids[0] and asks[0] without sorting,
which gave us the WRONG prices (lowest bid instead of highest bid)

This script:
1. Loads raw orderbook data
2. For each snapshot, finds ACTUAL best bid/ask
3. Saves corrected processed data

Usage:
    python reprocess_orderbook_data.py
"""

import json
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

def process_snapshot_correctly(snapshot):
    """Process a single snapshot with CORRECT best bid/ask"""
    timestamp_ms = snapshot.get('timestamp', 0)
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    bids = snapshot.get('bids', [])
    asks = snapshot.get('asks', [])

    if not bids or not asks:
        return None

    # Convert to numeric and sort properly
    bid_prices = [float(b['price']) for b in bids]
    bid_sizes = [float(b['size']) for b in bids]
    ask_prices = [float(a['price']) for a in asks]
    ask_sizes = [float(a['size']) for a in asks]

    # BEST BID = HIGHEST bid price
    best_bid_idx = bid_prices.index(max(bid_prices))
    best_bid_price = bid_prices[best_bid_idx]
    best_bid_size = bid_sizes[best_bid_idx]

    # BEST ASK = LOWEST ask price
    best_ask_idx = ask_prices.index(min(ask_prices))
    best_ask_price = ask_prices[best_ask_idx]
    best_ask_size = ask_sizes[best_ask_idx]

    # Calculate metrics
    mid_price = (best_bid_price + best_ask_price) / 2
    spread = best_ask_price - best_bid_price
    spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0

    # Total depth
    total_bid_size = sum(bid_sizes)
    total_ask_size = sum(ask_sizes)
    total_depth = total_bid_size + total_ask_size
    imbalance = (total_bid_size - total_ask_size) / total_depth if total_depth > 0 else 0

    # Number of levels
    bid_levels = len(bids)
    ask_levels = len(asks)

    return {
        'timestamp': timestamp,
        'timestamp_ms': timestamp_ms,
        'best_bid_price': best_bid_price,
        'best_bid_size': best_bid_size,
        'best_ask_price': best_ask_price,
        'best_ask_size': best_ask_size,
        'mid_price': mid_price,
        'spread': spread,
        'spread_pct': spread_pct,
        'bid_levels': bid_levels,
        'ask_levels': ask_levels,
        'total_bid_size': total_bid_size,
        'total_ask_size': total_ask_size,
        'total_depth': total_depth,
        'imbalance': imbalance
    }


def main():
    print("\n" + "=" * 80)
    print("REPROCESSING ORDERBOOK DATA WITH CORRECT BEST BID/ASK")
    print("=" * 80)

    # Load raw data
    print("\n📂 Loading raw orderbook data...")
    with open('data/orderbook_fed_oct15_31_raw.json', 'r') as f:
        snapshots = json.load(f)
    print(f"✅ Loaded {len(snapshots)} snapshots")

    # Process correctly
    print("\n⚙️  Processing snapshots with CORRECT best bid/ask logic...")
    records = []
    for snapshot in tqdm(snapshots, desc="Processing"):
        result = process_snapshot_correctly(snapshot)
        if result:
            records.append(result)

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Print summary
    print(f"\n✅ Processed {len(df)} snapshots")

    print("\n" + "=" * 80)
    print("CORRECTED DATA SUMMARY")
    print("=" * 80)

    print(f"\n📊 Best Bid/Ask (CORRECT):")
    print(f"   Best Bid range: {df['best_bid_price'].min():.4f} to {df['best_bid_price'].max():.4f}")
    print(f"   Best Ask range: {df['best_ask_price'].min():.4f} to {df['best_ask_price'].max():.4f}")
    print(f"   Mid-Price: {df['mid_price'].mean():.4f} (mean)")
    print(f"   Spread: {df['spread'].mean():.4f} ({df['spread_pct'].mean():.2f}%)")

    print(f"\n📈 Price Movement:")
    unique_bids = df['best_bid_price'].nunique()
    unique_asks = df['best_ask_price'].nunique()
    unique_mids = df['mid_price'].nunique()
    print(f"   Unique best bid prices: {unique_bids}")
    print(f"   Unique best ask prices: {unique_asks}")
    print(f"   Unique mid prices: {unique_mids}")

    if unique_bids > 50 and unique_asks > 50:
        print(f"   ✅ Excellent price movement for OFI analysis!")
    elif unique_bids > 10 and unique_asks > 10:
        print(f"   ⚠️  Moderate price movement")
    else:
        print(f"   ⚠️  Limited price movement")

    # Save corrected data
    print(f"\n💾 Saving corrected data...")
    output_file = 'data/orderbook_fed_oct15_31_processed_CORRECT.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to: {output_file}")

    # Comparison with old (wrong) data
    print("\n" + "=" * 80)
    print("COMPARISON: OLD (WRONG) vs NEW (CORRECT)")
    print("=" * 80)

    try:
        old_df = pd.read_csv('data/orderbook_fed_oct15_31_processed.csv')

        print(f"\nOLD (WRONG) - took first element without sorting:")
        print(f"   Best Bid: {old_df['best_bid_price'].iloc[0]:.4f}")
        print(f"   Best Ask: {old_df['best_ask_price'].iloc[0]:.4f}")
        print(f"   Mid: {old_df['mid_price'].iloc[0]:.4f}")
        print(f"   Spread: {old_df['spread'].iloc[0]:.4f}")

        print(f"\nNEW (CORRECT) - finds highest bid, lowest ask:")
        print(f"   Best Bid: {df['best_bid_price'].iloc[0]:.4f}")
        print(f"   Best Ask: {df['best_ask_price'].iloc[0]:.4f}")
        print(f"   Mid: {df['mid_price'].iloc[0]:.4f}")
        print(f"   Spread: {df['spread'].iloc[0]:.4f}")

    except:
        print("\n(Old file not found for comparison)")

    print("\n" + "=" * 80)
    print("✅ REPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\n📋 Next: Re-run analysis with corrected data\n")


if __name__ == "__main__":
    main()
