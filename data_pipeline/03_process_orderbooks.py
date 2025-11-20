"""
03_process_orderbooks.py
=========================
STEP 3: Process raw orderbook snapshots into clean CSV format

WHAT IT DOES:
-------------
Input:  Raw orderbook snapshots (from step 2)
Output: Clean CSV with best bid/ask, spreads, and depth metrics

HOW IT WORKS:
------------
1. Loads raw JSON snapshots
2. For each snapshot:
   - Finds BEST BID (highest bid price)
   - Finds BEST ASK (lowest ask price)
   - Calculates mid-price, spread, depth
3. Saves to CSV ready for analysis

IMPORTANT - Best Bid/Ask Definition:
------------------------------------
- Best Bid = HIGHEST bid price (max buyers will pay)
- Best Ask = LOWEST ask price (min sellers will accept)
- This is standard market microstructure convention

USAGE:
------
    python 03_process_orderbooks.py

WHAT YOU GET:
------------
- data/orderbook_processed.csv with columns:
  • timestamp, timestamp_ms
  • best_bid_price, best_bid_size
  • best_ask_price, best_ask_size
  • mid_price, spread, spread_pct
  • total_bid_size, total_ask_size, total_depth
  • imbalance, bid_levels, ask_levels
"""

import sys
import json
import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.config import *


def process_snapshot(snapshot):
    """
    Process a single orderbook snapshot

    Args:
        snapshot: Raw snapshot dict with timestamp, bids, asks

    Returns:
        dict: Processed snapshot with best bid/ask and metrics
    """
    timestamp_ms = snapshot.get('timestamp', 0)
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    bids = snapshot.get('bids', [])
    asks = snapshot.get('asks', [])

    # Skip if empty
    if not bids or not asks:
        return None

    # Convert to numeric
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

    # Number of price levels
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
    print("\n")
    print("=" * 80)
    print(" STEP 3: PROCESS ORDERBOOK DATA")
    print("=" * 80)

    # Check if raw file exists
    if not os.path.exists(RAW_ORDERBOOK_FILE):
        print(f"\n❌ Raw orderbook file not found: {RAW_ORDERBOOK_FILE}")
        print(f"\n💡 Run step 2 first: python 02_download_orderbooks.py")
        sys.exit(1)

    print(f"✅ Found raw orderbook file")

    # Load raw snapshots
    print(f"\n📂 Loading raw orderbook data...")
    with open(RAW_ORDERBOOK_FILE, 'r') as f:
        snapshots = json.load(f)

    print(f"✅ Loaded {len(snapshots)} snapshots")

    file_size_mb = os.path.getsize(RAW_ORDERBOOK_FILE) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")

    # Process snapshots
    print(f"\n⚙️  Processing snapshots...")
    print("   (Extracting best bid/ask for each snapshot)")

    records = []
    skipped = 0

    for snapshot in tqdm(snapshots, desc="Processing"):
        result = process_snapshot(snapshot)
        if result:
            records.append(result)
        else:
            skipped += 1

    if not records:
        print(f"\n❌ No valid snapshots processed!")
        print(f"\n💡 Possible reasons:")
        print(f"   1. All snapshots have empty bids or asks")
        print(f"   2. Raw data format is incorrect")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Processed {len(df)} snapshots")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} empty snapshots")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("ORDERBOOK SUMMARY")
    print("=" * 80)

    print(f"\n📅 Time Range:")
    print(f"   First: {df['timestamp'].min()}")
    print(f"   Last:  {df['timestamp'].max()}")
    duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f"   Duration: {duration:.1f} days")

    print(f"\n💰 Price Statistics:")
    print(f"   Best Bid range: ${df['best_bid_price'].min():.4f} to ${df['best_bid_price'].max():.4f}")
    print(f"   Best Ask range: ${df['best_ask_price'].min():.4f} to ${df['best_ask_price'].max():.4f}")
    print(f"   Mid-Price mean: ${df['mid_price'].mean():.4f}")
    print(f"   Spread mean: ${df['spread'].mean():.4f} ({df['spread_pct'].mean():.2f}%)")

    print(f"\n📊 Price Movement:")
    unique_bids = df['best_bid_price'].nunique()
    unique_asks = df['best_ask_price'].nunique()
    unique_mids = df['mid_price'].nunique()
    print(f"   Unique best bid prices: {unique_bids}")
    print(f"   Unique best ask prices: {unique_asks}")
    print(f"   Unique mid prices: {unique_mids}")

    if unique_bids > 50 and unique_asks > 50:
        print(f"   ✅ Excellent price movement for analysis!")
    elif unique_bids > 10 and unique_asks > 10:
        print(f"   ✅ Good price movement")
    else:
        print(f"   ⚠️  Limited price movement")

    print(f"\n📚 Orderbook Depth:")
    print(f"   Avg bid levels: {df['bid_levels'].mean():.1f}")
    print(f"   Avg ask levels: {df['ask_levels'].mean():.1f}")
    print(f"   Avg total depth: {df['total_depth'].mean():,.0f}")
    print(f"   Avg imbalance: {df['imbalance'].mean():.4f}")

    # Save processed data
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    # Create data directory
    Path(PROCESSED_ORDERBOOK_FILE).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to: {PROCESSED_ORDERBOOK_FILE}")
    df.to_csv(PROCESSED_ORDERBOOK_FILE, index=False)

    file_size_mb = os.path.getsize(PROCESSED_ORDERBOOK_FILE) / (1024 * 1024)
    print(f"✅ Saved {len(df)} rows × {len(df.columns)} columns")
    print(f"   File size: {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("✅ SUCCESS")
    print("=" * 80)

    print(f"\n📋 Processing Summary:")
    print(f"   Input:  {len(snapshots)} raw snapshots")
    print(f"   Output: {len(df)} processed snapshots")
    print(f"   Saved to: {PROCESSED_ORDERBOOK_FILE}")

    print(f"\n📝 Data Ready for Analysis!")
    print(f"   • Best bid/ask extracted correctly")
    print(f"   • {unique_mids} unique price points")
    print(f"   • Ready for OFI calculation or other analysis")

    print(f"\n💡 Next Steps:")
    print(f"   • Use this CSV for Order Flow Imbalance (OFI) calculation")
    print(f"   • Analyze price dynamics and liquidity patterns")
    print(f"   • Build prediction models")
    print()


if __name__ == "__main__":
    main()
