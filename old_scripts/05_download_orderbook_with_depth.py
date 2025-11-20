"""
05_download_orderbook_with_depth.py
====================================
Download orderbook data with FULL DEPTH for visualization

This script downloads orderbook snapshots for Oct 14-18, 2025 and saves:
1. Full orderbook depth (all bid/ask levels) as JSON
2. Processed metrics as CSV for analysis

Key difference from previous scripts:
- Saves COMPLETE orderbook (all levels), not just best bid/ask
- Enables classic orderbook depth chart visualization
- Stores raw snapshots for detailed analysis

Usage:
    python 05_download_orderbook_with_depth.py

Output:
    data/orderbook_oct14_18_raw.json - Full snapshots with all levels
    data/orderbook_oct14_18_processed.csv - Processed metrics
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()

# Configuration
BASE_URL = "https://api.domeapi.io/v1"
API_KEY = os.getenv('DOME_API_KEY')

# Load market info
with open('data/mamdani_market.json', 'r') as f:
    market_info = json.load(f)

TOKEN_ID = market_info['token_id']


def download_orderbook_day(token_id, date):
    """
    Download orderbook data for a single day

    Args:
        token_id: Polymarket token ID
        date: datetime object for the day

    Returns:
        List of orderbook snapshots
    """
    start = datetime(date.year, date.month, date.day, 0, 0, 0)
    end = datetime(date.year, date.month, date.day, 23, 59, 59)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    url = f"{BASE_URL}/polymarket/orderbooks"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    all_snapshots = []
    pagination_key = None
    page = 1

    print(f"\n📥 Downloading: {date.date()}")

    while True:
        params = {
            "token_id": token_id,
            "start_time": start_ms,
            "end_time": end_ms,
            "limit": 200
        }

        if pagination_key:
            params["pagination_key"] = pagination_key

        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"  ❌ Error {response.status_code}: {response.text}")
                break

            data = response.json()
            snapshots = data.get('snapshots', [])
            all_snapshots.extend(snapshots)

            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            pagination_key = pagination.get('pagination_key')

            if page % 5 == 0:  # Print every 5 pages
                print(f"  Page {page}: {len(all_snapshots)} snapshots so far...")

            if not has_more:
                break

            page += 1
            time.sleep(0.3)  # Rate limiting

        except Exception as e:
            print(f"  ❌ Error: {e}")
            break

    print(f"  ✅ Downloaded {len(all_snapshots)} snapshots for {date.date()}")
    return all_snapshots


def process_snapshots(snapshots):
    """
    Process raw snapshots into DataFrame with key metrics

    Args:
        snapshots: List of raw orderbook snapshots

    Returns:
        DataFrame with processed metrics
    """
    records = []

    for snap in tqdm(snapshots, desc="Processing snapshots"):
        timestamp_ms = snap.get('timestamp', 0)
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

        bids = snap.get('bids', [])
        asks = snap.get('asks', [])

        if not bids or not asks:
            continue

        # Best bid/ask
        best_bid_price = float(bids[0]['price'])
        best_bid_size = float(bids[0]['size'])
        best_ask_price = float(asks[0]['price'])
        best_ask_size = float(asks[0]['size'])

        # Aggregate metrics
        total_bid_size = sum(float(b['size']) for b in bids)
        total_ask_size = sum(float(a['size']) for a in asks)

        # Price metrics
        mid_price = (best_bid_price + best_ask_price) / 2
        spread = best_ask_price - best_bid_price
        spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0

        # Depth metrics
        bid_levels = len(bids)
        ask_levels = len(asks)
        total_depth = total_bid_size + total_ask_size
        imbalance = (total_bid_size - total_ask_size) / total_depth if total_depth > 0 else 0

        # Calculate depth at various price increments
        # This is useful for visualizing orderbook shape
        depth_1pct = {
            'bid': sum(float(b['size']) for b in bids if float(b['price']) >= best_bid_price * 0.99),
            'ask': sum(float(a['size']) for a in asks if float(a['price']) <= best_ask_price * 1.01)
        }

        record = {
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
            'imbalance': imbalance,
            'depth_1pct_bid': depth_1pct['bid'],
            'depth_1pct_ask': depth_1pct['ask']
        }

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "ORDERBOOK DATA DOWNLOAD - WITH FULL DEPTH" + " " * 27 + "║")
    print("║" + " " * 10 + "October 14-18, 2025" + " " * 48 + "║")
    print("╚" + "═" * 78 + "╝")

    # Date range
    start_date = datetime(2025, 10, 14)
    end_date = datetime(2025, 10, 18)

    print(f"\n📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"🎯 Market: {market_info['title']}")
    print(f"🔑 Token ID: {TOKEN_ID[:30]}...")

    # Download day by day
    all_snapshots = []
    current_date = start_date

    print("\n" + "=" * 80)
    print("DOWNLOADING DATA")
    print("=" * 80)

    while current_date <= end_date:
        snapshots = download_orderbook_day(TOKEN_ID, current_date)
        all_snapshots.extend(snapshots)
        current_date += timedelta(days=1)
        time.sleep(1)  # Rate limiting between days

    print(f"\n✅ Total snapshots downloaded: {len(all_snapshots)}")

    if not all_snapshots:
        print("\n❌ No data downloaded! Check:")
        print("   1. API key is valid")
        print("   2. Date range has available data")
        print("   3. Token ID is correct")
        return

    # Save raw data with FULL DEPTH
    print("\n" + "=" * 80)
    print("SAVING RAW DATA (with full orderbook depth)")
    print("=" * 80)

    raw_file = 'data/orderbook_oct14_18_raw.json'
    print(f"\n💾 Saving to: {raw_file}")
    print(f"   Size: {len(all_snapshots)} snapshots")

    with open(raw_file, 'w') as f:
        json.dump(all_snapshots, f, indent=2)

    # Check file size
    file_size_mb = os.path.getsize(raw_file) / (1024 * 1024)
    print(f"   ✅ Saved: {file_size_mb:.2f} MB")

    # Process data
    print("\n" + "=" * 80)
    print("PROCESSING DATA")
    print("=" * 80)

    df = process_snapshots(all_snapshots)

    # Save processed data
    processed_file = 'data/orderbook_oct14_18_processed.csv'
    print(f"\n💾 Saving to: {processed_file}")
    df.to_csv(processed_file, index=False)
    print(f"   ✅ Saved: {len(df)} rows")

    # Summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    print(f"\nTime range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"Total snapshots: {len(df):,}")

    # Snapshot frequency
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    print(f"\nSnapshot frequency:")
    print(f"  Mean: {df['time_diff'].mean():.1f} seconds")
    print(f"  Median: {df['time_diff'].median():.1f} seconds")

    # Price statistics
    print(f"\nPrice statistics:")
    print(f"  Best bid range: {df['best_bid_price'].min():.4f} to {df['best_bid_price'].max():.4f}")
    print(f"  Best ask range: {df['best_ask_price'].min():.4f} to {df['best_ask_price'].max():.4f}")
    print(f"  Mid-price range: {df['mid_price'].min():.4f} to {df['mid_price'].max():.4f}")
    print(f"  Spread range: {df['spread'].min():.4f} to {df['spread'].max():.4f}")
    print(f"  Spread %: {df['spread_pct'].mean():.2f}% (mean)")

    # Depth statistics
    print(f"\nDepth statistics:")
    print(f"  Bid levels: {df['bid_levels'].mean():.0f} (mean)")
    print(f"  Ask levels: {df['ask_levels'].mean():.0f} (mean)")
    print(f"  Total bid size: {df['total_bid_size'].mean():,.0f} (mean)")
    print(f"  Total ask size: {df['total_ask_size'].mean():,.0f} (mean)")
    print(f"  Imbalance: {df['imbalance'].mean():.4f} (mean, +ve = more bids)")

    # Check for price movement
    unique_bids = df['best_bid_price'].nunique()
    unique_asks = df['best_ask_price'].nunique()

    print(f"\nPrice movement:")
    print(f"  Unique bid prices: {unique_bids}")
    print(f"  Unique ask prices: {unique_asks}")

    if unique_bids == 1 and unique_asks == 1:
        print("\n⚠️  WARNING: Prices are constant (no movement)")
        print("   This will result in zero OFI values")
        print("   Consider using a different market for OFI analysis")
    else:
        print("\n✅ Price movement detected! OFI analysis will be meaningful")

    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print("\n📋 NEXT STEPS:")
    print("   1. Run: python 06_visualize_orderbook.py")
    print("   2. Run: python 07_calculate_hourly_metrics.py")
    print("   3. Run: python 08_dashboard.py")
    print("\n")


if __name__ == "__main__":
    main()
