"""
02_download_orderbook.py
=========================
Phase 2: Download Historical Orderbook Data

This script downloads all historical orderbook snapshots for the
Zohran Mamdani NYC mayoral election market.

Usage:
    python 02_download_orderbook.py

Output:
    data/orderbook_mamdani_raw.csv - All orderbook snapshots
    data/orderbook_mamdani_processed.csv - Processed with OFI-ready format
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


def download_orderbook_chunk(token_id, start_time, end_time, limit=200):
    """Download one chunk of orderbook data with pagination"""
    url = f"{BASE_URL}/polymarket/orderbooks"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    all_snapshots = []
    pagination_key = None

    while True:
        params = {
            "token_id": token_id,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit
        }

        if pagination_key:
            params["pagination_key"] = pagination_key

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            snapshots = data.get('snapshots', [])
            all_snapshots.extend(snapshots)

            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            pagination_key = pagination.get('pagination_key')

            print(f"  Got {len(snapshots)} snapshots (total: {len(all_snapshots)})")

            if not has_more:
                break

            # Rate limiting
            time.sleep(0.5)

        except requests.exceptions.HTTPError as e:
            print(f"  ❌ HTTP Error: {e.response.status_code}")
            print(f"  Response: {e.response.text}")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            break

    return all_snapshots


def download_all_orderbook_data(token_id, start_date, end_date):
    """Download all orderbook data in date chunks"""
    print("=" * 80)
    print("DOWNLOADING ORDERBOOK DATA")
    print("=" * 80)

    print(f"\nMarket: {market_info['title']}")
    print(f"Token ID: {token_id[:30]}...")
    print(f"Date range: {start_date} to {end_date}")

    # Split into weekly chunks to avoid rate limits
    current = start_date
    all_snapshots = []

    while current < end_date:
        chunk_end = min(current + timedelta(days=7), end_date)

        start_ms = int(current.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        print(f"\n📥 Downloading: {current.date()} to {chunk_end.date()}")

        snapshots = download_orderbook_chunk(token_id, start_ms, end_ms)
        all_snapshots.extend(snapshots)

        print(f"  ✅ Total snapshots so far: {len(all_snapshots)}")

        current = chunk_end
        time.sleep(1)  # Rate limiting between chunks

    print(f"\n✅ DOWNLOAD COMPLETE!")
    print(f"Total snapshots: {len(all_snapshots)}")

    return all_snapshots


def process_orderbook_data(snapshots):
    """Convert raw orderbook snapshots to DataFrame"""
    print("\n" + "=" * 80)
    print("PROCESSING ORDERBOOK DATA")
    print("=" * 80)

    records = []

    for snapshot in tqdm(snapshots, desc="Processing snapshots"):
        timestamp_ms = snapshot.get('timestamp', 0)
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

        # Get best bid and ask
        bids = snapshot.get('bids', [])
        asks = snapshot.get('asks', [])

        if not bids or not asks:
            continue

        best_bid_price = float(bids[0]['price'])
        best_bid_size = float(bids[0]['size'])
        best_ask_price = float(asks[0]['price'])
        best_ask_size = float(asks[0]['size'])

        # Calculate aggregates
        total_bid_size = sum(float(b['size']) for b in bids)
        total_ask_size = sum(float(a['size']) for a in asks)

        mid_price = (best_bid_price + best_ask_price) / 2
        spread = best_ask_price - best_bid_price

        record = {
            'timestamp': timestamp,
            'timestamp_ms': timestamp_ms,
            'best_bid_price': best_bid_price,
            'best_bid_size': best_bid_size,
            'best_ask_price': best_ask_price,
            'best_ask_size': best_ask_size,
            'mid_price': mid_price,
            'spread': spread,
            'bid_levels': len(bids),
            'ask_levels': len(asks),
            'total_bid_size': total_bid_size,
            'total_ask_size': total_ask_size,
            # Store full orderbook as JSON for later analysis
            'bids_json': json.dumps(bids[:10]),  # Top 10 levels
            'asks_json': json.dumps(asks[:10])
        }

        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Processed {len(df)} snapshots")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")

    return df


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "OFI ANALYSIS - PHASE 2: DATA COLLECTION" + " " * 23 + "║")
    print("║" + " " * 15 + "Orderbook Download" + " " * 44 + "║")
    print("╚" + "═" * 78 + "╝")

    # Date range: From Oct 15, 2025 (when data starts) to now
    start_date = datetime(2025, 10, 15)
    end_date = datetime.now()

    # Download data
    snapshots = download_all_orderbook_data(TOKEN_ID, start_date, end_date)

    if not snapshots:
        print("\n❌ No data downloaded!")
        return

    # Save raw data
    print("\n💾 Saving raw data...")
    with open('data/orderbook_mamdani_raw.json', 'w') as f:
        json.dump(snapshots, f)
    print(f"  ✅ Saved to: data/orderbook_mamdani_raw.json")

    # Process data
    df = process_orderbook_data(snapshots)

    # Save processed data
    print("\n💾 Saving processed data...")
    df.to_csv('data/orderbook_mamdani_processed.csv', index=False)
    print(f"  ✅ Saved to: data/orderbook_mamdani_processed.csv")

    # Summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"\nTotal snapshots: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

    # Time between snapshots
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    print(f"\nSnapshot frequency:")
    print(f"  Mean: {df['time_diff'].mean():.1f} seconds")
    print(f"  Median: {df['time_diff'].median():.1f} seconds")
    print(f"  Min: {df['time_diff'].min():.1f} seconds")
    print(f"  Max: {df['time_diff'].max():.1f} seconds")

    print(f"\nPrice statistics:")
    print(f"  Mid price range: {df['mid_price'].min():.4f} to {df['mid_price'].max():.4f}")
    print(f"  Mean mid price: {df['mid_price'].mean():.4f}")
    print(f"  Mean spread: {df['spread'].mean():.4f}")

    print(f"\nDepth statistics:")
    print(f"  Mean bid levels: {df['bid_levels'].mean():.1f}")
    print(f"  Mean ask levels: {df['ask_levels'].mean():.1f}")
    print(f"  Mean total bid size: {df['total_bid_size'].mean():,.0f}")
    print(f"  Mean total ask size: {df['total_ask_size'].mean():,.0f}")

    print("\n" + "=" * 80)
    print("✅ DATA COLLECTION COMPLETE!")
    print("=" * 80)
    print("\n📋 NEXT STEPS:")
    print("   1. Review data quality in orderbook_mamdani_processed.csv")
    print("   2. Run: python 03_calculate_ofi.py")
    print("\n")


if __name__ == "__main__":
    main()
