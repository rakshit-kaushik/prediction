"""
Quick test: Download 1 day of orderbook data
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.domeapi.io/v1"
API_KEY = os.getenv('DOME_API_KEY')
TOKEN_ID = "33945469250963963541781051637999677727672635213493648594066577298999471399137"

def download_one_day():
    """Download orderbook data for November 1, 2025"""

    print("\n" + "="*80)
    print("DOWNLOADING 1 DAY OF ORDERBOOK DATA")
    print("="*80)

    # November 1, 2025 (full day)
    start = datetime(2025, 11, 1, 0, 0, 0)
    end = datetime(2025, 11, 2, 0, 0, 0)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"\nDate: {start.date()}")
    print(f"Start: {start} ({start_ms})")
    print(f"End: {end} ({end_ms})")

    url = f"{BASE_URL}/polymarket/orderbooks"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    all_snapshots = []
    pagination_key = None
    page = 1

    print("\nDownloading...")

    while True:
        params = {
            "token_id": TOKEN_ID,
            "start_time": start_ms,
            "end_time": end_ms,
            "limit": 200
        }

        if pagination_key:
            params["pagination_key"] = pagination_key

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        data = response.json()
        snapshots = data.get('snapshots', [])
        all_snapshots.extend(snapshots)

        pagination = data.get('pagination', {})
        has_more = pagination.get('has_more', False)
        pagination_key = pagination.get('pagination_key')

        print(f"  Page {page}: Got {len(snapshots)} snapshots (total: {len(all_snapshots)})")

        if not has_more:
            break

        page += 1

    print(f"\n✅ Downloaded {len(all_snapshots)} snapshots")

    # Process into DataFrame
    print("\nProcessing...")
    records = []

    for snap in all_snapshots:
        ts_ms = snap['timestamp']
        ts = datetime.fromtimestamp(ts_ms / 1000)

        bids = snap.get('bids', [])
        asks = snap.get('asks', [])

        if not bids or not asks:
            continue

        records.append({
            'timestamp': ts,
            'timestamp_ms': ts_ms,
            'best_bid_price': float(bids[0]['price']),
            'best_bid_size': float(bids[0]['size']),
            'best_ask_price': float(asks[0]['price']),
            'best_ask_size': float(asks[0]['size']),
            'mid_price': (float(bids[0]['price']) + float(asks[0]['price'])) / 2,
            'spread': float(asks[0]['price']) - float(bids[0]['price']),
            'bid_levels': len(bids),
            'ask_levels': len(asks),
            'total_bid_size': sum(float(b['size']) for b in bids),
            'total_ask_size': sum(float(a['size']) for a in asks)
        })

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Save
    df.to_csv('data/orderbook_one_day.csv', index=False)
    print(f"✅ Saved to: data/orderbook_one_day.csv")

    # Stats
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"\nTotal snapshots: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Time between snapshots
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    print(f"\nSnapshot frequency:")
    print(f"  Mean: {df['time_diff'].mean():.1f} seconds")
    print(f"  Median: {df['time_diff'].median():.1f} seconds")

    print(f"\nPrice stats:")
    print(f"  Mid price: {df['mid_price'].mean():.4f} (±{df['mid_price'].std():.4f})")
    print(f"  Spread: {df['spread'].mean():.4f}")

    print(f"\nDepth stats:")
    print(f"  Bid levels: {df['bid_levels'].mean():.0f}")
    print(f"  Ask levels: {df['ask_levels'].mean():.0f}")
    print(f"  Total bid size: {df['total_bid_size'].mean():,.0f}")
    print(f"  Total ask size: {df['total_ask_size'].mean():,.0f}")

    print("\n✅ DONE! Ready to calculate OFI.")
    print("\n")

    return df

if __name__ == "__main__":
    df = download_one_day()
