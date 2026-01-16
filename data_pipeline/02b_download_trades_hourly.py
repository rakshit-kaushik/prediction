"""
02b_download_trades_hourly.py
==============================
Download trade history HOUR BY HOUR to avoid rate limits.

Each hour typically has ~200 trades = 2-3 API calls
21 days × 24 hours = 504 hours total
~1500 API calls, with 3-second delays = ~75 minutes

Usage:
    python data_pipeline/02b_download_trades_hourly.py
"""

import sys
import json
import os
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.config import *


def load_token_id():
    """Load token ID from market info file"""
    if not os.path.exists(MARKET_INFO_FILE):
        print(f"❌ Market info file not found: {MARKET_INFO_FILE}")
        sys.exit(1)

    with open(MARKET_INFO_FILE, 'r') as f:
        market_info = json.load(f)

    token_id = market_info.get('token_id')
    if not token_id:
        print(f"❌ No token_id found in market info")
        sys.exit(1)

    return token_id, market_info


def download_trades_hour(token_id, start_dt, end_dt, api_key):
    """
    Download trades for a single hour

    Args:
        token_id: Token ID
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        api_key: API key

    Returns:
        List of trades
    """
    start_sec = int(start_dt.timestamp())
    end_sec = int(end_dt.timestamp())

    all_trades = []
    offset = 0
    limit = 100

    headers = {
        'x-api-key': api_key,
        'accept': 'application/json'
    }

    while True:
        params = {
            'token_id': token_id,
            'start_time': start_sec,
            'end_time': end_sec,
            'limit': limit,
            'offset': offset
        }

        try:
            response = requests.get(
                f"{DOME_API_BASE_URL}/polymarket/orders",
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 10
                print(f"\n⚠️  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                print(f"\n❌ API error: {response.status_code}")
                break

            data = response.json()
            orders = data.get('orders', [])
            pagination = data.get('pagination', {})

            if not orders:
                break

            all_trades.extend(orders)

            if not pagination.get('has_more', False):
                break

            offset += limit
            time.sleep(1)  # Small delay between pages

        except Exception as e:
            print(f"\n❌ Request error: {e}")
            break

    return all_trades


def download_all_trades_hourly(token_id, start_date, end_date, api_key):
    """
    Download trades hour by hour for entire date range
    """
    all_trades = []

    # Calculate total hours
    current = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)

    total_hours = int((end - current).total_seconds() / 3600) + 1

    print(f"\n📊 Downloading {total_hours} hours of data...")

    pbar = tqdm(total=total_hours, desc="Downloading", unit="hour")

    hours_with_trades = 0

    while current < end:
        hour_end = current + timedelta(hours=1)

        # Download this hour
        hour_trades = download_trades_hour(token_id, current, hour_end, api_key)

        if hour_trades:
            all_trades.extend(hour_trades)
            hours_with_trades += 1

        pbar.set_postfix({
            'total': len(all_trades),
            'hours_w_trades': hours_with_trades
        })

        current = hour_end
        pbar.update(1)

        # Small delay between hours to avoid rate limits
        time.sleep(3)

    pbar.close()
    return all_trades


def main():
    print("\n" + "=" * 80)
    print("DOWNLOAD TRADES - HOUR BY HOUR")
    print("=" * 80)

    # Load token ID
    print("\n📋 Loading market info...")
    token_id, market_info = load_token_id()

    market_name = market_info.get('question', 'Unknown Market')
    print(f"✓ Market: {market_name}")
    print(f"✓ Token ID: {token_id[:20]}...")

    # Check API key
    api_key = os.getenv('DOME_API_KEY')
    if not api_key:
        print("\n❌ DOME_API_KEY not found")
        sys.exit(1)

    # Parse dates
    if len(sys.argv) >= 3:
        start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
        end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    else:
        start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    print(f"\n📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Days: {(end_date - start_date).days + 1}")
    print(f"   Hours: {((end_date - start_date).days + 1) * 24}")

    # Download
    print("\n📥 Starting hourly download...")
    all_trades = download_all_trades_hourly(token_id, start_date, end_date, api_key)

    if not all_trades:
        print("\n⚠️  No trades found")
        return

    # Statistics
    print("\n" + "=" * 80)
    print("TRADE STATISTICS")
    print("=" * 80)
    print(f"Total trades: {len(all_trades):,}")

    buy_trades = [t for t in all_trades if t.get('side') == 'BUY']
    sell_trades = [t for t in all_trades if t.get('side') == 'SELL']
    print(f"Buy trades: {len(buy_trades):,} ({len(buy_trades)/len(all_trades)*100:.1f}%)")
    print(f"Sell trades: {len(sell_trades):,} ({len(sell_trades)/len(all_trades)*100:.1f}%)")

    # Time range
    timestamps = [t['timestamp'] for t in all_trades if 'timestamp' in t]
    if timestamps:
        first_trade = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        last_trade = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)
        print(f"First trade: {first_trade.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Last trade:  {last_trade.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Trades per day
    print("\n📊 Trades per day:")
    from collections import Counter
    day_counts = Counter()
    for t in all_trades:
        if 'timestamp' in t:
            dt = datetime.fromtimestamp(t['timestamp'], tz=timezone.utc)
            day_counts[dt.strftime('%Y-%m-%d')] += 1

    for day in sorted(day_counts.keys()):
        print(f"   {day}: {day_counts[day]:,} trades")

    # Save
    print("\n💾 Saving to file...")
    with open(TRADES_RAW_FILE, 'w') as f:
        json.dump(all_trades, f, indent=2)

    print(f"✓ Saved to: {TRADES_RAW_FILE}")
    print(f"✓ File size: {os.path.getsize(TRADES_RAW_FILE) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\n💡 Next step: python data_pipeline/03b_process_trades.py")


if __name__ == "__main__":
    main()
