"""
02_download_orderbooks.py
==========================
STEP 2: Download orderbook snapshots for date range

WHAT IT DOES:
-------------
Input:  Token ID (from step 1) + Date range (from config)
Output: Raw orderbook snapshots saved to orderbook_raw.json

HOW IT WORKS:
------------
1. Reads token ID from market_info.json (created in step 1)
2. For each day in date range:
   - Queries Dome API for orderbook snapshots
   - Handles pagination (fetches all pages if >200 snapshots)
   - Saves raw JSON with full orderbook depth
3. Shows progress and statistics

USAGE:
------
    python 02_download_orderbooks.py

    # Or with custom dates (overrides config):
    python 02_download_orderbooks.py --start-date 2025-10-15 --end-date 2025-10-31

WHAT YOU GET:
------------
- data/orderbook_raw.json containing array of snapshots:
  [
    {
      "timestamp": 1729018714140,
      "bids": [{"price": "0.78", "size": "35000"}, ...],
      "asks": [{"price": "0.79", "size": "15000"}, ...]
    },
    ...
  ]

IMPORTANT:
----------
- Snapshots come at IRREGULAR intervals (event-driven, not time-driven)
- Days with high activity may have 1000+ snapshots
- Days with low activity may have <10 snapshots
- limit=200 means max per API call, NOT total per day
"""

import sys
import json
import os
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from tqdm import tqdm

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.config import *


def load_token_id():
    """Load token ID from market info file"""
    if not os.path.exists(MARKET_INFO_FILE):
        print(f"❌ Market info file not found: {MARKET_INFO_FILE}")
        print(f"\n💡 Run step 1 first: python 01_find_market.py")
        sys.exit(1)

    with open(MARKET_INFO_FILE, 'r') as f:
        market_info = json.load(f)

    token_id = market_info.get('token_id')
    if not token_id:
        print(f"❌ No token_id found in market info")
        sys.exit(1)

    return token_id, market_info


def download_orderbook_day(token_id, date, api_key):
    """
    Download all orderbook snapshots for a single day

    Args:
        token_id: Token ID to query
        date: datetime object for the day
        api_key: Dome API key

    Returns:
        List of orderbook snapshots
    """
    # Set time range for the day (UTC)
    start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    url = f"{DOME_API_BASE_URL}/polymarket/orderbooks"
    headers = {"Authorization": f"Bearer {api_key}"}

    all_snapshots = []
    pagination_key = None
    page = 1

    while True:
        params = {
            "token_id": token_id,
            "start_time": start_ms,
            "end_time": end_ms,
            "limit": PAGINATION_LIMIT  # Max per page
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

            # Check pagination
            pagination = data.get('pagination', {})
            has_more = pagination.get('has_more', False)
            pagination_key = pagination.get('pagination_key')

            # Show progress for days with many pages
            if page % 5 == 0:
                print(f"  Page {page}: {len(all_snapshots)} snapshots so far...")

            if not has_more:
                break  # No more pages

            page += 1
            time.sleep(DELAY_BETWEEN_PAGES)  # Rate limiting

        except Exception as e:
            print(f"  ❌ Error: {e}")
            break

    return all_snapshots


def main():
    print("\n")
    print("=" * 80)
    print(" STEP 2: DOWNLOAD ORDERBOOK DATA")
    print("=" * 80)

    # Get API key
    api_key = os.getenv('DOME_API_KEY')
    if not api_key:
        print("\n❌ DOME_API_KEY not found in .env file")
        sys.exit(1)

    print(f"✅ API key loaded")

    # Load token ID
    token_id, market_info = load_token_id()
    print(f"✅ Token ID loaded: {token_id[:30]}...")
    print(f"   Market: {market_info.get('question', 'N/A')}")

    # Parse date range
    if len(sys.argv) > 2 and sys.argv[1] == '--start-date':
        start_date_str = sys.argv[2]
        end_date_str = sys.argv[4] if len(sys.argv) > 4 else START_DATE
    else:
        start_date_str = START_DATE
        end_date_str = END_DATE

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    num_days = (end_date - start_date).days + 1

    print(f"\n📅 Date range:")
    print(f"   Start: {start_date.date()}")
    print(f"   End: {end_date.date()}")
    print(f"   Days: {num_days}")

    # Download day by day
    print("\n" + "=" * 80)
    print("DOWNLOADING ORDERBOOK SNAPSHOTS")
    print("=" * 80)

    all_snapshots = []
    current_date = start_date

    with tqdm(total=num_days, desc="Downloading") as pbar:
        while current_date <= end_date:
            pbar.set_description(f"Downloading {current_date.date()}")

            snapshots = download_orderbook_day(token_id, current_date, api_key)

            if snapshots:
                all_snapshots.extend(snapshots)
                pbar.write(f"  ✅ {current_date.date()}: {len(snapshots)} snapshots")
            else:
                pbar.write(f"  ⚠️  {current_date.date()}: No data")

            current_date += timedelta(days=1)
            pbar.update(1)

            if current_date <= end_date:
                time.sleep(DELAY_BETWEEN_DAYS)  # Rate limiting between days

    # Save raw data
    print("\n" + "=" * 80)
    print("SAVING RAW DATA")
    print("=" * 80)

    if not all_snapshots:
        print("\n❌ No data downloaded!")
        print("\n💡 Possible reasons:")
        print("   1. Date range has no data")
        print("   2. Token ID is incorrect")
        print("   3. Market wasn't active during this period")
        sys.exit(1)

    # Create data directory
    Path(RAW_ORDERBOOK_FILE).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving {len(all_snapshots)} snapshots...")
    with open(RAW_ORDERBOOK_FILE, 'w') as f:
        json.dump(all_snapshots, f, indent=2)

    file_size_mb = os.path.getsize(RAW_ORDERBOOK_FILE) / (1024 * 1024)
    print(f"✅ Saved to: {RAW_ORDERBOOK_FILE}")
    print(f"   File size: {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("✅ SUCCESS")
    print("=" * 80)

    print(f"\n📊 Download Summary:")
    print(f"   Total snapshots: {len(all_snapshots)}")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    print(f"   Days: {num_days}")
    print(f"   Avg per day: {len(all_snapshots) / num_days:.1f}")

    # Time span
    if all_snapshots:
        first_ts = all_snapshots[0]['timestamp']
        last_ts = all_snapshots[-1]['timestamp']
        first_time = datetime.fromtimestamp(first_ts / 1000)
        last_time = datetime.fromtimestamp(last_ts / 1000)
        print(f"\n   First snapshot: {first_time}")
        print(f"   Last snapshot: {last_time}")

    print(f"\n📝 Next Step:")
    print(f"   Run: python 03_process_orderbooks.py")
    print()


if __name__ == "__main__":
    main()
