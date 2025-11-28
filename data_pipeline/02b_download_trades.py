"""
02b_download_trades.py
=======================
STEP 2B: Download trade history for date range

WHAT IT DOES:
-------------
Input:  Token ID (from step 1) + Date range (from config)
Output: Raw trade data saved to trades_raw.json

HOW IT WORKS:
------------
1. Reads token ID from market_info.json (created in step 1)
2. For each day in date range:
   - Queries Dome API for trade history
   - Handles pagination (fetches all pages if >100 trades)
   - Saves raw JSON with all trade details
3. Shows progress and statistics

USAGE:
------
    python data_pipeline/02b_download_trades.py

    # Or with custom dates (overrides config):
    python data_pipeline/02b_download_trades.py --start-date 2025-10-15 --end-date 2025-10-31

WHAT YOU GET:
------------
- data/trades_raw.json containing array of trades:
  [
    {
      "token_id": "58519...",
      "side": "BUY",
      "price": 0.65,
      "shares_normalized": 4.995,
      "timestamp": 1757008834,
      ...
    },
    ...
  ]

IMPORTANT:
----------
- Trades are actual executions (not just orderbook updates)
- Timestamp is Unix time in SECONDS (not milliseconds like orderbook)
- side is either "BUY" or "SELL"
- shares_normalized is the actual trade size (shares / 1000000)
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
        print(f"\n💡 Run step 1 first: python data_pipeline/01_find_market.py")
        sys.exit(1)

    with open(MARKET_INFO_FILE, 'r') as f:
        market_info = json.load(f)

    token_id = market_info.get('token_id')
    if not token_id:
        print(f"❌ No token_id found in market info")
        sys.exit(1)

    return token_id, market_info


def make_request_with_retry(url, headers, params, max_retries=5):
    """Make API request with exponential backoff retry for rate limiting"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                print(f"\n⚠️  Rate limited (429). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            wait_time = (2 ** attempt) * 2
            print(f"\n❌ API error: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)

    return None  # All retries failed


def download_trades_day(token_id, date, api_key):
    """
    Download all trades for a single day

    Args:
        token_id: Token ID to query
        date: datetime object for the day
        api_key: Dome API key

    Returns:
        List of trades
    """
    # Set time range for the day (UTC)
    start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc)

    # Note: Trade API uses SECONDS not milliseconds
    start_sec = int(start.timestamp())
    end_sec = int(end.timestamp())

    all_trades = []
    offset = 0
    limit = 100  # Max trades per request

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

        data = make_request_with_retry(
            f"{DOME_API_BASE_URL}/polymarket/orders",
            headers,
            params
        )

        if data is None:
            print(f"\n❌ Max retries reached, stopping for this day")
            break

        orders = data.get('orders', [])
        if not orders:
            break  # No more data

        all_trades.extend(orders)

        # Check pagination
        pagination = data.get('pagination', {})
        has_more = pagination.get('has_more', False)

        if not has_more:
            break

        offset += limit
        time.sleep(1.0)  # Rate limiting between pagination requests

    return all_trades


def download_all_trades(token_id, start_date, end_date, api_key):
    """
    Download trades for entire date range

    Args:
        token_id: Token ID
        start_date: Start date (datetime)
        end_date: End date (datetime)
        api_key: Dome API key

    Returns:
        List of all trades
    """
    all_trades = []
    current_date = start_date

    # Create progress bar for days
    num_days = (end_date - start_date).days + 1
    pbar = tqdm(total=num_days, desc="Downloading trades", unit="day")

    while current_date <= end_date:
        # Download trades for this day
        day_trades = download_trades_day(token_id, current_date, api_key)

        if day_trades:
            all_trades.extend(day_trades)
            pbar.set_postfix({
                'total': len(all_trades),
                'today': len(day_trades)
            })

        current_date += timedelta(days=1)
        pbar.update(1)
        time.sleep(2.0)  # Delay between days to avoid rate limiting

    pbar.close()
    return all_trades


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("STEP 2B: DOWNLOAD TRADE HISTORY")
    print("=" * 80)

    # Load token ID
    print("\n📋 Loading market info...")
    token_id, market_info = load_token_id()

    market_name = market_info.get('question', 'Unknown Market')
    print(f"✓ Market: {market_name}")
    print(f"✓ Token ID: {token_id}")

    # Check API key
    api_key = os.getenv('DOME_API_KEY')
    if not api_key:
        print("\n❌ DOME_API_KEY not found in .env file")
        print("💡 Add your API key to .env file:")
        print("   DOME_API_KEY=your_key_here")
        sys.exit(1)

    # Parse dates (from command line or config)
    if len(sys.argv) >= 3:
        start_date_str = sys.argv[1]
        end_date_str = sys.argv[2]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        print(f"\n📅 Using command-line dates:")
    else:
        start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
        print(f"\n📅 Using config dates:")

    print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"   End:   {end_date.strftime('%Y-%m-%d')}")
    print(f"   Days:  {(end_date - start_date).days + 1}")

    # Download trades
    print("\n📥 Downloading trades...")
    all_trades = download_all_trades(token_id, start_date, end_date, api_key)

    if not all_trades:
        print("\n⚠️  No trades found for this date range")
        print("💡 This could mean:")
        print("   - Market had no trading activity")
        print("   - Date range is incorrect")
        print("   - API returned no data")
        return

    # Show statistics
    print("\n" + "=" * 80)
    print("TRADE STATISTICS")
    print("=" * 80)
    print(f"Total trades: {len(all_trades):,}")

    # Count buy/sell
    buy_trades = [t for t in all_trades if t.get('side') == 'BUY']
    sell_trades = [t for t in all_trades if t.get('side') == 'SELL']
    print(f"Buy trades: {len(buy_trades):,} ({len(buy_trades)/len(all_trades)*100:.1f}%)")
    print(f"Sell trades: {len(sell_trades):,} ({len(sell_trades)/len(all_trades)*100:.1f}%)")

    # Price range
    prices = [t['price'] for t in all_trades if 'price' in t]
    if prices:
        print(f"Price range: ${min(prices):.4f} - ${max(prices):.4f}")

    # Total volume
    volumes = [t.get('shares_normalized', 0) for t in all_trades]
    if volumes:
        print(f"Total volume: {sum(volumes):,.2f} shares")

    # Time range
    timestamps = [t['timestamp'] for t in all_trades if 'timestamp' in t]
    if timestamps:
        first_trade = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        last_trade = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)
        print(f"First trade: {first_trade.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Last trade:  {last_trade.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Save to file
    print("\n💾 Saving to file...")
    with open(TRADES_RAW_FILE, 'w') as f:
        json.dump(all_trades, f, indent=2)

    print(f"✓ Saved to: {TRADES_RAW_FILE}")
    print(f"✓ File size: {os.path.getsize(TRADES_RAW_FILE) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 80)
    print("✅ TRADE DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\n💡 Next step: python data_pipeline/03b_process_trades.py")


if __name__ == "__main__":
    main()
