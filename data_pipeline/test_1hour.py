"""Quick test: Download trades for just 1 hour to check if API caps at 1900"""
import sys
import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path
import time

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.config import DOME_API_BASE_URL

# Load token ID
with open('data/nyc_mayor_market_info.json', 'r') as f:
    market_info = json.load(f)
token_id = market_info['token_id']

api_key = os.getenv('DOME_API_KEY')

# Just 1 hour: Oct 16, 12:00-13:00 UTC
start = datetime(2025, 10, 16, 12, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 10, 16, 13, 0, 0, tzinfo=timezone.utc)

start_sec = int(start.timestamp())
end_sec = int(end.timestamp())

print(f"Testing: Oct 16, 12:00-13:00 UTC (1 hour)")
print(f"Start timestamp: {start_sec}")
print(f"End timestamp: {end_sec}")

headers = {
    'x-api-key': api_key,
    'accept': 'application/json'
}

all_trades = []
offset = 0
limit = 100

while True:
    params = {
        'token_id': token_id,
        'start_time': start_sec,
        'end_time': end_sec,
        'limit': limit,
        'offset': offset
    }

    response = requests.get(
        f"{DOME_API_BASE_URL}/polymarket/orders",
        headers=headers,
        params=params,
        timeout=30
    )

    if response.status_code == 429:
        print(f"Rate limited at offset {offset}")
        break

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        break

    data = response.json()
    orders = data.get('orders', [])
    pagination = data.get('pagination', {})

    if not orders:
        print(f"No more orders at offset {offset}")
        break

    all_trades.extend(orders)
    print(f"Page {offset//100 + 1}: Got {len(orders)} trades, total: {len(all_trades)}, has_more: {pagination.get('has_more')}")

    if not pagination.get('has_more', False):
        print("API says no more data")
        break

    offset += limit
    time.sleep(2)

print(f"\n=== RESULT ===")
print(f"Total trades for 1 hour: {len(all_trades)}")
if all_trades:
    first_ts = datetime.fromtimestamp(all_trades[0]['timestamp'], tz=timezone.utc)
    last_ts = datetime.fromtimestamp(all_trades[-1]['timestamp'], tz=timezone.utc)
    print(f"First trade: {first_ts}")
    print(f"Last trade: {last_ts}")
