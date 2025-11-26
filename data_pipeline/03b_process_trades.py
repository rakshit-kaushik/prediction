"""
03b_process_trades.py
=====================
STEP 3B: Process raw trade data into clean CSV format

WHAT IT DOES:
-------------
Input:  Raw trade history (from step 2b)
Output: Clean CSV with timestamp, side, price, size, direction

HOW IT WORKS:
------------
1. Loads raw JSON trade data
2. For each trade:
   - Converts Unix timestamp (seconds) to datetime
   - Adds direction: +1 for BUY, -1 for SELL
   - Extracts price, size, and other metadata
3. Saves to CSV ready for analysis

USAGE:
------
    python data_pipeline/03b_process_trades.py

WHAT YOU GET:
------------
- data/trades_processed.csv with columns:
  • timestamp, timestamp_sec
  • side (BUY/SELL)
  • direction (+1/-1)
  • price
  • shares_normalized (trade size)
  • Other trade metadata

IMPORTANT:
----------
- Trade timestamps are in SECONDS (not milliseconds like orderbook)
- direction: +1 = BUY, -1 = SELL (for calculating trade imbalance)
- shares_normalized is already divided by 1000000 in raw data
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


def process_trade(trade):
    """
    Process a single trade

    Args:
        trade: Raw trade dict with timestamp, side, price, shares

    Returns:
        dict: Processed trade with datetime and direction
    """
    # Trade timestamps are in SECONDS (not milliseconds)
    timestamp_sec = trade.get('timestamp', 0)
    if timestamp_sec == 0:
        return None

    timestamp = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)

    side = trade.get('side', '').upper()
    if side not in ['BUY', 'SELL']:
        return None

    # Direction: +1 for BUY, -1 for SELL
    direction = 1 if side == 'BUY' else -1

    price = float(trade.get('price', 0))
    shares_normalized = float(trade.get('shares_normalized', 0))

    return {
        'timestamp': timestamp,
        'timestamp_sec': timestamp_sec,
        'side': side,
        'direction': direction,
        'price': price,
        'shares_normalized': shares_normalized,
        'token_id': trade.get('token_id', ''),
        'transaction_hash': trade.get('transaction_hash', ''),
        'market': trade.get('market', ''),
    }


def main():
    print("\n")
    print("=" * 80)
    print(" STEP 3B: PROCESS TRADE DATA")
    print("=" * 80)

    # Check if raw file exists
    if not os.path.exists(TRADES_RAW_FILE):
        print(f"\n❌ Raw trade file not found: {TRADES_RAW_FILE}")
        print(f"\n💡 Run step 2b first: python data_pipeline/02b_download_trades.py")
        sys.exit(1)

    print(f"✅ Found raw trade file")

    # Load raw trades
    print(f"\n📂 Loading raw trade data...")
    with open(TRADES_RAW_FILE, 'r') as f:
        trades = json.load(f)

    print(f"✅ Loaded {len(trades)} trades")

    file_size_mb = os.path.getsize(TRADES_RAW_FILE) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")

    # Process trades
    print(f"\n⚙️  Processing trades...")
    print("   (Converting timestamps and adding direction)")

    records = []
    skipped = 0

    for trade in tqdm(trades, desc="Processing"):
        result = process_trade(trade)
        if result:
            records.append(result)
        else:
            skipped += 1

    if not records:
        print(f"\n❌ No valid trades processed!")
        print(f"\n💡 Possible reasons:")
        print(f"   1. All trades have invalid timestamps or sides")
        print(f"   2. Raw data format is incorrect")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\n✅ Processed {len(df)} trades")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} invalid trades")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("TRADE SUMMARY")
    print("=" * 80)

    print(f"\n📅 Time Range:")
    print(f"   First: {df['timestamp'].min()}")
    print(f"   Last:  {df['timestamp'].max()}")
    duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f"   Duration: {duration:.1f} days")

    print(f"\n📊 Trade Breakdown:")
    buy_trades = df[df['side'] == 'BUY']
    sell_trades = df[df['side'] == 'SELL']
    print(f"   Buy trades:  {len(buy_trades):,} ({len(buy_trades)/len(df)*100:.1f}%)")
    print(f"   Sell trades: {len(sell_trades):,} ({len(sell_trades)/len(df)*100:.1f}%)")

    print(f"\n💰 Price Statistics:")
    print(f"   Price range: ${df['price'].min():.4f} to ${df['price'].max():.4f}")
    print(f"   Mean price: ${df['price'].mean():.4f}")
    print(f"   Median price: ${df['price'].median():.4f}")

    print(f"\n📦 Volume Statistics:")
    total_volume = df['shares_normalized'].sum()
    buy_volume = buy_trades['shares_normalized'].sum()
    sell_volume = sell_trades['shares_normalized'].sum()
    print(f"   Total volume: {total_volume:,.2f} shares")
    print(f"   Buy volume:  {buy_volume:,.2f} shares ({buy_volume/total_volume*100:.1f}%)")
    print(f"   Sell volume: {sell_volume:,.2f} shares ({sell_volume/total_volume*100:.1f}%)")
    print(f"   Avg trade size: {df['shares_normalized'].mean():.2f} shares")
    print(f"   Median trade size: {df['shares_normalized'].median():.2f} shares")

    print(f"\n🔄 Trade Imbalance:")
    trade_imbalance = buy_volume - sell_volume
    print(f"   Net imbalance: {trade_imbalance:+,.2f} shares")
    print(f"   Imbalance ratio: {trade_imbalance/total_volume:+.4f}")

    # Trades per day
    trades_per_day = len(df) / duration if duration > 0 else 0
    print(f"\n📈 Activity:")
    print(f"   Avg trades per day: {trades_per_day:.1f}")
    print(f"   Unique prices: {df['price'].nunique()}")

    # Save processed data
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    # Create data directory
    Path(TRADES_PROCESSED_FILE).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to: {TRADES_PROCESSED_FILE}")
    df.to_csv(TRADES_PROCESSED_FILE, index=False)

    file_size_mb = os.path.getsize(TRADES_PROCESSED_FILE) / (1024 * 1024)
    print(f"✅ Saved {len(df)} rows × {len(df.columns)} columns")
    print(f"   File size: {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("✅ SUCCESS")
    print("=" * 80)

    print(f"\n📋 Processing Summary:")
    print(f"   Input:  {len(trades)} raw trades")
    print(f"   Output: {len(df)} processed trades")
    print(f"   Saved to: {TRADES_PROCESSED_FILE}")

    print(f"\n📝 Data Ready for Analysis!")
    print(f"   • Trade direction encoded: +1 (BUY), -1 (SELL)")
    print(f"   • {df['price'].nunique()} unique price points")
    print(f"   • Ready for trade imbalance calculation")

    print(f"\n💡 Next Steps:")
    print(f"   • Combine with orderbook data for comprehensive OFI analysis")
    print(f"   • Calculate trade imbalance metrics")
    print(f"   • Analyze trade flow patterns")
    print()


if __name__ == "__main__":
    main()
