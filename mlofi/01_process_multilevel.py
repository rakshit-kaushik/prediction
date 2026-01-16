"""
01_process_multilevel.py
========================
STEP 1: Extract multi-level orderbook data from raw JSON

WHAT IT DOES:
-------------
Input:  Raw orderbook snapshots (JSON with full depth)
Output: CSV with price/size for each level (1 to N) on both bid and ask sides

HOW IT WORKS:
-------------
1. Load raw orderbook JSON
2. For each snapshot:
   - Sort bids descending by price (best bid = highest)
   - Sort asks ascending by price (best ask = lowest)
   - Extract price/size for levels 1 to MAX_LEVELS
3. Calculate mid-price and spread
4. Save to CSV

OUTPUT COLUMNS:
--------------
timestamp, timestamp_ms,
bid_price_l1, bid_size_l1, ask_price_l1, ask_size_l1,
bid_price_l2, bid_size_l2, ask_price_l2, ask_size_l2,
...
bid_price_lN, bid_size_lN, ask_price_lN, ask_size_lN,
mid_price, spread,
n_bid_levels, n_ask_levels  (actual levels available in snapshot)

USAGE:
------
    python mlofi/01_process_multilevel.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    RAW_ORDERBOOK_FILE,
    MULTILEVEL_PROCESSED_FILE,
    MAX_LEVELS_TO_EXTRACT,
    EXISTING_PROCESSED_FILE,
    MLOFI_OUTPUT_DIR
)


def process_snapshot(snapshot, max_levels):
    """
    Process a single orderbook snapshot to extract multi-level data.

    Args:
        snapshot: Dict with 'bids', 'asks', 'timestamp' keys
        max_levels: Maximum number of levels to extract

    Returns:
        Dict with extracted data for this snapshot
    """
    # Extract timestamp
    timestamp_ms = snapshot.get('timestamp') or snapshot.get('indexedAt')
    if timestamp_ms is None:
        return None

    # Convert to datetime
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

    # Get bids and asks
    bids = snapshot.get('bids', [])
    asks = snapshot.get('asks', [])

    if not bids or not asks:
        return None

    # Sort bids descending (best bid = highest price)
    sorted_bids = sorted(bids, key=lambda x: -float(x['price']))

    # Sort asks ascending (best ask = lowest price)
    sorted_asks = sorted(asks, key=lambda x: float(x['price']))

    # Initialize result
    result = {
        'timestamp': timestamp,
        'timestamp_ms': timestamp_ms,
        'n_bid_levels': len(sorted_bids),
        'n_ask_levels': len(sorted_asks),
    }

    # Extract levels
    for level in range(1, max_levels + 1):
        # Bid level
        if level <= len(sorted_bids):
            result[f'bid_price_l{level}'] = float(sorted_bids[level - 1]['price'])
            result[f'bid_size_l{level}'] = float(sorted_bids[level - 1]['size'])
        else:
            result[f'bid_price_l{level}'] = np.nan
            result[f'bid_size_l{level}'] = 0.0

        # Ask level
        if level <= len(sorted_asks):
            result[f'ask_price_l{level}'] = float(sorted_asks[level - 1]['price'])
            result[f'ask_size_l{level}'] = float(sorted_asks[level - 1]['size'])
        else:
            result[f'ask_price_l{level}'] = np.nan
            result[f'ask_size_l{level}'] = 0.0

    # Calculate mid-price and spread (from level 1)
    best_bid = result['bid_price_l1']
    best_ask = result['ask_price_l1']

    if not np.isnan(best_bid) and not np.isnan(best_ask):
        result['mid_price'] = (best_bid + best_ask) / 2
        result['spread'] = best_ask - best_bid
        result['spread_pct'] = result['spread'] / result['mid_price'] * 100
    else:
        result['mid_price'] = np.nan
        result['spread'] = np.nan
        result['spread_pct'] = np.nan

    return result


def verify_against_existing(df_multilevel, existing_file):
    """
    Verify that Level 1 data matches existing processed data.

    Args:
        df_multilevel: DataFrame with multi-level data
        existing_file: Path to existing processed CSV

    Returns:
        bool: True if verification passes
    """
    print("\n" + "=" * 80)
    print("VERIFICATION: Comparing Level 1 with existing processed data")
    print("=" * 80)

    if not Path(existing_file).exists():
        print(f"⚠️  Existing file not found: {existing_file}")
        print("   Skipping verification (this is OK for first run)")
        return True

    # Load existing data
    df_existing = pd.read_csv(existing_file)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'], format='mixed')

    print(f"\n📊 Existing data: {len(df_existing)} snapshots")
    print(f"📊 Multi-level data: {len(df_multilevel)} snapshots")

    # Compare by timestamp_ms
    common_ts = set(df_multilevel['timestamp_ms']) & set(df_existing['timestamp_ms'])
    print(f"📊 Common timestamps: {len(common_ts)}")

    if len(common_ts) == 0:
        print("⚠️  No common timestamps found!")
        return False

    # Sample comparison
    sample_ts = list(common_ts)[:10]

    mismatches = 0
    for ts in sample_ts:
        ml_row = df_multilevel[df_multilevel['timestamp_ms'] == ts].iloc[0]
        ex_row = df_existing[df_existing['timestamp_ms'] == ts].iloc[0]

        # Compare best bid/ask prices
        ml_bid = ml_row['bid_price_l1']
        ex_bid = ex_row['best_bid_price']
        ml_ask = ml_row['ask_price_l1']
        ex_ask = ex_row['best_ask_price']

        if abs(ml_bid - ex_bid) > 0.0001 or abs(ml_ask - ex_ask) > 0.0001:
            mismatches += 1
            print(f"\n❌ Mismatch at ts={ts}:")
            print(f"   ML bid: {ml_bid:.4f}, Existing bid: {ex_bid:.4f}")
            print(f"   ML ask: {ml_ask:.4f}, Existing ask: {ex_ask:.4f}")

    if mismatches == 0:
        print(f"\n✅ All {len(sample_ts)} sampled timestamps match!")
        print("   Level 1 extraction is correct.")
        return True
    else:
        print(f"\n❌ {mismatches}/{len(sample_ts)} mismatches found!")
        return False


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "MULTI-LEVEL ORDERBOOK PROCESSING" + " " * 29 + "║")
    print("║" + " " * 20 + "Step 1: Extract All Levels" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not RAW_ORDERBOOK_FILE.exists():
        print(f"\n❌ Raw orderbook file not found: {RAW_ORDERBOOK_FILE}")
        print("   Please run the data pipeline first.")
        sys.exit(1)

    # Create output directory
    MLOFI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"\n📂 Loading raw orderbook data...")
    print(f"   File: {RAW_ORDERBOOK_FILE}")

    with open(RAW_ORDERBOOK_FILE, 'r') as f:
        raw_data = json.load(f)

    print(f"   Loaded {len(raw_data)} snapshots")

    # Analyze level distribution
    print("\n" + "-" * 80)
    print("ANALYZING ORDERBOOK DEPTH")
    print("-" * 80)

    bid_levels = [len(s.get('bids', [])) for s in raw_data[:1000]]
    ask_levels = [len(s.get('asks', [])) for s in raw_data[:1000]]

    print(f"\n📊 Bid levels (sample of 1000):")
    print(f"   Min: {min(bid_levels)}, Max: {max(bid_levels)}, Mean: {np.mean(bid_levels):.1f}")

    print(f"\n📊 Ask levels (sample of 1000):")
    print(f"   Min: {min(ask_levels)}, Max: {max(ask_levels)}, Mean: {np.mean(ask_levels):.1f}")

    # Determine max levels to extract
    max_available = max(max(bid_levels), max(ask_levels))
    levels_to_extract = min(MAX_LEVELS_TO_EXTRACT, max_available)

    print(f"\n📊 Will extract {levels_to_extract} levels (max available: {max_available})")

    # Process all snapshots
    print("\n" + "-" * 80)
    print("PROCESSING SNAPSHOTS")
    print("-" * 80)

    processed_data = []
    errors = 0

    for i, snapshot in enumerate(raw_data):
        if i % 10000 == 0:
            print(f"   Processing snapshot {i}/{len(raw_data)}...")

        try:
            result = process_snapshot(snapshot, levels_to_extract)
            if result:
                processed_data.append(result)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   ⚠️  Error at snapshot {i}: {e}")

    print(f"\n✓ Processed {len(processed_data)} snapshots ({errors} errors)")

    # Create DataFrame
    df = pd.DataFrame(processed_data)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Print statistics
    print("\n" + "-" * 80)
    print("DATA STATISTICS")
    print("-" * 80)

    print(f"\n📊 Total snapshots: {len(df)}")
    print(f"📊 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📊 Levels extracted: {levels_to_extract}")

    # Level availability
    print(f"\n📊 Level availability:")
    for level in [1, 5, 10, 20, levels_to_extract]:
        if level <= levels_to_extract:
            bid_avail = (df[f'bid_price_l{level}'].notna()).mean() * 100
            ask_avail = (df[f'ask_price_l{level}'].notna()).mean() * 100
            print(f"   Level {level}: Bid {bid_avail:.1f}%, Ask {ask_avail:.1f}%")

    # Price statistics at different levels
    print(f"\n📊 Best bid/ask (Level 1):")
    print(f"   Bid: {df['bid_price_l1'].mean():.4f} (mean), {df['bid_price_l1'].std():.4f} (std)")
    print(f"   Ask: {df['ask_price_l1'].mean():.4f} (mean), {df['ask_price_l1'].std():.4f} (std)")
    print(f"   Spread: {df['spread'].mean():.4f} (mean)")

    # Verify against existing data
    verify_against_existing(df, EXISTING_PROCESSED_FILE)

    # Save to CSV
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    df.to_csv(MULTILEVEL_PROCESSED_FILE, index=False)
    print(f"\n✓ Saved to: {MULTILEVEL_PROCESSED_FILE}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    # List columns
    print(f"\n📋 Columns saved:")
    print(f"   - timestamp, timestamp_ms")
    print(f"   - bid_price_l1 to bid_price_l{levels_to_extract}")
    print(f"   - bid_size_l1 to bid_size_l{levels_to_extract}")
    print(f"   - ask_price_l1 to ask_price_l{levels_to_extract}")
    print(f"   - ask_size_l1 to ask_size_l{levels_to_extract}")
    print(f"   - mid_price, spread, spread_pct")
    print(f"   - n_bid_levels, n_ask_levels")

    print("\n" + "=" * 80)
    print("✅ MULTI-LEVEL PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\n💡 Next step: python mlofi/02_calculate_mlofi.py")
    print("\n")

    return df


if __name__ == "__main__":
    main()
