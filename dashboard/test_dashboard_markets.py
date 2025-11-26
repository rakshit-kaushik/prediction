"""
Quick test to verify both markets can be loaded by the dashboard
"""
import pandas as pd
import json
from pathlib import Path

# Market configurations
MARKETS = {
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "ofi_oct15_nov20_combined.csv",
        "orderbook_file": "orderbook_oct15_nov20_combined.csv",
        "market_info_file": "market_info.json",
    },
    "NYC Mayoral Election 2025 (Zohran Mamdani)": {
        "ofi_file": "nyc_mayor_oct15_nov04_ofi.csv",
        "orderbook_file": "nyc_mayor_oct15_nov04_processed.csv",
        "market_info_file": "nyc_mayor_market_info.json",
    }
}

DATA_DIR = Path("data")

def test_market(market_name, config):
    """Test loading a single market"""
    print(f"\n{'='*80}")
    print(f"Testing: {market_name}")
    print('='*80)

    try:
        # Load OFI data
        ofi_path = DATA_DIR / config["ofi_file"]
        print(f"\nLoading OFI data: {ofi_path}")
        ofi_df = pd.read_csv(ofi_path)
        print(f"  ✓ Loaded {len(ofi_df):,} rows × {len(ofi_df.columns)} columns")
        print(f"  OFI range: {ofi_df['ofi'].min():.2f} to {ofi_df['ofi'].max():.2f}")

        # Load orderbook data
        orderbook_path = DATA_DIR / config["orderbook_file"]
        print(f"\nLoading orderbook data: {orderbook_path}")
        orderbook_df = pd.read_csv(orderbook_path)
        print(f"  ✓ Loaded {len(orderbook_df):,} rows × {len(orderbook_df.columns)} columns")
        print(f"  Price range: ${orderbook_df['mid_price'].min():.4f} to ${orderbook_df['mid_price'].max():.4f}")

        # Load market info
        info_path = DATA_DIR / config["market_info_file"]
        print(f"\nLoading market info: {info_path}")
        with open(info_path, 'r') as f:
            market_info = json.load(f)
        print(f"  ✓ Market: {market_info.get('question', 'N/A')}")
        print(f"  Volume: ${float(market_info.get('volume', 0)):,.2f}")

        print(f"\n✅ {market_name} - ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ {market_name} - FAILED: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("DASHBOARD MARKET LOADING TEST")
    print("="*80)

    results = {}
    for market_name, config in MARKETS.items():
        results[market_name] = test_market(market_name, config)

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for market_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {market_name}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL MARKETS CAN BE LOADED SUCCESSFULLY")
    else:
        print("❌ SOME MARKETS FAILED TO LOAD")
    print("="*80 + "\n")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
