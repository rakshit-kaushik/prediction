"""
01_find_market.py
=================
STEP 1: Find token ID for a given market slug

WHAT IT DOES:
-------------
Input:  Market slug (from Polymarket URL)
Output: Token ID + market metadata saved to market_info.json

HOW IT WORKS:
------------
1. Queries Polymarket Gamma API with the market slug
2. Extracts token ID from the response
3. Saves all market metadata for reference

USAGE:
------
    python 01_find_market.py

    # Or with custom slug:
    python 01_find_market.py "your-market-slug-here"

WHAT YOU GET:
------------
- data/market_info.json containing:
  • Token ID (needed for step 2)
  • Market title, description
  • Current prices
  • Volume, liquidity
  • All metadata
"""

import sys
import json
import requests
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.config import *


def find_market(market_slug):
    """
    Find market info using Polymarket Gamma API

    Args:
        market_slug: Market slug from URL

    Returns:
        dict: Market information including token_id
    """
    print(f"\n🔍 Searching for market: {market_slug}")

    url = f"{POLYMARKET_GAMMA_API}/markets?slug={market_slug}"

    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return None

        data = response.json()

        if not data or len(data) == 0:
            print(f"❌ No market found with slug: {market_slug}")
            print("\n💡 Tips:")
            print("   1. Check the slug is correct (copy from Polymarket URL)")
            print("   2. Market might not exist or be spelled differently")
            return None

        market = data[0]  # Gamma API returns array

        # Extract key information
        print(f"\n✅ Found market!")
        print(f"   Title: {market.get('question', 'N/A')}")
        print(f"   Market ID: {market.get('id', 'N/A')}")
        print(f"   Status: {'Active' if market.get('active') else 'Inactive'}")
        print(f"   Volume: ${market.get('volumeNum', 0):,.2f}")
        print(f"   Liquidity: ${market.get('liquidityNum', 0):,.2f}")

        # Get token IDs
        token_ids_str = market.get('clobTokenIds', '[]')
        token_ids = json.loads(token_ids_str)

        if len(token_ids) >= 2:
            print(f"\n🔑 Token IDs:")
            print(f"   Yes: {token_ids[0]}")
            print(f"   No:  {token_ids[1]}")

            # Store the Yes token (typically what we analyze)
            market['token_id'] = token_ids[0]
            market['token_id_yes'] = token_ids[0]
            market['token_id_no'] = token_ids[1]
        else:
            print(f"\n⚠️  Could not extract token IDs")
            market['token_id'] = None

        return market

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_market_info(market, output_path):
    """Save market info to JSON file"""
    # Create data directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(market, f, indent=2)

    print(f"\n💾 Saved market info to: {output_path}")


def main():
    print("\n")
    print("=" * 80)
    print(" STEP 1: FIND MARKET TOKEN ID")
    print("=" * 80)

    # Get slug from command line or config
    if len(sys.argv) > 1:
        slug = sys.argv[1]
        print(f"\nUsing slug from command line: {slug}")
    else:
        slug = MARKET_SLUG
        print(f"\nUsing slug from config: {slug}")

    # Find market
    market = find_market(slug)

    if not market:
        print("\n❌ FAILED: Could not find market")
        print("\n💡 Next steps:")
        print("   1. Check the market slug is correct")
        print("   2. Try searching on polymarket.com first")
        print("   3. Copy the exact slug from the URL")
        sys.exit(1)

    # Save to file
    save_market_info(market, MARKET_INFO_FILE)

    # Summary
    print("\n" + "=" * 80)
    print("✅ SUCCESS")
    print("=" * 80)

    print(f"\n📋 Market Details:")
    print(f"   Slug: {slug}")
    print(f"   Token ID: {market.get('token_id', 'N/A')}")
    print(f"   Saved to: {MARKET_INFO_FILE}")

    print(f"\n📝 Next Step:")
    print(f"   Run: python 02_download_orderbooks.py")
    print()


if __name__ == "__main__":
    main()
