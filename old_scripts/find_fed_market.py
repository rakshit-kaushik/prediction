"""
find_fed_market.py
==================
Find token ID for Fed interest rate market

This script:
1. Tests Dome API connection
2. Searches for the Fed rate decrease market
3. Extracts token_id and market metadata
4. Saves market info for later use

Usage:
    python find_fed_market.py

Target Market:
    fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting
"""

import os
from dotenv import load_dotenv
from datetime import datetime
import json
import requests

# Load environment variables
load_dotenv()

# Dome API configuration
BASE_URL = "https://api.domeapi.io/v1"


def find_fed_market():
    """Search for Fed interest rate market"""
    print("\n" + "=" * 80)
    print("Searching for Fed Interest Rate Market")
    print("=" * 80)

    api_key = os.getenv('DOME_API_KEY')

    if not api_key:
        print("\n❌ ERROR: DOME_API_KEY not found!")
        print("\nPlease set your API key in .env file")
        return None

    print(f"✅ API key loaded: {api_key[:10]}...")

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{BASE_URL}/polymarket/markets"

        # Target market slug
        target_slug = "fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting"

        print(f"\nSearching for market: {target_slug}")
        print("Trying multiple search strategies...")

        # Try different status filters
        for status in ["active", "closed", None]:
            params = {"limit": 2000}
            if status:
                params["status"] = status

            print(f"\n  Searching {status or 'all'} markets...")

            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"  ⚠️  API returned status {response.status_code}")
                continue

            data = response.json()
            markets = data.get('markets', data) if isinstance(data, dict) else data

            if not markets:
                print(f"  No markets found")
                continue

            print(f"  Retrieved {len(markets)} markets")

            # Search for exact match first
            for market in markets:
                if market.get('slug') == target_slug:
                    print(f"\n✅ FOUND EXACT MATCH!")
                    print_market_info(market)
                    save_market_info(market, 'fed_rates_market')
                    return market

            # Try case-insensitive match
            target_lower = target_slug.lower()
            for market in markets:
                if market.get('slug', '').lower() == target_lower:
                    print(f"\n✅ FOUND MATCH (case-insensitive)!")
                    print_market_info(market)
                    save_market_info(market, 'fed_rates_market')
                    return market

            # Try substring match for Fed/interest/rate keywords
            print("  Trying keyword search (fed, interest, rate, december)...")
            fed_markets = []
            for market in markets:
                slug = market.get('slug', '').lower()
                title = market.get('title', '').lower()
                if any(keyword in slug or keyword in title
                       for keyword in ['fed', 'interest', 'rate', 'december', '25-bps']):
                    fed_markets.append(market)

            if fed_markets:
                print(f"\n  Found {len(fed_markets)} Fed/interest rate markets:")
                for i, m in enumerate(fed_markets[:10], 1):
                    print(f"\n    {i}. {m.get('title', 'N/A')}")
                    print(f"       Slug: {m.get('slug', 'N/A')}")
                    print(f"       Status: {m.get('status', 'N/A')}")

                # Check if any match our target
                for market in fed_markets:
                    if target_slug in market.get('slug', ''):
                        print(f"\n✅ FOUND TARGET MARKET!")
                        print_market_info(market)
                        save_market_info(market, 'fed_rates_market')
                        return market

        print(f"\n❌ Could not find market: {target_slug}")
        print("\n💡 The market may:")
        print("   - Not exist yet")
        print("   - Have a different slug")
        print("   - Be resolved/closed")

        if fed_markets:
            print(f"\n💾 Saving {len(fed_markets)} Fed-related markets for manual review...")
            with open('data/fed_markets_all.json', 'w') as f:
                json.dump(fed_markets, f, indent=2)
            print("   Saved to: data/fed_markets_all.json")

        return None

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_market_info(market):
    """Print formatted market information"""
    print(f"\nTitle: {market.get('title', 'N/A')}")
    print(f"Slug: {market.get('slug', 'N/A')}")
    print(f"Token ID: {market.get('token_id', 'N/A')}")
    print(f"Market ID: {market.get('market_id', 'N/A')}")
    print(f"Condition ID: {market.get('condition_id', 'N/A')}")
    print(f"Status: {market.get('status', 'N/A')}")
    print(f"Volume: ${market.get('volume', 0):,.2f}")

    if 'created_at' in market and market['created_at']:
        created = datetime.fromtimestamp(market['created_at'])
        print(f"Created: {created}")

    if 'close_time' in market and market['close_time']:
        closes = datetime.fromtimestamp(market['close_time'])
        print(f"Closes: {closes}")


def save_market_info(market, filename):
    """Save market info to JSON file"""
    filepath = f'data/{filename}.json'
    with open(filepath, 'w') as f:
        json.dump(market, f, indent=2)
    print(f"\n💾 Market info saved to: {filepath}")


def main():
    """Main execution"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "FINDING FED INTEREST RATE MARKET" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")

    market = find_fed_market()

    print("\n" + "=" * 80)
    print("SEARCH SUMMARY")
    print("=" * 80)

    if market:
        print("\n✅ Market found and saved!")
        print(f"📁 Location: data/fed_rates_market.json")
        print(f"\n🔑 Token ID: {market.get('token_id', 'N/A')}")
        print("\n📋 Next step: Run download_fed_rates_data.py")
    else:
        print("\n❌ Market not found")
        print("📁 Review: data/fed_markets_all.json (if created)")
        print("\n💡 Try:")
        print("   1. Check if market slug is correct on Polymarket website")
        print("   2. Review fed_markets_all.json for similar markets")
        print("   3. Use Token ID directly if you have it")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
