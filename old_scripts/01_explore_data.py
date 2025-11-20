"""
01_explore_data.py
==================
Phase 1: Market Discovery and Data Exploration

This script:
1. Tests Dome API connection
2. Searches for the NYC mayor election market with Mamdani and Cuomo
3. Extracts market metadata
4. Identifies token_ids for each candidate
5. Verifies orderbook endpoints availability
6. Assesses data quality and availability

Usage:
    python 01_explore_data.py

Requirements:
    - DOME_API_KEY environment variable set
    - Or create .env file with: DOME_API_KEY=your_key_here
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


class DomeAPI:
    """Simple wrapper for Dome API using direct HTTP requests"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

    def get_markets(self, limit=1000, status="active"):
        """Get list of Polymarket markets"""
        url = f"{BASE_URL}/polymarket/markets"
        params = {"limit": limit}
        if status:
            params["status"] = status

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_orderbook_current(self, token_id):
        """Get current orderbook for a token"""
        url = f"{BASE_URL}/polymarket/orderbook/current"
        params = {"token_id": token_id}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_orderbook_history(self, token_id, start_time, end_time, interval=10):
        """Get historical orderbook snapshots"""
        url = f"{BASE_URL}/polymarket/orderbook/history"
        params = {
            "token_id": token_id,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()


def test_api_connection():
    """Test basic Dome API connectivity"""
    print("=" * 80)
    print("STEP 1: Testing Dome API Connection")
    print("=" * 80)

    api_key = os.getenv('DOME_API_KEY')

    if not api_key:
        print("\n❌ ERROR: DOME_API_KEY not found!")
        print("\nPlease set your API key:")
        print("  Option 1: export DOME_API_KEY='your_key_here'")
        print("  Option 2: Create .env file with DOME_API_KEY=your_key_here")
        print("\nGet your API key at: https://www.domeapi.io/")
        return None

    try:
        # Test connection with a simple API call
        dome = DomeAPI(api_key)
        print(f"✅ API key loaded: {api_key[:10]}...")

        # Test with a simple request
        test_url = f"{BASE_URL}/polymarket/markets"
        response = requests.get(
            test_url,
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": 1}
        )

        if response.status_code == 200:
            print("✅ API connection successful!")
            return dome
        else:
            print(f"❌ API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Connection failed - {e}")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


def find_nyc_mayor_market(dome):
    """Search for NYC mayor election market"""
    print("\n" + "=" * 80)
    print("STEP 2: Searching for NYC Mayor Election Market")
    print("=" * 80)

    try:
        # Get all active markets
        print("\nFetching markets from Dome API...")
        markets = dome.get_markets(limit=1000, status="active")

        # Handle different response formats
        if isinstance(markets, dict) and 'markets' in markets:
            markets = markets['markets']

        print(f"✅ Retrieved {len(markets)} active markets")

        # Search for NYC mayor markets
        print("\nSearching for NYC mayor markets...")
        nyc_markets = [
            m for m in markets
            if any(keyword in str(m.get('title', '')).lower()
                   for keyword in ['nyc', 'new york', 'mayor'])
        ]

        print(f"\nFound {len(nyc_markets)} NYC/mayor-related markets:")
        for i, market in enumerate(nyc_markets[:10], 1):  # Show first 10
            print(f"\n  {i}. {market.get('title', 'N/A')}")
            print(f"     Slug: {market.get('slug', 'N/A')}")
            print(f"     Status: {market.get('status', 'N/A')}")
            print(f"     Volume: ${market.get('volume', 0):,.2f}")

        # Find specific Mamdani/Cuomo market
        print("\n" + "-" * 80)
        print("Searching for Mamdani/Cuomo market...")
        target_market = None

        for market in nyc_markets:
            title_lower = str(market.get('title', '')).lower()
            if 'mamdani' in title_lower or 'cuomo' in title_lower:
                target_market = market
                break

        if target_market:
            print("\n✅ FOUND TARGET MARKET!")
            print(f"\nTitle: {target_market.get('title', 'N/A')}")
            print(f"Slug: {target_market.get('slug', 'N/A')}")
            print(f"Market ID: {target_market.get('market_id', 'N/A')}")
            print(f"Condition ID: {target_market.get('condition_id', 'N/A')}")
            print(f"Token ID: {target_market.get('token_id', 'N/A')}")
            print(f"Status: {target_market.get('status', 'N/A')}")
            print(f"Volume: ${target_market.get('volume', 0):,.2f}")

            # Convert timestamps
            if 'created_at' in target_market and target_market['created_at']:
                created = datetime.fromtimestamp(target_market['created_at'])
                print(f"Created: {created}")

            if 'close_time' in target_market and target_market['close_time']:
                closes = datetime.fromtimestamp(target_market['close_time'])
                print(f"Closes: {closes}")

            # Save market info
            with open('data/market_info.json', 'w') as f:
                json.dump(target_market, f, indent=2)
            print("\n💾 Market info saved to: data/market_info.json")

            return target_market

        else:
            print("\n⚠️  WARNING: Specific Mamdani/Cuomo market not found")
            print("\nPossible reasons:")
            print("  1. Market hasn't been created yet")
            print("  2. Market is resolved (try with resolved markets)")
            print("  3. Search terms don't match market title")

            if nyc_markets:
                print("\n💡 TIP: Review the NYC markets listed above")
                print("      You may need to manually identify the correct market")

                # Save all NYC markets for reference
                with open('data/nyc_markets_all.json', 'w') as f:
                    json.dump(nyc_markets, f, indent=2)
                print("\n💾 All NYC markets saved to: data/nyc_markets_all.json")

            return None

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        return None


def verify_orderbook_endpoint(dome, market):
    """Test if orderbook endpoints work"""
    print("\n" + "=" * 80)
    print("STEP 3: Verifying Orderbook Endpoints")
    print("=" * 80)

    if not market:
        print("\n⚠️  No market found - cannot test orderbook endpoints")
        return False

    # Try to get current orderbook
    token_id = market.get('token_id')

    if not token_id:
        print("\n⚠️  No token_id in market data")
        print("This is expected for multi-outcome markets")
        print("You'll need to find token_ids for each candidate separately")
        return None

    try:
        print(f"\nTesting current orderbook with token_id: {token_id[:20]}...")
        current_ob = dome.get_orderbook_current(token_id)
        print("✅ Current orderbook endpoint works!")
        print(f"\nSample data:")
        print(f"  Best bid: {current_ob.get('best_bid', 'N/A')}")
        print(f"  Best ask: {current_ob.get('best_ask', 'N/A')}")
        print(f"  Spread: {current_ob.get('spread', 'N/A')}")

        # Try historical orderbook
        print("\nTesting historical orderbook endpoint...")
        try:
            # Test with just 1 hour of data
            end_time = int(datetime.now().timestamp())
            start_time = end_time - 3600  # 1 hour ago

            history = dome.get_orderbook_history(
                token_id=token_id,
                start_time=start_time,
                end_time=end_time,
                interval=60  # 1 minute intervals for test
            )

            if isinstance(history, list):
                print(f"✅ Historical orderbook endpoint works!")
                print(f"   Retrieved {len(history)} snapshots (1 hour test)")

                if len(history) > 0:
                    print(f"\nSample snapshot:")
                    sample = history[0]
                    print(f"  Timestamp: {datetime.fromtimestamp(sample.get('timestamp', 0))}")
                    print(f"  Mid price: {sample.get('mid_price', 'N/A')}")
                    print(f"  Bid levels: {len(sample.get('bids', []))}")
                    print(f"  Ask levels: {len(sample.get('asks', []))}")

                return True
            else:
                print("⚠️  Unexpected response format from history endpoint")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("❌ Historical orderbook endpoint NOT FOUND (404)")
                print("\n📧 ACTION REQUIRED:")
                print("   Email: kurush@domeapi.com")
                print("   Subject: Request for Historical Orderbook Data")
                print("   Message: Mention academic research replicating Cont et al. (2011)")
                return False
            else:
                print(f"❌ HTTP Error: {e.response.status_code}")
                print(f"Response: {e.response.text}")
                return None

    except Exception as e:
        print(f"\n⚠️  Error testing orderbook: {e}")
        import traceback
        traceback.print_exc()
        return None


def identify_token_ids(market):
    """Guide for identifying token_ids for each candidate"""
    print("\n" + "=" * 80)
    print("STEP 4: Identifying Token IDs for Each Candidate")
    print("=" * 80)

    print("\n⚠️  MULTI-OUTCOME MARKET CHALLENGE:")
    print("   NYC mayor race has multiple candidates (Mamdani, Cuomo, others)")
    print("   Each candidate has a separate token_id")

    if market and market.get('token_id'):
        print(f"\n✅ Found one token_id in market data: {market['token_id']}")
        print("   This might be for one candidate")
        print("   You'll need to find token_ids for the other candidates")

    print("\n📋 MANUAL STEPS TO FIND TOKEN IDs:")
    print("\n1. Visit Polymarket website:")
    if market and market.get('slug'):
        print(f"   https://polymarket.com/event/{market['slug']}")
    else:
        print("   https://polymarket.com/ (search for NYC mayor)")

    print("\n2. Open browser DevTools (F12) → Network tab")
    print("   - Click on each candidate outcome")
    print("   - Look for API calls containing token_id")
    print("   - Copy the token_id for each candidate")

    print("\n3. Create data/token_ids.json with format:")
    print("   {")
    print('     "mamdani": "TOKEN_ID_HERE",')
    print('     "cuomo": "TOKEN_ID_HERE",')
    print('     "other_candidate": "TOKEN_ID_HERE"')
    print("   }")

    print("\n4. Alternative: Email Dome Support")
    print("   - kurush@domeapi.com")
    if market and market.get('slug'):
        print(f"   - Market slug: {market['slug']}")
    print("   - Request: Token IDs for all candidates in NYC mayor race")


def main():
    """Main exploration workflow"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "OFI ANALYSIS - PHASE 1: DATA EXPLORATION" + " " * 22 + "║")
    print("║" + " " * 15 + "NYC Mayor Election Market" + " " * 38 + "║")
    print("╚" + "═" * 78 + "╝")

    # Step 1: Test API
    dome = test_api_connection()
    if not dome:
        return

    # Step 2: Find market
    market = find_nyc_mayor_market(dome)

    # Step 3: Verify orderbook endpoint
    orderbook_works = verify_orderbook_endpoint(dome, market)

    # Step 4: Token ID guidance
    identify_token_ids(market)

    # Summary
    print("\n" + "=" * 80)
    print("EXPLORATION SUMMARY")
    print("=" * 80)

    if market:
        print("\n✅ Market found and metadata saved")
        print("📁 Location: data/market_info.json")
    else:
        print("\n⚠️  Market not found - review data/nyc_markets_all.json")

    if orderbook_works:
        print("✅ Orderbook endpoints working!")
    elif orderbook_works == False:
        print("❌ Historical orderbook endpoint not available - contact support")
    else:
        print("⚠️  Could not verify orderbook endpoint")

    print("\n📋 NEXT STEPS:")
    print("   1. Review market info and identify the correct market")
    print("   2. Obtain token_ids for each candidate (see Step 4 above)")
    print("   3. Create data/token_ids.json")
    print("   4. Run: python 02_download_orderbook.py")

    print("\n" + "=" * 80)
    print("Exploration complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
