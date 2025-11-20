"""
00_select_market_from_url.py
============================
Interactive market selection from Polymarket URL

WHAT IT DOES:
-------------
1. Takes a Polymarket URL (event or market)
2. Queries Gamma API to show all markets in that event
3. Let's you select which market you want
4. Updates config.py automatically
5. Optionally runs the rest of the pipeline

USAGE:
------
    # Interactive mode (prompts for URL)
    python 00_select_market_from_url.py

    # Command line with URL
    python 00_select_market_from_url.py "https://polymarket.com/event/fed-decision-in-december"

    # Auto-run pipeline after selection
    python 00_select_market_from_url.py --auto-run

    # Non-interactive: select market #1 and run pipeline
    python 00_select_market_from_url.py "URL" --select 1 --auto-run

EXAMPLE:
--------
    $ python 00_select_market_from_url.py

    Enter Polymarket URL: https://polymarket.com/event/fed-decision-in-december

    Found 5 markets for: fed-decision-in-december

    1. Fed decreases by 25 bps after December 2025 meeting?
       Yes: $0.30 | No: $0.70
       Volume: $14M
       Slug: fed-decreases-interest-rates-by-25-bps...

    2. Fed increases by 25 bps after December 2025 meeting?
       ...

    Select market (1-5): 1

    ✅ Selected: Fed decreases by 25 bps after December 2025 meeting?
    ✅ Updated config.py

    Run data pipeline now? (y/n): y
"""

import sys
import json
import requests
from urllib.parse import urlparse
from pathlib import Path
import subprocess
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = "data_pipeline/config.py"
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
TIMEOUT = 10


# ============================================================================
# URL PARSING
# ============================================================================

def parse_polymarket_url(url):
    """
    Extract event slug and optional market slug from Polymarket URL

    Args:
        url: Polymarket URL

    Returns:
        tuple: (event_slug, market_slug or None)

    Examples:
        "https://polymarket.com/event/fed-decision"
        -> ("fed-decision", None)

        "https://polymarket.com/event/fed-decision/fed-decreases-25bps"
        -> ("fed-decision", "fed-decreases-25bps")
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        parts = [p for p in path.split('/') if p]

        if 'event' in parts:
            idx = parts.index('event')

            # Check if market slug is in URL
            if len(parts) > idx + 2:
                event_slug = parts[idx + 1]
                market_slug = parts[idx + 2]
                return event_slug, market_slug
            elif len(parts) > idx + 1:
                event_slug = parts[idx + 1]
                return event_slug, None

        elif 'market' in parts:
            idx = parts.index('market')
            if len(parts) > idx + 1:
                market_slug = parts[idx + 1]
                return None, market_slug

        return None, None

    except Exception as e:
        print(f"❌ Error parsing URL: {e}")
        return None, None


# ============================================================================
# GAMMA API QUERIES
# ============================================================================

def query_gamma_api_markets(event_slug=None, market_slug=None):
    """
    Query Polymarket Gamma API for markets

    Args:
        event_slug: Optional event slug to filter by
        market_slug: Optional specific market slug

    Returns:
        list: List of market dictionaries
    """
    print(f"\n🔍 Querying Gamma API...")

    try:
        # Query all markets (Gamma API doesn't have great filtering)
        response = requests.get(GAMMA_API_URL, params={'limit': 1000}, timeout=TIMEOUT)

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return []

        all_markets = response.json()

        # Filter by event slug if provided
        if event_slug:
            # Match markets that contain the event slug
            filtered_markets = []
            for market in all_markets:
                market_slug_field = market.get('slug', market.get('market_slug', ''))

                # Check if event slug is in market slug or question
                if (event_slug.lower() in str(market_slug_field).lower() or
                    event_slug.lower() in str(market.get('question', '')).lower() or
                    event_slug.lower() in str(market.get('description', '')).lower()):
                    filtered_markets.append(market)

            return filtered_markets

        # Filter by specific market slug if provided
        elif market_slug:
            for market in all_markets:
                market_slug_field = market.get('slug', market.get('market_slug', ''))
                if market_slug.lower() == str(market_slug_field).lower():
                    return [market]

            return []

        return all_markets

    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {TIMEOUT} seconds")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error querying API: {e}")
        return []


# ============================================================================
# MARKET DISPLAY
# ============================================================================

def format_market_info(market, index):
    """Format a single market for display"""
    question = market.get('question', 'N/A')
    slug = market.get('slug', market.get('market_slug', 'N/A'))

    # Get prices
    outcome_prices = market.get('outcomePrices', '[]')
    try:
        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
        yes_price = float(prices[0]) if len(prices) > 0 else 0
        no_price = float(prices[1]) if len(prices) > 1 else 0
    except:
        yes_price = no_price = 0

    # Get volume
    volume = market.get('volumeNum', market.get('volume', 0))
    try:
        volume_str = f"${float(volume):,.0f}" if volume else "N/A"
    except:
        volume_str = "N/A"

    # Format slug (truncate if too long)
    slug_display = slug if len(slug) <= 60 else slug[:57] + "..."

    output = []
    output.append("─" * 80)
    output.append(f"{index}. {question}")
    output.append(f"   Prices: Yes ${yes_price:.2f} | No ${no_price:.2f}")
    output.append(f"   Volume: {volume_str}")
    output.append(f"   Slug: {slug_display}")
    output.append("─" * 80)

    return "\n".join(output)


def display_markets(markets):
    """Display all markets in a formatted list"""
    if not markets:
        print("\n❌ No markets found")
        return

    print(f"\n✅ Found {len(markets)} market{'s' if len(markets) > 1 else ''}\n")

    for i, market in enumerate(markets, 1):
        print(format_market_info(market, i))


# ============================================================================
# USER INTERACTION
# ============================================================================

def select_market_interactive(markets):
    """
    Interactive market selection

    Args:
        markets: List of market dicts

    Returns:
        dict: Selected market or None
    """
    if not markets:
        return None

    # If only one market, auto-select
    if len(markets) == 1:
        print(f"\n✅ Only one market found, auto-selecting:")
        print(f"   {markets[0].get('question', 'N/A')}")
        return markets[0]

    # Multiple markets - prompt user
    while True:
        try:
            choice = input(f"\nSelect market (1-{len(markets)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("❌ Cancelled by user")
                return None

            choice_num = int(choice)

            if 1 <= choice_num <= len(markets):
                selected = markets[choice_num - 1]
                print(f"\n✅ Selected: {selected.get('question', 'N/A')}")
                return selected
            else:
                print(f"❌ Please enter a number between 1 and {len(markets)}")

        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return None


def select_market_by_number(markets, number):
    """
    Select market by number (non-interactive)

    Args:
        markets: List of market dicts
        number: Market number (1-indexed)

    Returns:
        dict: Selected market or None
    """
    if not markets:
        return None

    if 1 <= number <= len(markets):
        selected = markets[number - 1]
        print(f"\n✅ Auto-selected market {number}: {selected.get('question', 'N/A')}")
        return selected
    else:
        print(f"❌ Invalid market number: {number} (valid: 1-{len(markets)})")
        return None


# ============================================================================
# CONFIG UPDATE
# ============================================================================

def update_config(market_slug):
    """
    Update config.py with selected market slug

    Args:
        market_slug: Market slug to set

    Returns:
        bool: Success
    """
    try:
        config_path = Path(CONFIG_FILE)

        if not config_path.exists():
            print(f"❌ Config file not found: {CONFIG_FILE}")
            return False

        # Read config
        with open(config_path, 'r') as f:
            lines = f.readlines()

        # Find and replace MARKET_SLUG line
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('MARKET_SLUG'):
                lines[i] = f'MARKET_SLUG = "{market_slug}"\n'
                updated = True
                break

        if not updated:
            print("❌ Could not find MARKET_SLUG in config.py")
            return False

        # Write back
        with open(config_path, 'w') as f:
            f.writelines(lines)

        print(f"\n✅ Updated config.py with market slug:")
        print(f"   {market_slug}")

        return True

    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_pipeline_steps():
    """
    Run the rest of the data pipeline (steps 01, 02, 03)

    Returns:
        bool: Success
    """
    print("\n" + "=" * 80)
    print("RUNNING DATA PIPELINE")
    print("=" * 80)

    steps = [
        ("01_find_market.py", "Find market token ID"),
        ("02_download_orderbooks.py", "Download orderbook snapshots"),
        ("03_process_orderbooks.py", "Process orderbooks"),
    ]

    for script, description in steps:
        print(f"\n📍 Step: {description}")
        print(f"   Running: {script}")

        try:
            result = subprocess.run(
                [sys.executable, f"data_pipeline/{script}"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per step
            )

            if result.returncode == 0:
                print(f"   ✅ {description} - Success")
            else:
                print(f"   ❌ {description} - Failed")
                print(f"\nError output:")
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print(f"   ❌ {description} - Timeout (>10 minutes)")
            return False
        except Exception as e:
            print(f"   ❌ {description} - Error: {e}")
            return False

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n📂 Data saved to: data/orderbook_processed.csv")

    return True


def ask_run_pipeline():
    """Ask user if they want to run the pipeline"""
    while True:
        try:
            response = input("\nRun data pipeline now? (y/n): ").strip().lower()

            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("\n💡 To run manually:")
                print("   python data_pipeline/01_find_market.py")
                print("   python data_pipeline/02_download_orderbooks.py")
                print("   python data_pipeline/03_process_orderbooks.py")
                return False
            else:
                print("Please enter 'y' or 'n'")

        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n")
    print("=" * 80)
    print(" POLYMARKET DATA PIPELINE - MARKET SELECTION")
    print("=" * 80)

    # Parse command line args
    auto_run = '--auto-run' in sys.argv
    select_number = None

    # Check for --select N argument
    for i, arg in enumerate(sys.argv):
        if arg == '--select' and i + 1 < len(sys.argv):
            try:
                select_number = int(sys.argv[i + 1])
            except ValueError:
                print(f"❌ Invalid --select value: {sys.argv[i + 1]}")
                sys.exit(1)

    # Get URL
    url = None
    for arg in sys.argv[1:]:
        if not arg.startswith('--') and not arg.isdigit():
            url = arg
            break

    if not url:
        try:
            url = input("\n📎 Enter Polymarket URL: ").strip()
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            sys.exit(0)

    if not url:
        print("❌ No URL provided")
        sys.exit(1)

    print(f"\n🔗 URL: {url}")

    # Parse URL
    event_slug, market_slug = parse_polymarket_url(url)

    if not event_slug and not market_slug:
        print("❌ Could not parse Polymarket URL")
        print("\n💡 Expected formats:")
        print("   https://polymarket.com/event/EVENT-SLUG")
        print("   https://polymarket.com/event/EVENT-SLUG/MARKET-SLUG")
        sys.exit(1)

    # If we have a market slug (either direct URL or from event/slug/market pattern),
    # use it directly without querying
    if market_slug:
        print(f"\n✅ Detected market URL with slug")
        print(f"   Market slug: {market_slug}")

        if update_config(market_slug):
            if auto_run or ask_run_pipeline():
                success = run_pipeline_steps()
                sys.exit(0 if success else 1)
        sys.exit(0)

    # Only event slug - need to query API for all markets in event
    print(f"\n🔍 Event: {event_slug}")
    markets = query_gamma_api_markets(event_slug=event_slug)

    if not markets:
        print(f"\n❌ No markets found for event: {event_slug}")
        print("\n💡 Tips:")
        print("   1. Check the event slug is correct")
        print("   2. Try the full market URL instead")
        print("   3. The market might not exist or be spelled differently")
        sys.exit(1)

    # Display markets
    display_markets(markets)

    # Select market
    if select_number is not None:
        selected_market = select_market_by_number(markets, select_number)
    else:
        selected_market = select_market_interactive(markets)

    if not selected_market:
        print("\n❌ No market selected")
        sys.exit(1)

    # Get slug from selected market
    selected_slug = selected_market.get('slug', selected_market.get('market_slug'))

    if not selected_slug:
        print("❌ Selected market has no slug")
        sys.exit(1)

    # Update config
    if not update_config(selected_slug):
        sys.exit(1)

    # Ask to run pipeline
    if auto_run or ask_run_pipeline():
        success = run_pipeline_steps()
        sys.exit(0 if success else 1)

    print("\n✅ Done! Market slug saved to config.py")
    print()


if __name__ == "__main__":
    main()
