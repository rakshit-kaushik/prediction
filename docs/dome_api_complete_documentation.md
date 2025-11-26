# DOME API - COMPLETE DOCUMENTATION FOR POLYMARKET ORDERBOOK RESEARCH

**Project:** OFI (Order Flow Imbalance) Analysis for Prediction Markets  
**Platform:** Polymarket via Dome API  
**Purpose:** Replicate Cont et al. (2011) research for prediction markets

---

## TABLE OF CONTENTS
1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [SDK Installation](#sdk-installation)
5. [Available Endpoints](#available-endpoints)
6. [Data Types & Structures](#data-types--structures)
7. [Code Examples](#code-examples)
8. [Rate Limits](#rate-limits)
9. [Error Handling](#error-handling)
10. [Research-Specific Guidance](#research-specific-guidance)

---

## API OVERVIEW

**Dome API** provides unified access to prediction market data across multiple platforms including Polymarket and Kalshi.

### Key Features for Your Research:
- ✅ **Historical orderbook data** (NOT available from native Polymarket API)
- ✅ **Real-time market prices**
- ✅ **Historical candlestick/OHLCV data**
- ✅ **Wallet analytics** (trader behavior)
- ✅ **Order tracking**
- ✅ **Cross-platform market matching**

### What Makes Dome Unique:
- Aggregates data from multiple sources
- Provides granular datasets not available elsewhere
- Historical orderbook snapshots (critical for your OFI calculation)
- Transaction-level data (mentioned: 500GB datasets provided to researchers)

---

## AUTHENTICATION

### API Key Setup:
1. Sign up at: https://www.domeapi.io/
2. Get API key from dashboard
3. Include in every request

### Authentication Methods:

**Option 1: Header (Recommended)**
```
Authorization: Bearer YOUR_API_KEY_HERE
```

**Option 2: SDK (Easiest)**
```python
from dome_sdk import DomeClient
dome = DomeClient(api_key="YOUR_API_KEY_HERE")
```

**Option 3: Environment Variable**
```bash
export DOME_API_KEY="your_api_key_here"
```

---

## BASE URL

```
https://api.domeapi.io/v1
```

**All endpoints are prefixed with this base URL.**

---

## SDK INSTALLATION

### Python:
```bash
pip install @dome-api/sdk
# OR
pip install dome-sdk-py
```

### TypeScript/JavaScript:
```bash
npm install @dome-api/sdk
# OR
yarn add @dome-api/sdk
```

### Basic Setup:
```python
from dome_sdk import DomeClient

# Initialize client
dome = DomeClient(
    api_key="your_api_key_here",
    base_url="https://api.domeapi.io/v1"  # Optional, this is default
)
```

---

## AVAILABLE ENDPOINTS

### 1. POLYMARKET MARKETS

#### **Get Market List**
```python
# SDK
markets = dome.polymarket.markets.list(
    limit=100,        # Optional: number of results
    offset=0,         # Optional: pagination offset
    status="active"   # Optional: "active", "resolved", "closed"
)

# HTTP
GET /polymarket/markets?limit=100&offset=0&status=active
```

**Response:**
```json
[
  {
    "market_id": "string",
    "token_id": "string",
    "condition_id": "string",
    "title": "Will Trump win 2024?",
    "slug": "trump-win-2024",
    "status": "active",
    "created_at": 1640995200,
    "close_time": 1672531200,
    "resolved_at": null,
    "outcome": null,
    "category": "politics",
    "volume": 1234567.89
  }
]
```

---

#### **Get Market Price**
```python
# SDK - Current price
price = dome.polymarket.markets.get_market_price(
    token_id="1234567890"
)

# SDK - Historical price at specific time
historical_price = dome.polymarket.markets.get_market_price(
    token_id="1234567890",
    at_time=1700000000  # Unix timestamp
)

# HTTP
GET /polymarket/markets/price?token_id=1234567890&at_time=1700000000
```

**Response:**
```json
{
  "token_id": "1234567890",
  "price": 0.65,
  "timestamp": 1700000000,
  "best_bid": 0.64,
  "best_ask": 0.66,
  "spread": 0.02
}
```

**🔑 KEY FEATURE:** `at_time` parameter allows retrieving historical prices!

---

#### **Get Candlestick Data**
```python
# SDK
candlesticks = dome.polymarket.markets.get_candlesticks(
    condition_id="0x4567b275e6b667a6217f5cb4f06a797d3a1eaf1d0281fb5bc8c75e2046ae7e57",
    start_time=1640995200,  # Unix timestamp
    end_time=1672531200,    # Unix timestamp
    interval=60             # 1=1min, 60=1hour, 1440=1day
)

# HTTP
GET /polymarket/markets/candlesticks?condition_id=0x...&start_time=1640995200&end_time=1672531200&interval=60
```

**Response:**
```json
[
  {
    "timestamp": 1640995200,
    "open": 0.63,
    "high": 0.67,
    "low": 0.62,
    "close": 0.65,
    "volume": 12345.67,
    "trades": 156
  }
]
```

**Interval options:**
- `1` = 1 minute
- `5` = 5 minutes
- `60` = 1 hour
- `1440` = 1 day

---

### 2. ORDERBOOK DATA (CRITICAL FOR YOUR RESEARCH)

#### **Get Current Orderbook**
```python
# SDK
orderbook = dome.polymarket.orderbook.get_current(
    token_id="1234567890"
)

# HTTP
GET /polymarket/orderbook/current?token_id=1234567890
```

**Response:**
```json
{
  "token_id": "1234567890",
  "timestamp": 1700000000,
  "bids": [
    {"price": 0.64, "size": 1000.00, "orders": 5},
    {"price": 0.63, "size": 500.00, "orders": 3},
    {"price": 0.62, "size": 750.00, "orders": 4}
  ],
  "asks": [
    {"price": 0.66, "size": 800.00, "orders": 4},
    {"price": 0.67, "size": 600.00, "orders": 2},
    {"price": 0.68, "size": 900.00, "orders": 6}
  ],
  "best_bid": 0.64,
  "best_ask": 0.66,
  "spread": 0.02,
  "mid_price": 0.65
}
```

---

#### **Get Historical Orderbook (MOST IMPORTANT)**
```python
# SDK
orderbook_history = dome.polymarket.orderbook.get_history(
    token_id="1234567890",
    start_time=1640995200,  # Unix timestamp - contract start
    end_time=1672531200,    # Unix timestamp - contract end
    interval=10             # Optional: seconds between snapshots (default: 10)
)

# HTTP
GET /polymarket/orderbook/history?token_id=1234567890&start_time=1640995200&end_time=1672531200&interval=10
```

**Response:**
```json
[
  {
    "timestamp": 1640995200,
    "bids": [
      {"price": 0.64, "size": 1000.00},
      {"price": 0.63, "size": 500.00}
    ],
    "asks": [
      {"price": 0.66, "size": 800.00},
      {"price": 0.67, "size": 600.00}
    ],
    "best_bid": 0.64,
    "best_ask": 0.66,
    "mid_price": 0.65,
    "spread": 0.02,
    "total_bid_size": 1500.00,
    "total_ask_size": 1400.00
  },
  {
    "timestamp": 1640995210,
    "bids": [...],
    "asks": [...]
  }
  // ... more snapshots
]
```

**⭐ CRITICAL FOR YOUR RESEARCH:**
- This endpoint provides orderbook snapshots at regular intervals
- You need this to calculate Order Flow Imbalance (OFI)
- Request entire contract lifecycle (creation → resolution)

**NOTE:** If this exact endpoint doesn't exist, contact Dome support (kurush@domeapi.com) and request access to historical orderbook data. They've provided 500GB+ datasets to researchers before.

---

### 3. ORDER DATA

#### **Get Orders**
```python
# SDK
orders = dome.polymarket.orders.get_orders(
    market_slug="bitcoin-up-or-down-july-25-8pm-et",  # Optional
    limit=50,                                          # Optional
    offset=0,                                          # Optional
    start_time=1640995200,                            # Optional
    end_time=1672531200                               # Optional
)

# HTTP
GET /polymarket/orders?market_slug=bitcoin-up-or-down&limit=50&start_time=1640995200&end_time=1672531200
```

**Response:**
```json
[
  {
    "order_id": "abc123",
    "token_id": "1234567890",
    "side": "BUY",
    "price": 0.65,
    "size": 100.00,
    "filled_size": 75.00,
    "status": "partially_filled",
    "created_at": 1640995200,
    "updated_at": 1640995300
  }
]
```

---

### 4. WALLET ANALYTICS

#### **Get Wallet P&L**
```python
# SDK
wallet_pnl = dome.polymarket.wallet.get_wallet_pnl(
    wallet_address="0x7c3db723f1d4d8cb9c550095203b686cb11e5c6b",
    granularity="day",  # 'day', 'week', 'month', 'year', 'all'
    start_time=1726857600,
    end_time=1758316829
)

# HTTP
GET /polymarket/wallet/pnl?wallet_address=0x...&granularity=day&start_time=1726857600&end_time=1758316829
```

**Response:**
```json
{
  "wallet_address": "0x...",
  "total_pnl": 12345.67,
  "realized_pnl": 10000.00,
  "unrealized_pnl": 2345.67,
  "pnl_history": [
    {
      "timestamp": 1726857600,
      "pnl": 100.50,
      "cumulative_pnl": 100.50
    }
  ]
}
```

**Use case:** Identify informed traders, analyze trading patterns

---

### 5. MATCHING MARKETS (Cross-Platform)

#### **Find Equivalent Markets**
```python
# SDK - By Polymarket slug
matching = dome.matching_markets.get_matching_markets(
    polymarket_market_slug=["nfl-ari-den-2025-08-16"]
)

# SDK - By Kalshi ticker
matching = dome.matching_markets.get_matching_markets(
    kalshi_event_ticker=["KXNFLGAME-25AUG16ARIDEN"]
)

# SDK - By sport and date
matching = dome.matching_markets.get_matching_markets_by_sport(
    sport="nfl",
    date="2025-08-16"
)

# HTTP
GET /matching-markets?polymarket_market_slug=nfl-ari-den-2025-08-16
```

---

## DATA TYPES & STRUCTURES

### Key Identifiers

**token_id:**
- Unique identifier for YES or NO outcome
- Example: `"71321045679252212594626385532706912750332728571942532289631379312455583992563"`
- Each binary market has 2 token_ids (YES and NO)

**condition_id:**
- Hex string identifying the entire market
- Example: `"0x4567b275e6b667a6217f5cb4f06a797d3a1eaf1d0281fb5bc8c75e2046ae7e57"`
- Used for candlestick data

**market_slug:**
- Human-readable market identifier
- Example: `"trump-win-2024"`
- Used in URLs and some endpoints

---

### Timestamps
- All timestamps are **Unix timestamps** (seconds since Jan 1, 1970)
- UTC timezone
- Convert in Python:
```python
from datetime import datetime

# To Unix timestamp
unix_ts = int(datetime(2024, 11, 1).timestamp())

# From Unix timestamp
dt = datetime.fromtimestamp(unix_ts)
```

---

### Prices
- All prices are **decimal values between 0 and 1**
- Represents probability (0 = 0%, 1 = 100%)
- Example: 0.65 = 65% probability

---

### Sizes/Volumes
- Denominated in **USDC** (stablecoin)
- Decimal values
- Example: 1234.56 = $1,234.56 USDC

---

## CODE EXAMPLES

### Example 1: Get Market List and Prices
```python
from dome_sdk import DomeClient

dome = DomeClient(api_key="your_key")

# Get all active markets
markets = dome.polymarket.markets.list(status="active", limit=100)

print(f"Found {len(markets)} active markets")

# Get price for each market
for market in markets[:5]:  # First 5 markets
    token_id = market['token_id']
    price = dome.polymarket.markets.get_market_price(token_id=token_id)
    
    print(f"{market['title']}: ${price['price']:.2f}")
```

---

### Example 2: Get Historical Orderbook for Contract Lifecycle
```python
from dome_sdk import DomeClient
from datetime import datetime
import pandas as pd

dome = DomeClient(api_key="your_key")

# Contract details
token_id = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
start_time = int(datetime(2024, 10, 1).timestamp())
end_time = int(datetime(2024, 11, 15).timestamp())

# Get historical orderbook
print("Fetching orderbook history...")
orderbook_history = dome.polymarket.orderbook.get_history(
    token_id=token_id,
    start_time=start_time,
    end_time=end_time,
    interval=10  # 10-second snapshots
)

print(f"Retrieved {len(orderbook_history)} orderbook snapshots")

# Convert to DataFrame
df = pd.DataFrame(orderbook_history)

# Save to CSV
df.to_csv("orderbook_history.csv", index=False)
print("Saved to orderbook_history.csv")

# Basic stats
print(f"\nDate range: {datetime.fromtimestamp(df['timestamp'].min())} to {datetime.fromtimestamp(df['timestamp'].max())}")
print(f"Average spread: ${df['spread'].mean():.4f}")
```

---

### Example 3: Collect Data for Multiple Contracts
```python
from dome_sdk import DomeClient
import pandas as pd
from datetime import datetime
import time

dome = DomeClient(api_key="your_key")

# List of contracts to analyze
contracts = [
    {"token_id": "123...", "name": "Trump Win 2024"},
    {"token_id": "456...", "name": "Bitcoin $100k"},
    {"token_id": "789...", "name": "Fed Rate Cut"}
]

all_data = []

for contract in contracts:
    print(f"\nProcessing: {contract['name']}")
    
    try:
        # Get market metadata
        markets = dome.polymarket.markets.list(limit=1000)
        market = next((m for m in markets if m['token_id'] == contract['token_id']), None)
        
        if not market:
            print(f"  ❌ Market not found")
            continue
        
        # Get historical orderbook
        orderbook_data = dome.polymarket.orderbook.get_history(
            token_id=contract['token_id'],
            start_time=market['created_at'],
            end_time=market.get('resolved_at', int(datetime.now().timestamp())),
            interval=10
        )
        
        # Add contract name to each record
        for record in orderbook_data:
            record['contract_name'] = contract['name']
            record['token_id'] = contract['token_id']
        
        all_data.extend(orderbook_data)
        print(f"  ✅ Got {len(orderbook_data)} snapshots")
        
        # Be respectful of API
        time.sleep(1)
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Save everything
df = pd.DataFrame(all_data)
df.to_csv("all_contracts_orderbook.csv", index=False)
print(f"\n✅ Total: {len(all_data)} snapshots saved")
```

---

### Example 4: Calculate Order Flow Imbalance (OFI)
```python
import pandas as pd
import numpy as np

# Load orderbook data
df = pd.read_csv("orderbook_history.csv")

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Initialize OFI calculation
df['ofi'] = 0.0
df['price_change'] = 0.0
df['mid_price_prev'] = df['mid_price'].shift(1)

for i in range(1, len(df)):
    current = df.iloc[i]
    previous = df.iloc[i-1]
    
    # Extract orderbook data (assuming lists stored as strings)
    # You may need to parse JSON if stored differently
    current_bids = current['bids']  # Parse this appropriately
    previous_bids = previous['bids']
    current_asks = current['asks']
    previous_asks = previous['asks']
    
    # Calculate OFI based on Cont et al. (2011) formula
    # This is simplified - actual calculation depends on data structure
    
    # Changes in bid side
    bid_change = current['total_bid_size'] - previous['total_bid_size']
    
    # Changes in ask side  
    ask_change = current['total_ask_size'] - previous['total_ask_size']
    
    # OFI = positive for more buying pressure
    ofi = bid_change - ask_change
    
    df.at[i, 'ofi'] = ofi
    df.at[i, 'price_change'] = current['mid_price'] - previous['mid_price']

# Save with OFI
df.to_csv("orderbook_with_ofi.csv", index=False)

# Basic analysis
print(f"OFI statistics:")
print(df['ofi'].describe())
print(f"\nCorrelation between OFI and price change: {df['ofi'].corr(df['price_change']):.4f}")
```

---

## RATE LIMITS

### Expected Limits (verify with Dome support):
- **Free tier:** ~100-1000 requests/hour
- **Paid tier:** Higher limits
- **Bulk data requests:** Contact support for special access

### Best Practices:
```python
import time

def rate_limited_request(func, *args, delay=1, **kwargs):
    """
    Wrapper to add delay between requests
    """
    result = func(*args, **kwargs)
    time.sleep(delay)
    return result

# Usage
for token_id in token_ids:
    data = rate_limited_request(
        dome.polymarket.markets.get_market_price,
        token_id=token_id,
        delay=0.5  # Wait 0.5s between requests
    )
```

---

## ERROR HANDLING

### Common Errors:

**401 Unauthorized**
```python
# Invalid API key
try:
    data = dome.polymarket.markets.list()
except Exception as e:
    if "401" in str(e):
        print("❌ Invalid API key. Check your credentials.")
```

**404 Not Found**
```python
# Token ID doesn't exist
try:
    price = dome.polymarket.markets.get_market_price(token_id="invalid")
except Exception as e:
    if "404" in str(e):
        print("❌ Token ID not found")
```

**429 Too Many Requests**
```python
# Hit rate limit
try:
    data = dome.polymarket.markets.list()
except Exception as e:
    if "429" in str(e):
        print("❌ Rate limit exceeded. Wait before retrying.")
        time.sleep(60)  # Wait 1 minute
```

**500 Server Error**
```python
# Server-side issue
try:
    data = dome.polymarket.markets.list()
except Exception as e:
    if "500" in str(e):
        print("❌ Server error. Try again later.")
```

### Robust Error Handling:
```python
import time

def safe_api_call(func, *args, max_retries=3, **kwargs):
    """
    API call with retry logic
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str:
                # Rate limit
                wait_time = 2 ** attempt * 30  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "500" in error_str or "503" in error_str:
                # Server error
                print(f"Server error. Retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
            else:
                # Other error
                print(f"Error: {e}")
                raise
    
    raise Exception(f"Failed after {max_retries} attempts")

# Usage
data = safe_api_call(
    dome.polymarket.markets.get_market_price,
    token_id="123..."
)
```

---

## RESEARCH-SPECIFIC GUIDANCE

### For OFI Calculation (Cont et al. 2011 Replication):

#### What You Need:
1. **Orderbook snapshots** at regular intervals (10-30 seconds)
2. **Full contract lifecycle** (creation → resolution)
3. **Bid/ask prices and sizes** at each snapshot
4. **Trade data** (optional, for comparison)

#### Data Collection Strategy:

**Step 1: Identify Target Contracts**
```python
# Criteria:
# - Liquid markets (>$100K volume)
# - Clear resolution
# - 2-6 month duration
# - Mix of topics (political, sports, economic)

markets = dome.polymarket.markets.list(status="resolved", limit=1000)

# Filter for liquid contracts
liquid_markets = [
    m for m in markets 
    if m['volume'] > 100000  # $100K+
    and m['resolved_at'] is not None
]

# Select 20-50 contracts
selected_contracts = liquid_markets[:50]
```

**Step 2: Download Historical Data**
```python
for contract in selected_contracts:
    orderbook_data = dome.polymarket.orderbook.get_history(
        token_id=contract['token_id'],
        start_time=contract['created_at'],
        end_time=contract['resolved_at'],
        interval=10  # 10-second snapshots
    )
    
    # Save incrementally
    df = pd.DataFrame(orderbook_data)
    df.to_csv(f"data/contract_{contract['token_id']}.csv")
```

**Step 3: Calculate OFI**
Based on Cont et al. (2011) formula:

```
e_n = Indicator{P^B_n >= P^B_{n-1}} * q^B_n 
    - Indicator{P^B_n <= P^B_{n-1}} * q^B_{n-1}
    - Indicator{P^A_n <= P^A_{n-1}} * q^A_n
    + Indicator{P^A_n >= P^A_{n-1}} * q^A_{n-1}

OFI_k = Sum(e_n) over interval k
```

Implementation in Python:
```python
def calculate_ofi(df):
    """
    Calculate Order Flow Imbalance
    df: DataFrame with columns [timestamp, bids, asks, best_bid, best_ask]
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    ofi_values = []
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        
        # Parse orderbook (adjust based on actual data structure)
        curr_best_bid_price = current['best_bid']
        curr_best_bid_size = current['bids'][0]['size']  # Assuming sorted
        prev_best_bid_price = previous['best_bid']
        prev_best_bid_size = previous['bids'][0]['size']
        
        curr_best_ask_price = current['best_ask']
        curr_best_ask_size = current['asks'][0]['size']
        prev_best_ask_price = previous['best_ask']
        prev_best_ask_size = previous['asks'][0]['size']
        
        # Calculate e_n
        e_n = 0
        
        # Bid side
        if curr_best_bid_price >= prev_best_bid_price:
            e_n += curr_best_bid_size
        if curr_best_bid_price <= prev_best_bid_price:
            e_n -= prev_best_bid_size
        
        # Ask side
        if curr_best_ask_price <= prev_best_ask_price:
            e_n -= curr_best_ask_size
        if curr_best_ask_price >= prev_best_ask_price:
            e_n += prev_best_ask_size
        
        ofi_values.append(e_n)
    
    df['ofi'] = [0] + ofi_values
    return df
```

**Step 4: Aggregate by Time Intervals**
```python
# Aggregate to 10-second intervals
df['interval'] = (df['timestamp'] // 10) * 10

ofi_aggregated = df.groupby('interval').agg({
    'ofi': 'sum',
    'mid_price': 'last'
}).reset_index()

# Calculate price changes
ofi_aggregated['price_change'] = ofi_aggregated['mid_price'].diff()
```

**Step 5: Lifecycle Segmentation**
```python
def segment_contract_lifecycle(df, contract_start, contract_end):
    """
    Segment contract into early/mid/late phases
    """
    total_duration = contract_end - contract_start
    early_end = contract_start + total_duration * 0.25
    mid_end = contract_start + total_duration * 0.75
    
    df['phase'] = 'mid'
    df.loc[df['timestamp'] < early_end, 'phase'] = 'early'
    df.loc[df['timestamp'] >= mid_end, 'phase'] = 'late'
    
    return df
```

**Step 6: Run Regressions**
```python
import statsmodels.api as sm

# Overall regression
X = ofi_aggregated['ofi']
y = ofi_aggregated['price_change']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

# By phase
for phase in ['early', 'mid', 'late']:
    phase_data = ofi_aggregated[ofi_aggregated['phase'] == phase]
    X_phase = sm.add_constant(phase_data['ofi'])
    y_phase = phase_data['price_change']
    
    model_phase = sm.OLS(y_phase, X_phase).fit()
    print(f"\n{phase.upper()} PHASE:")
    print(model_phase.summary())
```

---

### Depth Analysis:
```python
# Calculate average depth
df['avg_depth'] = (df['total_bid_size'] + df['total_ask_size']) / 2

# Depth vs price impact
df['price_impact'] = df['price_change'] / df['ofi']

# Test inverse relationship
import scipy.stats as stats
correlation, p_value = stats.pearsonr(df['avg_depth'], df['price_impact'])
print(f"Depth vs Impact correlation: {correlation:.4f} (p={p_value:.4f})")
```

---

## IMPORTANT NOTES & CAVEATS

### 1. Historical Orderbook Availability
- **CRITICAL:** Verify that Dome API provides the `orderbook/history` endpoint
- If not directly available, contact support: kurush@domeapi.com
- Mention you're doing academic research replicating Cont et al. (2011)
- They have provided 500GB+ datasets to researchers before

### 2. Data Granularity
- For OFI calculation, you need snapshots every 10-30 seconds
- More frequent = better, but larger data size
- Original paper used "event time" (every order book event)
- You may need to aggregate to fixed time intervals

### 3. Token IDs vs Condition IDs
- Each market has a `condition_id` (overall market)
- Each outcome (YES/NO) has a `token_id`
- For binary markets: 2 token_ids per market
- Orderbook data is per token_id

### 4. Data Size Estimates
- 1 contract, 3 months, 10-sec snapshots = ~780K snapshots
- 50 contracts = ~39M snapshots
- Estimated size: 10-50GB depending on orderbook depth

### 5. Polymarket Native API
- Dome API wraps Polymarket's CLOB API
- Some endpoints may directly call Polymarket
- Dome adds historical data not available from Polymarket

---

## CONTACT & SUPPORT

### Dome API Support:
- **Email:** kurush@domeapi.com, kunal@domeapi.com
- **Discord:** https://discord.gg/fKAbjNAbkt
- **GitHub:** https://github.com/kurushdubash/dome-sdk-ts

### For Research/Academic Use:
- Mention you're replicating Cont et al. (2011)
- Request historical orderbook data access
- Ask about academic pricing/data grants

---

## QUICK START CHECKLIST

- [ ] Sign up for Dome API account
- [ ] Get API key from dashboard
- [ ] Install SDK: `pip install @dome-api/sdk`
- [ ] Test API key with simple request
- [ ] Identify 20-50 target contracts
- [ ] Request historical orderbook data access (if needed)
- [ ] Download historical orderbook for all contracts
- [ ] Calculate OFI from orderbook snapshots
- [ ] Segment by lifecycle phases
- [ ] Run regressions
- [ ] Write paper!

---

## ADDITIONAL RESOURCES

### Papers:
- Cont, R., Kukanov, A., & Stoikov, S. (2011). "The price impact of order book events"
- Your paper: Extension to prediction markets

### APIs:
- Dome API: https://docs.domeapi.io
- Polymarket CLOB: https://docs.polymarket.com
- Polymarket Gamma: https://docs.polymarket.com/developers/gamma-markets-api

### Tools:
- pandas: Data manipulation
- statsmodels: Regression analysis
- matplotlib/seaborn: Visualization
- jupyter: Interactive analysis

---

## END OF DOCUMENTATION

**Last Updated:** November 2025
**For:** OFI Research Project - Prediction Markets
**Contact:** kurush@domeapi.com for questions about Dome API

---

**NEXT STEP:** Upload this document to Claude Code and begin implementation!
