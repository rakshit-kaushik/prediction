"""
config.py
=========
Configuration for the data pipeline

Modify these settings for your market and date range.
"""

# ============================================================================
# MARKET CONFIGURATION
# ============================================================================

# Market slug from Polymarket URL
# Example: https://polymarket.com/event/fed-decreases-interest-rates...
#          → slug = "fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting"
MARKET_SLUG = "will-zohran-mamdani-win-the-2025-nyc-mayoral-election"

# Optional: If you already know the token ID, set it here (otherwise leave None)
# The pipeline will find it automatically if None
TOKEN_ID = "33945469250963963541781051637999677727672635213493648594066577298999471399137"

# ============================================================================
# DATE RANGE
# ============================================================================

# Format: "YYYY-MM-DD"
START_DATE = "2025-10-15"
END_DATE = "2025-11-04"

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Dome API base URL
DOME_API_BASE_URL = "https://api.domeapi.io/v1"

# Polymarket Gamma API (for market lookup)
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"

# API key will be loaded from .env file
# Make sure you have: DOME_API_KEY=your_key_here in .env

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Where to save market info
MARKET_INFO_FILE = "data/nyc_mayor_market_info.json"

# Where to save raw orderbook data
RAW_ORDERBOOK_FILE = "data/nyc_mayor_oct15_nov04_raw.json"

# Where to save processed orderbook data
PROCESSED_ORDERBOOK_FILE = "data/nyc_mayor_oct15_nov04_processed.csv"

# Where to save raw trade data
TRADES_RAW_FILE = "data/nyc_mayor_oct15_nov04_trades_raw.json"

# Where to save processed trade data
TRADES_PROCESSED_FILE = "data/nyc_mayor_oct15_nov04_trades_processed.csv"

# ============================================================================
# DOWNLOAD SETTINGS
# ============================================================================

# Maximum snapshots per API call (don't change unless API limit changes)
PAGINATION_LIMIT = 200

# Rate limiting (seconds between requests)
DELAY_BETWEEN_PAGES = 0.3
DELAY_BETWEEN_DAYS = 1.0
