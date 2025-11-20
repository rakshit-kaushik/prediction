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
MARKET_SLUG = "fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting"

# Optional: If you already know the token ID, set it here (otherwise leave None)
# The pipeline will find it automatically if None
TOKEN_ID = None

# ============================================================================
# DATE RANGE
# ============================================================================

# Format: "YYYY-MM-DD"
START_DATE = "2025-10-15"
END_DATE = "2025-10-31"

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
MARKET_INFO_FILE = "data/market_info.json"

# Where to save raw orderbook data
RAW_ORDERBOOK_FILE = "data/orderbook_raw.json"

# Where to save processed orderbook data
PROCESSED_ORDERBOOK_FILE = "data/orderbook_processed.csv"

# ============================================================================
# DOWNLOAD SETTINGS
# ============================================================================

# Maximum snapshots per API call (don't change unless API limit changes)
PAGINATION_LIMIT = 200

# Rate limiting (seconds between requests)
DELAY_BETWEEN_PAGES = 0.3
DELAY_BETWEEN_DAYS = 1.0
