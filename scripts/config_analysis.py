"""
config_analysis.py
==================
Shared configuration for OFI analysis scripts

This centralizes key parameters so they can be easily changed without
modifying multiple scripts.
"""

# ============================================================================
# TIME AGGREGATION
# ============================================================================

# Time window for aggregating OFI and price changes
# Format: pandas frequency string
# Options: '10S' (10 seconds), '1T' (1 minute), '5T' (5 minutes), '10T' (10 minutes), etc.
TIME_WINDOW = '10T'  # 10 minutes (default)

# Note: This is used ONLY in analysis scripts for aggregation.
# Raw snapshot data in data/ folder is NEVER aggregated.

# ============================================================================
# PRICE NORMALIZATION
# ============================================================================

# Tick size for Polymarket (in dollars)
TICK_SIZE = 0.01  # $0.01

# Whether to use tick-normalized price changes in regressions
# If True: uses delta_mid_price_ticks (ΔP in ticks)
# If False: uses delta_mid_price (ΔP in dollars)
USE_TICK_NORMALIZED = True

# ============================================================================
# REGRESSION PARAMETERS
# ============================================================================

# Models to run
RUN_LINEAR_MODEL = True  # ΔP = α + β × OFI + ε
RUN_QUADRATIC_MODEL = True  # ΔP = α + β₁×OFI + β₂×OFI² + ε

# Minimum observations required for regression
MIN_OBS_FOR_REGRESSION = 30

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================

# Confidence level for statistical tests
CONFIDENCE_LEVEL = 0.95

# Significance threshold for p-values
ALPHA = 0.05

# ============================================================================
# 3-PHASE ANALYSIS
# ============================================================================

# Number of phases to divide market lifetime into
N_PHASES = 3  # Early, Middle, Late

# Minimum observations per phase
MIN_OBS_PER_PHASE = 50

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

from pathlib import Path

# Root directory (parent of scripts/)
ROOT_DIR = Path(__file__).parent.parent

# Data directory
DATA_DIR = ROOT_DIR / "data"

# Results directories
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# Create directories if they don't exist
for directory in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, ANALYSIS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dependent_variable_name():
    """Get the name of the dependent variable based on configuration"""
    return 'delta_mid_price_ticks' if USE_TICK_NORMALIZED else 'delta_mid_price'

def get_dependent_variable_label():
    """Get the label for plots based on configuration"""
    return 'ΔP (ticks)' if USE_TICK_NORMALIZED else 'ΔP ($)'

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("OFI ANALYSIS CONFIGURATION")
    print("=" * 80)
    print(f"Time Window: {TIME_WINDOW}")
    print(f"Tick Size: ${TICK_SIZE}")
    print(f"Use Tick Normalization: {USE_TICK_NORMALIZED}")
    print(f"Dependent Variable: {get_dependent_variable_name()}")
    print(f"Linear Model: {'Enabled' if RUN_LINEAR_MODEL else 'Disabled'}")
    print(f"Quadratic Model: {'Enabled' if RUN_QUADRATIC_MODEL else 'Disabled'}")
    print(f"Number of Phases: {N_PHASES}")
    print("=" * 80)
