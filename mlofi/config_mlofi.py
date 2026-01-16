"""
config_mlofi.py
===============
Configuration for Multi-Level OFI (MLOFI) analysis

This module defines parameters for:
- Level configurations (how many orderbook levels to analyze)
- Time windows for aggregation
- Regularization parameters for Ridge/Lasso/ElasticNet
- Input/output file paths
"""

from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# LEVEL CONFIGURATIONS
# ============================================================================

# Different depth configurations to test
# Each will be compared to see which provides best predictive power
LEVEL_CONFIGS = {
    'L5': 5,           # Top 5 levels on each side
    'L10': 10,         # Top 10 levels on each side
    'L50pct': '50%',   # 50% of available levels (dynamic per snapshot)
    'ALL': 'all',      # All available levels
}

# Maximum number of levels to extract from raw data
# This should be high enough to cover all configs above
MAX_LEVELS_TO_EXTRACT = 50  # Extract up to 50 levels (will use fewer if not available)

# ============================================================================
# TIME WINDOWS
# ============================================================================

# Time windows for aggregation (in minutes)
# Same as existing analysis for comparability
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]

# ============================================================================
# REGULARIZATION PARAMETERS
# ============================================================================

# Ridge regression alphas (L2 penalty)
# Higher alpha = more regularization = smaller coefficients
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Lasso regression alphas (L1 penalty)
# Lasso can zero out coefficients (feature selection)
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]

# ElasticNet parameters
# ElasticNet combines L1 and L2 penalties
ELASTICNET_ALPHAS = [0.001, 0.01, 0.1, 1.0]
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]  # 0 = Ridge, 1 = Lasso

# Cross-validation folds
CV_FOLDS = 5

# ============================================================================
# INPUT FILES (from existing pipeline)
# ============================================================================

# Raw orderbook JSON (contains all levels)
RAW_ORDERBOOK_FILE = PROJECT_ROOT / "data" / "nyc_mayor_oct15_nov04_raw.json"

# Existing processed data (for verification)
EXISTING_PROCESSED_FILE = PROJECT_ROOT / "data" / "nyc_mayor_oct15_nov04_processed.csv"
EXISTING_OFI_FILE = PROJECT_ROOT / "data" / "nyc_mayor_oct15_nov04_ofi.csv"

# ============================================================================
# OUTPUT FILES (MLOFI-specific)
# ============================================================================

MLOFI_OUTPUT_DIR = PROJECT_ROOT / "data" / "mlofi"

# Multi-level processed data (all levels extracted)
MULTILEVEL_PROCESSED_FILE = MLOFI_OUTPUT_DIR / "multilevel_processed.csv"

# MLOFI calculated data (OFI at each level)
MLOFI_CALCULATED_FILE = MLOFI_OUTPUT_DIR / "mlofi_calculated.csv"

# Regression results
REGRESSION_RESULTS_FILE = MLOFI_OUTPUT_DIR / "regression_results.csv"

# Comparison summary
COMPARISON_SUMMARY_FILE = MLOFI_OUTPUT_DIR / "comparison_summary.csv"

# Level importance analysis
LEVEL_IMPORTANCE_FILE = MLOFI_OUTPUT_DIR / "level_importance.csv"

# ============================================================================
# PRICE NORMALIZATION
# ============================================================================

# Tick size for price normalization (same as existing)
TICK_SIZE = 0.01

# Use tick-normalized price changes (recommended)
USE_TICK_NORMALIZED = True

# ============================================================================
# REGRESSION SETTINGS
# ============================================================================

# Minimum observations required for regression
MIN_OBS_FOR_REGRESSION = 30

# Whether to standardize features before regression
# (important for comparing coefficients across levels)
STANDARDIZE_FEATURES = True

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Number of decimal places for R² display
R2_DECIMALS = 4

# Number of decimal places for coefficients
COEF_DECIMALS = 8
