"""
Multi-Level Order Flow Imbalance (MLOFI) Module

This module implements Multi-Level OFI analysis following the research on
order flow imbalance prediction, extending the Cont, Kukanov & Stoikov (2011)
methodology to multiple orderbook depth levels.

Components:
- config_mlofi: Configuration parameters
- 01_process_multilevel: Extract multi-level data from raw orderbooks
- 02_calculate_mlofi: Calculate OFI at each level
- 03_regression_analysis: Ridge/Lasso/ElasticNet regression
- 04_compare_results: Compare MLOFI vs Level 1 OFI
"""

from . import config_mlofi
