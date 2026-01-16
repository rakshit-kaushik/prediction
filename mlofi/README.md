# Multi-Level Order Flow Imbalance (MLOFI)

This module implements Multi-Level OFI analysis, extending the Cont, Kukanov & Stoikov (2011) methodology to multiple orderbook depth levels.

## Overview

Standard OFI uses only the best bid/ask (Level 1). Research suggests that including deeper levels can improve predictive power by 15-74% depending on the instrument.

This module:
1. Extracts price/size data from all orderbook levels
2. Calculates OFI at each level using the Cont et al. formula
3. Runs regularized regression (Ridge/Lasso/ElasticNet)
4. Compares results to find optimal configuration

## Quick Start

```bash
# Run the full pipeline
python mlofi/01_process_multilevel.py
python mlofi/02_calculate_mlofi.py
python mlofi/03_regression_analysis.py
python mlofi/04_compare_results.py
```

## Pipeline Steps

### Step 1: Process Multi-Level Data
```bash
python mlofi/01_process_multilevel.py
```
- **Input:** `data/nyc_mayor_oct15_nov04_raw.json`
- **Output:** `data/mlofi/multilevel_processed.csv`
- Extracts price/size for levels 1-N from raw orderbook JSON
- Verifies Level 1 matches existing processed data

### Step 2: Calculate MLOFI
```bash
python mlofi/02_calculate_mlofi.py
```
- **Input:** `data/mlofi/multilevel_processed.csv`
- **Output:** `data/mlofi/mlofi_calculated.csv`
- Calculates OFI at each level using Cont et al. formula
- Computes cumulative OFI through each level
- Verifies ofi_l1 matches existing OFI calculation

### Step 3: Regression Analysis
```bash
python mlofi/03_regression_analysis.py
```
- **Input:** `data/mlofi/mlofi_calculated.csv`
- **Output:** `data/mlofi/regression_results.csv`, `data/mlofi/level_importance.csv`
- Runs 180 regressions (4 level configs × 9 time windows × 5 methods)
- Methods: OLS (L1), OLS (Cumulative), Ridge, Lasso, ElasticNet

### Step 4: Compare Results
```bash
python mlofi/04_compare_results.py
```
- **Input:** `data/mlofi/regression_results.csv`
- **Output:** `data/mlofi/comparison_summary.csv`, console report
- Generates comprehensive comparison report
- Identifies optimal configuration

## Configuration

Edit `mlofi/config_mlofi.py` to modify:

```python
# Level configurations
LEVEL_CONFIGS = {
    'L5': 5,           # Top 5 levels
    'L10': 10,         # Top 10 levels
    'L50pct': '50%',   # 50% of available levels
    'ALL': 'all',      # All available levels
}

# Time windows (minutes)
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]

# Regularization parameters
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
```

## Output Files

| File | Description |
|------|-------------|
| `multilevel_processed.csv` | Price/size for each level (1-N) |
| `mlofi_calculated.csv` | OFI at each level + cumulative |
| `regression_results.csv` | All 180 regression results |
| `level_importance.csv` | Coefficient values by level |
| `comparison_summary.csv` | Improvement vs Level 1 baseline |

## Key Metrics

The analysis compares:
- **R²**: Explained variance (higher is better)
- **RMSE**: Root mean squared error (lower is better)
- **Improvement %**: Gain over Level 1 OLS baseline

## Regression Methods

| Method | Penalty | Use Case |
|--------|---------|----------|
| OLS_L1 | None | Baseline (Level 1 only) |
| OLS_Cumulative | None | Sum of all levels |
| Ridge | L2 (λ‖β‖²) | Shrinks all coefficients |
| Lasso | L1 (λ‖β‖₁) | Feature selection |
| ElasticNet | L1 + L2 | Combines both |

## Why Regularization?

Research shows neighboring levels have 0.7-0.9 correlation. This multicollinearity causes:
- OLS coefficients to be unstable
- High variance in predictions

Ridge/Lasso/ElasticNet address this by penalizing coefficient magnitudes.

## Expected Results

Based on research:
- **Large-tick stocks:** 68-74% forecast error reduction
- **Small-tick instruments:** 15-31% improvement

For Polymarket (tick size $0.01), expect ~15-31% R² improvement.

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2011). The Price Impact of Order Book Events
- Research on MLOFI prediction: https://www.emergentmind.com/topics/order-flow-imbalance-prediction
