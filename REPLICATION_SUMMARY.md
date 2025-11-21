# Cont et al. (2011) Replication for Polymarket - Summary

## Overview
This project replicates the key findings from "The Price Impact of Order Book Events" by Cont, Kukanov & Stoikov (2011) for Polymarket prediction markets.

**Paper Reference:** Cont, R., Kukanov, A., & Stoikov, S. (2011). The Price Impact of Order Book Events. Journal of Financial Econometrics, 12(1), 47-88.

**Replication Period:**
- Fed Rate Decision: Oct 15 - Nov 20, 2025 (26,840 observations)
- NYC Mayoral Election: Oct 15 - Nov 04, 2025 (101,270 observations)

---

## Key Findings Summary

### 1. OFI-Price Relationship (Section 3.2)

**Linear Model: ΔP = α + β × OFI + ε**

| Market | β (×10⁻⁷) | R² | Correlation | p-value |
|--------|-----------|-------|-------------|---------|
| Fed | 1.977 | 0.0671 | 0.259 | < 0.0001 |
| NYC | 0.144 | 0.0497 | 0.223 | < 0.0001 |

**Key Insight:** OFI significantly predicts price changes in both markets, but with much lower R² (5-7%) compared to equities (65% in original paper). This reflects the event-driven nature of prediction markets vs continuous trading in equities.

---

### 2. Depth Analysis (Section 3.2)

**Power Law Model: β = a / AD^λ**

| Market | λ (exponent) | R² | Paper's Finding |
|--------|--------------|-----|-----------------|
| Fed | 0.53 ± 1.08 | 0.0025 | λ ≈ 1 |
| NYC | -0.08 ± 0.12 | 0.0011 | λ ≈ 1 |

**Key Insight:** Prediction markets do NOT show the strong inverse depth relationship found in equities. This suggests different market microstructure dynamics in binary outcome markets.

---

### 3. Variance Decomposition

**How much of price variance is explained by OFI?**

| Market | Total Variance | Explained by OFI | % Explained |
|--------|----------------|------------------|-------------|
| Fed | 8.35×10⁻⁶ | 5.60×10⁻⁷ | 6.7% |
| NYC | 1.21×10⁻⁷ | 6.04×10⁻⁹ | 5.0% |

**Comparison to Paper:** Original study found OFI explains ~65% of price variance in equities, vs ~5-7% in prediction markets.

**Interpretation:** Prediction market prices are driven more by news/information events than orderbook dynamics.

---

### 4. Event Pattern Analysis

**Event Distribution:**
- Fed: 93.5% of observations show no orderbook changes
- NYC: 89.0% of observations show no orderbook changes

**Most Impactful Events (Fed Market):**
- "both_down" (β = 4.72×10⁻⁸, R² = 0.041)
- "bid_up_ask_down" (β = 8.07×10⁻⁷, R² = 0.116)

**Most Impactful Events (NYC Market):**
- "both_down" (β = 1.17×10⁻⁸, R² = 0.049)
- "ask_up_only" (β = 9.29×10⁻⁹, R² = 0.012)

---

### 5. OFI vs Trade Imbalance Comparison

**Horse-Race Regressions:**

#### Fed Market
| Model | R² | β_OFI | β_TI |
|-------|-----|-------|------|
| OFI only | 0.0671 | 1.98×10⁻⁷ | - |
| TI only | 0.0034 | - | 7.52×10⁻⁴ |
| Both | 0.0706 | 1.98×10⁻⁷ | 7.79×10⁻⁴ |

**R² improvement from adding TI: 0.36%**

#### NYC Market
| Model | R² | β_OFI | β_TI |
|-------|-----|-------|------|
| OFI only | 0.0497 | 1.44×10⁻⁸ | - |
| TI only | 0.0017 | - | -4.65×10⁻⁵ |
| Both | 0.0514 | 1.44×10⁻⁸ | -4.55×10⁻⁵ |

**R² improvement from adding TI: 0.16%**

**Key Insight:** OFI dominates trade imbalance in predicting price changes, consistent with the paper's findings for equities.

---

## Differences from Original Paper

### 1. Lower R² Values (Major)
- **Paper:** R² ≈ 65% for equities
- **Our Results:** R² ≈ 5-7% for prediction markets
- **Reason:** Binary outcome markets are more news-driven; prices jump on information rather than evolving continuously through orderbook dynamics

### 2. Weak Depth Relationship (Major)
- **Paper:** Strong power law β ~ 1/AD with λ ≈ 1
- **Our Results:** λ ≈ 0.5 (Fed) or ≈ -0.08 (NYC), very low R²
- **Reason:** Prediction market depth doesn't buffer price impact the same way due to event resolution structure

### 3. High % No-Change Observations (Major)
- **Paper:** Continuous orderbook activity in equities
- **Our Results:** 89-93% of observations show no orderbook changes
- **Reason:** Prediction markets see bursts of activity around news, with long quiet periods

### 4. Lower Variance Explained (Major)
- **Paper:** OFI explains ~65% of price variance
- **Our Results:** OFI explains ~5-7% of price variance
- **Reason:** Exogenous news shocks dominate prediction market prices

---

## Files Generated

### Analysis Scripts (`scripts/`)
1. `01_regression_analysis.py` - Linear/quadratic OFI regression
2. `02_create_figure_2.py` - Scatter plots with regression lines
3. `03_residual_diagnostics.py` - Model validation
4. `04_depth_analysis.py` - Rolling window β vs depth analysis
5. `05_event_analysis.py` - Event patterns and variance decomposition
6. `06_trade_volume_analysis.py` - OFI vs TI horse-race regressions
7. `run_all_analyses.py` - Master script to run all analyses

### Tables (`results/tables/`)
1. `table_2_regression_statistics.csv` - Main regression results
2. `table_3_depth_analysis.csv` - Depth power law fits
3. `table_4_variance_decomposition.csv` - Variance breakdown
4. `table_5_ofi_ti_comparison.csv` - OFI vs TI comparison

### Figures (`results/figures/`)
1. `figure_2_*_ofi_vs_price.{png,pdf}` - Scatter plots (individual + combined)
2. `figure_3_*_residual_diagnostics.{png,pdf}` - Residual analysis (4-panel)
3. `figure_4_depth_analysis.{png,pdf}` - β vs depth time series + power law
4. `figure_5_event_analysis.{png,pdf}` - Event distribution + impact
5. `figure_6_ofi_ti_comparison.{png,pdf}` - Model comparison bar charts

### Detailed Analysis (`results/analysis/`)
- `*_regression_detailed.csv` - Full regression statistics
- `*_rolling_regression.csv` - Time-varying β estimates
- `*_event_statistics.csv` - Event-level breakdown
- `*_ti_comparison.csv` - Trade imbalance metrics
- `residual_diagnostics_summary.csv` - Diagnostic test results

---

## Methodology Notes

### OFI Calculation
Following Cont et al. (2011):
```
OFI_t = Σ (ΔQ_bid × I{P_bid ≥ P_mid,prev}) - Σ (ΔQ_ask × I{P_ask ≤ P_mid,prev})
```

Where:
- ΔQ_bid: Change in bid-side depth
- ΔQ_ask: Change in ask-side depth
- I{·}: Indicator function for price movement events

### Trade Imbalance Proxy
Since Polymarket doesn't provide direct trade data:
```
TI_proxy = (bid_up - bid_down) + (ask_down - ask_up)
```

Rationale:
- bid_up: Buying pressure (bullish signal)
- bid_down: Selling pressure (bearish signal)
- ask_down: Ask removed by buy order (bullish)
- ask_up: New ask added (bearish)

---

## Statistical Validation

### Residual Diagnostics

#### Fed Market
- Normality: Rejected (Jarque-Bera p < 0.001, high kurtosis)
- Autocorrelation: None detected (Durbin-Watson = 2.29)
- Heteroscedasticity: Homoscedastic (BP test p = 0.42)

#### NYC Market
- Normality: Rejected (Jarque-Bera p < 0.001, high kurtosis)
- Autocorrelation: None detected (Durbin-Watson = 2.21)
- Heteroscedasticity: Detected (BP test p < 0.001)

**Conclusion:** Despite non-normal residuals (typical in finance), no autocorrelation and use of robust standard errors ensure valid inference.

---

## Conclusions

### Confirmed from Original Paper
1. ✓ **OFI is statistically significant** in predicting price changes
2. ✓ **OFI dominates trade imbalance** as a price predictor
3. ✓ **Linear relationship** between OFI and price changes

### New Findings for Prediction Markets
1. ⚠ **Much lower explanatory power** (5-7% vs 65%)
2. ⚠ **Weak/absent depth relationship** (λ far from 1)
3. ⚠ **Event-driven dynamics** (89-93% no-change observations)
4. ⚠ **News dominates orderbook** in variance decomposition

### Implications
- **For Traders:** OFI signals work but are weaker in prediction markets; prioritize news/events over orderbook microstructure
- **For Researchers:** Prediction markets require different modeling approaches than continuous markets
- **For Market Design:** Orderbook depth matters less for price stability in binary outcome markets

---

## Reproducibility

To regenerate all results:
```bash
python scripts/run_all_analyses.py
```

Individual analyses:
```bash
python scripts/01_regression_analysis.py      # Tables 2, detailed stats
python scripts/02_create_figure_2.py          # Figure 2 (scatter plots)
python scripts/03_residual_diagnostics.py     # Figure 3 (diagnostics)
python scripts/04_depth_analysis.py           # Table 3, Figure 4
python scripts/05_event_analysis.py           # Table 4, Figure 5
python scripts/06_trade_volume_analysis.py    # Table 5, Figure 6
```

---

## Next Steps (Dashboard Integration)

Remaining tasks for full integration:
1. Add regression results table to main dashboard
2. Create depth analysis visualization tab
3. Add event pattern analysis tab
4. Implement multi-market comparison view

---

## References

**Original Paper:**
Cont, R., Kukanov, A., & Stoikov, S. (2011). The Price Impact of Order Book Events. *Journal of Financial Econometrics*, 12(1), 47-88.

**Data Source:**
Polymarket API (Dome orderbook snapshots)

**Markets Analyzed:**
- Fed Rate Decision (Dec 2025): `0x0000...` (26,840 obs)
- NYC Mayoral Election 2025: `0x0000...` (101,270 obs)

---

*Generated: 2025-11-21*
*Replication by: Claude Code*
