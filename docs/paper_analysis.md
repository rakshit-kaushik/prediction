# Cont, Kukanov & Stoikov (2011) - Section-by-Section Analysis
## For Adaptation to Polymarket Prediction Markets

---

## SECTION 1: INTRODUCTION

### What They Did:
- **Motivation**: Understanding price impact of orders is crucial for optimal execution
- **Problem**: Little agreement on how to model price impact (linear? square root? temporary? permanent?)
- **Gap in Literature**: Most studies focus only on trades, ignoring limit orders and cancelations
- **Their Contribution**: Study ALL order book events (market orders, limit orders, cancelations)

### Key Arguments:
1. Quotes provide more detailed picture than trades alone (40:1 quote updates vs trades ratio)
2. Limit orders play important role - they reduce trade impact and affect price dynamics
3. Market depth affects price impact

### What We'll Do for Polymarket:
- [ ] Explain why studying OFI in prediction markets matters
- [ ] Highlight differences: decentralized CLOB vs traditional exchange
- [ ] Motivation: prediction markets have unique characteristics (binary outcomes, event-driven)
- [ ] Note: Polymarket has Level 1 data (best bid/ask) via Dome API

---

## SECTION 2: MODEL FOR PRICE IMPACT

### 2.1 Variables (What They Did):

**Order Book Variables:**
- P^B_n = best bid price at time n
- q^B_n = size of bid queue (shares)
- P^A_n = best ask price at time n
- q^A_n = size of ask queue (shares)

**OFI Formula:**
```
e_n = I{P^B_n > P^B_{n-1}} × q^B_n
    - I{P^B_n < P^B_{n-1}} × q^B_{n-1}
    - I{P^A_n < P^A_{n-1}} × q^A_n
    + I{P^A_n > P^A_{n-1}} × q^A_{n-1}
```

**Interpretation:**
- OFI increases when: bid size ↑, ask size ↓, or prices improve
- OFI decreases when: bid size ↓, ask size ↑, or prices worsen
- Market sell order = Cancel buy order (same effect on bid queue)

**Aggregation:**
```
OFI_k = Σ e_n  (sum over interval [t_{k-1}, t_k])
ΔP_k = (P_k - P_{k-1})/δ  (price change in ticks)
```

### What We'll Do for Polymarket:
- [x] **DONE**: Already implemented in `old_scripts/calculate_ofi.py`
- [x] Used same OFI formula
- [x] Prices in cents (not ticks) - Polymarket has 0.01 cent minimum
- [ ] Document any differences in market structure

**Key Difference:**
- Equity: prices in dollars/cents
- Polymarket: prices are probabilities (0.00 to 1.00), represent market's belief

---

### 2.2 Stylized Model (What They Did):

**Assumptions:**
1. Depth D is constant at all price levels beyond best bid/ask
2. Limit orders and cancelations occur only at best bid/ask

**Result:**
Under these assumptions, they derive:
```
ΔP = OFI / (2D) + ε
```

**Intuition:**
- Price impact inversely proportional to depth
- More liquidity (higher D) = smaller price impact
- Linear relationship between OFI and price changes

### What We'll Do for Polymarket:
- [ ] Test if this stylized model holds
- [ ] Polymarket may have different depth distribution
- [ ] Document empirical depth patterns

---

### 2.3 Model Specification (What They Did):

**Main Model:**
```
ΔP_k = β × OFI_k + ε_k
```

Where:
- β = price impact coefficient
- ε_k = noise term (deeper book levels, rounding)

**Depth Relationship:**
```
β_i = c / (AD_i^λ) + ν_i
```

Where:
- AD_i = average depth in interval i
- c, λ = constants to estimate
- λ = 1 from stylized model (empirically verify)

### What We'll Do for Polymarket:
- [x] **DONE**: Implemented regression ΔP_k = β × OFI_k + ε_k
- [ ] Test depth relationship β vs AD
- [ ] Check if λ ≈ 1 holds for prediction markets

---

## SECTION 3: ESTIMATION AND RESULTS

### 3.1 Data (What They Did):

**Dataset:**
- NYSE TAQ data (April 2010)
- 50 randomly selected S&P 500 stocks
- Consolidated quotes (all exchanges)
- Consolidated trades
- Level 1 data (best bid/ask only)

**Processing:**
1. Filter quotes and trades (market hours, valid prices)
2. Construct NBBO (National Best Bid/Offer)
3. Match trades with quotes
4. Aggregate to 10-second intervals

**Time Aggregation:**
- Base timescale: Δt = 10 seconds
- Tested: 0.5 seconds to 10 minutes
- Results robust across timescales

### What We'll Do for Polymarket:
- [x] **DONE**: Downloaded Polymarket orderbook data via Dome API
- [x] **Markets**: Fed Rate Decision, NYC Mayoral Election
- [x] **Date Ranges**: Oct 15 - Nov 20 (Fed), Oct 15 - Nov 4 (NYC)
- [x] **Aggregation**: 10-second intervals (can test others)
- [ ] Document data source and processing steps
- [ ] Compare data characteristics to equity markets

**Data Comparison:**

| Feature | Equity (NYSE) | Polymarket |
|---------|--------------|------------|
| Data Source | TAQ | Dome API |
| Price Range | $0.01+ | $0.00 - $1.00 |
| Tick Size | $0.01 | $0.001 |
| Trading Hours | 9:30-4:00 ET | 24/7 |
| Market Type | Continuous | Continuous CLOB |
| Depth Visibility | Level 1 | Level 1 (best bid/ask) |

---

### 3.2 Empirical Findings (What They Did):

**Regression 1: OFI → Price Changes**
```
ΔP_k = α_i + β_i × OFI_k + ε_k
```

Estimated separately for each half-hour interval i.

**Results:**
- **Average R² = 65%** (surprisingly high!)
- β coefficient almost always significant (95% z-test)
- Intercept α mostly insignificant
- Linear model fits well

**Figure 2:** Scatter plot showing linear relationship

**Table 2:** Regression results for all 50 stocks
- R² ranges: 44% - 79%
- t-statistics very high (average 11.47)
- 97% of samples have significant β

### What We'll Do for Polymarket:
- [x] **DONE**: Ran same regression
- **Fed Market**: OFI correlation = 0.1656
- **NYC Market**: OFI correlation = 0.2230
- [ ] Create scatter plots (Figure 2 equivalent)
- [ ] Create results table (Table 2 equivalent)
- [ ] Analyze residuals (kurtosis, heteroscedasticity)

**Our Results So Far:**
```
Fed Rate Decision (Dec 2025):
- Snapshots: 26,842
- OFI-Price Correlation: 0.1656
- Price range: $0.175 - $0.745

NYC Mayoral Election 2025:
- Snapshots: 101,271
- OFI-Price Correlation: 0.2230
- Price range: $0.87 - $0.96
```

---

**Regression 2: Price Impact vs Depth**
```
log(β_i) = α_L - λ × log(AD_i) + ε_L
β_i = α_M + c/AD_i^λ + ε_M
```

**Results:**
- **Average λ ≈ 1** (35 of 50 stocks can't reject λ=1)
- **Average R² = 74%** for depth relationship
- Price impact inversely proportional to depth
- Confirms stylized model prediction

**Figure 4:** Log-log plot of β vs AD

**Table 3:** Depth regression results for all 50 stocks

### What We'll Do for Polymarket:
- [ ] Calculate average depth AD_i for each interval
- [ ] Regress β_i on AD_i
- [ ] Test if λ ≈ 1 holds
- [ ] Create log-log plots
- [ ] Compare depth levels: equity vs prediction markets

---

### 3.3 Intraday Patterns (What They Did):

**Analysis:**
- Averaged β_i and AD_i for each half-hour across all days
- Normalized by stock average
- Averaged across all 50 stocks

**Findings:**

**Figure 5: Intraday Seasonality**
- Depth: Low at open (50% of avg), high at close (250% of avg)
- β: High at open (200% of avg), low at close (50% of avg)
- Inverse relationship between β and AD

**Explanation:**
- Market open: Shallow book → high price impact → high volatility
- Market close: Deep book → low price impact → low volatility
- Price impact 5× higher at open vs close

**Figure 6: Variance Decomposition**
```
var[ΔP]_i = β_i² × var[OFI]_i + var[ε]_i
```

**Pattern:**
- Price volatility: Sharp peak at open (400% of avg)
- OFI volatility: Peak at close
- β² × var[OFI] closely matches var[ΔP]

**Insight:**
Observable quantities (OFI, depth) explain intraday volatility patterns.
No need for unobservable factors (information asymmetry, etc.)

### What We'll Do for Polymarket:
- [ ] Analyze intraday patterns (if applicable - markets are 24/7)
- [ ] Alternative: analyze patterns around market events
  - News announcements
  - Debates (for political markets)
  - Economic data releases (for Fed market)
- [ ] Test variance decomposition
- [ ] Document any unique patterns in prediction markets

**Note:** Polymarket trades 24/7, so traditional market open/close patterns won't exist.
Instead, we might see patterns around:
- U.S. business hours vs off-hours
- News cycles
- Event-specific catalysts

---

## SECTION 4: PRICE IMPACT OF TRADES

### 4.1 Trade Imbalance vs OFI (What They Did):

**Question:** Is trade volume really what moves prices?

**Trade Imbalance:**
```
TI_k = Σ b_n - Σ s_n
```
where b_n = buy trade size, s_n = sell trade size

**Comparison Regressions:**
```
(8a) ΔP_k = α_i + β_i × OFI_k + ε_k
(8b) ΔP_k = α_T,i + β_T,i × TI_k + ε_T,k
(8c) ΔP_k = α_D,i + θ_O,i × OFI_k + θ_T,i × TI_k + ε_D,k
```

**Panel A Results (Mid-prices):**
- OFI alone: Average R² = 65%
- TI alone: Average R² = 32%
- Both: R² = 67% (minimal improvement)
- TI coefficient becomes insignificant when OFI included
- OFI coefficient remains highly significant

**Panel B Results (Transaction prices):**
- Same pattern for L = 2, 5, 10 trades
- OFI explains 51%, TI explains 13%
- TI insignificant when both used

**Table 4:** Detailed comparison

**Conclusion:**
1. OFI explains price movements better than TI
2. TI effect is captured by OFI
3. Trades alone don't tell the full story

### What We'll Do for Polymarket:
- [ ] Calculate trade imbalance from Dome API data
- [ ] Run same 3 regressions
- [ ] Compare R² values
- [ ] Test if TI becomes insignificant with OFI
- [ ] Document findings

---

### 4.2 Volume-Price Relationship (What They Did):

**Question:** Does volume move prices?

**Scaling Argument:**
Under CLT and LLN assumptions:
```
OFI ~ ξ × √VOL
```
where ξ ~ N(0,1)

**Implication:**
If ΔP = β × OFI, then:
```
ΔP ≈ θ × √VOL
```

**But:** This is an artifact of aggregation!

**Power-Law Regression:**
```
log|ΔP| = log(θ) + H × log(VOL) + log(ξ)
```

**Results:**
- Exponent H varies (average 0.18, generally < 0.5)
- "Square root" relation is noisy and not robust

**Comparison:**
```
(17a) |ΔP| = α_O + β_O × |OFI| + ε_O
(17b) |ΔP| = α_V + β_V × VOL^H + ε_V
(17c) |ΔP| = α_W + φ_O × |OFI| + φ_V × VOL^H + ε_W
```

**Table 5 Results:**
- |OFI| alone: R² = 58%
- VOL^H alone: R² = 23%
- Both: R² = 61% (minimal improvement)
- VOL becomes insignificant when OFI included

**Conclusion:**
- Volume-price relation is indirect (through OFI)
- OFI is more fundamental
- "It takes volume to move prices" is misleading

### What We'll Do for Polymarket:
- [ ] Calculate volume from trades
- [ ] Estimate power-law exponent H
- [ ] Run all 3 regressions
- [ ] Compare |OFI| vs VOL^H
- [ ] Document findings

---

## SECTION 5: CONCLUSION

### Their Summary:

**Main Contributions:**
1. Introduced OFI as single variable capturing all order book events
2. Found linear relationship: ΔP = β × OFI (R² = 65%)
3. Showed β inversely proportional to depth: β ~ 1/D
4. Results robust across stocks and timescales
5. OFI > trade imbalance for explaining prices
6. Observable quantities explain intraday patterns

**Key Insight:**
Price impact is simpler than previous studies suggested -
linear model with single variable works remarkably well.

### What We'll Write:

**Our Contributions:**
- [ ] First application of OFI to prediction markets
- [ ] Adaptation to decentralized CLOB (Polymarket)
- [ ] Analysis of binary outcome markets
- [ ] Comparison to traditional markets
- [ ] Unique patterns in event-driven markets

---

## APPENDIX: DATA PROCESSING

### What They Did:

**Quote Filtering:**
1. Market hours only (9:30-4:00)
2. Positive bid, ask, sizes
3. Exclude certain quote modes

**Trade Filtering:**
1. Market hours only
2. Positive price and size
3. Correction indicator ≤ 2
4. Exclude certain conditions

**NBBO Construction:**
- Track best bid/ask across all exchanges
- Sum sizes at NBBO prices

**Trade Direction:**
- Quote test: compare trade price to NBBO
- If price ≥ ask → buy trade
- If price ≤ bid → sell trade
- Match within 1 second window

**Spread Filter:**
- Remove extreme spreads (top 5%)

### What We'll Do for Polymarket:

- [x] **DONE**: Downloaded via Dome API
- [x] Filtered valid snapshots
- [ ] Document data processing pipeline
- [ ] Describe any market-specific considerations
- [ ] Note: No need for NBBO (single exchange)
- [ ] Trade direction from API

---

## SUMMARY: ADAPTATION CHECKLIST

### Section 1: Introduction ✅
- [x] Understand motivation
- [ ] Write prediction market context
- [ ] Explain why OFI matters for Polymarket

### Section 2: Model ✅
- [x] OFI formula implemented
- [x] Calculate from orderbook
- [ ] Document model specification
- [ ] Stylized model discussion

### Section 3: Results 🔄
- [x] Download data
- [x] Calculate OFI
- [x] Basic regression
- [ ] Create all tables/figures
- [ ] Depth analysis
- [ ] Intraday/event patterns

### Section 4: Trade Analysis ❌
- [ ] Calculate trade imbalance
- [ ] Compare OFI vs TI
- [ ] Volume analysis

### Section 5: Conclusion ❌
- [ ] Summarize findings
- [ ] Compare to equities
- [ ] Discuss implications

### Appendix ✅
- [x] Data pipeline documented
- [ ] Add to paper appendix

---

## NEXT STEPS

1. **Immediate**: Create results tables and figures (Section 3.2)
2. **Next**: Depth analysis (Section 3.2, Figure 4, Table 3)
3. **Then**: Trade/volume comparison (Section 4)
4. **Finally**: Write full paper sections

**Priority Order:**
1. Section 3.2 empirical findings ← START HERE
2. Section 2 model specification
3. Section 1 introduction
4. Section 4 trade analysis
5. Section 5 conclusion
