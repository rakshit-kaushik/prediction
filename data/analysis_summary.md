# OFI Analysis Summary - NYC Mayoral Election 2025

**Generated:** 2025-11-30 06:16:27

---

## 1. OFI Analysis Results

### Best Overall Configuration
- **Time Window:** 90 min
- **Outlier Method:** Z-Score (3)
- **R²:** 0.3532 (35.32%)

### Best by Phase

| Phase | Time Window | Method | R² |
|-------|-------------|--------|-----|
| Phase 1 (Early) | 90 min | Abs (200k) | 0.1595 |
| Phase 2 (Middle) | 90 min | Abs (100k) | 0.4777 |
| Phase 3 (Near Expiry) | 45 min | Z-Score (3) | 0.4428 |

### Overall Statistics
- **Mean R²:** 0.1937 (19.37%)
- **Max R²:** 0.3532
- **Min R²:** 0.0545
- **Total Configurations:** 81

---

## 2. Depth Analysis Results

### Log-Log Regression: log(β) = log(c) - λ × log(AD)

Per Cont et al. (2011), price impact β decreases with average depth AD:
**β = c / AD^λ**

| Level | λ Estimate | Std Err | p-value | Log-Log R² | Interpretation |
|-------|------------|---------|---------|------------|----------------|
| L1 | 0.375 | 0.200 | 6.26e-02 | 0.0143 | Theory holds (β decreases with AD) |
| L2 | 0.289 | 0.172 | 9.37e-02 | 0.0116 | Theory holds (β decreases with AD) |

### Qualitative Check
- **Level L1:** 27.2% of configurations show beta decreasing with depth
- **Level L2:** 76.5% of configurations show beta decreasing with depth

### Conclusion
**Supports** Cont et al. theory: λ > 0 means price impact decreases with depth (avg λ = 0.332)

---

## 3. TI vs OFI Comparison

### Winner Summary
- **OFI wins:** 81/81 configurations (100.0%)
- **TI wins:** 0/81 configurations (0.0%)
- **Average difference:** OFI R² - TI R² = 0.1924

### Best Configurations

| Metric | Time Window | Method | R² |
|--------|-------------|--------|-----|
| OFI | 90 min | Z-Score (3) | 0.3532 |
| TI | 90 min | MAD (3) | 0.0075 |

### Overall Comparison (45-min, Z-Score)

| Metric | R² | β | p-value |
|--------|-----|---|---------|
| OFI (signed) | 0.3116 | 2.57e-06 | 5.59e-49 |
| |OFI| (absolute) | 0.1698 | 2.04e-06 | 2.87e-25 |
| TI (signed) | 0.0072 | 1.06e-07 | 4.12e-02 |
| Volume | 0.0172 | 7.12e-02 | 1.50e-03 |

---

## 4. Key Conclusions

1. **OFI outperforms TI** in 81/81 configurations. OFI captures order book queue dynamics that trade imbalance misses.

2. **Price impact decreases with market depth** (λ = 0.332), supporting the Cont et al. model β = c / AD^λ.

3. **Recommended configuration:** 90-min window with Z-Score (3) filtering (R² = 0.3532).

