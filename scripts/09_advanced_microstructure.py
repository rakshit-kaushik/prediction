"""
09_advanced_microstructure.py
==============================
Three advanced analyses extending Silantyev (2019):

1. Combined OFI + TFI model: ΔMP = α + β₁·OFI + β₂·TFI + ε (per phase)
2. Quote-to-trade ratio analysis (per phase + per day)
3. Spread dynamics correlated with R² (per phase)

Usage:
    python scripts/09_advanced_microstructure.py
"""

import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from config_analysis import TICK_SIZE

# File paths
OFI_FILE = BASE_DIR / "data" / "nyc_mayor_oct15_nov04_ofi.csv"
DOME_FILE = BASE_DIR / "DOME_zohran-oct-15_2025-11-29.csv"
OFI_PHASE_FILE = BASE_DIR / "data" / "ofi_phase_analysis.csv"
TFI_PHASE_FILE = BASE_DIR / "results" / "tables" / "tfi_phase_analysis.csv"
OUTPUT_DIR = BASE_DIR / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SHARED: Data Loading & Phase Division
# ============================================================================

def load_ofi_data():
    """Load raw OFI orderbook snapshots."""
    df = pd.read_csv(OFI_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_dome_trades_yes_only():
    """Load DOME trades, YES token only."""
    raw = pd.read_csv(DOME_FILE, usecols=['primary_token_id', 'token_id', 'block_timestamp',
                                           'shares', 'price', 'side'])
    primary = str(raw['primary_token_id'].iloc[0])
    trades = raw[raw['token_id'].astype(str) == primary].copy()
    trades['timestamp'] = pd.to_datetime(trades['block_timestamp'], utc=True)
    trades['shares_normalized'] = trades['shares'] / 1e6
    trades['signed_volume'] = np.where(
        trades['side'] == 'BUY',
        trades['shares_normalized'],
        -trades['shares_normalized']
    )
    trades = trades.sort_values('timestamp').reset_index(drop=True)
    return trades


def split_into_phases(df):
    """Split sorted dataframe into 3 equal phases by observation count.

    Matches the dashboard and 06_export_all_analysis.py snapshot method.
    """
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    n = len(df)
    phase_size = n // 3
    df = df.copy()
    df['phase'] = 'Phase 3: Near Expiry'
    df.loc[:phase_size - 1, 'phase'] = 'Phase 1: Early'
    df.loc[phase_size:2 * phase_size - 1, 'phase'] = 'Phase 2: Middle'
    return df


def normalize_phase_name(name):
    """Normalize phase names: 'Phase 1 (Early)' and 'Phase 1: Early' → 'Phase 1: Early'."""
    m = re.match(r'Phase\s+(\d+)\s*[\(:]?\s*(.*?)\s*\)?$', name)
    if m:
        return f'Phase {m.group(1)}: {m.group(2)}'
    return name


# ============================================================================
# ANALYSIS 1: Combined OFI + TFI Model
# ============================================================================

def filter_zscore(df, column, threshold=3):
    """Remove rows where |z-score| > threshold for the given column."""
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return df
    z = (df[column] - mean) / std
    return df[z.abs() <= threshold]


def run_combined_model(ofi_raw, trades, outlier_method='Raw'):
    """Run ΔMP = α + β₁·OFI + β₂·TFI + ε per phase, across multiple time windows."""
    print("\n" + "=" * 80)
    print(f"ANALYSIS 1: COMBINED OFI + TFI MODEL  [{outlier_method}]")
    print("  ΔMP_k = α + β₁·OFI_k + β₂·TFI_k + ε")
    print(f"  (ΔMP = inter-window price change, outlier: {outlier_method})")
    print("=" * 80)

    TIME_WINDOWS_MIN = [1, 5, 10, 15, 20, 30, 45, 60, 90]
    all_results = []

    for tw_min in TIME_WINDOWS_MIN:
        freq = f'{tw_min}min'

        # Aggregate OFI to time windows
        ofi = ofi_raw.copy()
        ofi['window'] = ofi['timestamp'].dt.floor(freq)
        ofi_agg = ofi.groupby('window').agg(
            ofi=('ofi', 'sum'),
            mid_price_last=('mid_price', 'last'),
        ).reset_index()
        # Inter-window price change: MP_k - MP_{k-1} (matches 07_three_phase_analysis.py)
        ofi_agg['delta_mp_ticks'] = ofi_agg['mid_price_last'].diff() / TICK_SIZE
        ofi_agg = ofi_agg.dropna()

        # Outlier filtering on OFI
        if outlier_method == 'Z-Score (3)':
            ofi_agg = filter_zscore(ofi_agg, 'ofi', threshold=3)

        # Split phases on OFI-only data first (matches dashboard & OFI phase CSV)
        ofi_agg = split_into_phases(ofi_agg)

        # Aggregate TFI to same windows
        tr = trades.copy()
        tr['window'] = tr['timestamp'].dt.floor(freq)
        tfi_agg = tr.groupby('window').agg(
            tfi=('signed_volume', 'sum'),
        ).reset_index()

        # Merge — left join preserves OFI phase assignments; fill missing TFI with 0
        merged = pd.merge(ofi_agg, tfi_agg, on='window', how='left')
        merged['tfi'] = merged['tfi'].fillna(0)

        # Outlier filtering on TFI (per phase)
        if outlier_method == 'Z-Score (3)':
            parts = []
            for pn in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
                part = merged[merged['phase'] == pn].copy()
                part = filter_zscore(part, 'tfi', threshold=3)
                parts.append(part)
            merged = pd.concat(parts, ignore_index=True)

        for phase_name in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
            pdf = merged[merged['phase'] == phase_name].copy()
            if len(pdf) < 30:
                continue

            y = pdf['delta_mp_ticks'].values

            # Model 1: OFI only
            X_ofi = sm.add_constant(pdf[['ofi']].values)
            res_ofi = sm.OLS(y, X_ofi).fit(cov_type='HC0')

            # Model 2: TFI only
            X_tfi = sm.add_constant(pdf[['tfi']].values)
            res_tfi = sm.OLS(y, X_tfi).fit(cov_type='HC0')

            # Model 3: Combined
            X_both = sm.add_constant(pdf[['ofi', 'tfi']].values)
            res_both = sm.OLS(y, X_both).fit(cov_type='HC0')

            all_results.append({
                'outlier_method': outlier_method,
                'time_window': tw_min,
                'phase': phase_name,
                'n_obs': len(pdf),
                'r2_ofi_only': res_ofi.rsquared,
                'r2_tfi_only': res_tfi.rsquared,
                'r2_combined': res_both.rsquared,
                'r2_gain_from_tfi': res_both.rsquared - res_ofi.rsquared,
                'beta_ofi_combined': res_both.params[1],
                'beta_tfi_combined': res_both.params[2],
                'p_ofi_combined': res_both.pvalues[1],
                'p_tfi_combined': res_both.pvalues[2],
                'beta_ofi_solo': res_ofi.params[1],
                'p_ofi_solo': res_ofi.pvalues[1],
                'beta_tfi_solo': res_tfi.params[1],
                'p_tfi_solo': res_tfi.pvalues[1],
            })

    all_df = pd.DataFrame(all_results)

    # Display best combined model per phase
    sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

    print(f"\n  Best Combined Model per Phase (by combined R²):")
    print(f"  {'Phase':<25s} {'Window':>8s} {'N':>6s} {'OFI R²':>10s} {'TFI R²':>10s} {'Comb R²':>10s} {'R² gain':>10s}")
    print(f"  {'-'*75}")

    for phase_name in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
        phase_df = all_df[all_df['phase'] == phase_name]
        if len(phase_df) == 0:
            continue
        best_idx = phase_df['r2_combined'].idxmax()
        best = phase_df.loc[best_idx]

        print(f"  {phase_name:<25s} {int(best['time_window']):>6d}m {int(best['n_obs']):>6d} "
              f"{best['r2_ofi_only']*100:>9.2f}% {best['r2_tfi_only']*100:>9.2f}% "
              f"{best['r2_combined']*100:>9.2f}% {best['r2_gain_from_tfi']*100:>+9.2f}pp")

    # Detailed breakdown for best window per phase
    for phase_name in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
        phase_df = all_df[all_df['phase'] == phase_name]
        if len(phase_df) == 0:
            continue
        best_idx = phase_df['r2_combined'].idxmax()
        best = phase_df.loc[best_idx]

        print(f"\n  {phase_name} (best window: {int(best['time_window'])}min, N={int(best['n_obs'])})")
        print(f"  {'Model':<20s} {'R²':>8s}  {'β_OFI':>12s}  {'p(OFI)':>10s}  {'β_TFI':>12s}  {'p(TFI)':>10s}")
        print(f"  {'-'*74}")

        print(f"  {'OFI only':<20s} {best['r2_ofi_only']*100:7.2f}%  {best['beta_ofi_solo']:12.2e}  {best['p_ofi_solo']:10.2e}{sig(best['p_ofi_solo'])}")
        print(f"  {'TFI only':<20s} {best['r2_tfi_only']*100:7.2f}%  {'':>12s}  {'':>10s}  {best['beta_tfi_solo']:12.2e}  {best['p_tfi_solo']:10.2e}{sig(best['p_tfi_solo'])}")
        print(f"  {'OFI + TFI':<20s} {best['r2_combined']*100:7.2f}%  {best['beta_ofi_combined']:12.2e}  {best['p_ofi_combined']:10.2e}{sig(best['p_ofi_combined'])}  {best['beta_tfi_combined']:12.2e}  {best['p_tfi_combined']:10.2e}{sig(best['p_tfi_combined'])}")

        r2_gain = best['r2_gain_from_tfi'] * 100
        print(f"  R² gain from adding TFI: {r2_gain:+.2f} percentage points")

    return all_df


# ============================================================================
# ANALYSIS 2: Quote-to-Trade Ratio
# ============================================================================

def run_quote_trade_ratio(ofi_raw, trades):
    """Compute quote-to-trade ratio per phase and per day."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: QUOTE-TO-TRADE RATIO")
    print("  Silantyev's BitMex: 2.08 (low → TFI wins)")
    print("=" * 80)

    # Split OFI by observation count (snapshot method, matches dashboard)
    ofi_phased = split_into_phases(ofi_raw)
    # Get phase time boundaries from OFI split, apply to trades
    phase_boundaries = []
    for pname in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
        pdf = ofi_phased[ofi_phased['phase'] == pname]
        phase_boundaries.append((pname, pdf['timestamp'].min(), pdf['timestamp'].max() + pd.Timedelta(seconds=1)))
    trades_phased = trades.copy()
    trades_phased['phase'] = None
    for name, start, end in phase_boundaries:
        mask = (trades_phased['timestamp'] >= start) & (trades_phased['timestamp'] < end)
        trades_phased.loc[mask, 'phase'] = name

    print(f"\n  {'Phase':<25s}  {'Quotes':>10s}  {'Trades':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*58}")

    results = []
    for phase_name in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
        n_quotes = (ofi_phased['phase'] == phase_name).sum()
        n_trades = (trades_phased['phase'] == phase_name).sum()
        ratio = n_quotes / n_trades if n_trades > 0 else float('inf')

        print(f"  {phase_name:<25s}  {n_quotes:>10,}  {n_trades:>10,}  {ratio:>8.2f}")
        results.append({
            'phase': phase_name,
            'n_quote_events': n_quotes,
            'n_trade_events': n_trades,
            'quote_trade_ratio': ratio,
        })

    overall_q = len(ofi_raw)
    overall_t = len(trades)
    overall_ratio = overall_q / overall_t
    print(f"  {'OVERALL':<25s}  {overall_q:>10,}  {overall_t:>10,}  {overall_ratio:>8.2f}")
    results.append({
        'phase': 'OVERALL',
        'n_quote_events': overall_q,
        'n_trade_events': overall_t,
        'quote_trade_ratio': overall_ratio,
    })

    print(f"\n  Silantyev BitMex ratio: 2.08")
    print(f"  Our Polymarket ratio:  {overall_ratio:.2f}")
    if overall_ratio > 2.08:
        print(f"  → Higher ratio = richer order book = OFI has more signal → explains why OFI >> TFI here")
    else:
        print(f"  → Lower ratio = fewer quotes per trade")

    # Per-day breakdown
    print(f"\n  Daily quote-to-trade ratios:")
    ofi_raw_copy = ofi_raw.copy()
    ofi_raw_copy['date'] = ofi_raw_copy['timestamp'].dt.date
    trades_copy = trades.copy()
    trades_copy['date'] = trades_copy['timestamp'].dt.date

    daily_quotes = ofi_raw_copy.groupby('date').size()
    daily_trades = trades_copy.groupby('date').size()
    daily = pd.DataFrame({'quotes': daily_quotes, 'trades': daily_trades}).dropna()
    daily['ratio'] = daily['quotes'] / daily['trades']

    for date, row in daily.iterrows():
        print(f"    {date}:  {int(row['quotes']):>6,} quotes / {int(row['trades']):>6,} trades = {row['ratio']:.2f}")

    return pd.DataFrame(results)


# ============================================================================
# ANALYSIS 3: Spread Dynamics
# ============================================================================

def run_spread_analysis(ofi_raw):
    """Compute spread stats per phase and correlate with best R² from existing results."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: SPREAD DYNAMICS")
    print("  Silantyev: wide spreads degrade OFI power")
    print("=" * 80)

    ofi_phased = split_into_phases(ofi_raw)

    # Spread stats per phase
    print(f"\n  Spread Statistics by Phase:")
    print(f"  {'Phase':<25s}  {'Mean($)':>10s}  {'Median($)':>10s}  {'Std($)':>10s}  {'Mean(%)':>10s}  {'Std(%)':>10s}")
    print(f"  {'-'*80}")

    spread_results = []
    for phase_name in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
        pdf = ofi_phased[ofi_phased['phase'] == phase_name]
        s = pdf['spread']
        sp = pdf['spread_pct']
        row = {
            'phase': phase_name,
            'spread_mean': s.mean(),
            'spread_median': s.median(),
            'spread_std': s.std(),
            'spread_pct_mean': sp.mean(),
            'spread_pct_std': sp.std(),
            'n_snapshots': len(pdf),
        }
        spread_results.append(row)
        print(f"  {phase_name:<25s}  {row['spread_mean']:10.5f}  {row['spread_median']:10.5f}  {row['spread_std']:10.5f}  {row['spread_pct_mean']:9.4f}%  {row['spread_pct_std']:9.4f}%")

    spread_df = pd.DataFrame(spread_results)

    # Load best R² per phase from existing CSV results (across all configs)
    ofi_best = {}  # {normalized_phase: {'r2': float, 'config': str}}
    tfi_best = {}

    if OFI_PHASE_FILE.exists():
        ofi_phases_csv = pd.read_csv(OFI_PHASE_FILE)
        for phase in ofi_phases_csv['phase'].unique():
            norm = normalize_phase_name(phase)
            rows = ofi_phases_csv[ofi_phases_csv['phase'] == phase]
            best = rows.loc[rows['r_squared'].idxmax()]
            ofi_best[norm] = {
                'r2': best['r_squared'],
                'config': f"{int(best['time_window'])}min/{best['outlier_method']}",
            }

    if TFI_PHASE_FILE.exists():
        tfi_phases_csv = pd.read_csv(TFI_PHASE_FILE)
        for phase in tfi_phases_csv['phase'].unique():
            norm = normalize_phase_name(phase)
            rows = tfi_phases_csv[tfi_phases_csv['phase'] == phase]
            best = rows.loc[rows['r_squared'].idxmax()]
            tfi_best[norm] = {
                'r2': best['r_squared'],
                'config': f"{int(best['time_window'])}min/{best['outlier_method']}",
            }

    # Display spread vs best R²
    print(f"\n  Spread vs Best R² (across all configs):")
    print(f"  {'Phase':<25s}  {'Spread(%)':>10s}  {'OFI R²':>10s} {'OFI Config':>20s}  {'TFI R²':>10s} {'TFI Config':>20s}")
    print(f"  {'-'*100}")

    for _, row in spread_df.iterrows():
        phase = row['phase']
        ofi_info = ofi_best.get(phase, {'r2': float('nan'), 'config': 'N/A'})
        tfi_info = tfi_best.get(phase, {'r2': float('nan'), 'config': 'N/A'})
        print(f"  {phase:<25s}  {row['spread_pct_mean']:9.4f}%  {ofi_info['r2']*100:9.2f}% {ofi_info['config']:>20s}  {tfi_info['r2']*100:9.2f}% {tfi_info['config']:>20s}")

    # Add R² and config to spread_df
    spread_df['ofi_r2'] = spread_df['phase'].map(lambda p: ofi_best.get(p, {}).get('r2', float('nan')))
    spread_df['ofi_config'] = spread_df['phase'].map(lambda p: ofi_best.get(p, {}).get('config', 'N/A'))
    spread_df['tfi_r2'] = spread_df['phase'].map(lambda p: tfi_best.get(p, {}).get('r2', float('nan')))
    spread_df['tfi_config'] = spread_df['phase'].map(lambda p: tfi_best.get(p, {}).get('config', 'N/A'))

    # Compute correlation if we have 3 phases
    if len(spread_df) == 3 and not spread_df['ofi_r2'].isna().any():
        from scipy import stats as sp_stats
        r_ofi, p_ofi = sp_stats.pearsonr(spread_df['spread_pct_mean'], spread_df['ofi_r2'])
        r_tfi, p_tfi = sp_stats.pearsonr(spread_df['spread_pct_mean'], spread_df['tfi_r2'])
        print(f"\n  Correlation(spread%, OFI R²): r={r_ofi:.3f}  (p={p_ofi:.3f})")
        print(f"  Correlation(spread%, TFI R²): r={r_tfi:.3f}  (p={p_tfi:.3f})")
        if r_ofi < 0:
            print(f"  -> Negative correlation: tighter spread -> higher OFI R² (matches Silantyev)")
        else:
            print(f"  -> Positive correlation: wider spread -> higher OFI R²")

    return spread_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("ADVANCED MICROSTRUCTURE ANALYSES")
    print("Following Silantyev (2019) — Applied to Polymarket Prediction Market")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    ofi_raw = load_ofi_data()
    trades = load_dome_trades_yes_only()
    print(f"  OFI snapshots: {len(ofi_raw):,}")
    print(f"  YES trades:    {len(trades):,}")

    # Analysis 1: Combined model — run Raw and Z-Score(3) separately
    combined_raw = run_combined_model(ofi_raw, trades, outlier_method='Raw')
    combined_zscore = run_combined_model(ofi_raw, trades, outlier_method='Z-Score (3)')

    # Analysis 2 & 3
    qtr_df = run_quote_trade_ratio(ofi_raw, trades)
    spread_df = run_spread_analysis(ofi_raw)

    # Save all results
    out1_raw = OUTPUT_DIR / "combined_ofi_tfi_model_raw.csv"
    out1_zs = OUTPUT_DIR / "combined_ofi_tfi_model_zscore3.csv"
    out2 = OUTPUT_DIR / "quote_trade_ratio.csv"
    out3 = OUTPUT_DIR / "spread_dynamics.csv"

    combined_raw.to_csv(out1_raw, index=False)
    combined_zscore.to_csv(out1_zs, index=False)
    qtr_df.to_csv(out2, index=False)
    spread_df.to_csv(out3, index=False)

    print(f"\n  Saved: {out1_raw}")
    print(f"  Saved: {out1_zs}")
    print(f"  Saved: {out2}")
    print(f"  Saved: {out3}")

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
