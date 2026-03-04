"""
08_tfi_phase_analysis.py
=========================
Three-Phase TFI Analysis: Does TFI explanatory power change as the market
approaches expiry?

Divides the timeline into 3 equal phases and runs the full TFI 81-config
grid (9 time windows x 9 outlier methods) for each phase separately.

Usage:
    python scripts/08_tfi_phase_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "data_pipeline"))

import importlib.util
spec = importlib.util.spec_from_file_location("ti", BASE_DIR / "data_pipeline" / "04_process_trades_ti.py")
ti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ti)

OUTPUT_DIR = BASE_DIR / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the core time windows (skip very long ones that won't have enough data per phase)
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]


def divide_into_phases(trades, n_phases=3):
    """Divide trades into equal-duration time phases."""
    min_t = trades['timestamp'].min()
    max_t = trades['timestamp'].max()
    duration = (max_t - min_t).total_seconds()
    phase_dur = duration / n_phases

    phases = {}
    labels = ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']
    for i in range(n_phases):
        start = min_t + pd.Timedelta(seconds=i * phase_dur)
        end = min_t + pd.Timedelta(seconds=(i + 1) * phase_dur)
        mask = (trades['timestamp'] >= start) & (trades['timestamp'] < end)
        if i == n_phases - 1:  # include last timestamp
            mask = (trades['timestamp'] >= start) & (trades['timestamp'] <= end)
        phase_df = trades[mask].copy()
        phases[labels[i]] = {
            'df': phase_df,
            'start': start,
            'end': end,
            'description': f"{start.strftime('%b %d')} - {end.strftime('%b %d')}"
        }
    return phases


def run_phase_grid(phase_trades, time_windows, mp_method='last_trade'):
    """Run TFI regression grid for a single phase's trades."""
    results = []
    for tw in time_windows:
        data = ti.calculate_ti_and_price_per_window(phase_trades, tw, mp_method)
        valid = data.dropna(subset=['delta_mid_price_ticks'])
        if len(valid) < 10:
            continue

        for method_idx, method_name in enumerate(ti.OUTLIER_METHODS):
            filtered = ti.apply_outlier_method(data, method_idx, 'trade_imbalance')
            if len(filtered.dropna(subset=['delta_mid_price_ticks'])) < 10:
                continue
            reg = ti.run_ti_regression(filtered)
            if reg is not None:
                results.append({
                    'time_window': tw,
                    'outlier_method': method_name,
                    'r_squared': reg['r_squared'],
                    'beta': reg['beta'],
                    'p_value': reg['p_value'],
                    'n_windows': reg['n_windows'],
                })
    return pd.DataFrame(results)


def main():
    print("\n" + "=" * 80)
    print("TFI THREE-PHASE ANALYSIS")
    print("Does TFI explanatory power improve near expiry?")
    print("=" * 80)

    # Load trades
    trades = ti.load_dome_trades()

    # Divide into phases
    phases = divide_into_phases(trades)

    all_phase_results = []

    for phase_name, phase_info in phases.items():
        phase_trades = phase_info['df']
        print(f"\n{'=' * 80}")
        print(f"{phase_name}  ({phase_info['description']})")
        print(f"  Trades: {len(phase_trades):,}")
        print(f"{'=' * 80}")

        results = run_phase_grid(phase_trades, TIME_WINDOWS, 'last_trade')

        if len(results) == 0:
            print("  No valid configs")
            continue

        results['phase'] = phase_name

        # Best config
        best = results.loc[results['r_squared'].idxmax()]
        print(f"\n  BEST:  {int(best['time_window'])}min / {best['outlier_method']} → R²={best['r_squared']*100:.2f}%  (p={best['p_value']:.2e}, N={int(best['n_windows'])})")

        # By time window
        print(f"\n  R² by Time Window (avg / max):")
        tw_stats = results.groupby('time_window')['r_squared'].agg(['mean', 'max'])
        for tw in TIME_WINDOWS:
            if tw in tw_stats.index:
                print(f"    {tw:3d}min: {tw_stats.loc[tw,'mean']*100:6.2f}% / {tw_stats.loc[tw,'max']*100:6.2f}%")

        print(f"\n  Overall: mean={results['r_squared'].mean()*100:.2f}%, max={results['r_squared'].max()*100:.2f}%")

        all_phase_results.append(results)

    # Combine
    combined = pd.concat(all_phase_results, ignore_index=True)

    # Summary comparison
    print("\n" + "=" * 80)
    print("PHASE COMPARISON SUMMARY")
    print("=" * 80)

    phase_summary = combined.groupby('phase')['r_squared'].agg(['mean', 'median', 'max', 'count'])
    phase_summary.columns = ['mean_r2', 'median_r2', 'max_r2', 'n_configs']

    for phase_name in ['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry']:
        if phase_name in phase_summary.index:
            row = phase_summary.loc[phase_name]
            print(f"\n  {phase_name}:")
            print(f"    Mean R²:   {row['mean_r2']*100:.2f}%")
            print(f"    Median R²: {row['median_r2']*100:.2f}%")
            print(f"    Max R²:    {row['max_r2']*100:.2f}%")
            print(f"    Configs:   {int(row['n_configs'])}")

    # Best config per phase at each time window (Raw outlier only for clean comparison)
    print("\n" + "-" * 80)
    print("R² PROGRESSION BY TIME WINDOW (Raw, no outlier filtering)")
    print("-" * 80)
    raw_only = combined[combined['outlier_method'] == 'Raw']
    if len(raw_only) > 0:
        pivot = raw_only.pivot(index='time_window', columns='phase', values='r_squared')
        pivot = pivot.reindex(columns=['Phase 1: Early', 'Phase 2: Middle', 'Phase 3: Near Expiry'])
        print(f"\n  {'Window':>8s}  {'Phase 1':>10s}  {'Phase 2':>10s}  {'Phase 3':>10s}")
        for tw in TIME_WINDOWS:
            if tw in pivot.index:
                vals = []
                for col in pivot.columns:
                    v = pivot.loc[tw, col]
                    vals.append(f"{v*100:9.2f}%" if pd.notna(v) else f"{'N/A':>10s}")
                print(f"  {tw:5d}min  {'  '.join(vals)}")

    # Save
    output_file = OUTPUT_DIR / "tfi_phase_analysis.csv"
    combined.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")

    print("\n" + "=" * 80)
    print("TFI PHASE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
