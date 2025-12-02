"""
06_export_all_analysis.py
=========================
Export all dashboard analysis data to CSV files for writing conclusions.

This script runs the same computations as the dashboard and saves results locally.

Usage:
    python data_pipeline/06_export_all_analysis.py

Output Files:
    data/ofi_overall_81configs.csv   - OFI R² for 81 configs (no phase split)
    data/ofi_phase_analysis.csv      - OFI R² by phase (243 configs)
    data/ofi_best_configs.csv        - Best configs per category
    data/depth_analysis.csv          - Beta vs depth data
    data/depth_summary.csv           - Depth correlation summary
    data/ofi_81_configs.csv          - OFI configs (matches TI format)
    data/ti_vs_ofi_comparison.csv    - Side-by-side OFI vs TI
    data/overall_comparison.csv      - Final OFI vs TI vs Vol comparison
    data/analysis_summary.md         - Key findings markdown
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION (Same as dashboard)
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
TICK_SIZE = 0.01

TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]

OUTLIER_METHODS = [
    'Raw',
    'IQR (1.5x)',
    'Pctl (1%-99%)',
    'Z-Score (3)',
    'Winsorized',
    'Abs (200k)',
    'Abs (100k)',
    'MAD (3)',
    'Pctl (5%-95%)'
]

# ============================================================================
# OUTLIER FILTERING FUNCTIONS (Same as dashboard)
# ============================================================================

def filter_outliers_iqr(df, column='ofi', multiplier=1.5):
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_percentile(df, column='ofi', lower_pct=0.01, upper_pct=0.99):
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_zscore(df, column='ofi', threshold=3):
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return df
    z_scores = (df[column] - mean) / std
    return df[z_scores.abs() <= threshold]

def winsorize_data(df, column='ofi', limits=(0.01, 0.01)):
    df = df.copy()
    lower = df[column].quantile(limits[0])
    upper = df[column].quantile(1 - limits[1])
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df

def filter_absolute_threshold(df, column='ofi', lower=-200000, upper=200000):
    df = df.copy()
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_mad(df, column='ofi', threshold=3):
    df = df.copy()
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    if mad == 0:
        return df
    modified_z = 0.6745 * (df[column] - median) / mad
    return df[np.abs(modified_z) <= threshold]

def filter_outliers_percentile_aggressive(df, column='ofi', lower_pct=0.05, upper_pct=0.95):
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]

def apply_outlier_method(df, method_idx, column='ofi'):
    """Apply outlier method by index (0-8)"""
    if method_idx == 0:
        return df.copy()
    elif method_idx == 1:
        return filter_outliers_iqr(df, column)
    elif method_idx == 2:
        return filter_outliers_percentile(df, column)
    elif method_idx == 3:
        return filter_outliers_zscore(df, column)
    elif method_idx == 4:
        return winsorize_data(df, column)
    elif method_idx == 5:
        return filter_absolute_threshold(df, column, -200000, 200000)
    elif method_idx == 6:
        return filter_absolute_threshold(df, column, -100000, 100000)
    elif method_idx == 7:
        return filter_outliers_mad(df, column, 3)
    elif method_idx == 8:
        return filter_outliers_percentile_aggressive(df, column)
    return df.copy()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ofi_data():
    """Load OFI data for NYC market"""
    ofi_file = DATA_DIR / "nyc_mayor_oct15_nov04_ofi.csv"
    if not ofi_file.exists():
        raise FileNotFoundError(f"OFI file not found: {ofi_file}")

    df = pd.read_csv(ofi_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    return df

def load_orderbook_data():
    """Load orderbook data for NYC market"""
    ob_file = DATA_DIR / "nyc_mayor_oct15_nov04_processed.csv"
    if not ob_file.exists():
        raise FileNotFoundError(f"Orderbook file not found: {ob_file}")

    df = pd.read_csv(ob_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    return df

def load_ti_data():
    """Load pre-computed TI 81-config results"""
    ti_file = DATA_DIR / "ti_81_configs.csv"
    if not ti_file.exists():
        raise FileNotFoundError(f"TI file not found: {ti_file}")
    return pd.read_csv(ti_file)

def load_ti_aggregated():
    """Load pre-computed TI aggregated data"""
    ti_file = DATA_DIR / "ti_aggregated_45min.csv"
    if not ti_file.exists():
        return None
    df = pd.read_csv(ti_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_ofi_data(df, time_window_minutes):
    """Aggregate OFI data to time windows"""
    df = df.copy()
    tw_str = f'{time_window_minutes}min'

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_bin'] = df['timestamp'].dt.floor(tw_str)

    # Aggregate
    agg_dict = {
        'ofi': 'sum',
        'mid_price': ['first', 'last'],
    }

    # Add depth columns if they exist
    for col in ['total_bid_size', 'total_ask_size', 'best_bid_size', 'best_ask_size']:
        if col in df.columns:
            agg_dict[col] = 'mean'

    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten columns
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[0] == 'mid_price':
                new_cols.append(f'mid_price_{col[1]}')
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Calculate price change
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    # Calculate average depth
    if 'total_bid_size' in aggregated.columns and 'total_ask_size' in aggregated.columns:
        aggregated['avg_depth_l2'] = (aggregated['total_bid_size'] + aggregated['total_ask_size']) / 2
    if 'best_bid_size' in aggregated.columns and 'best_ask_size' in aggregated.columns:
        aggregated['avg_depth_l1'] = (aggregated['best_bid_size'] + aggregated['best_ask_size']) / 2

    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})

    return aggregated

def split_into_phases(df, split_method='snapshot'):
    """Split into 3 phases"""
    n = len(df)
    phase_size = n // 3
    return {
        'Phase 1 (Early)': df.iloc[:phase_size].copy(),
        'Phase 2 (Middle)': df.iloc[phase_size:2*phase_size].copy(),
        'Phase 3 (Near Expiry)': df.iloc[2*phase_size:].copy()
    }

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_ofi_analysis(raw_ofi_df):
    """Export OFI analysis results"""
    print("\n" + "="*60)
    print("EXPORTING OFI ANALYSIS")
    print("="*60)

    overall_results = []
    phase_results = []

    for tw in TIME_WINDOWS:
        print(f"  Processing {tw}-min window...")
        aggregated = aggregate_ofi_data(raw_ofi_df.copy(), tw)

        if aggregated is None or len(aggregated) < 10:
            continue

        for method_idx, method_name in enumerate(OUTLIER_METHODS):
            filtered = apply_outlier_method(aggregated, method_idx, 'ofi')

            if len(filtered) < 10:
                continue

            df_clean = filtered.dropna(subset=['ofi', 'delta_mid_price_ticks'])

            if len(df_clean) < 10:
                continue

            # Overall regression (no phase split)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_clean['ofi'], df_clean['delta_mid_price_ticks']
            )

            overall_results.append({
                'time_window': tw,
                'outlier_method': method_name,
                'r_squared': r_value ** 2,
                'beta': slope,
                'p_value': p_value,
                'n_obs': len(df_clean),
                'std_err': std_err
            })

            # Phase-by-phase regression
            if len(df_clean) >= 30:
                phases = split_into_phases(df_clean)

                for phase_name, phase_df in phases.items():
                    if len(phase_df) >= 5:
                        p_clean = phase_df.dropna(subset=['ofi', 'delta_mid_price_ticks'])
                        if len(p_clean) >= 5:
                            slope_p, _, r_p, p_val_p, _ = stats.linregress(
                                p_clean['ofi'], p_clean['delta_mid_price_ticks']
                            )
                            phase_results.append({
                                'time_window': tw,
                                'outlier_method': method_name,
                                'phase': phase_name,
                                'r_squared': r_p ** 2,
                                'beta': slope_p,
                                'p_value': p_val_p,
                                'n_obs': len(p_clean)
                            })

    # Save overall results
    overall_df = pd.DataFrame(overall_results)
    overall_file = DATA_DIR / "ofi_overall_81configs.csv"
    overall_df.to_csv(overall_file, index=False)
    print(f"  ✓ Saved {overall_file.name} ({len(overall_df)} rows)")

    # Save phase results
    phase_df = pd.DataFrame(phase_results)
    phase_file = DATA_DIR / "ofi_phase_analysis.csv"
    phase_df.to_csv(phase_file, index=False)
    print(f"  ✓ Saved {phase_file.name} ({len(phase_df)} rows)")

    # Save best configs
    best_configs = []
    if len(overall_df) > 0:
        best_overall = overall_df.loc[overall_df['r_squared'].idxmax()]
        best_configs.append({
            'category': 'Overall Best',
            'time_window': best_overall['time_window'],
            'method': best_overall['outlier_method'],
            'r_squared': best_overall['r_squared']
        })

    if len(phase_df) > 0:
        for phase in phase_df['phase'].unique():
            phase_data = phase_df[phase_df['phase'] == phase]
            if len(phase_data) > 0:
                best = phase_data.loc[phase_data['r_squared'].idxmax()]
                best_configs.append({
                    'category': f'Best {phase}',
                    'time_window': best['time_window'],
                    'method': best['outlier_method'],
                    'r_squared': best['r_squared']
                })

    best_df = pd.DataFrame(best_configs)
    best_file = DATA_DIR / "ofi_best_configs.csv"
    best_df.to_csv(best_file, index=False)
    print(f"  ✓ Saved {best_file.name} ({len(best_df)} rows)")

    return overall_df, phase_df, best_df


def export_depth_analysis(raw_ofi_df):
    """
    Export depth analysis results.

    Per Cont et al. (2011), the relationship is:
        β = c / AD^λ

    In log form:
        log(β) = log(c) - λ * log(AD)

    We estimate λ via log-log regression.
    """
    print("\n" + "="*60)
    print("EXPORTING DEPTH ANALYSIS")
    print("="*60)

    depth_results = []

    for tw in TIME_WINDOWS:
        print(f"  Processing {tw}-min window...")
        aggregated = aggregate_ofi_data(raw_ofi_df.copy(), tw)

        if aggregated is None or len(aggregated) < 10:
            continue

        # Check if depth columns exist
        has_l1 = 'avg_depth_l1' in aggregated.columns
        has_l2 = 'avg_depth_l2' in aggregated.columns

        if not has_l1 and not has_l2:
            print(f"    Warning: No depth columns for {tw}-min")
            continue

        for method_idx, method_name in enumerate(OUTLIER_METHODS):
            filtered = apply_outlier_method(aggregated, method_idx, 'ofi')

            if len(filtered) < 30:
                continue

            df_clean = filtered.dropna(subset=['ofi', 'delta_mid_price_ticks'])

            if len(df_clean) < 30:
                continue

            # Split into phases
            phases = split_into_phases(df_clean)

            for phase_name, phase_df in phases.items():
                if len(phase_df) >= 5:
                    p_clean = phase_df.dropna(subset=['ofi', 'delta_mid_price_ticks'])
                    if len(p_clean) >= 5:
                        # Run OFI regression to get beta
                        slope, _, r_value, p_value, _ = stats.linregress(
                            p_clean['ofi'], p_clean['delta_mid_price_ticks']
                        )

                        # Get average depths
                        avg_l1 = p_clean['avg_depth_l1'].mean() if has_l1 else None
                        avg_l2 = p_clean['avg_depth_l2'].mean() if has_l2 else None

                        depth_results.append({
                            'time_window': tw,
                            'outlier_method': method_name,
                            'phase': phase_name,
                            'beta': slope,
                            'r_squared': r_value ** 2,
                            'p_value': p_value,
                            'avg_depth_l1': avg_l1,
                            'avg_depth_l2': avg_l2,
                            'n_obs': len(p_clean)
                        })

    # Save depth results
    depth_df = pd.DataFrame(depth_results)
    depth_file = DATA_DIR / "depth_analysis.csv"
    depth_df.to_csv(depth_file, index=False)
    print(f"  ✓ Saved {depth_file.name} ({len(depth_df)} rows)")

    # Calculate summary with LOG-LOG regression (per Cont et al.)
    # log(β) = log(c) - λ * log(AD)
    summary_results = []

    for level in ['l1', 'l2']:
        depth_col = f'avg_depth_{level}'
        if depth_col not in depth_df.columns or depth_df[depth_col].isna().all():
            continue

        # Filter for positive beta and positive depth (required for log)
        valid_df = depth_df[(depth_df['beta'] > 0) & (depth_df[depth_col] > 0)].copy()

        if len(valid_df) < 10:
            continue

        # Log-log regression: log(β) = log(c) - λ * log(AD)
        log_beta = np.log(valid_df['beta'])
        log_depth = np.log(valid_df[depth_col])

        slope_lambda, intercept, r_value, p_value, std_err = stats.linregress(
            log_depth, log_beta
        )

        # λ = -slope (since log(β) = log(c) - λ * log(AD))
        lambda_estimate = -slope_lambda

        # Count % where beta decreases with depth (qualitative check)
        negative_corr_count = 0
        total_count = 0

        for (tw, method), group in depth_df.groupby(['time_window', 'outlier_method']):
            if len(group) >= 2:
                group_valid = group[(group['beta'] > 0) & (group[depth_col] > 0)]
                if len(group_valid) >= 2:
                    corr = group_valid['beta'].corr(group_valid[depth_col])
                    if not pd.isna(corr):
                        total_count += 1
                        if corr < 0:
                            negative_corr_count += 1

        pct_negative = (negative_corr_count / total_count * 100) if total_count > 0 else 0

        summary_results.append({
            'level': level.upper(),
            'lambda_estimate': lambda_estimate,
            'lambda_std_err': std_err,
            'lambda_p_value': p_value,
            'log_log_r_squared': r_value ** 2,
            'pct_negative_correlation': pct_negative,
            'n_configs': total_count,
            'n_datapoints': len(valid_df),
            'interpretation': 'Theory holds (β decreases with AD)' if lambda_estimate > 0 else 'Theory does not hold'
        })

    summary_df = pd.DataFrame(summary_results)
    summary_file = DATA_DIR / "depth_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ Saved {summary_file.name} ({len(summary_df)} rows)")

    # Print key findings
    for _, row in summary_df.iterrows():
        print(f"    Level {row['level']}: λ = {row['lambda_estimate']:.3f} (p={row['lambda_p_value']:.2e})")
        print(f"      Log-log R² = {row['log_log_r_squared']:.4f}")
        print(f"      {row['interpretation']}")

    return depth_df, summary_df


def export_ti_comparison(raw_ofi_df, ti_df):
    """Export TI vs OFI comparison results"""
    print("\n" + "="*60)
    print("EXPORTING TI VS OFI COMPARISON")
    print("="*60)

    # Compute OFI 81-configs (to match TI format)
    ofi_results = []

    for tw in TIME_WINDOWS:
        print(f"  Processing {tw}-min window...")
        aggregated = aggregate_ofi_data(raw_ofi_df.copy(), tw)

        if aggregated is None or len(aggregated) < 10:
            continue

        for method_idx, method_name in enumerate(OUTLIER_METHODS):
            filtered = apply_outlier_method(aggregated, method_idx, 'ofi')

            if len(filtered) < 10:
                continue

            df_clean = filtered.dropna(subset=['ofi', 'delta_mid_price_ticks'])

            if len(df_clean) < 10:
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_clean['ofi'], df_clean['delta_mid_price_ticks']
            )

            ofi_results.append({
                'time_window': tw,
                'outlier_method': method_name,
                'r_squared': r_value ** 2,
                'beta': slope,
                'p_value': p_value,
                'n_windows': len(df_clean),
                'std_err': std_err
            })

    ofi_df = pd.DataFrame(ofi_results)
    ofi_file = DATA_DIR / "ofi_81_configs.csv"
    ofi_df.to_csv(ofi_file, index=False)
    print(f"  ✓ Saved {ofi_file.name} ({len(ofi_df)} rows)")

    # Create side-by-side comparison
    comparison_results = []

    for _, ofi_row in ofi_df.iterrows():
        tw = ofi_row['time_window']
        method = ofi_row['outlier_method']

        ti_match = ti_df[(ti_df['time_window'] == tw) & (ti_df['outlier_method'] == method)]

        if len(ti_match) > 0:
            ti_r2 = ti_match.iloc[0]['r_squared']
            ofi_r2 = ofi_row['r_squared']

            comparison_results.append({
                'time_window': tw,
                'outlier_method': method,
                'ofi_r2': ofi_r2,
                'ti_r2': ti_r2,
                'difference': ofi_r2 - ti_r2,
                'winner': 'OFI' if ofi_r2 > ti_r2 else 'TI'
            })

    comparison_df = pd.DataFrame(comparison_results)
    comparison_file = DATA_DIR / "ti_vs_ofi_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  ✓ Saved {comparison_file.name} ({len(comparison_df)} rows)")

    # Overall comparison (45-min, Z-Score)
    overall_results = []

    # Get 45-min Z-Score data
    aggregated_45 = aggregate_ofi_data(raw_ofi_df.copy(), 45)
    filtered_45 = apply_outlier_method(aggregated_45, 3, 'ofi')  # Z-Score is index 3

    # Load TI aggregated data
    ti_agg = load_ti_aggregated()

    if ti_agg is not None:
        # Merge OFI and TI
        merged = pd.merge(filtered_45, ti_agg, on='timestamp', how='inner')
        merged = merged.dropna(subset=['ofi', 'delta_mid_price_ticks'])

        # OFI regression (signed)
        if len(merged) >= 10:
            slope_ofi, _, r_ofi, p_ofi, _ = stats.linregress(
                merged['ofi'], merged['delta_mid_price_ticks']
            )
            overall_results.append({
                'metric': 'OFI (signed)',
                'r_squared': r_ofi ** 2,
                'beta': slope_ofi,
                'p_value': p_ofi,
                'notes': 'ΔP = α + β*OFI'
            })

            # |OFI| regression (absolute)
            slope_ofi_abs, _, r_ofi_abs, p_ofi_abs, _ = stats.linregress(
                merged['ofi'].abs(), merged['delta_mid_price_ticks'].abs()
            )
            overall_results.append({
                'metric': '|OFI| (absolute)',
                'r_squared': r_ofi_abs ** 2,
                'beta': slope_ofi_abs,
                'p_value': p_ofi_abs,
                'notes': '|ΔP| = α + β*|OFI| (Table 5 style)'
            })

        # TI regression
        ti_clean = merged.dropna(subset=['trade_imbalance', 'delta_mid_price_ticks'])
        if len(ti_clean) >= 10:
            slope_ti, _, r_ti, p_ti, _ = stats.linregress(
                ti_clean['trade_imbalance'], ti_clean['delta_mid_price_ticks']
            )
            overall_results.append({
                'metric': 'TI (signed)',
                'r_squared': r_ti ** 2,
                'beta': slope_ti,
                'p_value': p_ti,
                'notes': 'ΔP = α + β*TI'
            })

        # Volume regression
        vol_clean = merged.dropna(subset=['total_volume', 'delta_mid_price_ticks'])
        if len(vol_clean) >= 10:
            mask = (vol_clean['total_volume'] > 0) & (vol_clean['delta_mid_price_ticks'].abs() > 0)
            vol_log = vol_clean[mask]

            if len(vol_log) >= 10:
                log_vol = np.log(vol_log['total_volume'])
                log_abs_price = np.log(vol_log['delta_mid_price_ticks'].abs())
                H_estimate = stats.linregress(log_vol, log_abs_price).slope

                vol_H = np.power(vol_clean['total_volume'] + 1e-10, H_estimate)
                _, _, r_vol, p_vol, _ = stats.linregress(
                    vol_H, vol_clean['delta_mid_price_ticks'].abs()
                )

                overall_results.append({
                    'metric': 'Volume',
                    'r_squared': r_vol ** 2,
                    'beta': H_estimate,
                    'p_value': p_vol,
                    'notes': f'|ΔP| = α + β*VOL^H, H={H_estimate:.3f}'
                })

    overall_comp_df = pd.DataFrame(overall_results)
    overall_file = DATA_DIR / "overall_comparison.csv"
    overall_comp_df.to_csv(overall_file, index=False)
    print(f"  ✓ Saved {overall_file.name} ({len(overall_comp_df)} rows)")

    return ofi_df, comparison_df, overall_comp_df


def generate_summary_markdown(ofi_overall, ofi_phase, ofi_best, depth_df, depth_summary,
                               comparison_df, overall_comp):
    """Generate analysis_summary.md with key findings"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY MARKDOWN")
    print("="*60)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build markdown content
    md = f"""# OFI Analysis Summary - NYC Mayoral Election 2025

**Generated:** {timestamp}

---

## 1. OFI Analysis Results

### Best Overall Configuration
"""

    if len(ofi_best) > 0:
        best_overall = ofi_best[ofi_best['category'] == 'Overall Best']
        if len(best_overall) > 0:
            row = best_overall.iloc[0]
            md += f"""- **Time Window:** {int(row['time_window'])} min
- **Outlier Method:** {row['method']}
- **R²:** {row['r_squared']:.4f} ({row['r_squared']*100:.2f}%)
"""

    md += """
### Best by Phase

| Phase | Time Window | Method | R² |
|-------|-------------|--------|-----|
"""

    for _, row in ofi_best.iterrows():
        if 'Phase' in row['category']:
            md += f"| {row['category'].replace('Best ', '')} | {int(row['time_window'])} min | {row['method']} | {row['r_squared']:.4f} |\n"

    # OFI statistics
    if len(ofi_overall) > 0:
        md += f"""
### Overall Statistics
- **Mean R²:** {ofi_overall['r_squared'].mean():.4f} ({ofi_overall['r_squared'].mean()*100:.2f}%)
- **Max R²:** {ofi_overall['r_squared'].max():.4f}
- **Min R²:** {ofi_overall['r_squared'].min():.4f}
- **Total Configurations:** {len(ofi_overall)}
"""

    # Depth Analysis
    md += """
---

## 2. Depth Analysis Results

### Log-Log Regression: log(β) = log(c) - λ × log(AD)

Per Cont et al. (2011), price impact β decreases with average depth AD:
**β = c / AD^λ**

"""

    if len(depth_summary) > 0:
        md += """| Level | λ Estimate | Std Err | p-value | Log-Log R² | Interpretation |
|-------|------------|---------|---------|------------|----------------|
"""
        for _, row in depth_summary.iterrows():
            md += f"| {row['level']} | {row['lambda_estimate']:.3f} | {row['lambda_std_err']:.3f} | {row['lambda_p_value']:.2e} | {row['log_log_r_squared']:.4f} | {row['interpretation']} |\n"

        md += f"""
### Qualitative Check
"""
        for _, row in depth_summary.iterrows():
            md += f"- **Level {row['level']}:** {row['pct_negative_correlation']:.1f}% of configurations show beta decreasing with depth\n"

        # Overall interpretation
        avg_lambda = depth_summary['lambda_estimate'].mean()
        if avg_lambda > 0:
            interpretation = f"**Supports** Cont et al. theory: λ > 0 means price impact decreases with depth (avg λ = {avg_lambda:.3f})"
        else:
            interpretation = f"**Does not support** Cont et al. theory: λ < 0 (avg λ = {avg_lambda:.3f})"
        md += f"""
### Conclusion
{interpretation}
"""

    # TI vs OFI Comparison
    md += """
---

## 3. TI vs OFI Comparison

### Winner Summary
"""

    if len(comparison_df) > 0:
        ofi_wins = (comparison_df['winner'] == 'OFI').sum()
        ti_wins = (comparison_df['winner'] == 'TI').sum()
        total = len(comparison_df)
        avg_diff = comparison_df['difference'].mean()

        md += f"""- **OFI wins:** {ofi_wins}/{total} configurations ({ofi_wins/total*100:.1f}%)
- **TI wins:** {ti_wins}/{total} configurations ({ti_wins/total*100:.1f}%)
- **Average difference:** OFI R² - TI R² = {avg_diff:.4f}
"""

    # Best configs for each
    md += """
### Best Configurations

| Metric | Time Window | Method | R² |
|--------|-------------|--------|-----|
"""

    if len(comparison_df) > 0:
        best_ofi = comparison_df.loc[comparison_df['ofi_r2'].idxmax()]
        best_ti = comparison_df.loc[comparison_df['ti_r2'].idxmax()]
        md += f"| OFI | {int(best_ofi['time_window'])} min | {best_ofi['outlier_method']} | {best_ofi['ofi_r2']:.4f} |\n"
        md += f"| TI | {int(best_ti['time_window'])} min | {best_ti['outlier_method']} | {best_ti['ti_r2']:.4f} |\n"

    # Overall Comparison
    md += """
### Overall Comparison (45-min, Z-Score)

| Metric | R² | β | p-value |
|--------|-----|---|---------|
"""

    if len(overall_comp) > 0:
        for _, row in overall_comp.iterrows():
            md += f"| {row['metric']} | {row['r_squared']:.4f} | {row['beta']:.2e} | {row['p_value']:.2e} |\n"

    # Key Conclusions
    md += """
---

## 4. Key Conclusions

"""

    # Auto-generate conclusions based on data
    conclusions = []

    if len(comparison_df) > 0:
        ofi_wins = (comparison_df['winner'] == 'OFI').sum()
        ti_wins = (comparison_df['winner'] == 'TI').sum()
        if ofi_wins > ti_wins:
            conclusions.append(f"**OFI outperforms TI** in {ofi_wins}/{len(comparison_df)} configurations. OFI captures order book queue dynamics that trade imbalance misses.")
        else:
            conclusions.append(f"**TI outperforms OFI** in {ti_wins}/{len(comparison_df)} configurations.")

    if len(overall_comp) > 0:
        vol_row = overall_comp[overall_comp['metric'] == 'Volume']
        if len(vol_row) > 0:
            vol_r2 = vol_row.iloc[0]['r_squared']
            if vol_r2 < 0.01:
                conclusions.append(f"**Volume is a poor predictor** (R² = {vol_r2:.4f}), consistent with Cont et al. finding that volume becomes insignificant when OFI is included.")

    if len(depth_summary) > 0:
        avg_lambda = depth_summary['lambda_estimate'].mean()
        if avg_lambda > 0:
            conclusions.append(f"**Price impact decreases with market depth** (λ = {avg_lambda:.3f}), supporting the Cont et al. model β = c / AD^λ.")
        else:
            conclusions.append(f"**Depth relationship is inverted** (λ = {avg_lambda:.3f}), suggesting different dynamics in prediction markets vs traditional equities.")

    if len(ofi_best) > 0:
        best = ofi_best[ofi_best['category'] == 'Overall Best'].iloc[0]
        conclusions.append(f"**Recommended configuration:** {int(best['time_window'])}-min window with {best['method']} filtering (R² = {best['r_squared']:.4f}).")

    for i, conc in enumerate(conclusions, 1):
        md += f"{i}. {conc}\n\n"

    # Save markdown
    md_file = DATA_DIR / "analysis_summary.md"
    with open(md_file, 'w') as f:
        f.write(md)
    print(f"  ✓ Saved {md_file.name}")

    return md


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("EXPORTING ALL DASHBOARD ANALYSIS DATA")
    print("="*70)

    # Load data
    print("\nLoading data...")
    raw_ofi_df = load_ofi_data()
    print(f"  Loaded {len(raw_ofi_df):,} OFI records")

    ti_df = load_ti_data()
    print(f"  Loaded {len(ti_df):,} TI configs")

    # Export OFI Analysis
    ofi_overall, ofi_phase, ofi_best = export_ofi_analysis(raw_ofi_df)

    # Export Depth Analysis
    depth_df, depth_summary = export_depth_analysis(raw_ofi_df)

    # Export TI vs OFI Comparison
    ofi_81, comparison_df, overall_comp = export_ti_comparison(raw_ofi_df, ti_df)

    # Generate Summary
    generate_summary_markdown(
        ofi_overall, ofi_phase, ofi_best,
        depth_df, depth_summary,
        comparison_df, overall_comp
    )

    print("\n" + "="*70)
    print("✓ ALL ANALYSIS DATA EXPORTED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutput files saved to: {DATA_DIR}")
    print("\nFiles generated:")
    print("  - ofi_overall_81configs.csv")
    print("  - ofi_phase_analysis.csv")
    print("  - ofi_best_configs.csv")
    print("  - depth_analysis.csv")
    print("  - depth_summary.csv")
    print("  - ofi_81_configs.csv")
    print("  - ti_vs_ofi_comparison.csv")
    print("  - overall_comparison.csv")
    print("  - analysis_summary.md")


if __name__ == "__main__":
    main()
