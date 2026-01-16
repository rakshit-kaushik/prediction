"""
04_compare_results.py
=====================
STEP 4: Compare MLOFI results with Level 1 OFI baseline

WHAT IT DOES:
-------------
Input:  Regression results (from step 3)
Output: Comprehensive comparison report and summary

ANALYSIS DIMENSIONS:
-------------------
1. By Level Configuration: L5 vs L10 vs L50% vs ALL
2. By Regression Method: OLS vs Ridge vs Lasso vs ElasticNet
3. By Time Window: Which aggregation works best
4. Feature Importance: Which levels matter most (from Lasso coefficients)

OUTPUTS:
--------
1. comparison_summary.csv - Detailed comparisons
2. Console report with formatted tables
3. Recommendations for optimal configuration

USAGE:
------
    python mlofi/04_compare_results.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    REGRESSION_RESULTS_FILE,
    COMPARISON_SUMMARY_FILE,
    LEVEL_IMPORTANCE_FILE,
    MLOFI_OUTPUT_DIR
)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "═" * 80)
    print(f" {title}")
    print("═" * 80)


def print_subheader(title):
    """Print a formatted subheader."""
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80)


def format_table(df, columns, headers=None):
    """Format a DataFrame as a text table."""
    if headers is None:
        headers = columns

    # Calculate column widths
    widths = []
    for col, header in zip(columns, headers):
        max_width = max(len(header), df[col].astype(str).str.len().max())
        widths.append(min(max_width + 2, 20))

    # Print header
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in widths) + "-|"

    print(header_row)
    print(separator)

    # Print rows
    for _, row in df.iterrows():
        values = []
        for col, width in zip(columns, widths):
            val = row[col]
            if isinstance(val, float):
                if abs(val) < 0.0001 and val != 0:
                    val_str = f"{val:.2e}"
                else:
                    val_str = f"{val:.4f}" if val < 1 else f"{val:.2f}"
            else:
                val_str = str(val)
            values.append(val_str.ljust(width))
        print("| " + " | ".join(values) + " |")


def analyze_level_improvement(df_results):
    """
    Analyze improvement from Level 1 to multi-level configurations.

    Returns:
        DataFrame with improvement analysis
    """
    comparisons = []

    # Get baseline (OLS_L1 for each time window)
    baseline = df_results[df_results['method'] == 'OLS_L1'].copy()
    baseline = baseline.rename(columns={'r_squared': 'baseline_r2', 'rmse': 'baseline_rmse'})

    for _, base_row in baseline.iterrows():
        tw = base_row['time_window']
        base_r2 = base_row['baseline_r2']
        base_rmse = base_row['baseline_rmse']

        # Compare with other configurations at same time window
        others = df_results[
            (df_results['time_window'] == tw) &
            (df_results['method'] != 'OLS_L1')
        ]

        for _, other_row in others.iterrows():
            improvement_r2 = ((other_row['r_squared'] - base_r2) / base_r2 * 100) if base_r2 > 0 else 0
            improvement_rmse = ((base_rmse - other_row['rmse']) / base_rmse * 100) if base_rmse > 0 else 0

            comparisons.append({
                'time_window': tw,
                'baseline': 'OLS_L1',
                'baseline_r2': base_r2,
                'comparison': f"{other_row['level_config']}_{other_row['method']}",
                'level_config': other_row['level_config'],
                'method': other_row['method'],
                'n_levels': other_row['n_levels'],
                'r_squared': other_row['r_squared'],
                'rmse': other_row['rmse'],
                'r2_improvement_pct': improvement_r2,
                'rmse_improvement_pct': improvement_rmse,
            })

    return pd.DataFrame(comparisons)


def find_optimal_configuration(df_results, df_comparisons):
    """
    Find the optimal configuration based on R² improvement.

    Returns:
        Dictionary with recommendations
    """
    recommendations = {}

    # Best overall
    best_idx = df_results['r_squared'].idxmax()
    best = df_results.loc[best_idx]
    recommendations['best_overall'] = {
        'level_config': best['level_config'],
        'method': best['method'],
        'time_window': best['time_window'],
        'r_squared': best['r_squared'],
        'rmse': best['rmse'],
    }

    # Best by level config
    best_by_config = df_results.loc[df_results.groupby('level_config')['r_squared'].idxmax()]
    recommendations['best_by_level_config'] = best_by_config[[
        'level_config', 'method', 'time_window', 'r_squared', 'rmse'
    ]].to_dict('records')

    # Best by method
    best_by_method = df_results.loc[df_results.groupby('method')['r_squared'].idxmax()]
    recommendations['best_by_method'] = best_by_method[[
        'method', 'level_config', 'time_window', 'r_squared', 'rmse'
    ]].to_dict('records')

    # Average improvement by level config (vs L1 baseline)
    if len(df_comparisons) > 0:
        avg_improvement = df_comparisons.groupby('level_config')['r2_improvement_pct'].mean()
        recommendations['avg_improvement_by_config'] = avg_improvement.to_dict()

    return recommendations


def generate_report(df_results, df_comparisons, recommendations):
    """Generate the comprehensive comparison report."""

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "MLOFI vs LEVEL 1 OFI - COMPREHENSIVE COMPARISON" + " " * 19 + "║")
    print("╚" + "═" * 78 + "╝")

    # Section 1: Best Results by Level Configuration
    print_header("BEST RESULTS BY LEVEL CONFIGURATION")

    df_best_config = pd.DataFrame(recommendations['best_by_level_config'])
    df_best_config = df_best_config.sort_values('r_squared', ascending=False)
    format_table(
        df_best_config,
        ['level_config', 'method', 'time_window', 'r_squared', 'rmse'],
        ['Config', 'Best Method', 'Window (min)', 'R²', 'RMSE']
    )

    # Section 2: Best Results by Method
    print_header("BEST RESULTS BY REGRESSION METHOD")

    df_best_method = pd.DataFrame(recommendations['best_by_method'])
    df_best_method = df_best_method.sort_values('r_squared', ascending=False)
    format_table(
        df_best_method,
        ['method', 'level_config', 'time_window', 'r_squared', 'rmse'],
        ['Method', 'Best Config', 'Window (min)', 'R²', 'RMSE']
    )

    # Section 3: Improvement Analysis
    print_header("IMPROVEMENT vs LEVEL 1 BASELINE")

    if 'avg_improvement_by_config' in recommendations:
        print("\n📊 Average R² improvement by level configuration:")
        for config, improvement in sorted(
            recommendations['avg_improvement_by_config'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"   {config}: {improvement:+.1f}%")

    # Top 10 improvements
    if len(df_comparisons) > 0:
        print_subheader("Top 10 Configurations by R² Improvement")

        df_top = df_comparisons.nlargest(10, 'r2_improvement_pct')
        format_table(
            df_top,
            ['comparison', 'time_window', 'r_squared', 'baseline_r2', 'r2_improvement_pct'],
            ['Configuration', 'Window', 'R²', 'Baseline R²', 'Improvement %']
        )

    # Section 4: Method Comparison at Fixed Config
    print_header("METHOD COMPARISON (L10 Config, 10-min Window)")

    df_l10_10 = df_results[
        (df_results['level_config'] == 'L10') &
        (df_results['time_window'] == 10)
    ].copy()

    if len(df_l10_10) > 0:
        df_l10_10 = df_l10_10.sort_values('r_squared', ascending=False)
        format_table(
            df_l10_10,
            ['method', 'r_squared', 'rmse', 'n_nonzero_coefs', 'best_alpha'],
            ['Method', 'R²', 'RMSE', 'Non-zero Coefs', 'Best Alpha']
        )

    # Section 5: Time Window Analysis
    print_header("BEST R² BY TIME WINDOW")

    best_by_window = df_results.loc[df_results.groupby('time_window')['r_squared'].idxmax()]
    best_by_window = best_by_window.sort_values('time_window')
    format_table(
        best_by_window,
        ['time_window', 'level_config', 'method', 'r_squared'],
        ['Window (min)', 'Best Config', 'Best Method', 'R²']
    )

    # Section 6: Feature Importance (from Lasso)
    print_header("FEATURE IMPORTANCE (Lasso Feature Selection)")

    lasso_results = df_results[df_results['method'] == 'Lasso'].copy()
    if len(lasso_results) > 0:
        print("\n📊 Non-zero coefficients by configuration:")
        for _, row in lasso_results.nlargest(5, 'r_squared').iterrows():
            print(f"   {row['level_config']} @ {row['time_window']}min: "
                  f"{row['n_nonzero_coefs']}/{row['n_levels']} levels kept, R²={row['r_squared']:.4f}")

    # Level importance file
    if LEVEL_IMPORTANCE_FILE.exists():
        print_subheader("Coefficient Values by Level")
        df_importance = pd.read_csv(LEVEL_IMPORTANCE_FILE)
        if len(df_importance) > 0:
            print(df_importance.head(10).to_string(index=False))

    # Section 7: Recommendations
    print_header("RECOMMENDATIONS")

    best = recommendations['best_overall']
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"   Level Config: {best['level_config']}")
    print(f"   Method: {best['method']}")
    print(f"   Time Window: {best['time_window']} minutes")
    print(f"   R²: {best['r_squared']:.4f}")
    print(f"   RMSE: {best['rmse']:.6f}")

    # Calculate improvement over baseline
    baseline_r2 = df_results[
        (df_results['method'] == 'OLS_L1') &
        (df_results['time_window'] == best['time_window'])
    ]['r_squared'].values

    if len(baseline_r2) > 0:
        improvement = ((best['r_squared'] - baseline_r2[0]) / baseline_r2[0] * 100)
        print(f"\n📈 Improvement over Level 1 OLS baseline: {improvement:+.1f}%")

    # Key findings
    print("\n📋 KEY FINDINGS:")

    # Check if multi-level helps
    l1_best = df_results[df_results['level_config'] == 'L5']['r_squared'].max()
    multi_best = df_results[df_results['level_config'].isin(['L10', 'L50pct', 'ALL'])]['r_squared'].max()

    if multi_best > l1_best * 1.05:
        print("   ✅ Multi-level OFI significantly outperforms Level 1")
    else:
        print("   ⚠️  Limited improvement from deeper levels")

    # Check which method wins
    ridge_best = df_results[df_results['method'] == 'Ridge']['r_squared'].max()
    lasso_best = df_results[df_results['method'] == 'Lasso']['r_squared'].max()
    elastic_best = df_results[df_results['method'] == 'ElasticNet']['r_squared'].max()

    if ridge_best >= max(lasso_best, elastic_best):
        print("   ✅ Ridge regression performs best (all coefficients contribute)")
    elif lasso_best > ridge_best:
        print("   ✅ Lasso performs best (some levels can be dropped)")

    # Check optimal depth
    l5_best = df_results[df_results['level_config'] == 'L5']['r_squared'].max()
    l10_best = df_results[df_results['level_config'] == 'L10']['r_squared'].max()
    all_best = df_results[df_results['level_config'] == 'ALL']['r_squared'].max()

    if l5_best >= l10_best * 0.95 and l5_best >= all_best * 0.95:
        print("   ✅ 5 levels sufficient (diminishing returns beyond)")
    elif l10_best >= all_best * 0.95:
        print("   ✅ 10 levels optimal (diminishing returns beyond)")
    else:
        print("   ✅ Deeper levels continue to help")


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "MLOFI COMPARISON ANALYSIS" + " " * 37 + "║")
    print("║" + " " * 18 + "Generating Final Report" + " " * 36 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not REGRESSION_RESULTS_FILE.exists():
        print(f"\n❌ Regression results not found: {REGRESSION_RESULTS_FILE}")
        print("   Please run step 3 first: python mlofi/03_regression_analysis.py")
        sys.exit(1)

    # Load results
    print(f"\n📂 Loading regression results...")
    df_results = pd.read_csv(REGRESSION_RESULTS_FILE)
    print(f"   Loaded {len(df_results)} regression results")

    # Analyze improvements
    print(f"\n📊 Analyzing improvements vs baseline...")
    df_comparisons = analyze_level_improvement(df_results)

    # Find optimal configuration
    print(f"\n🔍 Finding optimal configuration...")
    recommendations = find_optimal_configuration(df_results, df_comparisons)

    # Save comparison summary
    if len(df_comparisons) > 0:
        df_comparisons.to_csv(COMPARISON_SUMMARY_FILE, index=False)
        print(f"\n✓ Comparison summary saved to: {COMPARISON_SUMMARY_FILE}")

    # Generate report
    generate_report(df_results, df_comparisons, recommendations)

    print("\n" + "=" * 80)
    print("✅ COMPARISON ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📂 Output files:")
    print(f"   - {REGRESSION_RESULTS_FILE}")
    print(f"   - {COMPARISON_SUMMARY_FILE}")
    if LEVEL_IMPORTANCE_FILE.exists():
        print(f"   - {LEVEL_IMPORTANCE_FILE}")
    print("\n")

    return df_results, df_comparisons, recommendations


if __name__ == "__main__":
    main()
