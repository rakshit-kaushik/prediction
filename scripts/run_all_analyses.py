"""
run_all_analyses.py
===================
Master Script - Run All Paper Replication Analyses

Executes all analysis scripts in the correct order:
1. Regression analysis (OFI vs price)
2. Figure 2 (scatter plots)
3. Residual diagnostics
4. Depth analysis
5. Event pattern analysis
6. Trade imbalance comparison

Usage:
    python scripts/run_all_analyses.py
"""

import subprocess
import sys
from pathlib import Path
import time

# Script execution order
SCRIPTS = [
    {
        "name": "Regression Analysis",
        "file": "scripts/01_regression_analysis.py",
        "description": "Linear and quadratic OFI regression models"
    },
    {
        "name": "Figure 2 - Scatter Plots",
        "file": "scripts/02_create_figure_2.py",
        "description": "OFI vs price change scatter plots"
    },
    {
        "name": "Residual Diagnostics",
        "file": "scripts/03_residual_diagnostics.py",
        "description": "Regression assumption validation"
    },
    {
        "name": "Depth Analysis",
        "file": "scripts/04_depth_analysis.py",
        "description": "Price impact vs market depth relationship"
    },
    {
        "name": "Event Pattern Analysis",
        "file": "scripts/05_event_analysis.py",
        "description": "Orderbook event patterns and variance decomposition"
    },
    {
        "name": "Trade Imbalance & Volume",
        "file": "scripts/06_trade_volume_analysis.py",
        "description": "OFI vs TI comparison and volume-price analysis"
    }
]


def run_script(script_info):
    """Run a single analysis script"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_info['name']}")
    print(f"{'='*80}")
    print(f"Description: {script_info['description']}")
    print(f"Script: {script_info['file']}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_info['file']],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✓ SUCCESS - Completed in {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n❌ ERROR - Script failed with return code {result.returncode}")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"\n❌ TIMEOUT - Script exceeded 10 minute limit")
        return False
    except Exception as e:
        print(f"\n❌ EXCEPTION - {str(e)}")
        return False


def main():
    print("\n" + "="*80)
    print("CONT ET AL. (2011) REPLICATION - FULL ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis script will run all paper replication analyses in sequence.")
    print(f"Total scripts to run: {len(SCRIPTS)}\n")

    # Check all script files exist
    print("Checking script files...")
    missing_files = []
    for script in SCRIPTS:
        if not Path(script['file']).exists():
            missing_files.append(script['file'])
            print(f"  ❌ Missing: {script['file']}")
        else:
            print(f"  ✓ Found: {script['file']}")

    if missing_files:
        print(f"\n❌ ERROR: {len(missing_files)} script file(s) missing. Aborting.")
        sys.exit(1)

    print("\n✓ All script files found.\n")

    # Run all scripts
    start_time = time.time()
    results = []

    for idx, script in enumerate(SCRIPTS, 1):
        print(f"\n{'#'*80}")
        print(f"# STEP {idx}/{len(SCRIPTS)}")
        print(f"{'#'*80}")

        success = run_script(script)
        results.append({
            'script': script['name'],
            'file': script['file'],
            'success': success
        })

        if not success:
            print(f"\n⚠ Warning: {script['name']} failed, but continuing with remaining scripts...")

    # Summary
    total_time = time.time() - start_time

    print(f"\n\n{'='*80}")
    print("ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*80}\n")

    print("Summary of Results:")
    print(f"{'Script':<40} {'Status':<10}")
    print(f"{'-'*50}")

    success_count = 0
    for result in results:
        status = "✓ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['script']:<40} {status:<10}")
        if result['success']:
            success_count += 1

    print(f"\n{'-'*50}")
    print(f"Success Rate: {success_count}/{len(SCRIPTS)} ({100*success_count/len(SCRIPTS):.0f}%)")
    print(f"Total Time: {total_time/60:.1f} minutes")

    print(f"\n{'='*80}")
    print("OUTPUTS GENERATED")
    print(f"{'='*80}\n")

    print("Tables (results/tables/):")
    print("  - table_2_regression_statistics.csv")
    print("  - table_3_depth_analysis.csv")
    print("  - table_4_variance_decomposition.csv")
    print("  - table_5_ofi_ti_comparison.csv")

    print("\nFigures (results/figures/):")
    print("  - figure_2_*_ofi_vs_price.{png,pdf} (scatter plots)")
    print("  - figure_2_combined_comparison.{png,pdf}")
    print("  - figure_3_*_residual_diagnostics.{png,pdf}")
    print("  - figure_4_depth_analysis.{png,pdf}")
    print("  - figure_5_event_analysis.{png,pdf}")
    print("  - figure_6_ofi_ti_comparison.{png,pdf}")

    print("\nAnalysis Files (results/analysis/):")
    print("  - *_regression_detailed.csv")
    print("  - *_rolling_regression.csv")
    print("  - *_event_statistics.csv")
    print("  - *_ti_comparison.csv")
    print("  - residual_diagnostics_summary.csv")

    if success_count == len(SCRIPTS):
        print(f"\n{'='*80}")
        print("✓ ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*80}")
        print(f"⚠ {len(SCRIPTS) - success_count} ANALYSIS/ES FAILED")
        print(f"{'='*80}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
