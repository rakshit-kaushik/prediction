"""
Generate OFI vs TI R² Heatmap for Research Poster
Output: poster/ofi_ti_heatmap.png (300 DPI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).parent.parent / "data" / "ti_vs_ofi_comparison.csv"
OUTPUT_FILE = Path(__file__).parent / "ofi_ti_heatmap.png"

# Short names for outlier methods (for poster readability)
METHOD_SHORT = {
    'Raw': 'Raw',
    'IQR (1.5x)': 'IQR',
    'Pctl (1%-99%)': 'P1-99',
    'Z-Score (3)': 'Z-Score',
    'Winsorized': 'Winsor',
    'Abs (200k)': '|200k|',
    'Abs (100k)': '|100k|',
    'MAD (3)': 'MAD',
    'Pctl (5%-95%)': 'P5-95'
}

# Order for methods (same as dashboard)
METHOD_ORDER = ['Raw', 'IQR (1.5x)', 'Pctl (1%-99%)', 'Z-Score (3)', 'Winsorized',
                'Abs (200k)', 'Abs (100k)', 'MAD (3)', 'Pctl (5%-95%)']

TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]


def main():
    print("Generating OFI vs TI Heatmap for Poster...")

    # Load data
    df = pd.read_csv(DATA_FILE)

    # Create pivot tables
    ofi_pivot = df.pivot(index='time_window', columns='outlier_method', values='ofi_r2')
    ti_pivot = df.pivot(index='time_window', columns='outlier_method', values='ti_r2')

    # Reorder columns and rows
    ofi_pivot = ofi_pivot.reindex(index=TIME_WINDOWS, columns=METHOD_ORDER)
    ti_pivot = ti_pivot.reindex(index=TIME_WINDOWS, columns=METHOD_ORDER)

    # Convert to percentages
    ofi_pivot = ofi_pivot * 100
    ti_pivot = ti_pivot * 100

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Common settings
    vmin_ofi, vmax_ofi = 0, 40  # OFI ranges 5-35%
    vmin_ti, vmax_ti = 0, 1    # TI ranges 0-0.8%

    # OFI Heatmap (left)
    im1 = ax1.imshow(ofi_pivot.values, cmap='YlGnBu', aspect='auto', vmin=vmin_ofi, vmax=vmax_ofi)
    ax1.set_title('OFI R² (%)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Outlier Method', fontsize=11)
    ax1.set_ylabel('Time Window (min)', fontsize=11)

    # Set tick labels
    ax1.set_xticks(range(len(METHOD_ORDER)))
    ax1.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER], rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(TIME_WINDOWS)))
    ax1.set_yticklabels(TIME_WINDOWS, fontsize=10)

    # Add value annotations
    for i in range(len(TIME_WINDOWS)):
        for j in range(len(METHOD_ORDER)):
            val = ofi_pivot.values[i, j]
            color = 'white' if val > 25 else 'black'
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=color)

    # Colorbar for OFI
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('R² (%)', fontsize=10)

    # TI Heatmap (right)
    im2 = ax2.imshow(ti_pivot.values, cmap='YlOrRd', aspect='auto', vmin=vmin_ti, vmax=vmax_ti)
    ax2.set_title('TI R² (%)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Outlier Method', fontsize=11)
    ax2.set_ylabel('Time Window (min)', fontsize=11)

    # Set tick labels
    ax2.set_xticks(range(len(METHOD_ORDER)))
    ax2.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER], rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(TIME_WINDOWS)))
    ax2.set_yticklabels(TIME_WINDOWS, fontsize=10)

    # Add value annotations
    for i in range(len(TIME_WINDOWS)):
        for j in range(len(METHOD_ORDER)):
            val = ti_pivot.values[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    # Colorbar for TI
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('R² (%)', fontsize=10)

    # Add summary annotation
    ofi_max = ofi_pivot.values.max()
    ti_max = ti_pivot.values.max()
    fig.text(0.5, 0.02, f'OFI Best: {ofi_max:.1f}%  |  TI Best: {ti_max:.2f}%  |  OFI wins 81/81 configs',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  OFI max R²: {ofi_max:.1f}%")
    print(f"  TI max R²: {ti_max:.2f}%")

    plt.close()


if __name__ == "__main__":
    main()
