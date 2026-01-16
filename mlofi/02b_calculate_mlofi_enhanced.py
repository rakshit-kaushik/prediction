"""
02b_calculate_mlofi_enhanced.py
===============================
ENHANCED MLOFI Calculation with Paper's Full Methodology

ENHANCEMENTS OVER BASIC VERSION:
--------------------------------
1. DEPTH NORMALIZATION - Normalize OFI at each level by depth
2. EXPONENTIAL WEIGHTING - Weight deeper levels less
3. PCA COMPRESSION - Reduce dimensionality of MLOFI vector

METHODOLOGY:
-----------
Following the research methodology from:
https://www.emergentmind.com/topics/order-flow-imbalance-mlofi

OUTPUT COLUMNS:
--------------
- ofi_l1, ofi_l2, ... (raw per-level OFI)
- ofi_norm_l1, ofi_norm_l2, ... (depth-normalized OFI)
- ofi_weighted (exponentially weighted sum)
- ofi_pca_1, ofi_pca_2, ofi_pca_3 (PCA components)

USAGE:
------
    python mlofi/02b_calculate_mlofi_enhanced.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlofi.config_mlofi import (
    MULTILEVEL_PROCESSED_FILE,
    MLOFI_OUTPUT_DIR,
    TICK_SIZE
)

# Enhanced output file
MLOFI_ENHANCED_FILE = MLOFI_OUTPUT_DIR / "mlofi_enhanced.csv"

# Exponential weighting parameter (decay rate)
# Higher lambda = faster decay = less weight on deeper levels
LAMBDA_DECAY = 0.1  # exp(-0.1 * level)

# PCA components to keep
N_PCA_COMPONENTS = 5


def calculate_ofi_for_level(df, level):
    """
    Calculate OFI for a specific level using Cont et al. formula.
    """
    bid_price_col = f'bid_price_l{level}'
    bid_size_col = f'bid_size_l{level}'
    ask_price_col = f'ask_price_l{level}'
    ask_size_col = f'ask_size_l{level}'

    if bid_price_col not in df.columns:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    # Current values
    bid_price = df[bid_price_col]
    bid_size = df[bid_size_col]
    ask_price = df[ask_price_col]
    ask_size = df[ask_size_col]

    # Previous values
    prev_bid_price = bid_price.shift(1)
    prev_bid_size = bid_size.shift(1)
    prev_ask_price = ask_price.shift(1)
    prev_ask_size = ask_size.shift(1)

    # Indicator functions
    bid_up = (bid_price >= prev_bid_price).astype(float).fillna(0)
    bid_down = (bid_price <= prev_bid_price).astype(float).fillna(0)
    ask_up = (ask_price >= prev_ask_price).astype(float).fillna(0)
    ask_down = (ask_price <= prev_ask_price).astype(float).fillna(0)

    # OFI formula
    ofi = (
        bid_up * bid_size.fillna(0) -
        bid_down * prev_bid_size.fillna(0) -
        ask_down * ask_size.fillna(0) +
        ask_up * prev_ask_size.fillna(0)
    )

    # Depth at this level (average of bid and ask size)
    depth = (bid_size.fillna(0) + ask_size.fillna(0)) / 2

    return ofi, depth


def apply_depth_normalization(df, ofi_cols, max_level):
    """
    Normalize OFI at each level by the depth at that level.

    This ensures consistent units across levels and prevents
    deeper levels with larger sizes from dominating.

    Formula: OFI_norm_l = OFI_l / avg_depth_l
    """
    print("\n" + "-" * 80)
    print("APPLYING DEPTH NORMALIZATION")
    print("-" * 80)

    normalized_cols = []

    for level in range(1, max_level + 1):
        ofi_col = f'ofi_l{level}'
        depth_col = f'depth_l{level}'
        norm_col = f'ofi_norm_l{level}'

        if ofi_col in df.columns and depth_col in df.columns:
            # Avoid division by zero - use small epsilon
            depth_safe = df[depth_col].replace(0, np.nan)

            # Normalize: OFI / depth
            df[norm_col] = df[ofi_col] / depth_safe

            # Fill NaN with 0 (where depth was 0)
            df[norm_col] = df[norm_col].fillna(0)

            normalized_cols.append(norm_col)

            if level <= 5 or level == max_level:
                mean_raw = df[ofi_col].mean()
                mean_norm = df[norm_col].mean()
                print(f"   Level {level}: Raw mean={mean_raw:.2f}, Normalized mean={mean_norm:.4f}")

    print(f"\n✓ Created {len(normalized_cols)} normalized OFI columns")
    return normalized_cols


def apply_exponential_weighting(df, ofi_cols, max_level, lambda_decay=LAMBDA_DECAY):
    """
    Apply exponential weighting across levels.

    Deeper levels get less weight since they're further from execution.

    Formula: weight_m = exp(-λ * (m-1))
    Weighted OFI = Σ weight_m × OFI_m
    """
    print("\n" + "-" * 80)
    print("APPLYING EXPONENTIAL WEIGHTING")
    print("-" * 80)

    # Calculate weights
    weights = {}
    print(f"\n   Decay parameter λ = {lambda_decay}")
    print(f"\n   Level weights:")

    for level in range(1, max_level + 1):
        weight = np.exp(-lambda_decay * (level - 1))
        weights[level] = weight
        if level <= 10 or level == max_level:
            print(f"      Level {level}: weight = {weight:.4f}")

    # Apply weights to raw OFI
    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = 0

    for level in range(1, max_level + 1):
        ofi_col = f'ofi_l{level}'
        if ofi_col in df.columns:
            weighted_sum += weights[level] * df[ofi_col].fillna(0)
            total_weight += weights[level]

    # Normalize by total weight
    df['ofi_weighted'] = weighted_sum / total_weight

    # Also create weighted sum of normalized OFI
    weighted_norm_sum = pd.Series(0.0, index=df.index)

    for level in range(1, max_level + 1):
        norm_col = f'ofi_norm_l{level}'
        if norm_col in df.columns:
            weighted_norm_sum += weights[level] * df[norm_col].fillna(0)

    df['ofi_norm_weighted'] = weighted_norm_sum / total_weight

    print(f"\n✓ Created ofi_weighted (raw) and ofi_norm_weighted (normalized)")
    print(f"   ofi_weighted mean: {df['ofi_weighted'].mean():.2f}")
    print(f"   ofi_norm_weighted mean: {df['ofi_norm_weighted'].mean():.4f}")

    return ['ofi_weighted', 'ofi_norm_weighted']


def apply_pca_compression(df, ofi_cols, n_components=N_PCA_COMPONENTS):
    """
    Apply PCA to reduce dimensionality of MLOFI vector.

    This captures the main patterns across all levels while reducing noise.
    """
    print("\n" + "-" * 80)
    print("APPLYING PCA COMPRESSION")
    print("-" * 80)

    # Get OFI matrix (rows = observations, cols = levels)
    ofi_matrix = df[ofi_cols].fillna(0).values

    print(f"\n   Input shape: {ofi_matrix.shape}")
    print(f"   Components to extract: {n_components}")

    # Standardize before PCA
    scaler = StandardScaler()
    ofi_scaled = scaler.fit_transform(ofi_matrix)

    # Apply PCA
    pca = PCA(n_components=n_components)
    ofi_pca = pca.fit_transform(ofi_scaled)

    # Print explained variance
    print(f"\n   Explained variance ratio:")
    total_var = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        total_var += var
        print(f"      PC{i+1}: {var:.4f} ({total_var:.4f} cumulative)")

    # Add PCA components to dataframe
    pca_cols = []
    for i in range(n_components):
        col_name = f'ofi_pca_{i+1}'
        df[col_name] = ofi_pca[:, i]
        pca_cols.append(col_name)

    print(f"\n✓ Created {n_components} PCA components")

    # Also do PCA on normalized OFI
    norm_cols = [f'ofi_norm_l{i}' for i in range(1, len(ofi_cols) + 1) if f'ofi_norm_l{i}' in df.columns]

    if len(norm_cols) >= n_components:
        ofi_norm_matrix = df[norm_cols].fillna(0).values
        ofi_norm_scaled = scaler.fit_transform(ofi_norm_matrix)

        pca_norm = PCA(n_components=n_components)
        ofi_norm_pca = pca_norm.fit_transform(ofi_norm_scaled)

        print(f"\n   Normalized OFI PCA explained variance:")
        total_var = 0
        for i, var in enumerate(pca_norm.explained_variance_ratio_):
            total_var += var
            print(f"      PC{i+1}: {var:.4f} ({total_var:.4f} cumulative)")

        for i in range(n_components):
            col_name = f'ofi_norm_pca_{i+1}'
            df[col_name] = ofi_norm_pca[:, i]
            pca_cols.append(col_name)

    return pca_cols


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 12 + "ENHANCED MLOFI CALCULATION" + " " * 39 + "║")
    print("║" + " " * 8 + "Depth Normalization + Exponential Weighting + PCA" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")

    # Check input file
    if not MULTILEVEL_PROCESSED_FILE.exists():
        print(f"\n❌ Multi-level processed file not found: {MULTILEVEL_PROCESSED_FILE}")
        print("   Please run step 1 first: python mlofi/01_process_multilevel.py")
        sys.exit(1)

    # Load multi-level data
    print(f"\n📂 Loading multi-level data...")
    df = pd.read_csv(MULTILEVEL_PROCESSED_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   Loaded {len(df)} snapshots")

    # Determine max levels
    level_cols = [c for c in df.columns if c.startswith('bid_price_l')]
    max_level = max([int(c.replace('bid_price_l', '')) for c in level_cols])
    print(f"   Max levels available: {max_level}")

    # Calculate OFI and depth at each level
    print("\n" + "=" * 80)
    print("CALCULATING OFI AND DEPTH AT EACH LEVEL")
    print("=" * 80)

    ofi_cols = []
    depth_cols = []

    for level in range(1, max_level + 1):
        ofi, depth = calculate_ofi_for_level(df, level)

        ofi_col = f'ofi_l{level}'
        depth_col = f'depth_l{level}'

        df[ofi_col] = ofi
        df[depth_col] = depth

        ofi_cols.append(ofi_col)
        depth_cols.append(depth_col)

        if level <= 5 or level % 10 == 0 or level == max_level:
            non_zero = (df[ofi_col].abs() > 0).sum()
            mean_depth = df[depth_col].mean()
            print(f"   Level {level}: {non_zero} non-zero OFI, avg depth = {mean_depth:.0f}")

    # Calculate cumulative OFI (for comparison)
    print("\n" + "-" * 80)
    print("CALCULATING CUMULATIVE OFI")
    print("-" * 80)

    for level in [1, 5, 10, 25, max_level]:
        if level <= max_level:
            ofi_cols_to_sum = [f'ofi_l{l}' for l in range(1, level + 1)]
            df[f'ofi_cumulative_l{level}'] = df[ofi_cols_to_sum].sum(axis=1)
            print(f"   Cumulative L1-L{level}: mean = {df[f'ofi_cumulative_l{level}'].mean():.2f}")

    # Apply enhancements
    # 1. Depth Normalization
    norm_cols = apply_depth_normalization(df, ofi_cols, max_level)

    # 2. Exponential Weighting
    weighted_cols = apply_exponential_weighting(df, ofi_cols, max_level)

    # 3. PCA Compression
    pca_cols = apply_pca_compression(df, ofi_cols)

    # Calculate price changes
    print("\n" + "-" * 80)
    print("CALCULATING PRICE CHANGES")
    print("-" * 80)

    df['prev_mid_price'] = df['mid_price'].shift(1)
    df['delta_mid_price'] = df['mid_price'] - df['prev_mid_price']
    df['delta_mid_price_ticks'] = df['delta_mid_price'] / TICK_SIZE
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    # Remove first row
    df = df.iloc[1:].reset_index(drop=True)

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION WITH PRICE CHANGE")
    print("=" * 80)

    print("\n📊 Raw OFI correlation:")
    for level in [1, 5, 10, max_level]:
        if level <= max_level:
            corr = df[f'ofi_l{level}'].corr(df['delta_mid_price_ticks'])
            print(f"   Level {level}: {corr:.4f}")

    print("\n📊 Cumulative OFI correlation:")
    for level in [1, 5, 10, 25, max_level]:
        if level <= max_level and f'ofi_cumulative_l{level}' in df.columns:
            corr = df[f'ofi_cumulative_l{level}'].corr(df['delta_mid_price_ticks'])
            print(f"   Cumulative L1-L{level}: {corr:.4f}")

    print("\n📊 Normalized OFI correlation:")
    for level in [1, 5, 10, max_level]:
        if level <= max_level and f'ofi_norm_l{level}' in df.columns:
            corr = df[f'ofi_norm_l{level}'].corr(df['delta_mid_price_ticks'])
            print(f"   Normalized L{level}: {corr:.4f}")

    print("\n📊 Weighted OFI correlation:")
    if 'ofi_weighted' in df.columns:
        corr = df['ofi_weighted'].corr(df['delta_mid_price_ticks'])
        print(f"   Exponentially weighted (raw): {corr:.4f}")
    if 'ofi_norm_weighted' in df.columns:
        corr = df['ofi_norm_weighted'].corr(df['delta_mid_price_ticks'])
        print(f"   Exponentially weighted (norm): {corr:.4f}")

    print("\n📊 PCA components correlation:")
    for i in range(1, N_PCA_COMPONENTS + 1):
        if f'ofi_pca_{i}' in df.columns:
            corr = df[f'ofi_pca_{i}'].corr(df['delta_mid_price_ticks'])
            print(f"   PC{i}: {corr:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Select columns to save
    base_cols = ['timestamp', 'timestamp_ms', 'mid_price', 'spread', 'time_diff',
                 'delta_mid_price', 'delta_mid_price_ticks']

    # Raw OFI (first 10 levels)
    raw_ofi_cols = [f'ofi_l{i}' for i in range(1, min(11, max_level + 1))]

    # Cumulative
    cumul_cols = [f'ofi_cumulative_l{i}' for i in [1, 5, 10, 25, max_level] if f'ofi_cumulative_l{i}' in df.columns]

    # Normalized (first 10)
    norm_ofi_cols = [f'ofi_norm_l{i}' for i in range(1, min(11, max_level + 1)) if f'ofi_norm_l{i}' in df.columns]

    # Weighted and PCA
    enhanced_cols = weighted_cols + pca_cols

    output_cols = base_cols + raw_ofi_cols + cumul_cols + norm_ofi_cols + enhanced_cols
    output_cols = [c for c in output_cols if c in df.columns]

    df_output = df[output_cols]
    df_output.to_csv(MLOFI_ENHANCED_FILE, index=False)

    print(f"\n✓ Saved to: {MLOFI_ENHANCED_FILE}")
    print(f"   Rows: {len(df_output)}")
    print(f"   Columns: {len(output_cols)}")

    print(f"\n📋 Column groups saved:")
    print(f"   - Base: timestamp, mid_price, delta_mid_price_ticks, etc.")
    print(f"   - Raw OFI: ofi_l1 to ofi_l10")
    print(f"   - Cumulative: ofi_cumulative_l1, l5, l10, l25, l{max_level}")
    print(f"   - Normalized: ofi_norm_l1 to ofi_norm_l10")
    print(f"   - Weighted: ofi_weighted, ofi_norm_weighted")
    print(f"   - PCA: ofi_pca_1 to ofi_pca_{N_PCA_COMPONENTS}, ofi_norm_pca_1 to ofi_norm_pca_{N_PCA_COMPONENTS}")

    print("\n" + "=" * 80)
    print("✅ ENHANCED MLOFI CALCULATION COMPLETE")
    print("=" * 80)
    print(f"\n💡 Next step: python mlofi/03b_regression_enhanced.py")
    print("\n")

    return df_output


if __name__ == "__main__":
    main()
