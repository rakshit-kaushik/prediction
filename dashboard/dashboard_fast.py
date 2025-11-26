"""
dashboard_fast.py
=================
Lightweight OFI Analysis Dashboard - Tables Only (No Graphs)

Fast loading version that shows only summary tables without scatter plots.
Computes all 243 regressions and displays R² heatmaps.

USAGE:
------
    streamlit run dashboard_fast.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import sys
from scipy import stats

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import config, with fallback defaults for Streamlit Cloud
try:
    from scripts.config_analysis import (
        TICK_SIZE,
        USE_TICK_NORMALIZED,
        get_dependent_variable_name,
    )
except ImportError:
    TICK_SIZE = 0.01
    USE_TICK_NORMALIZED = True

    def get_dependent_variable_name():
        return 'delta_mid_price_ticks' if USE_TICK_NORMALIZED else 'delta_mid_price'

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Time windows to analyze (in minutes)
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]

# Pre-configured markets
MARKETS = {
    "NYC Mayoral Election 2025 (Zohran Mamdani)": {
        "ofi_file": "nyc_mayor_oct15_nov04_ofi.csv",
        "description": "Will Zohran Mamdani win the 2025 NYC mayoral election?",
    },
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "ofi_oct15_nov20_combined.csv",
        "description": "Fed decreases interest rates by 25 bps after December 2025 meeting?",
    }
}

# ============================================================================
# OUTLIER FILTERING FUNCTIONS
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

def get_outlier_methods(df):
    return {
        'Raw (No Filter)': df.copy(),
        'IQR (1.5x)': filter_outliers_iqr(df, 'ofi'),
        'Pctl (1%-99%)': filter_outliers_percentile(df, 'ofi'),
        'Z-Score (3)': filter_outliers_zscore(df, 'ofi'),
        'Winsorized (1%)': winsorize_data(df, 'ofi'),
        'Abs (+-200k)': filter_absolute_threshold(df, 'ofi', -200000, 200000),
        'Abs (+-100k)': filter_absolute_threshold(df, 'ofi', -100000, 100000),
        'MAD (3s)': filter_outliers_mad(df, 'ofi', 3),
        'Pctl (5%-95%)': filter_outliers_percentile_aggressive(df, 'ofi'),
    }

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def aggregate_ofi_data(df, time_window='10min'):
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    else:
        raise ValueError("No timestamp column found")

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Convert time window format
    tw_clean = time_window.replace('T', 'min') if isinstance(time_window, str) else time_window
    df['time_bin'] = df['timestamp'].dt.floor(tw_clean)

    # Aggregate
    agg_dict = {'ofi': 'sum', 'mid_price': ['first', 'last']}
    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten columns
    aggregated.columns = ['timestamp', 'ofi', 'mid_price_first', 'mid_price_last']
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    return aggregated


@st.cache_data
def load_market_data(market_key):
    market_config = MARKETS[market_key]
    ofi_file = DATA_DIR / market_config["ofi_file"]
    if ofi_file.exists():
        ofi_df = pd.read_csv(ofi_file)
        ofi_df['timestamp'] = pd.to_datetime(ofi_df['timestamp'], format='mixed')
        return ofi_df
    return None


@st.cache_data
def compute_all_regressions(market_key):
    """Compute all 243 regressions and return summary dataframe"""
    raw_ofi_df = load_market_data(market_key)
    if raw_ofi_df is None:
        return None

    dep_var = get_dependent_variable_name()
    all_results = []

    for tw in TIME_WINDOWS:
        time_window_str = f'{tw}min'
        aggregated_df = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)

        if aggregated_df is None or len(aggregated_df) == 0:
            continue

        outlier_methods = get_outlier_methods(aggregated_df)

        for method_name, method_df in outlier_methods.items():
            df_clean = method_df.dropna(subset=['ofi', dep_var]).copy()
            if len(df_clean) < 150:
                continue

            n = len(df_clean)
            phase_size = n // 3
            phases = {
                'P1 (Early)': df_clean.iloc[:phase_size],
                'P2 (Middle)': df_clean.iloc[phase_size:2*phase_size],
                'P3 (Near Expiry)': df_clean.iloc[2*phase_size:]
            }

            for phase_name, phase_df in phases.items():
                if len(phase_df) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        phase_df['ofi'], phase_df[dep_var]
                    )
                    all_results.append({
                        'TimeWindow': tw,
                        'Method': method_name,
                        'Phase': phase_name,
                        'N': len(phase_df),
                        'Beta': slope,
                        'R2': r_value**2,
                        'p-value': p_value
                    })

    return pd.DataFrame(all_results) if all_results else None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="OFI Analysis (Fast)",
        layout="wide"
    )

    st.title("OFI Multi-Time Window Analysis (Fast)")
    st.markdown("**Tables only - No graphs for faster loading**")
    st.caption("9 time windows x 9 outlier methods x 3 phases = 243 regressions")

    # Sidebar
    with st.sidebar:
        st.header("Market Selection")
        market_options = list(MARKETS.keys())
        selected_market = st.selectbox(
            "Choose Market",
            options=market_options,
            index=0
        )
        st.info(f"**{MARKETS[selected_market]['description']}**")

    # Compute regressions
    with st.spinner("Computing 243 regressions..."):
        master_df = compute_all_regressions(selected_market)

    if master_df is None or len(master_df) == 0:
        st.error("No data available")
        st.stop()

    st.success(f"Computed {len(master_df)} regression results")

    # Table 1: Average R² across all phases
    st.markdown("## 1. Average R² Across All Phases")
    avg_r2 = master_df.groupby(['TimeWindow', 'Method'])['R2'].mean().reset_index()
    pivot_avg = avg_r2.pivot(index='TimeWindow', columns='Method', values='R2')
    st.dataframe(
        pivot_avg.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
        use_container_width=True,
        height=380
    )
    best_avg = avg_r2.loc[avg_r2['R2'].idxmax()]
    st.success(f"**Best Overall**: {best_avg['TimeWindow']} min + {best_avg['Method']} (Avg R² = {best_avg['R2']:.4f})")

    # Table 2: Phase 1 (Early)
    st.markdown("## 2. Phase 1 (Early Market) R²")
    p1_data = master_df[master_df['Phase'] == 'P1 (Early)']
    if len(p1_data) > 0:
        pivot_p1 = p1_data.pivot(index='TimeWindow', columns='Method', values='R2')
        st.dataframe(
            pivot_p1.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
            use_container_width=True,
            height=380
        )
        best_p1 = p1_data.loc[p1_data['R2'].idxmax()]
        st.info(f"**Best Phase 1**: {best_p1['TimeWindow']} min + {best_p1['Method']} (R² = {best_p1['R2']:.4f})")

    # Table 3: Phase 2 (Middle)
    st.markdown("## 3. Phase 2 (Middle Market) R²")
    p2_data = master_df[master_df['Phase'] == 'P2 (Middle)']
    if len(p2_data) > 0:
        pivot_p2 = p2_data.pivot(index='TimeWindow', columns='Method', values='R2')
        st.dataframe(
            pivot_p2.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
            use_container_width=True,
            height=380
        )
        best_p2 = p2_data.loc[p2_data['R2'].idxmax()]
        st.info(f"**Best Phase 2**: {best_p2['TimeWindow']} min + {best_p2['Method']} (R² = {best_p2['R2']:.4f})")

    # Table 4: Phase 3 (Near Expiry)
    st.markdown("## 4. Phase 3 (Near Expiry) R²")
    p3_data = master_df[master_df['Phase'] == 'P3 (Near Expiry)']
    if len(p3_data) > 0:
        pivot_p3 = p3_data.pivot(index='TimeWindow', columns='Method', values='R2')
        st.dataframe(
            pivot_p3.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
            use_container_width=True,
            height=380
        )
        best_p3 = p3_data.loc[p3_data['R2'].idxmax()]
        st.info(f"**Best Phase 3**: {best_p3['TimeWindow']} min + {best_p3['Method']} (R² = {best_p3['R2']:.4f})")

    # Download
    st.markdown("---")
    st.markdown("## Download Results")
    csv = master_df.to_csv(index=False)
    st.download_button(
        label="Download Full CSV",
        data=csv,
        file_name=f"ofi_analysis_{selected_market.split()[0].lower()}.csv",
        mime="text/csv"
    )

    # Summary stats
    st.markdown("---")
    st.markdown("## Quick Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Regressions", len(master_df))
    with col2:
        st.metric("Avg R² (all)", f"{master_df['R2'].mean():.4f}")
    with col3:
        st.metric("Max R²", f"{master_df['R2'].max():.4f}")


if __name__ == "__main__":
    main()
