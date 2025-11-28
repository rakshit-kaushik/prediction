"""
dashboard_depth.py
==================
Depth Analysis Dashboard - Testing beta = c / AD^lambda (Cont et al. 2011)

Tests whether price impact (beta) is inversely proportional to market depth.

FEATURES:
---------
- Level 1 depth analysis (best bid/ask sizes)
- Level 2 depth analysis (total orderbook depth)
- 81 configurations: 9 time windows x 9 outlier methods
- 3 phases per config: Early, Middle, Near Expiry
- Visual validation of beta ~ 1/AD relationship

USAGE:
------
    streamlit run dashboard/dashboard_depth.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
from scipy import stats
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from dashboard_simple for shared functionality
try:
    from scripts.config_analysis import TICK_SIZE, get_dependent_variable_name
except ImportError:
    TICK_SIZE = 0.01
    def get_dependent_variable_name():
        return 'delta_mid_price_ticks'

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

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
# OUTLIER FILTERING FUNCTIONS (copied from dashboard_simple.py)
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

def apply_outlier_method(df, method_idx):
    """Apply outlier method by index (0-8)"""
    if method_idx == 0:
        return df.copy()
    elif method_idx == 1:
        return filter_outliers_iqr(df, 'ofi')
    elif method_idx == 2:
        return filter_outliers_percentile(df, 'ofi')
    elif method_idx == 3:
        return filter_outliers_zscore(df, 'ofi')
    elif method_idx == 4:
        return winsorize_data(df, 'ofi')
    elif method_idx == 5:
        return filter_absolute_threshold(df, 'ofi', -200000, 200000)
    elif method_idx == 6:
        return filter_absolute_threshold(df, 'ofi', -100000, 100000)
    elif method_idx == 7:
        return filter_outliers_mad(df, 'ofi', 3)
    elif method_idx == 8:
        return filter_outliers_percentile_aggressive(df, 'ofi')
    return df.copy()

# ============================================================================
# TIME AGGREGATION
# ============================================================================

def aggregate_ofi_data(df, time_window):
    """Aggregate raw OFI data to specified time window"""
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Convert time window format
    tw_clean = time_window.replace('T', 'min') if isinstance(time_window, str) else time_window
    df['time_bin'] = df['timestamp'].dt.floor(tw_clean)

    agg_dict = {
        'ofi': 'sum',
        'mid_price': ['first', 'last'],
        'best_bid_size': 'mean',
        'best_ask_size': 'mean',
        'total_bid_size': 'mean',
        'total_ask_size': 'mean',
        'total_depth': 'mean',
        'spread': 'mean',
    }

    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten column names
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[0] == 'mid_price':
                new_cols.append('_'.join(col))
            elif col[1]:
                new_cols.append(col[0])
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Calculate price changes
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})
    aggregated = aggregated.drop(columns=['mid_price_first', 'mid_price_last'], errors='ignore')

    return aggregated

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_market_data(market_key):
    """Load market data from CSV"""
    market_config = MARKETS[market_key]
    ofi_file = DATA_DIR / market_config["ofi_file"]

    if ofi_file.exists():
        df = pd.read_csv(ofi_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        return df
    return None

def filter_by_date(df, start_date, end_date):
    """Filter dataframe by date range"""
    if df is None or len(df) == 0:
        return df
    import pytz
    start_date_utc = pytz.utc.localize(start_date)
    end_date_utc = pytz.utc.localize(end_date)
    mask = (df['timestamp'] >= start_date_utc) & (df['timestamp'] <= end_date_utc)
    return df[mask].copy()

# ============================================================================
# PHASE SPLITTING
# ============================================================================

def split_into_phases(df, split_method='snapshot'):
    """Split dataframe into 3 phases"""
    if split_method == 'date':
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        total_duration = max_time - min_time
        one_third = total_duration / 3

        cutoff1 = min_time + one_third
        cutoff2 = min_time + 2 * one_third

        phases = {
            'Early': df[df['timestamp'] < cutoff1],
            'Middle': df[(df['timestamp'] >= cutoff1) & (df['timestamp'] < cutoff2)],
            'Near Expiry': df[df['timestamp'] >= cutoff2]
        }
    else:
        n = len(df)
        phase_size = n // 3
        phases = {
            'Early': df.iloc[:phase_size],
            'Middle': df.iloc[phase_size:2*phase_size],
            'Near Expiry': df.iloc[2*phase_size:]
        }
    return phases

# ============================================================================
# CORE DEPTH ANALYSIS
# ============================================================================

def calculate_beta_and_depth(phase_df):
    """
    Calculate beta (OFI regression coefficient) and average depth for a phase.

    Returns dict with: beta, r_squared, p_value, ad_level1, ad_level2, n_obs
    """
    dep_var = get_dependent_variable_name()
    df_clean = phase_df.dropna(subset=['ofi', dep_var])

    if len(df_clean) < 3:
        return None

    # Calculate beta from regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_clean['ofi'], df_clean[dep_var]
    )

    # Calculate average depths
    # Level 1: best bid/ask sizes
    ad_level1 = (df_clean['best_bid_size'].mean() + df_clean['best_ask_size'].mean()) / 2

    # Level 2: total orderbook depth
    ad_level2 = (df_clean['total_bid_size'].mean() + df_clean['total_ask_size'].mean()) / 2

    return {
        'beta': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'ad_level1': ad_level1,
        'ad_level2': ad_level2,
        'n_obs': len(df_clean)
    }


@st.cache_data
def compute_all_depth_analysis(raw_ofi_df_hash, raw_ofi_df, split_method):
    """
    Compute beta and AD for all 81 configurations x 3 phases.

    Returns DataFrame with all results.
    """
    results = []

    for tw in TIME_WINDOWS:
        time_window_str = f'{tw}min'
        aggregated = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)

        if aggregated is None or len(aggregated) < 50:
            continue

        for method_idx, method_name in enumerate(OUTLIER_METHODS):
            filtered = apply_outlier_method(aggregated, method_idx)

            if len(filtered) < 50:
                continue

            phases = split_into_phases(filtered, split_method)

            for phase_name, phase_df in phases.items():
                if len(phase_df) < 10:
                    continue

                result = calculate_beta_and_depth(phase_df)

                if result is not None:
                    results.append({
                        'time_window': tw,
                        'outlier_method': method_name,
                        'phase': phase_name,
                        **result
                    })

    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_beta_vs_depth(data_points, title, depth_type='level1'):
    """
    Plot 3 points (Early, Middle, Near Expiry) showing beta vs AD relationship.
    """
    fig = go.Figure()

    ad_col = f'ad_{depth_type}'
    colors = {'Early': '#2E86AB', 'Middle': '#A23B72', 'Near Expiry': '#F77F00'}

    # Plot each phase point
    for point in data_points:
        fig.add_trace(go.Scatter(
            x=[point[ad_col]],
            y=[point['beta']],
            mode='markers+text',
            marker=dict(size=14, color=colors.get(point['phase'], 'gray')),
            text=[point['phase'][0]],  # E, M, N
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name=point['phase'],
            showlegend=False,
            hovertemplate=f"<b>{point['phase']}</b><br>AD: %{{x:,.0f}}<br>Beta: %{{y:.2e}}<extra></extra>"
        ))

    # Calculate and add trend line
    x_vals = [p[ad_col] for p in data_points]
    y_vals = [p['beta'] for p in data_points]

    if len(x_vals) == 3 and all(x > 0 for x in x_vals) and all(y != 0 for y in y_vals):
        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
        x_line = [min(x_vals), max(x_vals)]
        y_line = [slope * x + intercept for x in x_line]

        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Annotation: is slope negative? (beta decreasing with AD = theory holds)
        if slope < 0:
            direction = "Beta decreases as AD increases"
            color = 'green'
        else:
            direction = "Beta increases as AD increases"
            color = 'red'

        fig.add_annotation(
            text=direction,
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor='center'
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        xaxis_title="Average Depth (AD)",
        yaxis_title="Beta (Price Impact)",
        height=280,
        margin=dict(l=60, r=20, t=40, b=60),
        showlegend=False
    )

    return fig


def plot_summary_heatmap(results_df, depth_type='level1'):
    """
    Create heatmap showing % of configs where beta decreases with AD.
    """
    # For each time window x outlier method, check if beta decreases as AD increases
    summary = []

    for tw in TIME_WINDOWS:
        for method in OUTLIER_METHODS:
            subset = results_df[
                (results_df['time_window'] == tw) &
                (results_df['outlier_method'] == method)
            ]

            if len(subset) == 3:
                # Sort by AD and check if beta trend is negative
                ad_col = f'ad_{depth_type}'
                sorted_data = subset.sort_values(ad_col)
                betas = sorted_data['beta'].values

                # Simple check: does beta decrease from lowest AD to highest AD?
                decreases = betas[0] > betas[2]  # Early has lower AD typically

                # Also calculate correlation
                correlation = subset[ad_col].corr(subset['beta'])

                summary.append({
                    'TimeWindow': tw,
                    'Method': method,
                    'BetaDecreases': decreases,
                    'Correlation': correlation
                })

    summary_df = pd.DataFrame(summary)

    # Create pivot for heatmap
    pivot = summary_df.pivot(index='TimeWindow', columns='Method', values='Correlation')

    return pivot, summary_df


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Depth Analysis Dashboard",
        layout="wide"
    )

    st.title("Depth Analysis: Beta vs Average Depth")
    st.markdown("**Testing: Beta = c / AD^lambda (Cont et al. 2011)**")
    st.markdown("*Does price impact decrease as market depth increases?*")

    # Sidebar
    with st.sidebar:
        st.header("Market Selection")

        market_options = list(MARKETS.keys())
        default_idx = 0  # NYC Mayor first
        selected_market = st.selectbox(
            "Choose Market",
            options=market_options,
            index=default_idx
        )

        st.info(f"**{MARKETS[selected_market]['description']}**")

        st.divider()

        st.header("Phase Split Method")
        split_method = st.selectbox(
            "How to divide into 3 phases?",
            options=['snapshot', 'date'],
            format_func=lambda x: 'By Observation Count' if x == 'snapshot' else 'By Calendar Time'
        )

        st.divider()

        # Load data
        with st.spinner("Loading data..."):
            raw_ofi_df = load_market_data(selected_market)

        if raw_ofi_df is None:
            st.error("No data found")
            st.stop()

        st.success(f"Loaded {len(raw_ofi_df):,} raw snapshots")

        st.divider()
        st.header("Analysis Info")
        st.metric("Time Windows", len(TIME_WINDOWS))
        st.metric("Outlier Methods", len(OUTLIER_METHODS))
        st.metric("Total Configs", len(TIME_WINDOWS) * len(OUTLIER_METHODS))

    # Compute all depth analysis
    with st.spinner("Computing 243 beta-depth calculations (81 configs x 3 phases)..."):
        # Create a hash of the dataframe for caching
        df_hash = hash(tuple(raw_ofi_df['timestamp'].head(10).astype(str)))
        results_df = compute_all_depth_analysis(df_hash, raw_ofi_df, split_method)

    if results_df is None or len(results_df) == 0:
        st.error("No results computed")
        st.stop()

    st.success(f"Computed {len(results_df)} beta-depth pairs")

    # Create tabs
    tabs = st.tabs(["Level 1 Depth", "Level 2 Depth", "Comparison", "Summary"])

    # Tab 1: Level 1 Depth (best bid/ask)
    with tabs[0]:
        st.subheader("Level 1 Depth Analysis")
        st.markdown("**AD = (best_bid_size + best_ask_size) / 2**")
        st.markdown("Using depth at the best bid/ask prices only")

        # Show 81 charts organized by time window
        for tw in TIME_WINDOWS:
            st.markdown(f"### {tw} Minute Aggregation")

            # Get data for this time window
            tw_data = results_df[results_df['time_window'] == tw]

            if len(tw_data) == 0:
                st.warning(f"No data for {tw} min window")
                continue

            # Create 3x3 grid for 9 outlier methods
            for row_idx in range(3):
                cols = st.columns(3)
                for col_idx in range(3):
                    method_idx = row_idx * 3 + col_idx
                    if method_idx < len(OUTLIER_METHODS):
                        method_name = OUTLIER_METHODS[method_idx]
                        method_data = tw_data[tw_data['outlier_method'] == method_name]

                        with cols[col_idx]:
                            if len(method_data) == 3:
                                data_points = method_data.to_dict('records')
                                fig = plot_beta_vs_depth(data_points, method_name, 'level1')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(f"{method_name}: Insufficient data")

            st.markdown("---")

    # Tab 2: Level 2 Depth (full orderbook)
    with tabs[1]:
        st.subheader("Level 2 Depth Analysis")
        st.markdown("**AD = (total_bid_size + total_ask_size) / 2**")
        st.markdown("Using total depth across all price levels")

        for tw in TIME_WINDOWS:
            st.markdown(f"### {tw} Minute Aggregation")

            tw_data = results_df[results_df['time_window'] == tw]

            if len(tw_data) == 0:
                st.warning(f"No data for {tw} min window")
                continue

            for row_idx in range(3):
                cols = st.columns(3)
                for col_idx in range(3):
                    method_idx = row_idx * 3 + col_idx
                    if method_idx < len(OUTLIER_METHODS):
                        method_name = OUTLIER_METHODS[method_idx]
                        method_data = tw_data[tw_data['outlier_method'] == method_name]

                        with cols[col_idx]:
                            if len(method_data) == 3:
                                data_points = method_data.to_dict('records')
                                fig = plot_beta_vs_depth(data_points, method_name, 'level2')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(f"{method_name}: Insufficient data")

            st.markdown("---")

    # Tab 3: Comparison
    with tabs[2]:
        st.subheader("Level 1 vs Level 2 Comparison")
        st.markdown("Which depth measure better predicts price impact?")

        # Summary statistics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Level 1 (Best Bid/Ask)")
            pivot_l1, summary_l1 = plot_summary_heatmap(results_df, 'level1')
            pct_decreases_l1 = summary_l1['BetaDecreases'].mean() * 100
            st.metric("% Configs Where Beta Decreases with AD", f"{pct_decreases_l1:.1f}%")

            # Show correlation heatmap
            st.markdown("**Beta-AD Correlation by Config**")
            st.dataframe(
                pivot_l1.style.format('{:.3f}').background_gradient(cmap='RdYlGn_r', axis=None, vmin=-1, vmax=1),
                use_container_width=True,
                height=380
            )

        with col2:
            st.markdown("### Level 2 (Full Orderbook)")
            pivot_l2, summary_l2 = plot_summary_heatmap(results_df, 'level2')
            pct_decreases_l2 = summary_l2['BetaDecreases'].mean() * 100
            st.metric("% Configs Where Beta Decreases with AD", f"{pct_decreases_l2:.1f}%")

            st.markdown("**Beta-AD Correlation by Config**")
            st.dataframe(
                pivot_l2.style.format('{:.3f}').background_gradient(cmap='RdYlGn_r', axis=None, vmin=-1, vmax=1),
                use_container_width=True,
                height=380
            )

        # Interpretation
        st.markdown("---")
        st.markdown("### Interpretation")
        st.markdown("""
        - **Negative correlation** (green) = Beta decreases as AD increases = **Theory holds**
        - **Positive correlation** (red) = Beta increases as AD increases = **Theory doesn't hold**

        If Cont et al. (2011) is correct, we expect most cells to be **green (negative)**.
        """)

        # Which is better?
        if pct_decreases_l1 > pct_decreases_l2:
            st.success(f"**Level 1 depth** shows stronger support for the theory ({pct_decreases_l1:.1f}% vs {pct_decreases_l2:.1f}%)")
        elif pct_decreases_l2 > pct_decreases_l1:
            st.success(f"**Level 2 depth** shows stronger support for the theory ({pct_decreases_l2:.1f}% vs {pct_decreases_l1:.1f}%)")
        else:
            st.info("Both depth measures show similar support for the theory")

    # Tab 4: Summary
    with tabs[3]:
        st.subheader("Summary Statistics")

        # Overall findings
        st.markdown("### Key Findings")

        # Average beta by phase
        avg_beta_by_phase = results_df.groupby('phase')['beta'].mean()
        avg_ad_l1_by_phase = results_df.groupby('phase')['ad_level1'].mean()
        avg_ad_l2_by_phase = results_df.groupby('phase')['ad_level2'].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Average Beta by Phase**")
            for phase in ['Early', 'Middle', 'Near Expiry']:
                if phase in avg_beta_by_phase.index:
                    st.metric(phase, f"{avg_beta_by_phase[phase]:.2e}")

        with col2:
            st.markdown("**Avg Level 1 Depth by Phase**")
            for phase in ['Early', 'Middle', 'Near Expiry']:
                if phase in avg_ad_l1_by_phase.index:
                    st.metric(phase, f"{avg_ad_l1_by_phase[phase]:,.0f}")

        with col3:
            st.markdown("**Avg Level 2 Depth by Phase**")
            for phase in ['Early', 'Middle', 'Near Expiry']:
                if phase in avg_ad_l2_by_phase.index:
                    st.metric(phase, f"{avg_ad_l2_by_phase[phase]:,.0f}")

        st.markdown("---")

        # Full results table
        st.markdown("### Full Results Table")
        st.dataframe(
            results_df.style.format({
                'beta': '{:.2e}',
                'r_squared': '{:.4f}',
                'p_value': '{:.2e}',
                'ad_level1': '{:,.0f}',
                'ad_level2': '{:,.0f}',
                'n_obs': '{:,.0f}'
            }),
            use_container_width=True,
            height=500
        )

        # Download
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="depth_analysis_results.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("### Methodology Notes")
        st.markdown("""
        **What we're testing:**
        - Cont et al. (2011) found that price impact (beta) is inversely proportional to market depth
        - Formula: beta = c / AD^lambda, where lambda ~ 1

        **Our approach:**
        - For each of 81 configurations (9 time windows x 9 outlier methods):
            - Split data into 3 phases (Early, Middle, Near Expiry)
            - Calculate beta from OFI regression for each phase
            - Calculate average depth (AD) for each phase
            - Check if beta decreases as AD increases

        **Depth measures:**
        - Level 1: AD = (best_bid_size + best_ask_size) / 2
        - Level 2: AD = (total_bid_size + total_ask_size) / 2
        """)


if __name__ == "__main__":
    main()
