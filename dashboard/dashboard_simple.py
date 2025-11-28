"""
dashboard_simple.py
===================
Simplified OFI Analysis Dashboard - Explore Pre-Existing Data

FEATURES:
---------
- Market dropdown selection (pre-configured markets)
- Date/time range sliders for filtering
- Live interactive visualizations
- No API calls - uses existing CSV data
- Fast and reliable data exploration

USAGE:
------
    streamlit run dashboard_simple.py

Then open browser to http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import json
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import config, with fallback defaults for Streamlit Cloud
try:
    from scripts.config_analysis import (
        TIME_WINDOW,
        TICK_SIZE,
        USE_TICK_NORMALIZED,
        get_dependent_variable_name,
        get_dependent_variable_label
    )
except ImportError:
    # Fallback defaults if config import fails (e.g., on Streamlit Cloud)
    TIME_WINDOW = '10min'
    TICK_SIZE = 0.01
    USE_TICK_NORMALIZED = True

    def get_dependent_variable_name():
        return 'delta_mid_price_ticks' if USE_TICK_NORMALIZED else 'delta_mid_price'

    def get_dependent_variable_label():
        return 'Price Change (ticks)' if USE_TICK_NORMALIZED else 'Price Change ($)'

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Pre-configured markets
MARKETS = {
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "ofi_oct15_nov20_combined.csv",
        "orderbook_file": "orderbook_oct15_nov20_combined.csv",
        "market_info_file": "market_info.json",
        "description": "Fed decreases interest rates by 25 bps after December 2025 meeting?",
        "date_range": ("2025-10-15", "2025-11-20")
    },
    "NYC Mayoral Election 2025 (Zohran Mamdani)": {
        "ofi_file": "nyc_mayor_oct15_nov04_ofi.csv",
        "orderbook_file": "nyc_mayor_oct15_nov04_processed.csv",
        "market_info_file": "nyc_mayor_market_info.json",
        "trades_file": "nyc_mayor_oct15_nov04_trades_processed.csv",
        "description": "Will Zohran Mamdani win the 2025 NYC mayoral election?",
        "date_range": ("2025-10-15", "2025-11-04")
    }
}

# ============================================================================
# OUTLIER FILTERING FUNCTIONS
# ============================================================================

def filter_outliers_iqr(df, column='ofi', multiplier=1.5):
    """Remove outliers using IQR method"""
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_percentile(df, column='ofi', lower_pct=0.01, upper_pct=0.99):
    """Remove outliers using percentile trimming"""
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_zscore(df, column='ofi', threshold=3):
    """Remove outliers using Z-score method"""
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return df
    z_scores = (df[column] - mean) / std
    return df[z_scores.abs() <= threshold]

def winsorize_data(df, column='ofi', limits=(0.01, 0.01)):
    """Cap extreme values at percentile limits (doesn't remove, just caps)"""
    df = df.copy()
    lower = df[column].quantile(limits[0])
    upper = df[column].quantile(1 - limits[1])
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df

def filter_absolute_threshold(df, column='ofi', lower=-200000, upper=200000):
    """Remove outliers using absolute threshold - keep only values within range"""
    df = df.copy()
    return df[(df[column] >= lower) & (df[column] <= upper)]

def filter_outliers_mad(df, column='ofi', threshold=3):
    """Remove outliers using Median Absolute Deviation (more robust than Z-score)"""
    df = df.copy()
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    if mad == 0:
        return df
    modified_z = 0.6745 * (df[column] - median) / mad
    return df[np.abs(modified_z) <= threshold]

def filter_outliers_percentile_aggressive(df, column='ofi', lower_pct=0.05, upper_pct=0.95):
    """Remove outliers using aggressive percentile trimming (5% each tail)"""
    df = df.copy()
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]

# ============================================================================
# TIME AGGREGATION (Per Cont et al. 2011)
# ============================================================================

def aggregate_ofi_data(df, time_window=None):
    """
    Aggregate raw orderbook snapshots using configurable time window following
    Cont et al. (2011) methodology.

    This matches the aggregation used in all analysis scripts.

    Args:
        df: DataFrame with raw orderbook data
        time_window: Time window string (e.g., '10T' for 10 minutes).
                     If None, uses TIME_WINDOW from config.
    """
    if df is None or len(df) == 0:
        return df

    # Use provided time_window or fall back to config
    tw = time_window if time_window is not None else TIME_WINDOW

    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    else:
        raise ValueError("No timestamp column found")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create time bins using provided time window
    # Convert 'T' format to 'min' format (pandas deprecation)
    tw_clean = tw.replace('T', 'min') if isinstance(tw, str) else tw
    df['time_bin'] = df['timestamp'].dt.floor(tw_clean)

    # Define aggregation rules
    agg_dict = {
        'ofi': 'sum',  # Sum OFI over window
        'mid_price': ['first', 'last'],
        'total_depth': 'mean',
        'spread': 'mean',
        'spread_pct': 'mean',
        'best_bid_price': 'last',
        'best_ask_price': 'last',
        'best_bid_size': 'last',
        'best_ask_size': 'last',
        'total_bid_size': 'mean',
        'total_ask_size': 'mean'
    }

    # Add event columns if they exist
    for col in ['bid_up', 'bid_down', 'ask_up', 'ask_down']:
        if col in df.columns:
            agg_dict[col] = 'max'

    # Aggregate
    aggregated = df.groupby('time_bin').agg(agg_dict).reset_index()

    # Flatten multi-level column names
    new_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[0] in ['mid_price']:
                new_cols.append('_'.join(col))
            elif col[1]:
                new_cols.append(col[0])
            else:
                new_cols.append(col[0])
        else:
            new_cols.append(col)
    aggregated.columns = new_cols

    # Calculate delta_mid_price as BETWEEN-window change (per Cont et al. 2011)
    # ΔPk = Pk - Pk-1 (change from previous window's end to current window's end)
    aggregated['mid_price'] = aggregated['mid_price_last']
    aggregated['delta_mid_price'] = aggregated['mid_price'].diff()

    # Calculate tick-normalized price change
    aggregated['delta_mid_price_ticks'] = aggregated['delta_mid_price'] / TICK_SIZE

    # Rename timestamp
    aggregated = aggregated.rename(columns={'time_bin': 'timestamp'})

    # Drop temporary columns
    aggregated = aggregated.drop(columns=['mid_price_first', 'mid_price_last'], errors='ignore')

    return aggregated


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_market_data(market_key):
    """
    Load pre-existing market data from CSV files

    Args:
        market_key: Key from MARKETS dict

    Returns:
        tuple: (ofi_df, orderbook_df, market_info)
    """
    market_config = MARKETS[market_key]

    # Load OFI results
    ofi_file = DATA_DIR / market_config["ofi_file"]
    if ofi_file.exists():
        ofi_df = pd.read_csv(ofi_file)
        ofi_df['timestamp'] = pd.to_datetime(ofi_df['timestamp'], format='mixed')
    else:
        ofi_df = None

    # Load orderbook data
    orderbook_file = DATA_DIR / market_config["orderbook_file"]
    if orderbook_file.exists():
        orderbook_df = pd.read_csv(orderbook_file)
        orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'], format='mixed')
    else:
        orderbook_df = None

    # Load market info
    market_info_file = DATA_DIR / market_config["market_info_file"]
    if market_info_file.exists():
        with open(market_info_file, 'r') as f:
            market_info = json.load(f)
    else:
        market_info = {}

    return ofi_df, orderbook_df, market_info


def filter_by_date(df, start_date, end_date):
    """
    Filter dataframe by date range

    Args:
        df: DataFrame with timestamp column
        start_date: Start datetime (naive)
        end_date: End datetime (naive)

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df is None or len(df) == 0:
        return df

    # Make datetimes timezone-aware (UTC) to match dataframe timestamps
    import pytz
    start_date_utc = pytz.utc.localize(start_date)
    end_date_utc = pytz.utc.localize(end_date)

    mask = (df['timestamp'] >= start_date_utc) & (df['timestamp'] <= end_date_utc)
    return df[mask].copy()


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_price_evolution(df):
    """Plot mid-price evolution over time"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:.4f}<extra></extra>'
    ))

    # Add best bid/ask as shaded area
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['best_ask_price'],
        mode='lines',
        name='Best Ask',
        line=dict(color='#A23B72', width=1, dash='dot'),
        hovertemplate='<b>%{x}</b><br>Ask: $%{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['best_bid_price'],
        mode='lines',
        name='Best Bid',
        line=dict(color='#06A77D', width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(46, 134, 171, 0.1)',
        hovertemplate='<b>%{x}</b><br>Bid: $%{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Market Price Evolution",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    return fig


def plot_spread_analysis(df):
    """Plot bid-ask spread over time"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Spread (Absolute)', 'Spread (%)'),
        vertical_spacing=0.15
    )

    # Absolute spread
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['spread'],
            mode='lines',
            name='Spread ($)',
            line=dict(color='#E63946', width=1),
            hovertemplate='<b>%{x}</b><br>Spread: $%{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Percentage spread
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['spread_pct'],
            mode='lines',
            name='Spread (%)',
            line=dict(color='#F77F00', width=1),
            hovertemplate='<b>%{x}</b><br>Spread: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Spread ($)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=2, col=1)

    fig.update_layout(
        title="Bid-Ask Spread Analysis",
        hovermode='x unified',
        height=500,
        showlegend=False
    )

    return fig


def plot_depth_evolution(df):
    """Plot order book depth over time"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_bid_size'],
        mode='lines',
        name='Bid Depth',
        line=dict(color='#06A77D', width=2),
        stackgroup='one',
        hovertemplate='<b>%{x}</b><br>Bid Depth: %{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['total_ask_size'],
        mode='lines',
        name='Ask Depth',
        line=dict(color='#A23B72', width=2),
        stackgroup='two',
        hovertemplate='<b>%{x}</b><br>Ask Depth: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title="Order Book Depth Over Time",
        xaxis_title="Time",
        yaxis_title="Depth (contracts)",
        hovermode='x unified',
        height=400
    )

    return fig


def split_into_phases(df, split_method='snapshot'):
    """
    Split dataframe into 3 phases based on chosen method.

    Args:
        df: DataFrame with timestamp column
        split_method: 'snapshot' (by observation count) or 'date' (by calendar time)

    Returns:
        dict of phase_name -> DataFrame
    """
    if split_method == 'date':
        # Split by calendar time
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        total_duration = max_time - min_time
        one_third = total_duration / 3

        cutoff1 = min_time + one_third
        cutoff2 = min_time + 2 * one_third

        phases = {
            'Phase 1 (Early)': df[df['timestamp'] < cutoff1],
            'Phase 2 (Middle)': df[(df['timestamp'] >= cutoff1) & (df['timestamp'] < cutoff2)],
            'Phase 3 (Near Expiry)': df[df['timestamp'] >= cutoff2]
        }
    else:
        # Split by observation count (default)
        n = len(df)
        phase_size = n // 3
        phases = {
            'Phase 1 (Early)': df.iloc[:phase_size],
            'Phase 2 (Middle)': df.iloc[phase_size:2*phase_size],
            'Phase 3 (Near Expiry)': df.iloc[2*phase_size:]
        }

    return phases


def plot_ofi_three_phases(df, split_method='snapshot'):
    """Plot OFI vs price change for 3 phases of market"""
    # Get dependent variable from config
    dep_var = get_dependent_variable_name()
    dep_var_label = get_dependent_variable_label()

    # Remove NaN values
    df_clean = df.dropna(subset=['ofi', dep_var]).copy()

    if len(df_clean) < 150:  # Need at least 50 per phase
        return None, None

    # Divide into 3 phases using chosen method
    phases = split_into_phases(df_clean, split_method)

    # Create 3 subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(phases.keys()),
        horizontal_spacing=0.08
    )

    colors = ['#2E86AB', '#A23B72', '#F77F00']
    regression_results = []

    from scipy import stats

    for idx, (phase_name, phase_df) in enumerate(phases.items(), 1):
        # Filter non-zero for visualization
        phase_viz = phase_df[(phase_df['ofi'] != 0) | (phase_df[dep_var] != 0)].copy()

        if len(phase_viz) == 0:
            continue

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=phase_viz['ofi'],
                y=phase_viz[dep_var],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[idx-1],
                    opacity=0.5
                ),
                name=phase_name,
                showlegend=False,
                hovertemplate=f'<b>OFI:</b> %{{x:.2f}}<br><b>{dep_var_label}:</b> %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=idx
        )

        # Regression
        if len(phase_df) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                phase_df['ofi'], phase_df[dep_var]
            )

            x_range = np.array([phase_viz['ofi'].min(), phase_viz['ofi'].max()])
            y_pred = slope * x_range + intercept

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hovertemplate='<b>Regression Line</b><extra></extra>'
                ),
                row=1, col=idx
            )

            # Add annotation with BLACK background
            # Calculate x position based on subplot index (paper coordinates)
            x_pos = (idx - 1) / 3 + 0.17  # Center of each subplot
            fig.add_annotation(
                x=x_pos, y=0.98,
                xref='paper', yref='paper',
                text=f'β = {slope:.2e}<br>R² = {r_value**2:.3f}<br>p = {p_value:.2e}',
                showarrow=False,
                bgcolor='black',
                font=dict(color='white', size=10),
                bordercolor='white',
                borderwidth=1,
                xanchor='center'
            )

            regression_results.append({
                'Phase': phase_name,
                'N': len(phase_df),
                'Beta': slope,
                'R²': r_value**2,
                'p-value': p_value
            })

    # Update axes
    for idx in range(1, 4):
        fig.update_xaxes(title_text="OFI", row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title_text=dep_var_label, row=1, col=idx)

    fig.update_layout(
        title="OFI Price Impact Across Market Phases",
        height=400,
        showlegend=False
    )

    return fig, pd.DataFrame(regression_results)


def plot_ofi_distribution(df):
    """Plot OFI distribution histogram"""
    # Filter non-zero OFI
    ofi_nonzero = df[df['ofi'] != 0]['ofi']

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=ofi_nonzero,
        nbinsx=50,
        marker=dict(
            color=ofi_nonzero,
            colorscale='RdYlGn',
            showscale=False
        ),
        hovertemplate='<b>OFI Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title="OFI Distribution",
        xaxis_title="OFI",
        yaxis_title="Frequency",
        height=400
    )

    return fig


# ============================================================================
# TIME WINDOW ANALYSIS FUNCTIONS
# ============================================================================

# Time windows to analyze (in minutes)
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]


def filter_last_day(df):
    """Filter dataframe to only include the last 24 hours of data"""
    if df is None or len(df) == 0:
        return df
    max_time = df['timestamp'].max()
    cutoff = max_time - pd.Timedelta(days=1)
    return df[df['timestamp'] >= cutoff].copy()


def plot_single_regression(df, title):
    """Create a single OFI vs price change scatter with regression line"""
    from scipy import stats

    dep_var = get_dependent_variable_name()
    dep_var_label = get_dependent_variable_label()
    df_clean = df.dropna(subset=['ofi', dep_var])

    fig = go.Figure()

    if len(df_clean) == 0:
        fig.update_layout(
            title=dict(text=f"{title} (No data)", font=dict(size=11)),
            height=250,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        return fig, None

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df_clean['ofi'],
        y=df_clean[dep_var],
        mode='markers',
        marker=dict(size=4, opacity=0.5, color='#2E86AB'),
        hovertemplate=f'<b>OFI:</b> %{{x:.2f}}<br><b>{dep_var_label}:</b> %{{y:.3f}}<extra></extra>'
    ))

    result = None
    # Regression line
    if len(df_clean) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_clean['ofi'], df_clean[dep_var]
        )
        x_range = [df_clean['ofi'].min(), df_clean['ofi'].max()]
        y_pred = [slope * x + intercept for x in x_range]

        fig.add_trace(go.Scatter(
            x=x_range, y=y_pred,
            mode='lines',
            line=dict(color='red', width=2)
        ))

        # Add stats annotation
        fig.add_annotation(
            text=f"R²={r_value**2:.3f}<br>β={slope:.2e}",
            xref="paper", yref="paper",
            x=0.95, y=0.95,
            showarrow=False,
            bgcolor='black',
            font=dict(color='white', size=9),
            xanchor='right',
            yanchor='top'
        )

        result = {
            'N': len(df_clean),
            'Beta': slope,
            'R2': r_value**2,
            'p-value': p_value
        }

    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        xaxis_title="OFI",
        yaxis_title=dep_var_label
    )

    return fig, result


def render_last_day_analysis(raw_ofi_df):
    """
    Render 81 scatter plots for last day analysis.
    Layout: 9 time windows, each showing 9 outlier methods in 3x3 grid.
    """
    from scipy import stats

    st.subheader("Last Day Analysis (Final 24 Hours Before Expiry)")
    st.markdown("Analyzing OFI-price relationships in the final day when trading intensifies")

    # Filter to last day
    last_day_df = filter_last_day(raw_ofi_df)

    if last_day_df is None or len(last_day_df) == 0:
        st.warning("No data available for last day analysis")
        return

    st.info(f"Last day data: {len(last_day_df):,} raw snapshots")

    # Collect all results for summary heatmap
    all_results = []
    dep_var = get_dependent_variable_name()

    # First, compute all regressions for summary
    with st.spinner("Computing 81 regressions for last day..."):
        for tw in TIME_WINDOWS:
            time_window_str = f'{tw}min'
            aggregated = aggregate_ofi_data(last_day_df.copy(), time_window_str)

            if aggregated is None or len(aggregated) == 0:
                continue

            outlier_methods = get_outlier_methods(aggregated)

            for method_name, method_df in outlier_methods.items():
                df_clean = method_df.dropna(subset=['ofi', dep_var])
                if len(df_clean) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df_clean['ofi'], df_clean[dep_var]
                    )
                    short_name = method_name.replace('Filtered', '').replace('Trimmed', '').replace('Data', '').strip()
                    all_results.append({
                        'TimeWindow': tw,
                        'Method': short_name,
                        'N': len(df_clean),
                        'Beta': slope,
                        'R2': r_value**2,
                        'p-value': p_value
                    })

    # Show summary heatmap at top
    if all_results:
        st.markdown("### Summary: R² Heatmap (Last Day)")
        results_df = pd.DataFrame(all_results)
        pivot_r2 = results_df.pivot(index='TimeWindow', columns='Method', values='R2')
        st.dataframe(
            pivot_r2.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
            use_container_width=True,
            height=380
        )

        # Find best combination
        best_idx = results_df['R2'].idxmax()
        best_row = results_df.loc[best_idx]
        st.success(f"**Best Last Day**: {best_row['TimeWindow']} min + {best_row['Method']} (R² = {best_row['R2']:.4f})")

    st.markdown("---")
    st.markdown("### Detailed Scatter Plots (81 Total)")

    # Now show all 81 scatter plots organized by time window
    for tw in TIME_WINDOWS:
        st.markdown(f"#### {tw} Minute Aggregation")

        time_window_str = f'{tw}min'
        aggregated = aggregate_ofi_data(last_day_df.copy(), time_window_str)

        if aggregated is None or len(aggregated) == 0:
            st.warning(f"No data for {tw} min window")
            continue

        st.caption(f"{len(aggregated)} intervals")

        outlier_methods = get_outlier_methods(aggregated)
        method_names = list(outlier_methods.keys())

        # Create 3x3 grid
        for row_idx in range(3):
            cols = st.columns(3)
            for col_idx in range(3):
                method_idx = row_idx * 3 + col_idx
                if method_idx < len(method_names):
                    method_name = method_names[method_idx]
                    method_df = outlier_methods[method_name]
                    short_name = method_name.replace('Filtered', '').replace('Trimmed', '').replace('Data', '').strip()

                    with cols[col_idx]:
                        fig, _ = plot_single_regression(method_df, short_name)
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")


def get_outlier_methods(df):
    """Return dictionary of all outlier filtering methods applied to df"""
    return {
        'Raw Data (No Filtering)': df.copy(),
        'IQR Filtered (1.5xIQR)': filter_outliers_iqr(df, 'ofi'),
        'Percentile Trimmed (1%-99%)': filter_outliers_percentile(df, 'ofi'),
        'Z-Score Filtered (|Z|<=3)': filter_outliers_zscore(df, 'ofi'),
        'Winsorized (1% caps)': winsorize_data(df, 'ofi'),
        'Absolute Threshold (+-200k)': filter_absolute_threshold(df, 'ofi', -200000, 200000),
        'Absolute Threshold (+-100k)': filter_absolute_threshold(df, 'ofi', -100000, 100000),
        'MAD Filtered (3s)': filter_outliers_mad(df, 'ofi', 3),
        'Percentile Trimmed (5%-95%)': filter_outliers_percentile_aggressive(df, 'ofi'),
    }


def render_time_window_analysis(raw_ofi_df, time_window_minutes, start_datetime, end_datetime, split_method='snapshot'):
    """
    Render all 9 outlier methods for a specific time window.

    Args:
        raw_ofi_df: Raw OFI DataFrame (not yet aggregated)
        time_window_minutes: Time window in minutes (e.g., 10)
        start_datetime: Start datetime for filtering
        end_datetime: End datetime for filtering
        split_method: 'snapshot' or 'date' for phase splitting

    Returns:
        list: Summary statistics for all methods
    """
    time_window_str = f'{time_window_minutes}min'

    # Aggregate data with this specific time window
    aggregated_df = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)

    # Filter by date range
    filtered_ofi = filter_by_date(aggregated_df, start_datetime, end_datetime)

    if filtered_ofi is None or len(filtered_ofi) == 0:
        st.warning(f"No data available for {time_window_minutes} min window")
        return []

    st.info(f"Aggregated to {len(filtered_ofi):,} intervals ({time_window_minutes} min each)")

    # Get all outlier methods
    outlier_methods = get_outlier_methods(filtered_ofi)

    # Collect summary stats
    summary_rows = []

    # Display each method
    for method_name, method_df in outlier_methods.items():
        n_original = len(filtered_ofi)
        n_filtered = len(method_df)
        n_removed = n_original - n_filtered

        st.markdown(f"---")
        st.markdown(f"### {method_name}")
        if n_removed > 0:
            st.caption(f"{n_removed:,} observations removed ({100*n_removed/n_original:.1f}%)")
        else:
            caption = f"{n_filtered:,} observations (values capped)" if 'Winsorized' in method_name else f"{n_filtered:,} observations"
            st.caption(caption)

        # Generate 3-phase plot
        fig_phases, phase_results = plot_ofi_three_phases(method_df, split_method)

        if fig_phases:
            st.plotly_chart(fig_phases, use_container_width=True)

            # Add to summary
            if phase_results is not None and len(phase_results) > 0:
                for _, row in phase_results.iterrows():
                    short_name = method_name.replace('Filtered', '').replace('Trimmed', '').replace('Data', '').strip()
                    summary_rows.append({
                        'Method': short_name,
                        'Phase': row['Phase'].replace('Phase ', 'P'),
                        'N': row['N'],
                        'Beta': row['Beta'],
                        'R2': row['R²'],
                        'p-value': row['p-value']
                    })
        else:
            st.warning(f"Not enough data after filtering for 3-phase analysis")

    # Summary comparison table for this time window
    if summary_rows:
        st.markdown("---")
        st.markdown("### Summary for This Time Window")
        summary_df = pd.DataFrame(summary_rows)

        # Pivot tables
        pivot_r2 = summary_df.pivot(index='Method', columns='Phase', values='R2')
        pivot_beta = summary_df.pivot(index='Method', columns='Phase', values='Beta')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**R² by Method and Phase**")
            st.dataframe(
                pivot_r2.style.format('{:.4f}').background_gradient(cmap='Greens', axis=None),
                use_container_width=True
            )
        with col2:
            st.markdown("**β by Method and Phase**")
            st.dataframe(
                pivot_beta.style.format('{:.2e}'),
                use_container_width=True
            )

    return summary_rows


# ============================================================================
# DEPTH ANALYSIS RENDER FUNCTION
# ============================================================================

def compute_depth_analysis(raw_ofi_df, split_method):
    """Compute beta vs average depth for all configurations"""
    results = []
    dep_var = get_dependent_variable_name()

    for tw in TIME_WINDOWS:
        time_window_str = f'{tw}min'
        aggregated_df = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)

        if aggregated_df is None or len(aggregated_df) == 0:
            continue

        # Apply outlier methods
        outlier_methods = get_outlier_methods(aggregated_df)

        for method_idx, (method_name, method_df) in enumerate(outlier_methods.items()):
            df_clean = method_df.dropna(subset=['ofi', dep_var]).copy()
            if len(df_clean) < 50:
                continue

            # Split into phases
            phases_dict = split_into_phases(df_clean, split_method)
            phases = {
                'Early': phases_dict['Phase 1 (Early)'],
                'Middle': phases_dict['Phase 2 (Middle)'],
                'Near Expiry': phases_dict['Phase 3 (Near Expiry)']
            }

            for phase_name, phase_df in phases.items():
                if len(phase_df) < 10:
                    continue

                # Run regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    phase_df['ofi'], phase_df[dep_var]
                )

                # Calculate average depths
                ad_level1 = (phase_df['best_bid_size'].mean() + phase_df['best_ask_size'].mean()) / 2
                ad_level2 = (phase_df['total_bid_size'].mean() + phase_df['total_ask_size'].mean()) / 2

                results.append({
                    'time_window': tw,
                    'outlier_method': method_name,
                    'phase': phase_name,
                    'beta': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'ad_level1': ad_level1,
                    'ad_level2': ad_level2,
                    'n_obs': len(phase_df)
                })

    return pd.DataFrame(results) if results else None


def plot_beta_vs_depth(data_points, method_name, depth_type='level1'):
    """Create scatter plot of beta vs average depth"""
    import plotly.graph_objects as go

    ad_col = f'ad_{depth_type}'
    fig = go.Figure()

    colors = {'Early': '#2E86AB', 'Middle': '#A23B72', 'Near Expiry': '#F18F01'}
    markers = {'Early': 'circle', 'Middle': 'square', 'Near Expiry': 'diamond'}

    for point in data_points:
        fig.add_trace(go.Scatter(
            x=[point[ad_col]],
            y=[point['beta']],
            mode='markers+text',
            marker=dict(
                color=colors.get(point['phase'], '#333'),
                size=12,
                symbol=markers.get(point['phase'], 'circle')
            ),
            text=[point['phase'][0]],
            textposition='top center',
            name=point['phase'],
            showlegend=False
        ))

    # Add trend line
    if len(data_points) >= 2:
        x_vals = [p[ad_col] for p in data_points]
        y_vals = [p['beta'] for p in data_points]

        if len(x_vals) >= 2:
            slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
            x_line = [min(x_vals), max(x_vals)]
            y_line = [slope * x + intercept for x in x_line]

            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False
            ))

            direction = "↓ β decreases" if slope < 0 else "↑ β increases"
            color = "green" if slope < 0 else "red"
            fig.add_annotation(
                text=direction,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=9, color=color),
                xanchor='center'
            )

    fig.update_layout(
        title=dict(text=method_name, font=dict(size=11)),
        xaxis_title="Average Depth (AD)",
        yaxis_title="Beta (Price Impact)",
        height=280,
        margin=dict(l=60, r=20, t=40, b=60),
        showlegend=False
    )

    return fig


def render_depth_analysis(raw_ofi_df, split_method, start_datetime, end_datetime):
    """Render the Depth Analysis content"""
    st.header("Depth Analysis: Beta vs Average Depth")
    st.markdown("**Testing: Beta = c / AD^lambda (Cont et al. 2011)**")
    st.markdown("*Does price impact decrease as market depth increases?*")

    with st.spinner("Computing beta-depth analysis (81 configs x 3 phases)..."):
        results_df = compute_depth_analysis(raw_ofi_df, split_method)

    if results_df is None or len(results_df) == 0:
        st.error("No results computed. Check data availability.")
        return

    st.success(f"Computed {len(results_df)} beta-depth pairs")

    # Create tabs
    tabs = st.tabs(["Level 1 Depth", "Level 2 Depth", "Comparison", "Summary"])

    # Tab 1: Level 1 Depth
    with tabs[0]:
        st.subheader("Level 1 Depth Analysis")
        st.markdown("**AD = (best_bid_size + best_ask_size) / 2**")

        for tw in TIME_WINDOWS:
            st.markdown(f"### {tw} Minute Aggregation")
            tw_data = results_df[results_df['time_window'] == tw]

            if len(tw_data) == 0:
                st.warning(f"No data for {tw} min window")
                continue

            outlier_methods = list(get_outlier_methods(pd.DataFrame({'ofi': [0]})).keys())
            for row_idx in range(3):
                cols = st.columns(3)
                for col_idx in range(3):
                    method_idx = row_idx * 3 + col_idx
                    if method_idx < len(outlier_methods):
                        method_name = outlier_methods[method_idx]
                        method_data = tw_data[tw_data['outlier_method'] == method_name]

                        with cols[col_idx]:
                            if len(method_data) == 3:
                                data_points = method_data.to_dict('records')
                                fig = plot_beta_vs_depth(data_points, method_name, 'level1')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(f"{method_name}: Insufficient data")
            st.markdown("---")

    # Tab 2: Level 2 Depth
    with tabs[1]:
        st.subheader("Level 2 Depth Analysis")
        st.markdown("**AD = (total_bid_size + total_ask_size) / 2**")

        for tw in TIME_WINDOWS:
            st.markdown(f"### {tw} Minute Aggregation")
            tw_data = results_df[results_df['time_window'] == tw]

            if len(tw_data) == 0:
                continue

            outlier_methods = list(get_outlier_methods(pd.DataFrame({'ofi': [0]})).keys())
            for row_idx in range(3):
                cols = st.columns(3)
                for col_idx in range(3):
                    method_idx = row_idx * 3 + col_idx
                    if method_idx < len(outlier_methods):
                        method_name = outlier_methods[method_idx]
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

        # Calculate % where beta decreases with AD
        def calc_decrease_pct(df, ad_col):
            count = 0
            total = 0
            for tw in TIME_WINDOWS:
                outlier_methods = list(get_outlier_methods(pd.DataFrame({'ofi': [0]})).keys())
                for method in outlier_methods:
                    subset = df[(df['time_window'] == tw) & (df['outlier_method'] == method)]
                    if len(subset) == 3:
                        sorted_data = subset.sort_values(ad_col)
                        betas = sorted_data['beta'].values
                        if betas[0] > betas[2]:
                            count += 1
                        total += 1
            return (count / total * 100) if total > 0 else 0

        pct_l1 = calc_decrease_pct(results_df, 'ad_level1')
        pct_l2 = calc_decrease_pct(results_df, 'ad_level2')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Level 1 (Best Bid/Ask)")
            st.metric("% Configs Where Beta Decreases with AD", f"{pct_l1:.1f}%")
        with col2:
            st.markdown("### Level 2 (Full Orderbook)")
            st.metric("% Configs Where Beta Decreases with AD", f"{pct_l2:.1f}%")

        st.markdown("---")
        st.markdown("### Interpretation")
        st.markdown("""
        - **Beta decreases as AD increases** = **Theory holds** (Cont et al. 2011)
        - **Beta increases as AD increases** = **Theory doesn't hold**
        """)

        if pct_l1 > pct_l2:
            st.success(f"**Level 1 depth** shows stronger support ({pct_l1:.1f}% vs {pct_l2:.1f}%)")
        elif pct_l2 > pct_l1:
            st.success(f"**Level 2 depth** shows stronger support ({pct_l2:.1f}% vs {pct_l1:.1f}%)")
        else:
            st.info("Both depth measures show similar support")

    # Tab 4: Summary
    with tabs[3]:
        st.subheader("Summary Statistics")

        avg_beta_by_phase = results_df.groupby('phase')['beta'].mean()
        avg_ad_l1 = results_df.groupby('phase')['ad_level1'].mean()
        avg_ad_l2 = results_df.groupby('phase')['ad_level2'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Avg Beta by Phase**")
            for phase in ['Early', 'Middle', 'Near Expiry']:
                if phase in avg_beta_by_phase.index:
                    st.metric(phase, f"{avg_beta_by_phase[phase]:.2e}")
        with col2:
            st.markdown("**Avg L1 Depth by Phase**")
            for phase in ['Early', 'Middle', 'Near Expiry']:
                if phase in avg_ad_l1.index:
                    st.metric(phase, f"{avg_ad_l1[phase]:,.0f}")
        with col3:
            st.markdown("**Avg L2 Depth by Phase**")
            for phase in ['Early', 'Middle', 'Near Expiry']:
                if phase in avg_ad_l2.index:
                    st.metric(phase, f"{avg_ad_l2[phase]:,.0f}")

        st.markdown("---")
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
            height=400
        )

        csv = results_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "depth_analysis.csv", "text/csv")


# ============================================================================
# TI COMPARISON RENDER FUNCTION
# ============================================================================

def aggregate_trades_by_time_window(df, time_window_minutes):
    """Aggregate trade data to calculate Trade Imbalance (TI)"""
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    tw_str = f'{time_window_minutes}min'
    df['time_bin'] = df['timestamp'].dt.floor(tw_str)

    df['signed_volume'] = df['direction'] * df['shares_normalized']

    agg = df.groupby('time_bin').agg({
        'signed_volume': 'sum',
        'shares_normalized': 'sum',
        'direction': 'count',
        'price': ['mean', 'first', 'last']
    }).reset_index()

    agg.columns = ['timestamp', 'trade_imbalance', 'total_volume', 'num_trades',
                   'avg_price', 'first_price', 'last_price']

    return agg


def aggregate_ofi_by_time_window(df, time_window_minutes):
    """Aggregate OFI data by time window for TI comparison"""
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    tw_str = f'{time_window_minutes}min'
    df['time_bin'] = df['timestamp'].dt.floor(tw_str)

    agg = df.groupby('time_bin').agg({
        'ofi': 'sum',
        'mid_price': ['first', 'last']
    }).reset_index()

    agg.columns = ['timestamp', 'ofi', 'mid_price_first', 'mid_price_last']
    agg['mid_price'] = agg['mid_price_last']
    agg['delta_mid_price'] = agg['mid_price'].diff()
    agg['delta_mid_price_ticks'] = agg['delta_mid_price'] / TICK_SIZE
    agg = agg.iloc[1:].reset_index(drop=True)

    return agg


def run_ti_regression(x, y):
    """Run OLS regression for TI comparison"""
    if len(x) < 3:
        return None

    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'n_obs': len(x_clean)
    }


def render_ti_comparison(market_config, raw_ofi_df):
    """Render the TI vs OFI Comparison content

    Configuration:
    - Time Window: 45 minutes (fixed)
    - Outlier Method: Z-Score (3σ)
    - Phases: Early, Middle, Near Expiry
    """
    st.header("Trade Imbalance vs OFI Comparison")
    st.markdown("""
    **Cont et al. (2011) Section 3.3**: Comparing OFI with Trade Imbalance (TI).

    - **OFI** = Order flow imbalance from orderbook changes
    - **TI** = Trade Imbalance = Σ(buy_volume) - Σ(sell_volume)

    The paper found OFI outperforms TI because OFI captures queue dynamics.
    """)

    st.info("**Configuration:** 45-min time window | Z-Score (3σ) outlier removal | 3 phases")

    # Load trades data
    trades_file = market_config.get('trades_file')
    if not trades_file:
        st.warning("Trade data not configured for this market")
        return

    trades_path = DATA_DIR / trades_file
    if not trades_path.exists():
        st.warning(f"Trade data file not found: {trades_file}")
        return

    trades_df = pd.read_csv(trades_path)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], format='mixed', utc=True)

    st.success(f"Loaded {len(raw_ofi_df):,} OFI records and {len(trades_df):,} trades")

    # Fixed configuration
    TIME_WINDOW = 45  # minutes

    # Aggregate OFI data with 45-min window
    ofi_agg = aggregate_ofi_by_time_window(raw_ofi_df, TIME_WINDOW)
    ti_agg = aggregate_trades_by_time_window(trades_df, TIME_WINDOW)

    if ofi_agg is None or ti_agg is None:
        st.error("Could not aggregate data")
        return

    # Merge OFI and TI data
    merged = pd.merge(ofi_agg, ti_agg, on='timestamp', how='inner')
    merged = merged.dropna(subset=['ofi', 'trade_imbalance', 'delta_mid_price_ticks'])

    if len(merged) < 30:
        st.error(f"Not enough data after merge: {len(merged)} observations")
        return

    # Apply Z-score outlier filtering to OFI
    mean_ofi = merged['ofi'].mean()
    std_ofi = merged['ofi'].std()
    if std_ofi > 0:
        z_scores = (merged['ofi'] - mean_ofi) / std_ofi
        merged = merged[z_scores.abs() <= 3].copy()

    st.caption(f"After Z-score filtering: {len(merged)} observations")

    # Split into 3 phases
    n = len(merged)
    phase_size = n // 3

    phases = {
        'Early': merged.iloc[:phase_size].copy(),
        'Middle': merged.iloc[phase_size:2*phase_size].copy(),
        'Near Expiry': merged.iloc[2*phase_size:].copy()
    }

    # Calculate R² for each phase
    results = []
    for phase_name, phase_df in phases.items():
        if len(phase_df) < 10:
            continue

        ofi_reg = run_ti_regression(phase_df['ofi'].values, phase_df['delta_mid_price_ticks'].values)
        ti_reg = run_ti_regression(phase_df['trade_imbalance'].values, phase_df['delta_mid_price_ticks'].values)
        vol_reg = run_ti_regression(phase_df['total_volume'].values, phase_df['delta_mid_price_ticks'].values)

        if ofi_reg and ti_reg:
            results.append({
                'Phase': phase_name,
                'N': len(phase_df),
                'OFI R²': ofi_reg['r_squared'],
                'TI R²': ti_reg['r_squared'],
                'Volume R²': vol_reg['r_squared'] if vol_reg else 0,
                'Winner': 'OFI' if ofi_reg['r_squared'] > ti_reg['r_squared'] else 'TI'
            })

    if not results:
        st.error("Not enough data for phase comparison")
        return

    results_df = pd.DataFrame(results)

    # Summary metrics
    st.subheader("Results by Phase")

    # Display results table
    st.dataframe(
        results_df.style.format({
            'OFI R²': '{:.4f}',
            'TI R²': '{:.4f}',
            'Volume R²': '{:.4f}'
        }).apply(lambda x: ['background-color: #d4edda' if v == 'OFI' else 'background-color: #f8d7da'
                           for v in x] if x.name == 'Winner' else [''] * len(x), axis=0),
        use_container_width=True
    )

    # Summary statistics
    st.markdown("---")
    st.subheader("Summary")

    ofi_wins = (results_df['Winner'] == 'OFI').sum()
    ti_wins = (results_df['Winner'] == 'TI').sum()
    avg_ofi_r2 = results_df['OFI R²'].mean()
    avg_ti_r2 = results_df['TI R²'].mean()
    avg_vol_r2 = results_df['Volume R²'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg OFI R²", f"{avg_ofi_r2:.4f}")
    with col2:
        st.metric("Avg TI R²", f"{avg_ti_r2:.4f}")
    with col3:
        st.metric("Avg Volume R²", f"{avg_vol_r2:.4f}")
    with col4:
        st.metric("OFI wins", f"{ofi_wins}/{len(results_df)} phases")

    # Conclusion
    st.markdown("---")
    if avg_ofi_r2 > avg_ti_r2:
        st.success(f"**OFI outperforms Trade Imbalance** (consistent with Cont et al. 2011)")
    else:
        st.warning(f"**Trade Imbalance outperforms OFI** (contrary to Cont et al. 2011)")

    if avg_ofi_r2 > avg_vol_r2:
        st.success("**OFI outperforms raw Volume** - direction matters!")
    else:
        st.info("Volume performs comparably to OFI")

    # Chart
    st.markdown("---")
    st.subheader("R² by Phase")

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df['Phase'],
        y=results_df['OFI R²'],
        name='OFI',
        marker_color='#2E86AB',
        text=[f"{r:.4f}" for r in results_df['OFI R²']],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=results_df['Phase'],
        y=results_df['TI R²'],
        name='Trade Imbalance',
        marker_color='#06A77D',
        text=[f"{r:.4f}" for r in results_df['TI R²']],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=results_df['Phase'],
        y=results_df['Volume R²'],
        name='Volume',
        marker_color='#A23B72',
        text=[f"{r:.4f}" for r in results_df['Volume R²']],
        textposition='outside'
    ))
    fig.update_layout(
        title="R² Comparison: OFI vs TI vs Volume by Phase (45-min, Z-score filtered)",
        xaxis_title="Phase",
        yaxis_title="R²",
        barmode='group',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="OFI Multi-Time Analysis",
        layout="wide"
    )

    st.title("OFI Multi-Time Window Analysis")
    st.markdown("**Comparing 9 time windows x 9 outlier methods x 3 phases = 243 regressions**")

    dep_var_label = get_dependent_variable_label()
    st.caption(f"Price Units: {dep_var_label}")

    # Sidebar - Page Selection, Market Selection and Date/Time Filter
    with st.sidebar:
        st.header("Dashboard")

        # Page selector
        page_options = ["OFI Analysis", "Depth Analysis", "TI vs OFI Comparison"]
        selected_page = st.radio(
            "Select Analysis Type",
            options=page_options,
            index=0,
            help="OFI: 243 regressions. Depth: β vs Market Depth. TI: Trade Imbalance vs OFI."
        )

        st.divider()

        st.header("Market Selection")

        # Market dropdown - default to NYC
        market_options = list(MARKETS.keys())
        default_idx = market_options.index("NYC Mayoral Election 2025 (Zohran Mamdani)")
        selected_market = st.selectbox(
            "Choose Market",
            options=market_options,
            index=default_idx,
            help="Select market for analysis"
        )

        market_config = MARKETS[selected_market]
        st.info(f"**{market_config['description']}**")

        st.divider()

        # Phase splitting method
        st.header("Phase Split Method")
        split_method = st.selectbox(
            "How to divide into 3 phases?",
            options=['snapshot', 'date'],
            format_func=lambda x: 'By Observation Count' if x == 'snapshot' else 'By Calendar Time',
            help="Snapshot: equal # of observations per phase. Date: equal time periods."
        )

        st.divider()

        # Load raw data (not aggregated yet - we'll aggregate per tab)
        with st.spinner("Loading raw data..."):
            raw_ofi_df, orderbook_df, market_info = load_market_data(selected_market)

        if raw_ofi_df is None or len(raw_ofi_df) == 0:
            st.error("No data found for NYC market")
            st.stop()

        # Ensure timestamp is datetime for date range calculation
        if 'timestamp' not in raw_ofi_df.columns and 'timestamp_ms' in raw_ofi_df.columns:
            raw_ofi_df['timestamp'] = pd.to_datetime(raw_ofi_df['timestamp_ms'], unit='ms', utc=True)
        elif 'timestamp' in raw_ofi_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(raw_ofi_df['timestamp']):
                raw_ofi_df['timestamp'] = pd.to_datetime(raw_ofi_df['timestamp'], format='mixed', utc=True)

        st.success(f"Loaded {len(raw_ofi_df):,} raw snapshots")

        # Date range sliders
        st.header("Date/Time Filter")

        min_date = raw_ofi_df['timestamp'].min()
        max_date = raw_ofi_df['timestamp'].max()

        st.caption(f"Available: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")

        # Date selection
        col1, col2 = st.columns(2)

        with col1:
            start_date_only = st.date_input(
                "Start Date",
                value=min_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )

        with col2:
            end_date_only = st.date_input(
                "End Date",
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )

        # Time selection
        col3, col4 = st.columns(2)

        with col3:
            start_time = st.time_input(
                "Start Time (UTC)",
                value=min_date.time()
            )

        with col4:
            end_time = st.time_input(
                "End Time (UTC)",
                value=max_date.time()
            )

        # Combine date and time
        start_datetime = datetime.combine(start_date_only, start_time)
        end_datetime = datetime.combine(end_date_only, end_time)

        # Validation
        if start_datetime >= end_datetime:
            st.error("Start date/time must be before end date/time")
            st.stop()

        # Clamp to data boundaries
        start_datetime = max(start_datetime, min_date.replace(tzinfo=None))
        end_datetime = min(end_datetime, max_date.replace(tzinfo=None))

        st.divider()

        # Data info
        st.header("Data Info")
        st.metric("Raw Snapshots", f"{len(raw_ofi_df):,}")
        st.metric("Date Range", f"{(max_date - min_date).days} days")

    # ========================================================================
    # CONDITIONAL RENDERING BASED ON SELECTED PAGE
    # ========================================================================

    if selected_page == "Depth Analysis":
        # Render Depth Analysis content
        render_depth_analysis(raw_ofi_df, split_method, start_datetime, end_datetime)

    elif selected_page == "TI vs OFI Comparison":
        # Render TI Comparison content
        render_ti_comparison(market_config, raw_ofi_df)

    else:
        # Default: OFI Analysis
        # Create tabs: Home + Last Day + 9 time windows
        tab_names = ["Home", "Last Day"] + [f"{t} min" for t in TIME_WINDOWS]
        tabs = st.tabs(tab_names)

        # Home tab - Price & Depth analysis + Master Summary
        with tabs[0]:
            st.subheader("Price & Market Overview")
            st.markdown("Overview of price evolution, spread, and order book depth")

            # Aggregate with default 10min window for overview
            overview_df = aggregate_ofi_data(raw_ofi_df.copy(), '10min')
            filtered_overview = filter_by_date(overview_df, start_datetime, end_datetime)

            if filtered_overview is not None and len(filtered_overview) > 0:
                st.info(f"Showing {len(filtered_overview):,} intervals (10-min aggregation)")

                # Price Evolution
                st.subheader("Price Evolution")
                fig_price = plot_price_evolution(filtered_overview)
                st.plotly_chart(fig_price, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Spread Analysis")
                    fig_spread = plot_spread_analysis(filtered_overview)
                    st.plotly_chart(fig_spread, use_container_width=True)

                with col2:
                    st.subheader("Order Book Depth")
                    fig_depth = plot_depth_evolution(filtered_overview)
                    st.plotly_chart(fig_depth, use_container_width=True)

                # OFI Distribution
                st.subheader("OFI Distribution")
                fig_ofi_dist = plot_ofi_distribution(filtered_overview)
                st.plotly_chart(fig_ofi_dist, use_container_width=True)
            else:
                st.warning("No data available for overview")

            # Master Summary Section (in Home tab)
            st.markdown("---")
            st.markdown("# Master Summary: All Time Windows x All Methods")
            st.markdown("*Collecting data from all 9 time windows...*")

            # Collect all summary data from all time windows
            all_summary_data = []
            with st.spinner("Calculating 243 regressions across all time windows..."):
                for tw in TIME_WINDOWS:
                    time_window_str = f'{tw}min'
                    aggregated_df = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)
                    filtered_ofi = filter_by_date(aggregated_df, start_datetime, end_datetime)

                    if filtered_ofi is None or len(filtered_ofi) == 0:
                        continue

                    outlier_methods = get_outlier_methods(filtered_ofi)
                    dep_var = get_dependent_variable_name()

                    for method_name, method_df in outlier_methods.items():
                        df_clean = method_df.dropna(subset=['ofi', dep_var]).copy()
                        if len(df_clean) < 150:
                            continue

                        # Use the chosen split method
                        phases_dict = split_into_phases(df_clean, split_method)
                        # Rename keys to short form
                        phases = {
                            'P1 (Early)': phases_dict['Phase 1 (Early)'],
                            'P2 (Middle)': phases_dict['Phase 2 (Middle)'],
                            'P3 (Near Expiry)': phases_dict['Phase 3 (Near Expiry)']
                        }

                        from scipy import stats
                        for phase_name, phase_df in phases.items():
                            if len(phase_df) > 2:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(
                                    phase_df['ofi'], phase_df[dep_var]
                                )
                                short_name = method_name.replace('Filtered', '').replace('Trimmed', '').replace('Data', '').strip()
                                all_summary_data.append({
                                    'TimeWindow': tw,
                                    'Method': short_name,
                                    'Phase': phase_name,
                                    'N': len(phase_df),
                                    'Beta': slope,
                                    'R2': r_value**2,
                                    'p-value': p_value
                                })

            if all_summary_data:
                master_df = pd.DataFrame(all_summary_data)

                # Table 1: Average R² across all phases
                st.markdown("### 1. Average R² Across All Phases")
                avg_r2_by_tw_method = master_df.groupby(['TimeWindow', 'Method'])['R2'].mean().reset_index()
                pivot_avg = avg_r2_by_tw_method.pivot(index='TimeWindow', columns='Method', values='R2')
                st.dataframe(
                    pivot_avg.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
                    use_container_width=True,
                    height=380
                )

                # Find best overall
                best_idx = avg_r2_by_tw_method['R2'].idxmax()
                best_row = avg_r2_by_tw_method.loc[best_idx]
                st.success(f"**Best Overall**: {best_row['TimeWindow']} min + {best_row['Method']} (Avg R² = {best_row['R2']:.4f})")

                # Table 2: Phase 1 (Early) only
                st.markdown("### 2. Phase 1 (Early Market) R²")
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

                # Table 3: Phase 2 (Middle) only
                st.markdown("### 3. Phase 2 (Middle Market) R²")
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

                # Table 4: Phase 3 (Near Expiry) only
                st.markdown("### 4. Phase 3 (Near Expiry) R²")
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

                # Download full results
                st.markdown("### Download Full Results")
                csv = master_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="ofi_multi_time_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for summary")

        # Last Day tab (tabs[1])
        with tabs[1]:
            render_last_day_analysis(raw_ofi_df)

        # Time window tabs (tabs[2] through tabs[10])
        for i, tw in enumerate(TIME_WINDOWS):
            with tabs[i + 2]:  # +2 because tabs[0] is Home, tabs[1] is Last Day
                st.subheader(f"Time Window: {tw} Minutes")
                st.markdown(f"Aggregating OFI data into {tw}-minute intervals, then applying 9 outlier methods")

                # Render analysis for this time window
                render_time_window_analysis(raw_ofi_df, tw, start_datetime, end_datetime, split_method)


if __name__ == "__main__":
    main()
