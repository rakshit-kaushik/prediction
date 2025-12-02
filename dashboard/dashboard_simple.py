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


def compute_daily_depth_analysis(raw_ofi_df):
    """
    Compute beta and average depth for each day.

    Uses fixed config: 45-min time window + Z-score outlier removal.
    This gives ~21 data points (one per day) instead of just 3 phases.
    """
    dep_var = get_dependent_variable_name()

    # Config: 45-min window, Z-score filtering (matches TI comparison)
    TIME_WINDOW = 45

    # Aggregate to 45-min intervals
    aggregated = aggregate_ofi_data(raw_ofi_df.copy(), f'{TIME_WINDOW}min')

    if aggregated is None or len(aggregated) == 0:
        return None

    # Apply Z-score filtering
    filtered = filter_outliers_zscore(aggregated, 'ofi', threshold=3)

    if filtered is None or len(filtered) == 0:
        return None

    # Group by date
    filtered = filtered.copy()
    filtered['date'] = filtered['timestamp'].dt.date

    results = []
    for date, day_df in filtered.groupby('date'):
        # Need enough points for regression
        day_clean = day_df.dropna(subset=['ofi', dep_var])
        if len(day_clean) < 5:
            continue

        # Run regression for this day
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            day_clean['ofi'], day_clean[dep_var]
        )

        # Calculate average depths for this day
        ad_level1 = (day_df['best_bid_size'].mean() + day_df['best_ask_size'].mean()) / 2
        ad_level2 = (day_df['total_bid_size'].mean() + day_df['total_ask_size'].mean()) / 2

        results.append({
            'date': date,
            'beta': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'ad_level1': ad_level1,
            'ad_level2': ad_level2,
            'n_obs': len(day_clean)
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
    tabs = st.tabs(["Level 1 Depth", "Level 2 Depth", "Comparison", "Summary", "Daily Analysis"])

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

    # Tab 5: Daily Analysis
    with tabs[4]:
        st.subheader("Daily Analysis: 45-min Window + Z-Score")
        st.markdown("**Config:** 45-minute time window | Z-Score (3σ) outlier removal")
        st.markdown("*Each point = one day's beta and average depth*")
        st.markdown("**Per Cont et al. (2011):** β = c / AD^λ → log(β) = log(c) - λ × log(AD)")

        with st.spinner("Computing daily beta-depth analysis..."):
            daily_df = compute_daily_depth_analysis(raw_ofi_df)

        if daily_df is None or len(daily_df) == 0:
            st.error("Could not compute daily analysis. Check data availability.")
        else:
            st.success(f"Computed {len(daily_df)} daily data points")

            # Create side-by-side scatter plots for Level 1 and Level 2
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Level 1 Depth (Best Bid/Ask)")

                # Filter positive values for log-log regression
                valid_df1 = daily_df[(daily_df['ad_level1'] > 0) & (daily_df['beta'] > 0)].copy()

                # Create scatter plot (log-log scale)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=valid_df1['ad_level1'],
                    y=valid_df1['beta'],
                    mode='markers+text',
                    marker=dict(size=10, color='#2E86AB'),
                    text=valid_df1['date'].astype(str).str[-5:],  # Show MM-DD
                    textposition='top center',
                    textfont=dict(size=8),
                    hovertemplate='<b>Date:</b> %{text}<br><b>AD:</b> %{x:,.0f}<br><b>Beta:</b> %{y:.2e}<extra></extra>'
                ))

                # Log-log regression: log(β) = log(c) - λ * log(AD)
                if len(valid_df1) >= 2:
                    log_ad = np.log(valid_df1['ad_level1'])
                    log_beta = np.log(valid_df1['beta'])
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ad, log_beta)

                    # λ = -slope (since log(β) = log(c) - λ * log(AD))
                    lambda_est = -slope

                    # Add fitted line (in original space)
                    x_range = np.linspace(valid_df1['ad_level1'].min(), valid_df1['ad_level1'].max(), 100)
                    y_fitted = np.exp(intercept) * np.power(x_range, slope)
                    fig1.add_trace(go.Scatter(
                        x=x_range, y=y_fitted,
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Fitted: β = c/AD^λ'
                    ))

                    # Annotation with λ
                    result = "Theory HOLDS" if lambda_est > 0 else "Theory DOESN'T hold"
                    color = "green" if lambda_est > 0 else "red"
                    fig1.add_annotation(
                        text=f"<b>λ = {lambda_est:.3f}</b> ± {std_err:.3f}<br>Log-Log R²: {r_value**2:.3f}<br>p-value: {p_value:.3f}<br><b>{result}</b>",
                        xref="paper", yref="paper",
                        x=0.95, y=0.95,
                        showarrow=False,
                        bgcolor='black',
                        font=dict(color=color, size=10),
                        xanchor='right',
                        yanchor='top',
                        bordercolor='white',
                        borderwidth=1
                    )

                fig1.update_layout(
                    title="Beta vs Level 1 Depth (Log-Log)",
                    xaxis_title="Average Depth (Best Bid/Ask)",
                    yaxis_title="Beta (Price Impact)",
                    xaxis_type="log",
                    yaxis_type="log",
                    height=450,
                    showlegend=False
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("### Level 2 Depth (Full Orderbook)")

                # Filter positive values for log-log regression
                valid_df2 = daily_df[(daily_df['ad_level2'] > 0) & (daily_df['beta'] > 0)].copy()

                # Create scatter plot (log-log scale)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=valid_df2['ad_level2'],
                    y=valid_df2['beta'],
                    mode='markers+text',
                    marker=dict(size=10, color='#A23B72'),
                    text=valid_df2['date'].astype(str).str[-5:],  # Show MM-DD
                    textposition='top center',
                    textfont=dict(size=8),
                    hovertemplate='<b>Date:</b> %{text}<br><b>AD:</b> %{x:,.0f}<br><b>Beta:</b> %{y:.2e}<extra></extra>'
                ))

                # Log-log regression: log(β) = log(c) - λ * log(AD)
                if len(valid_df2) >= 2:
                    log_ad2 = np.log(valid_df2['ad_level2'])
                    log_beta2 = np.log(valid_df2['beta'])
                    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(log_ad2, log_beta2)

                    # λ = -slope (since log(β) = log(c) - λ * log(AD))
                    lambda_est2 = -slope2

                    # Add fitted line (in original space)
                    x_range2 = np.linspace(valid_df2['ad_level2'].min(), valid_df2['ad_level2'].max(), 100)
                    y_fitted2 = np.exp(intercept2) * np.power(x_range2, slope2)
                    fig2.add_trace(go.Scatter(
                        x=x_range2, y=y_fitted2,
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Fitted: β = c/AD^λ'
                    ))

                    # Annotation with λ
                    result2 = "Theory HOLDS" if lambda_est2 > 0 else "Theory DOESN'T hold"
                    color2 = "green" if lambda_est2 > 0 else "red"
                    fig2.add_annotation(
                        text=f"<b>λ = {lambda_est2:.3f}</b> ± {std_err2:.3f}<br>Log-Log R²: {r_value2**2:.3f}<br>p-value: {p_value2:.3f}<br><b>{result2}</b>",
                        xref="paper", yref="paper",
                        x=0.95, y=0.95,
                        showarrow=False,
                        bgcolor='black',
                        font=dict(color=color2, size=10),
                        xanchor='right',
                        yanchor='top',
                        bordercolor='white',
                        borderwidth=1
                    )

                fig2.update_layout(
                    title="Beta vs Level 2 Depth (Log-Log)",
                    xaxis_title="Average Depth (Full Orderbook)",
                    yaxis_title="Beta (Price Impact)",
                    xaxis_type="log",
                    yaxis_type="log",
                    height=450,
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Daily results table
            st.markdown("---")
            st.markdown("### Daily Results Table")
            st.dataframe(
                daily_df.style.format({
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

            # Download button
            csv_daily = daily_df.to_csv(index=False)
            st.download_button("Download Daily CSV", csv_daily, "daily_depth_analysis.csv", "text/csv")


# ============================================================================
# TI COMPARISON RENDER FUNCTION
# ============================================================================

# TI Analysis Configuration (same as OFI)
TI_OUTLIER_METHODS = [
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


def load_ti_81_configs():
    """Load pre-computed TI 81-config results"""
    ti_file = DATA_DIR / "ti_81_configs.csv"
    if ti_file.exists():
        return pd.read_csv(ti_file)
    return None


def compute_ofi_81_configs(raw_ofi_df):
    """Compute OFI R² for all 81 configurations to match TI analysis"""
    dep_var = get_dependent_variable_name()
    results = []

    for tw in TIME_WINDOWS:
        time_window_str = f'{tw}min'
        aggregated = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)

        if aggregated is None or len(aggregated) < 10:
            continue

        outlier_methods = get_outlier_methods(aggregated)

        for method_idx, (method_full_name, method_df) in enumerate(outlier_methods.items()):
            if len(method_df) < 10:
                continue

            df_clean = method_df.dropna(subset=['ofi', dep_var])
            if len(df_clean) < 10:
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_clean['ofi'], df_clean[dep_var]
            )

            # Map to TI method name format
            method_name = TI_OUTLIER_METHODS[method_idx] if method_idx < len(TI_OUTLIER_METHODS) else method_full_name

            results.append({
                'time_window': tw,
                'outlier_method': method_name,
                'r_squared': r_value ** 2,
                'beta': slope,
                'p_value': p_value,
                'n_windows': len(df_clean),
                'std_err': std_err
            })

    return pd.DataFrame(results) if results else None


def render_ti_comparison(market_config, raw_ofi_df):
    """Render the TI vs OFI Comparison content with 81-config heatmaps

    Features:
    - TI R² Heatmap (81 configs)
    - OFI R² Heatmap (81 configs)
    - Difference Heatmap (OFI R² - TI R²)
    - Full TI metrics (beta, p-value, n_obs)
    """
    st.header("Trade Imbalance vs OFI Comparison")
    st.markdown("""
    **Cont et al. (2011) Section 3.3**: Comparing OFI with Trade Imbalance (TI).

    - **OFI** = Order flow imbalance from orderbook changes
    - **TI** = Trade Imbalance = Σ(buy_volume) - Σ(sell_volume)

    The paper found OFI (R²=65%) significantly outperforms TI (R²=32%) because OFI captures queue dynamics.
    """)

    # Create tabs for different views
    ti_tabs = st.tabs(["81-Config Heatmaps", "TI Full Results", "Overall Comparison"])

    # ========================================================================
    # TAB 1: 81-Config Heatmaps
    # ========================================================================
    with ti_tabs[0]:
        st.subheader("81-Configuration Analysis: 9 Time Windows x 9 Outlier Methods")

        # Load TI results
        ti_df = load_ti_81_configs()

        if ti_df is None:
            st.warning("TI 81-config results not found. Run `python data_pipeline/04_process_trades_ti.py` first.")
            return

        # Compute OFI 81-config results
        with st.spinner("Computing OFI 81-config results..."):
            ofi_df = compute_ofi_81_configs(raw_ofi_df)

        if ofi_df is None:
            st.error("Could not compute OFI results")
            return

        st.success(f"Loaded {len(ti_df)} TI configs, computed {len(ofi_df)} OFI configs")

        # Summary metrics at top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("OFI Best R²", f"{ofi_df['r_squared'].max():.4f}")
        with col2:
            st.metric("TI Best R²", f"{ti_df['r_squared'].max():.4f}")
        with col3:
            st.metric("OFI Mean R²", f"{ofi_df['r_squared'].mean():.4f}")
        with col4:
            st.metric("TI Mean R²", f"{ti_df['r_squared'].mean():.4f}")

        # Best configs
        ofi_best = ofi_df.loc[ofi_df['r_squared'].idxmax()]
        ti_best = ti_df.loc[ti_df['r_squared'].idxmax()]
        st.info(f"**OFI Best:** {int(ofi_best['time_window'])}min + {ofi_best['outlier_method']} (R²={ofi_best['r_squared']:.4f})")
        st.info(f"**TI Best:** {int(ti_best['time_window'])}min + {ti_best['outlier_method']} (R²={ti_best['r_squared']:.4f})")

        st.markdown("---")

        # Create pivot tables for heatmaps
        ti_pivot = ti_df.pivot(index='time_window', columns='outlier_method', values='r_squared')
        ofi_pivot = ofi_df.pivot(index='time_window', columns='outlier_method', values='r_squared')

        # Reorder columns to match TI_OUTLIER_METHODS
        ti_pivot = ti_pivot.reindex(columns=TI_OUTLIER_METHODS)
        ofi_pivot = ofi_pivot.reindex(columns=TI_OUTLIER_METHODS)

        # 1. TI R² Heatmap
        st.markdown("### 1. Trade Imbalance R² Heatmap")
        st.markdown("*TI = Σ(buy_volume) - Σ(sell_volume) per time window*")

        # Use Plotly heatmap for better visualization
        fig_ti = go.Figure(data=go.Heatmap(
            z=ti_pivot.values * 100,  # Convert to percentage
            x=ti_pivot.columns,
            y=[f"{tw}min" for tw in ti_pivot.index],
            colorscale='RdYlGn',
            text=[[f"{v:.2f}%" if pd.notna(v) else "N/A" for v in row] for row in ti_pivot.values * 100],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Time: %{y}<br>Method: %{x}<br>R²: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="R² (%)")
        ))
        fig_ti.update_layout(
            title="TI R² (%) - Trade Imbalance vs Price Change",
            xaxis_title="Outlier Method",
            yaxis_title="Time Window",
            height=450
        )
        st.plotly_chart(fig_ti, use_container_width=True)

        # 2. OFI R² Heatmap
        st.markdown("### 2. OFI R² Heatmap")
        st.markdown("*OFI = Order Flow Imbalance from orderbook changes*")

        fig_ofi = go.Figure(data=go.Heatmap(
            z=ofi_pivot.values * 100,
            x=ofi_pivot.columns,
            y=[f"{tw}min" for tw in ofi_pivot.index],
            colorscale='RdYlGn',
            text=[[f"{v:.2f}%" if pd.notna(v) else "N/A" for v in row] for row in ofi_pivot.values * 100],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Time: %{y}<br>Method: %{x}<br>R²: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="R² (%)")
        ))
        fig_ofi.update_layout(
            title="OFI R² (%) - Order Flow Imbalance vs Price Change",
            xaxis_title="Outlier Method",
            yaxis_title="Time Window",
            height=450
        )
        st.plotly_chart(fig_ofi, use_container_width=True)

        # 3. Difference Heatmap (OFI - TI)
        st.markdown("### 3. Difference Heatmap (OFI R² - TI R²)")
        st.markdown("*Positive (green) = OFI wins, Negative (red) = TI wins*")

        # Calculate difference
        diff_pivot = ofi_pivot - ti_pivot

        fig_diff = go.Figure(data=go.Heatmap(
            z=diff_pivot.values * 100,
            x=diff_pivot.columns,
            y=[f"{tw}min" for tw in diff_pivot.index],
            colorscale='RdYlGn',
            zmid=0,  # Center at 0
            text=[[f"{v:+.2f}%" if pd.notna(v) else "N/A" for v in row] for row in diff_pivot.values * 100],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Time: %{y}<br>Method: %{x}<br>Diff: %{z:+.2f}%<extra></extra>",
            colorbar=dict(title="OFI - TI (%)")
        ))
        fig_diff.update_layout(
            title="OFI R² minus TI R² (Positive = OFI wins)",
            xaxis_title="Outlier Method",
            yaxis_title="Time Window",
            height=450
        )
        st.plotly_chart(fig_diff, use_container_width=True)

        # Summary statistics
        st.markdown("---")
        st.markdown("### Summary Statistics")

        # Merge for comparison
        merged_stats = pd.merge(
            ofi_df[['time_window', 'outlier_method', 'r_squared']].rename(columns={'r_squared': 'ofi_r2'}),
            ti_df[['time_window', 'outlier_method', 'r_squared']].rename(columns={'r_squared': 'ti_r2'}),
            on=['time_window', 'outlier_method'],
            how='inner'
        )
        merged_stats['diff'] = merged_stats['ofi_r2'] - merged_stats['ti_r2']
        merged_stats['winner'] = np.where(merged_stats['ofi_r2'] > merged_stats['ti_r2'], 'OFI', 'TI')

        ofi_wins = (merged_stats['winner'] == 'OFI').sum()
        ti_wins = (merged_stats['winner'] == 'TI').sum()
        total = len(merged_stats)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("OFI Wins", f"{ofi_wins}/{total} ({100*ofi_wins/total:.1f}%)")
        with col2:
            st.metric("TI Wins", f"{ti_wins}/{total} ({100*ti_wins/total:.1f}%)")
        with col3:
            avg_diff = merged_stats['diff'].mean() * 100
            st.metric("Avg Difference", f"{avg_diff:+.2f}%")

        if ofi_wins > ti_wins:
            st.success(f"**OFI outperforms TI in {ofi_wins}/{total} configurations** (consistent with Cont et al. 2011)")
        else:
            st.warning(f"**TI outperforms OFI in {ti_wins}/{total} configurations** (contrary to Cont et al. 2011)")

    # ========================================================================
    # TAB 2: TI Full Results (Paper Metrics)
    # ========================================================================
    with ti_tabs[1]:
        st.subheader("TI Full Results (All Paper Metrics)")
        st.markdown("*Complete regression results including beta, p-value, n_obs*")

        ti_df = load_ti_81_configs()

        if ti_df is None:
            st.warning("TI 81-config results not found.")
            return

        # Summary tables by time window
        st.markdown("### Results by Time Window")

        for tw in TIME_WINDOWS:
            tw_data = ti_df[ti_df['time_window'] == tw]
            if len(tw_data) == 0:
                continue

            st.markdown(f"#### {tw} Minute Window")

            # Format the data
            display_df = tw_data[['outlier_method', 'r_squared', 'beta', 'p_value', 'n_windows']].copy()
            display_df.columns = ['Outlier Method', 'R²', 'Beta (β)', 'p-value', 'N obs']

            st.dataframe(
                display_df.style.format({
                    'R²': '{:.4f}',
                    'Beta (β)': '{:.2e}',
                    'p-value': '{:.2e}',
                    'N obs': '{:.0f}'
                }).background_gradient(subset=['R²'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )

        # Full results table
        st.markdown("---")
        st.markdown("### Complete Results Table")

        full_display = ti_df[['time_window', 'outlier_method', 'r_squared', 'beta', 'p_value', 'n_windows', 'std_err']].copy()
        full_display.columns = ['Time Window', 'Outlier Method', 'R²', 'Beta', 'p-value', 'N obs', 'Std Error']

        st.dataframe(
            full_display.style.format({
                'R²': '{:.4f}',
                'Beta': '{:.2e}',
                'p-value': '{:.2e}',
                'N obs': '{:.0f}',
                'Std Error': '{:.2e}'
            }),
            use_container_width=True,
            height=500
        )

        # Download button
        csv = ti_df.to_csv(index=False)
        st.download_button(
            label="Download TI Results CSV",
            data=csv,
            file_name="ti_81_configs.csv",
            mime="text/csv"
        )

        # Significance analysis
        st.markdown("---")
        st.markdown("### Significance Analysis")

        sig_001 = (ti_df['p_value'] < 0.001).sum()
        sig_01 = ((ti_df['p_value'] >= 0.001) & (ti_df['p_value'] < 0.01)).sum()
        sig_05 = ((ti_df['p_value'] >= 0.01) & (ti_df['p_value'] < 0.05)).sum()
        not_sig = (ti_df['p_value'] >= 0.05).sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("p < 0.001", f"{sig_001}/{len(ti_df)}")
        with col2:
            st.metric("p < 0.01", f"{sig_01}/{len(ti_df)}")
        with col3:
            st.metric("p < 0.05", f"{sig_05}/{len(ti_df)}")
        with col4:
            st.metric("Not Significant", f"{not_sig}/{len(ti_df)}")

    # ========================================================================
    # TAB 3: Overall Comparison (OFI vs TI vs Volume)
    # ========================================================================
    with ti_tabs[2]:
        st.subheader("Overall Comparison: OFI vs TI vs Volume")
        st.markdown("*Fixed config: 45-min time window | Z-Score (3σ) outlier removal*")
        st.markdown("*Per Cont et al. (2014) methodology*")

        # Load pre-computed TI data (small file, ~37KB)
        ti_file = DATA_DIR / "ti_aggregated_45min.csv"
        if not ti_file.exists():
            st.warning("Pre-computed TI data not found. Run: `python data_pipeline/05_precompute_ti_windows.py`")
            return

        ti_agg = pd.read_csv(ti_file)
        ti_agg['timestamp'] = pd.to_datetime(ti_agg['timestamp'], utc=True)

        # Fixed configuration
        TIME_WINDOW_PHASE = 45
        TIME_WINDOW_STR = f'{TIME_WINDOW_PHASE}min'

        # Aggregate OFI data
        ofi_agg = aggregate_ofi_data(raw_ofi_df.copy(), TIME_WINDOW_STR)

        if ofi_agg is None:
            st.error("Could not aggregate OFI data")
            return

        st.success(f"Loaded {len(ofi_agg):,} OFI windows and {len(ti_agg):,} TI windows")

        # Merge OFI and TI
        merged = pd.merge(ofi_agg, ti_agg, on='timestamp', how='inner')
        merged = merged.dropna(subset=['ofi', 'delta_mid_price_ticks'])

        # Apply Z-score filtering
        merged = filter_outliers_zscore(merged, 'ofi', threshold=3)

        st.caption(f"After filtering: {len(merged)} observations")

        # ---- Calculate Overall Metrics ----

        # 1. OFI regression: ΔP = α + β*OFI + ε (signed)
        ofi_clean = merged.dropna(subset=['ofi', 'delta_mid_price_ticks'])
        if len(ofi_clean) >= 3:
            slope_ofi, intercept_ofi, r_ofi, p_ofi, se_ofi = stats.linregress(
                ofi_clean['ofi'], ofi_clean['delta_mid_price_ticks']
            )
        else:
            st.error("Not enough OFI data")
            return

        # 2. |OFI| regression: |ΔP| = α + β*|OFI| + ε (Table 5 style)
        if len(ofi_clean) >= 3:
            slope_ofi_abs, _, r_ofi_abs, p_ofi_abs, _ = stats.linregress(
                ofi_clean['ofi'].abs(),
                ofi_clean['delta_mid_price_ticks'].abs()
            )
        else:
            r_ofi_abs = 0

        # 3. TI regression: ΔP = α + β*TI + ε (signed)
        ti_clean = merged.dropna(subset=['trade_imbalance', 'delta_mid_price_ticks'])
        if len(ti_clean) >= 3:
            slope_ti, intercept_ti, r_ti, p_ti, se_ti = stats.linregress(
                ti_clean['trade_imbalance'], ti_clean['delta_mid_price_ticks']
            )
        else:
            r_ti = 0
            slope_ti = 0
            p_ti = 1

        # 4. Volume regression with exponent H (per Cont et al. eq. 16-17)
        vol_clean = merged.dropna(subset=['total_volume', 'delta_mid_price_ticks'])
        H_estimate = 0.5  # Default

        if len(vol_clean) >= 10:
            # Estimate H via log-log regression
            mask = (vol_clean['total_volume'] > 0) & (vol_clean['delta_mid_price_ticks'].abs() > 0)
            vol_log = vol_clean[mask]
            if len(vol_log) >= 10:
                log_vol = np.log(vol_log['total_volume'])
                log_abs_price = np.log(vol_log['delta_mid_price_ticks'].abs())
                H_estimate = stats.linregress(log_vol, log_abs_price).slope

        if len(vol_clean) >= 3:
            vol_H = np.power(vol_clean['total_volume'] + 1e-10, H_estimate)
            slope_vol, _, r_vol, p_vol, _ = stats.linregress(
                vol_H, vol_clean['delta_mid_price_ticks'].abs()
            )
        else:
            r_vol = 0
            p_vol = 1

        # ---- Display Results ----
        st.markdown("### Regression Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("OFI R²", f"{r_ofi**2:.4f}", help="Signed OFI: ΔP = α + β*OFI")
            st.caption(f"β = {slope_ofi:.2e}")
            st.caption(f"p = {p_ofi:.2e}")

        with col2:
            st.metric("|OFI| R²", f"{r_ofi_abs**2:.4f}", help="Absolute OFI: |ΔP| = α + β*|OFI|")
            st.caption(f"(Table 5 style)")

        with col3:
            st.metric("TI R²", f"{r_ti**2:.4f}", help="Trade Imbalance: ΔP = α + β*TI")
            st.caption(f"β = {slope_ti:.2e}")
            st.caption(f"p = {p_ti:.2e}")

        with col4:
            st.metric("Vol R²", f"{r_vol**2:.4f}", help=f"Volume: |ΔP| = α + β*VOL^H")
            st.caption(f"H = {H_estimate:.3f}")
            st.caption(f"p = {p_vol:.2e}")

        # Bar chart comparison
        fig = go.Figure()
        metrics = ['OFI', '|OFI|', 'TI', 'Volume']
        r2_values = [r_ofi**2, r_ofi_abs**2, r_ti**2, r_vol**2]
        colors = ['#2E86AB', '#A23B72', '#06A77D', '#F18F01']

        fig.add_trace(go.Bar(
            x=metrics,
            y=r2_values,
            marker_color=colors,
            text=[f"{r:.4f}" for r in r2_values],
            textposition='outside'
        ))

        fig.update_layout(
            title="R² Comparison: OFI vs TI vs Volume (45-min, Z-score filtered)",
            xaxis_title="Metric",
            yaxis_title="R²",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.markdown("---")
        st.markdown("### Summary")

        winner = 'OFI' if r_ofi**2 > r_ti**2 else 'TI'
        if winner == 'OFI':
            st.success(f"**OFI outperforms TI** (R²: OFI={r_ofi**2:.4f} vs TI={r_ti**2:.4f})")
        else:
            st.warning(f"**TI outperforms OFI** (R²: TI={r_ti**2:.4f} vs OFI={r_ofi**2:.4f})")

        st.info(f"""
        **Per Cont et al. (2014) Section 4.2:**
        - Volume alone has very low R² ({r_vol**2:.4f}) - consistent with paper's finding
        - Exponent H = {H_estimate:.3f} (paper found H < 0.5 typically)
        - OFI captures queue dynamics that volume misses
        """)


# ============================================================================
# PRESENTATION TAB (White background plots for poster)
# ============================================================================

# Subset configurations for presentation
PRESENTATION_TIME_WINDOWS = [15, 45, 90]  # 3 time windows
PRESENTATION_OUTLIER_METHODS = ['Raw', 'Z-Score (3)', 'Abs (200k)']  # 3 outlier methods


def apply_white_background(fig):
    """Apply white background styling for poster-ready plots"""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', size=12),
        title_font=dict(color='black', size=14),
        xaxis=dict(
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True,
            tickfont=dict(color='black', size=11),
            title_font=dict(color='black', size=12)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True,
            tickfont=dict(color='black', size=11),
            title_font=dict(color='black', size=12)
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        )
    )
    # Also update any colorbar on traces
    fig.update_coloraxes(
        colorbar_tickfont=dict(color='black'),
        colorbar_title_font=dict(color='black')
    )
    return fig


def render_presentation(market_config, raw_ofi_df, split_method, start_datetime, end_datetime):
    """Render the Presentation tab with white-background plots for poster

    Features:
    - White background for all plots (poster-ready)
    - 3×3 configuration subset (15, 45, 90 min × Raw, Abs (200k), Z-Score)
    - Use camera icon on each plot to download as PNG
    """
    st.header("Presentation Mode")
    st.markdown("""
    **Poster-ready plots with white backgrounds**

    Configuration: 3 Time Windows (15, 45, 90 min) × 3 Outlier Methods (Raw, Abs 200k, Z-Score)

    *Use the camera icon (📷) on each plot to download as PNG*
    """)

    # Width control for poster sizing
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        plot_width = st.slider("Plot Width (px)", min_value=300, max_value=1200, value=600, step=50)
    with col_ctrl2:
        st.caption(f"Current width: {plot_width}px - Adjust to fit your poster")

    # Load pre-computed data
    phase_df = None
    ofi_81_df = None
    ti_81_df = None
    overall_df = None

    phase_file = DATA_DIR / "ofi_phase_analysis.csv"
    ofi_81_file = DATA_DIR / "ofi_81_configs.csv"
    ti_81_file = DATA_DIR / "ti_81_configs.csv"
    overall_file = DATA_DIR / "overall_comparison.csv"

    if phase_file.exists():
        phase_df = pd.read_csv(phase_file)
    if ofi_81_file.exists():
        ofi_81_df = pd.read_csv(ofi_81_file)
    if ti_81_file.exists():
        ti_81_df = pd.read_csv(ti_81_file)
    if overall_file.exists():
        overall_df = pd.read_csv(overall_file)

    # Check if data is available
    if ofi_81_df is None or phase_df is None:
        st.error("Data files not found. Run the export script first: `python data_pipeline/06_export_all_analysis.py`")
        return

    # Filter to 3×3 subset (do this outside tabs to ensure variables are in scope)
    ofi_subset = ofi_81_df[
        (ofi_81_df['time_window'].isin(PRESENTATION_TIME_WINDOWS)) &
        (ofi_81_df['outlier_method'].isin(PRESENTATION_OUTLIER_METHODS))
    ].copy()

    phase_subset = phase_df[
        (phase_df['time_window'].isin(PRESENTATION_TIME_WINDOWS)) &
        (phase_df['outlier_method'].isin(PRESENTATION_OUTLIER_METHODS))
    ].copy()

    # Create pivot table for overall OFI (used in multiple tabs)
    pivot_overall = ofi_subset.pivot(index='time_window', columns='outlier_method', values='r_squared')
    pivot_overall = pivot_overall.reindex(columns=PRESENTATION_OUTLIER_METHODS)

    # Create tabs for different visualization sections
    pres_tabs = st.tabs([
        "Price Evolution",
        "OFI Heatmaps",
        "OFI Scatter Plots",
        "TI Comparison",
        "Summary Charts"
    ])

    # ========================================================================
    # TAB 0: Price Evolution (Mid Price over time)
    # ========================================================================
    with pres_tabs[0]:
        st.subheader("Price Evolution")
        st.markdown("*Mid price over the trading period with phase boundaries*")

        # Aggregate to 10-min for cleaner visualization
        price_df = aggregate_ofi_data(raw_ofi_df.copy(), '10min')
        price_df = filter_by_date(price_df, start_datetime, end_datetime)

        if price_df is not None and len(price_df) > 0:
            # Calculate phase boundaries (by snapshot count)
            total = len(price_df)
            third = total // 3
            phase1_end = price_df.iloc[third - 1]['timestamp']
            phase2_end = price_df.iloc[2*third - 1]['timestamp']

            fig_price = go.Figure()

            # Add mid price line
            fig_price.add_trace(go.Scatter(
                x=price_df['timestamp'],
                y=price_df['mid_price'],
                mode='lines',
                line=dict(color='#2E86AB', width=2),
                name='Mid Price'
            ))

            # Add phase boundary lines
            fig_price.add_vline(x=phase1_end, line_dash="dash", line_color="gray", line_width=1)
            fig_price.add_vline(x=phase2_end, line_dash="dash", line_color="gray", line_width=1)

            # Add phase labels
            fig_price.add_annotation(x=price_df.iloc[third//2]['timestamp'], y=price_df['mid_price'].max(),
                                     text="Phase 1<br>(Early)", showarrow=False, font=dict(size=10, color='black'))
            fig_price.add_annotation(x=price_df.iloc[third + third//2]['timestamp'], y=price_df['mid_price'].max(),
                                     text="Phase 2<br>(Middle)", showarrow=False, font=dict(size=10, color='black'))
            fig_price.add_annotation(x=price_df.iloc[2*third + (total-2*third)//2]['timestamp'], y=price_df['mid_price'].max(),
                                     text="Phase 3<br>(Near Expiry)", showarrow=False, font=dict(size=10, color='black'))

            fig_price.update_layout(
                title="Mid Price Over Trading Period",
                xaxis_title="Date",
                yaxis_title="Mid Price ($)",
                height=450,
                showlegend=False
            )
            fig_price = apply_white_background(fig_price)
            fig_price.update_layout(width=plot_width)
            st.plotly_chart(fig_price, use_container_width=False, key="price_evolution")
        else:
            st.warning("No price data available")

    # ========================================================================
    # TAB 1: OFI Heatmaps (Overall + 3 phases) - 3×3 subset
    # ========================================================================
    with pres_tabs[1]:
        st.subheader("OFI R² Heatmaps (3×3 Configuration)")
        st.markdown("*Time Windows: 15, 45, 90 min | Outlier Methods: Raw, Abs (200k), Z-Score*")

        # 1. Overall R² Heatmap
        st.markdown("### 1. Overall Market R²")

        fig_overall = go.Figure(data=go.Heatmap(
            z=pivot_overall.values * 100,
            x=pivot_overall.columns,
            y=[f"{tw} min" for tw in pivot_overall.index],
            colorscale='RdYlGn',
            text=[[f"{v:.1f}%" for v in row] for row in pivot_overall.values * 100],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hovertemplate="Time: %{y}<br>Method: %{x}<br>R²: %{z:.2f}%<extra></extra>",
            colorbar=dict(title=dict(text="R² (%)", font=dict(color='black')), tickfont=dict(color='black'))
        ))
        fig_overall.update_layout(
            title="OFI R² (%) - Overall Market",
            xaxis_title="Outlier Method",
            yaxis_title="Time Window",
            height=350
        )
        fig_overall = apply_white_background(fig_overall)
        fig_overall.update_layout(width=plot_width)
        st.plotly_chart(fig_overall, use_container_width=False, key="overall_heatmap")

        # 2-4. Phase Heatmaps
        phases = ["Phase 1 (Early)", "Phase 2 (Middle)", "Phase 3 (Near Expiry)"]
        phase_titles = ["Early Market (Phase 1)", "Middle Market (Phase 2)", "Near Expiry (Phase 3)"]

        for i, (phase, title) in enumerate(zip(phases, phase_titles)):
            st.markdown(f"### {i+2}. {title} R²")

            phase_data = phase_subset[phase_subset['phase'] == phase]
            if len(phase_data) == 0:
                st.warning(f"No data for {phase}")
                continue

            pivot_phase = phase_data.pivot(index='time_window', columns='outlier_method', values='r_squared')
            pivot_phase = pivot_phase.reindex(columns=PRESENTATION_OUTLIER_METHODS)

            fig_phase = go.Figure(data=go.Heatmap(
                z=pivot_phase.values * 100,
                x=pivot_phase.columns,
                y=[f"{tw} min" for tw in pivot_phase.index],
                colorscale='RdYlGn',
                text=[[f"{v:.1f}%" for v in row] for row in pivot_phase.values * 100],
                texttemplate="%{text}",
                textfont={"size": 14, "color": "black"},
                hovertemplate="Time: %{y}<br>Method: %{x}<br>R²: %{z:.2f}%<extra></extra>",
                colorbar=dict(title=dict(text="R² (%)", font=dict(color='black')), tickfont=dict(color='black'))
            ))
            fig_phase.update_layout(
                title=f"OFI R² (%) - {title}",
                xaxis_title="Outlier Method",
                yaxis_title="Time Window",
                height=350
            )
            fig_phase = apply_white_background(fig_phase)
            fig_phase.update_layout(width=plot_width)
            st.plotly_chart(fig_phase, use_container_width=False, key=f"phase_{i}_heatmap")

    # ========================================================================
    # TAB 2: OFI Scatter Plots (45-min, Z-Score by phase)
    # ========================================================================
    with pres_tabs[2]:
        st.subheader("OFI Price Impact Scatter Plots")
        st.markdown("*Configuration: 45-minute window, Z-Score outlier removal*")

        # Aggregate data for 45-min window
        time_window_str = '45min'
        aggregated_df = aggregate_ofi_data(raw_ofi_df.copy(), time_window_str)
        filtered_ofi = filter_by_date(aggregated_df, start_datetime, end_datetime)

        if filtered_ofi is None or len(filtered_ofi) == 0:
            st.error("No data available for scatter plots")
        else:
            # Apply Z-Score filtering
            filtered_data = filter_outliers_zscore(filtered_ofi, 'ofi', 3)
            dep_var = get_dependent_variable_name()

            # Split into phases
            phases_dict = split_into_phases(filtered_data, split_method)

            # Get phase-specific R² values from pre-computed data
            phase_r2_data = phase_subset[
                (phase_subset['time_window'] == 45) &
                (phase_subset['outlier_method'] == 'Z-Score (3)')
            ]

            colors = ['#2E86AB', '#A23B72', '#F77F00']  # Blue, Purple, Orange
            phase_names = ["Phase 1 (Early)", "Phase 2 (Middle)", "Phase 3 (Near Expiry)"]
            phase_labels = ["Early Market", "Middle Market", "Near Expiry"]

            for i, (phase_name, phase_label, color) in enumerate(zip(phase_names, phase_labels, colors)):
                st.markdown(f"### {phase_label} (45-min, Z-Score)")

                phase_data = phases_dict.get(phase_name)
                if phase_data is None or len(phase_data) < 10:
                    st.warning(f"Insufficient data for {phase_label}")
                    continue

                df_clean = phase_data.dropna(subset=['ofi', dep_var]).copy()

                # Run regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_clean['ofi'], df_clean[dep_var]
                )

                # Create scatter plot
                fig = go.Figure()

                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=df_clean['ofi'],
                    y=df_clean[dep_var],
                    mode='markers',
                    marker=dict(size=6, color=color, opacity=0.6),
                    name='Data Points',
                    hovertemplate='<b>OFI:</b> %{x:.2f}<br><b>ΔPrice:</b> %{y:.4f}<extra></extra>'
                ))

                # Add regression line
                x_range = np.linspace(df_clean['ofi'].min(), df_clean['ofi'].max(), 100)
                y_pred = slope * x_range + intercept
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Regression Line'
                ))

                # Add annotation
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=f'β = {slope:.2e}<br>R² = {r_value**2:.3f}<br>p = {p_value:.2e}<br>n = {len(df_clean)}',
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=11, color='black'),
                    align='left'
                )

                fig.update_layout(
                    title=f"{phase_label}: OFI vs Price Change",
                    xaxis_title="Order Flow Imbalance (OFI)",
                    yaxis_title="Price Change (ticks)",
                    height=450,
                    showlegend=False
                )
                fig = apply_white_background(fig)
                fig.update_layout(width=plot_width)
                st.plotly_chart(fig, use_container_width=False, key=f"scatter_{i}")

            # R² by Phase bar chart
            st.markdown("### R² Progression Across Phases")

            r2_values = []
            for phase_name in phase_names:
                phase_data = phases_dict.get(phase_name)
                if phase_data is not None and len(phase_data) >= 10:
                    df_clean = phase_data.dropna(subset=['ofi', dep_var])
                    _, _, r_value, _, _ = stats.linregress(df_clean['ofi'], df_clean[dep_var])
                    r2_values.append(r_value**2)
                else:
                    r2_values.append(0)

            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=phase_labels,
                y=[r * 100 for r in r2_values],
                marker_color=colors,
                text=[f"{r*100:.1f}%" for r in r2_values],
                textposition='outside',
                textfont=dict(size=14, color='black')
            ))
            fig_r2.update_layout(
                title="R² Increases as Market Approaches Expiry (45-min, Z-Score)",
                xaxis_title="Market Phase",
                yaxis_title="R² (%)",
                height=400,
                showlegend=False
            )
            fig_r2 = apply_white_background(fig_r2)
            fig_r2.update_layout(width=plot_width)
            st.plotly_chart(fig_r2, use_container_width=False, key="r2_progression")

    # ========================================================================
    # TAB 3: TI Comparison
    # ========================================================================
    with pres_tabs[3]:
        st.subheader("Trade Imbalance (TI) Comparison")
        st.markdown("*Comparing OFI vs TI across 3×3 configuration*")

        if ti_81_df is None:
            st.error("TI data not found. Run the export script first.")
        else:
            # Filter TI to 3×3 subset
            ti_subset = ti_81_df[
                (ti_81_df['time_window'].isin(PRESENTATION_TIME_WINDOWS)) &
                (ti_81_df['outlier_method'].isin(PRESENTATION_OUTLIER_METHODS))
            ].copy()

            # TI Heatmap
            st.markdown("### 1. TI R² Heatmap")
            pivot_ti = ti_subset.pivot(index='time_window', columns='outlier_method', values='r_squared')
            pivot_ti = pivot_ti.reindex(columns=PRESENTATION_OUTLIER_METHODS)

            fig_ti = go.Figure(data=go.Heatmap(
                z=pivot_ti.values * 100,
                x=pivot_ti.columns,
                y=[f"{tw} min" for tw in pivot_ti.index],
                colorscale='RdYlGn',
                text=[[f"{v:.2f}%" for v in row] for row in pivot_ti.values * 100],
                texttemplate="%{text}",
                textfont={"size": 14, "color": "black"},
                hovertemplate="Time: %{y}<br>Method: %{x}<br>R²: %{z:.2f}%<extra></extra>",
                colorbar=dict(title=dict(text="R² (%)", font=dict(color='black')), tickfont=dict(color='black'))
            ))
            fig_ti.update_layout(
                title="TI R² (%) - Trade Imbalance vs Price Change",
                xaxis_title="Outlier Method",
                yaxis_title="Time Window",
                height=350
            )
            fig_ti = apply_white_background(fig_ti)
            fig_ti.update_layout(width=plot_width)
            st.plotly_chart(fig_ti, use_container_width=False, key="ti_heatmap")

            # Side-by-side comparison
            st.markdown("### 2. OFI vs TI Side-by-Side")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**OFI R² (0-40% scale)**")
                fig_ofi_side = go.Figure(data=go.Heatmap(
                    z=pivot_overall.values * 100,
                    x=pivot_overall.columns,
                    y=[f"{tw} min" for tw in pivot_overall.index],
                    colorscale='RdYlGn',
                    zmin=0, zmax=40,
                    text=[[f"{v:.1f}%" for v in row] for row in pivot_overall.values * 100],
                    texttemplate="%{text}",
                    textfont={"size": 12, "color": "black"},
                    colorbar=dict(title="R² (%)")
                ))
                fig_ofi_side.update_layout(title="OFI", height=300, width=plot_width//2)
                fig_ofi_side = apply_white_background(fig_ofi_side)
                st.plotly_chart(fig_ofi_side, use_container_width=False, key="ofi_side")

            with col2:
                st.markdown("**TI R² (0-1% scale)**")
                fig_ti_side = go.Figure(data=go.Heatmap(
                    z=pivot_ti.values * 100,
                    x=pivot_ti.columns,
                    y=[f"{tw} min" for tw in pivot_ti.index],
                    colorscale='RdYlGn',
                    zmin=0, zmax=1,
                    text=[[f"{v:.2f}%" for v in row] for row in pivot_ti.values * 100],
                    texttemplate="%{text}",
                    textfont={"size": 12, "color": "black"},
                    colorbar=dict(title="R² (%)")
                ))
                fig_ti_side.update_layout(title="TI", height=300, width=plot_width//2)
                fig_ti_side = apply_white_background(fig_ti_side)
                st.plotly_chart(fig_ti_side, use_container_width=False, key="ti_side")

            st.success("**Key Finding:** OFI R² ranges from 15-35% while TI R² is <1% across all configurations")

    # ========================================================================
    # TAB 4: Summary Charts
    # ========================================================================
    with pres_tabs[4]:
        st.subheader("Summary Comparison Charts")
        st.markdown("*Final comparison: OFI vs TI (45-min, Z-Score configuration)*")

        # Get consistent 45-min Z-Score values from per-config data
        ofi_45_zscore = ofi_81_df[
            (ofi_81_df['time_window'] == 45) &
            (ofi_81_df['outlier_method'] == 'Z-Score (3)')
        ]
        ti_45_zscore = ti_81_df[
            (ti_81_df['time_window'] == 45) &
            (ti_81_df['outlier_method'] == 'Z-Score (3)')
        ] if ti_81_df is not None else None

        if len(ofi_45_zscore) > 0:
            ofi_r2 = ofi_45_zscore['r_squared'].values[0]
            ofi_beta = ofi_45_zscore['beta'].values[0]
            ofi_pval = ofi_45_zscore['p_value'].values[0]
            ofi_n = ofi_45_zscore['n_windows'].values[0]

            ti_r2 = ti_45_zscore['r_squared'].values[0] if ti_45_zscore is not None and len(ti_45_zscore) > 0 else 0
            ti_beta = ti_45_zscore['beta'].values[0] if ti_45_zscore is not None and len(ti_45_zscore) > 0 else 0
            ti_pval = ti_45_zscore['p_value'].values[0] if ti_45_zscore is not None and len(ti_45_zscore) > 0 else 1

            # Comparison bar chart
            st.markdown("### R² Comparison (45-min, Z-Score)")

            metrics = ['OFI (signed)', 'TI (signed)']
            r2_values = [ofi_r2, ti_r2]
            colors = ['#2E86AB', '#06A77D']

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=metrics,
                y=[r * 100 for r in r2_values],
                marker_color=colors,
                text=[f"{r*100:.2f}%" for r in r2_values],
                textposition='outside',
                textfont=dict(size=14, color='black')
            ))
            fig_compare.update_layout(
                title="R² Comparison: OFI vs TI (45-min, Z-Score)",
                xaxis_title="Metric",
                yaxis_title="R² (%)",
                height=450,
                showlegend=False
            )
            fig_compare = apply_white_background(fig_compare)
            fig_compare.update_layout(width=plot_width)
            st.plotly_chart(fig_compare, use_container_width=False, key="compare_bar")

            # Results table
            st.markdown("### Detailed Results Table (45-min, Z-Score)")

            results_data = pd.DataFrame({
                'Metric': ['OFI (signed)', 'TI (signed)'],
                'R²': [f"{ofi_r2*100:.2f}%", f"{ti_r2*100:.2f}%"],
                'β': [f"{ofi_beta:.2e}", f"{ti_beta:.2e}"],
                'p-value': [f"{ofi_pval:.2e}" if ofi_pval < 0.001 else f"{ofi_pval:.4f}",
                           f"{ti_pval:.2e}" if ti_pval < 0.001 else f"{ti_pval:.4f}"],
                'n': [int(ofi_n), int(ofi_n)],
                'Model': ['ΔP = α + β×OFI', 'ΔP = α + β×TI']
            })

            st.dataframe(results_data, use_container_width=True, hide_index=True)

            st.success(f"""
            **Key Takeaways (45-min, Z-Score configuration):**
            - **OFI ({ofi_r2*100:.1f}%)** dramatically outperforms TI
            - **TI ({ti_r2*100:.2f}%)** has minimal explanatory power
            - OFI explains **{ofi_r2/ti_r2:.0f}×** more variance than TI
            - OFI captures order book dynamics that trade-based metrics miss
            """)
        else:
            st.warning("Overall comparison data not found. Run the export script first.")


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
        page_options = ["OFI Analysis", "Depth Analysis", "TI vs OFI Comparison", "Presentation"]
        selected_page = st.radio(
            "Select Analysis Type",
            options=page_options,
            index=0,
            help="OFI: 243 regressions. Depth: β vs Market Depth. TI: Trade Imbalance vs OFI. Presentation: Poster-ready plots."
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

    elif selected_page == "Presentation":
        # Render Presentation tab (poster-ready plots)
        render_presentation(market_config, raw_ofi_df, split_method, start_datetime, end_datetime)

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
            overall_data = []  # For overall R² (no phase split)
            with st.spinner("Calculating regressions across all time windows..."):
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
                        if len(df_clean) < 10:
                            continue

                        short_name = method_name.replace('Filtered', '').replace('Trimmed', '').replace('Data', '').strip()

                        # Overall R² (no phase split) - same as TI vs OFI comparison
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            df_clean['ofi'], df_clean[dep_var]
                        )
                        overall_data.append({
                            'TimeWindow': tw,
                            'Method': short_name,
                            'N': len(df_clean),
                            'Beta': slope,
                            'R2': r_value**2,
                            'p-value': p_value
                        })

                        # Phase-by-phase data (for tables 2-4)
                        if len(df_clean) >= 150:
                            phases_dict = split_into_phases(df_clean, split_method)
                            phases = {
                                'P1 (Early)': phases_dict['Phase 1 (Early)'],
                                'P2 (Middle)': phases_dict['Phase 2 (Middle)'],
                                'P3 (Near Expiry)': phases_dict['Phase 3 (Near Expiry)']
                            }

                            for phase_name, phase_df in phases.items():
                                if len(phase_df) > 2:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        phase_df['ofi'], phase_df[dep_var]
                                    )
                                    all_summary_data.append({
                                        'TimeWindow': tw,
                                        'Method': short_name,
                                        'Phase': phase_name,
                                        'N': len(phase_df),
                                        'Beta': slope,
                                        'R2': r_value**2,
                                        'p-value': p_value
                                    })

            if overall_data:
                overall_df = pd.DataFrame(overall_data)
                master_df = pd.DataFrame(all_summary_data) if all_summary_data else None

                # Table 1: Overall R² (no phase split - consistent with TI vs OFI comparison)
                st.markdown("### 1. Overall R² (All Data Combined)")
                st.caption("*Same methodology as TI vs OFI Comparison tab*")
                pivot_overall = overall_df.pivot(index='TimeWindow', columns='Method', values='R2')
                st.dataframe(
                    pivot_overall.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None),
                    use_container_width=True,
                    height=380
                )

                # Find best overall
                best_idx = overall_df['R2'].idxmax()
                best_row = overall_df.loc[best_idx]
                st.success(f"**Best Overall**: {best_row['TimeWindow']} min + {best_row['Method']} (R² = {best_row['R2']:.4f})")

                # Table 2: Phase 1 (Early) only
                st.markdown("### 2. Phase 1 (Early Market) R²")
                if master_df is not None and len(master_df) > 0:
                    p1_data = master_df[master_df['Phase'] == 'P1 (Early)']
                else:
                    p1_data = pd.DataFrame()
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
                if master_df is not None and len(master_df) > 0:
                    p2_data = master_df[master_df['Phase'] == 'P2 (Middle)']
                else:
                    p2_data = pd.DataFrame()
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
                if master_df is not None and len(master_df) > 0:
                    p3_data = master_df[master_df['Phase'] == 'P3 (Near Expiry)']
                else:
                    p3_data = pd.DataFrame()
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
                if master_df is not None and len(master_df) > 0:
                    csv = master_df.to_csv(index=False)
                    st.download_button(
                        label="Download Phase Results CSV",
                        data=csv,
                        file_name="ofi_phase_analysis.csv",
                        mime="text/csv"
                    )
                csv_overall = overall_df.to_csv(index=False)
                st.download_button(
                    label="Download Overall Results CSV",
                    data=csv_overall,
                    file_name="ofi_overall_analysis.csv",
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
