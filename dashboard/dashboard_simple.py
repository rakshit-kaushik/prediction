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
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")

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
        "description": "Will Zohran Mamdani win the 2025 NYC mayoral election?",
        "date_range": ("2025-10-15", "2025-11-04")
    }
}

# ============================================================================
# TIME AGGREGATION (Per Cont et al. 2011)
# ============================================================================

def aggregate_to_10_minutes(df):
    """
    Aggregate raw orderbook snapshots to 10-minute intervals following
    Cont et al. (2011) methodology.

    This matches the aggregation used in all analysis scripts.
    """
    if df is None or len(df) == 0:
        return df

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

    # Create 10-minute bins
    df['time_bin'] = df['timestamp'].dt.floor('10T')

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


def plot_ofi_analysis(df):
    """Plot OFI vs price change"""
    # Filter non-zero OFI
    df_nonzero = df[(df['ofi'] != 0) | (df['delta_mid_price'] != 0)].copy()

    if len(df_nonzero) == 0:
        return None

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df_nonzero['ofi'],
        y=df_nonzero['delta_mid_price'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_nonzero['delta_mid_price'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="ΔP"),
            opacity=0.6
        ),
        hovertemplate='<b>OFI:</b> %{x:.2f}<br><b>ΔP:</b> %{y:.6f}<extra></extra>'
    ))

    # Add regression line
    from scipy import stats
    if len(df_nonzero) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_nonzero['ofi'], df_nonzero['delta_mid_price']
        )

        x_range = np.array([df_nonzero['ofi'].min(), df_nonzero['ofi'].max()])
        y_pred = slope * x_range + intercept

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'Fit: R²={r_value**2:.4f}',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Regression Line</b><extra></extra>'
        ))

        # Add annotation
        fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=f'R² = {r_value**2:.4f}<br>p-value = {p_value:.2e}',
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )

    fig.update_layout(
        title="Order Flow Imbalance vs Price Change",
        xaxis_title="OFI",
        yaxis_title="ΔP (Price Change)",
        hovermode='closest',
        height=500
    )

    return fig


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
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="OFI Analysis Dashboard",
        layout="wide"
    )

    st.title("OFI Analysis Dashboard")
    st.markdown("**Interactive data exploration for Polymarket order flow analysis**")

    # Sidebar - Market Selection
    with st.sidebar:
        st.header("Market Selection")

        # Market dropdown
        selected_market = st.selectbox(
            "Choose Market",
            options=list(MARKETS.keys()),
            help="Select from pre-configured markets"
        )

        market_config = MARKETS[selected_market]
        st.info(f"**{market_config['description']}**")

        st.divider()

        # Load data
        with st.spinner("Loading data..."):
            ofi_df, orderbook_df, market_info = load_market_data(selected_market)

        # Check if data loaded
        if ofi_df is None or len(ofi_df) == 0:
            st.error("No data found for this market")
            st.stop()

        # Apply 10-minute aggregation (per Cont et al. 2011)
        with st.spinner("Aggregating to 10-minute intervals..."):
            raw_count = len(ofi_df)
            ofi_df = aggregate_to_10_minutes(ofi_df)
            st.success(f"✓ Aggregated {raw_count:,} snapshots → {len(ofi_df):,} 10-minute intervals")

        # Date range sliders
        st.header("Date/Time Filter")

        min_date = ofi_df['timestamp'].min()
        max_date = ofi_df['timestamp'].max()

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

        # Validation: ensure start is before end
        if start_datetime >= end_datetime:
            st.error("Start date/time must be before end date/time")
            st.stop()

        # Clamp to data boundaries
        start_datetime = max(start_datetime, min_date.replace(tzinfo=None))
        end_datetime = min(end_datetime, max_date.replace(tzinfo=None))

        st.divider()

        # Data info
        st.header("Data Info")
        st.metric("Total Snapshots", f"{len(ofi_df):,}")
        st.metric("Full Date Range", f"{(max_date - min_date).days} days")

    # Filter data by selected date range
    filtered_ofi = filter_by_date(ofi_df, start_datetime, end_datetime)
    filtered_orderbook = filter_by_date(orderbook_df, start_datetime, end_datetime)

    if filtered_ofi is None or len(filtered_ofi) == 0:
        st.warning("No data in selected date range. Please adjust filters.")
        st.stop()

    # Main content area - tabs
    tab1, tab2 = st.tabs([
        "📈 Price & Depth",
        "📊 OFI Analysis"
    ])

    with tab1:
        st.subheader("Price Evolution")
        fig_price = plot_price_evolution(filtered_ofi)
        st.plotly_chart(fig_price, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Spread Analysis")
            fig_spread = plot_spread_analysis(filtered_ofi)
            st.plotly_chart(fig_spread, use_container_width=True)

        with col2:
            st.subheader("Order Book Depth")
            fig_depth = plot_depth_evolution(filtered_ofi)
            st.plotly_chart(fig_depth, use_container_width=True)

    with tab2:
        st.subheader("OFI vs Price Change")
        fig_ofi = plot_ofi_analysis(filtered_ofi)
        if fig_ofi:
            st.plotly_chart(fig_ofi, use_container_width=True)
        else:
            st.info("No OFI data available for selected period")

        st.subheader("OFI Distribution")
        fig_ofi_dist = plot_ofi_distribution(filtered_ofi)
        st.plotly_chart(fig_ofi_dist, use_container_width=True)


if __name__ == "__main__":
    main()
