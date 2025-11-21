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
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

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
# RESEARCH DATA LOADING (NEW!)
# ============================================================================

@st.cache_data
def load_research_tables():
    """Load all research tables"""
    tables = {}
    table_files = {
        'regression': 'table_2_regression_statistics.csv',
        'depth': 'table_3_depth_analysis.csv',
        'variance': 'table_4_variance_decomposition.csv',
        'ti_comparison': 'table_5_ofi_ti_comparison.csv'
    }

    for key, filename in table_files.items():
        file_path = TABLES_DIR / filename
        if file_path.exists():
            tables[key] = pd.read_csv(file_path)
        else:
            tables[key] = None

    return tables

@st.cache_data
def load_figure_image(filename):
    """Load a figure image"""
    file_path = FIGURES_DIR / filename
    if file_path.exists():
        return Image.open(file_path)
    return None


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


def plot_intraday_patterns(df):
    """Plot intraday OFI patterns by hour"""
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.hour

    hourly_stats = df_copy.groupby('hour').agg({
        'ofi': ['mean', 'std', 'count'],
        'delta_mid_price': ['mean', 'std']
    }).reset_index()

    hourly_stats.columns = ['hour', 'ofi_mean', 'ofi_std', 'ofi_count',
                            'price_change_mean', 'price_change_std']

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average OFI by Hour', 'Average Price Change by Hour'),
        vertical_spacing=0.15
    )

    # OFI by hour
    fig.add_trace(
        go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['ofi_mean'],
            name='Avg OFI',
            marker=dict(color='#2E86AB'),
            error_y=dict(type='data', array=hourly_stats['ofi_std']),
            hovertemplate='<b>Hour:</b> %{x}<br><b>Avg OFI:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Price change by hour
    fig.add_trace(
        go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['price_change_mean'],
            name='Avg ΔP',
            marker=dict(color='#06A77D'),
            error_y=dict(type='data', array=hourly_stats['price_change_std']),
            hovertemplate='<b>Hour:</b> %{x}<br><b>Avg ΔP:</b> %{y:.6f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Hour of Day (UTC)", row=2, col=1)
    fig.update_yaxes(title_text="OFI", row=1, col=1)
    fig.update_yaxes(title_text="ΔP", row=2, col=1)

    fig.update_layout(
        title="Intraday Patterns",
        height=500,
        showlegend=False
    )

    return fig


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def calculate_summary_stats(ofi_df, orderbook_df):
    """Calculate summary statistics for selected period"""
    stats = {}

    if ofi_df is not None and len(ofi_df) > 0:
        stats['Total Snapshots'] = len(ofi_df)
        stats['Date Range'] = f"{ofi_df['timestamp'].min().strftime('%Y-%m-%d')} to {ofi_df['timestamp'].max().strftime('%Y-%m-%d')}"
        stats['Duration (days)'] = (ofi_df['timestamp'].max() - ofi_df['timestamp'].min()).days

        # Price stats
        stats['Price Range'] = f"${ofi_df['mid_price'].min():.4f} - ${ofi_df['mid_price'].max():.4f}"
        stats['Avg Price'] = f"${ofi_df['mid_price'].mean():.4f}"
        stats['Price Std Dev'] = f"${ofi_df['mid_price'].std():.4f}"

        # OFI stats
        ofi_nonzero = ofi_df[ofi_df['ofi'] != 0]
        stats['Non-zero OFI Events'] = f"{len(ofi_nonzero):,} ({len(ofi_nonzero)/len(ofi_df)*100:.1f}%)"
        stats['OFI Range'] = f"{ofi_df['ofi'].min():,.2f} to {ofi_df['ofi'].max():,.2f}"
        stats['Mean OFI'] = f"{ofi_df['ofi'].mean():.2f}"

        # Spread stats
        stats['Avg Spread'] = f"{ofi_df['spread'].mean():.4f} ({ofi_df['spread_pct'].mean():.2f}%)"
        stats['Spread Range'] = f"{ofi_df['spread_pct'].min():.2f}% - {ofi_df['spread_pct'].max():.2f}%"

        # Depth stats
        stats['Avg Total Depth'] = f"{ofi_df['total_depth'].mean():,.0f}"
        stats['Avg Bid Depth'] = f"{ofi_df['total_bid_size'].mean():,.0f}"
        stats['Avg Ask Depth'] = f"{ofi_df['total_ask_size'].mean():,.0f}"

        # Correlation
        if len(ofi_nonzero) > 2:
            from scipy import stats as sp_stats
            correlation = ofi_nonzero['ofi'].corr(ofi_nonzero['delta_mid_price'])
            result = sp_stats.linregress(ofi_nonzero['ofi'], ofi_nonzero['delta_mid_price'])
            p_value = result.pvalue
            stats['OFI-Price Correlation'] = f"{correlation:.4f} (p={p_value:.2e})"

    return stats


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="OFI Analysis Dashboard - Cont et al. (2011) Replication",
        layout="wide"
    )

    st.title("📊 OFI Analysis Dashboard")
    st.markdown("**Data exploration + Complete Cont et al. (2011) replication results**")

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Price & Depth",
        "OFI Analysis",
        "Intraday Patterns",
        "Summary Stats",
        "📊 Research Results",
        "📉 Depth & Events",
        "🖼️ Figures Gallery"
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

    with tab3:
        st.subheader("Hourly Patterns")
        fig_intraday = plot_intraday_patterns(filtered_ofi)
        st.plotly_chart(fig_intraday, use_container_width=True)

    with tab4:
        st.subheader("Summary Statistics")

        stats = calculate_summary_stats(filtered_ofi, filtered_orderbook)

        # Display in columns
        cols = st.columns(3)

        stat_items = list(stats.items())
        for idx, (key, value) in enumerate(stat_items):
            with cols[idx % 3]:
                st.metric(key, value)

        # Raw data preview
        with st.expander("View Raw Data"):
            st.dataframe(filtered_ofi.head(100), use_container_width=True)

    # ========================================================================
    # NEW RESEARCH RESULTS TAB
    # ========================================================================

    with tab5:
        st.subheader("📊 Research Results - Cont et al. (2011) Replication")

        st.info("""
        **Complete replication of "The Price Impact of Order Book Events"**

        This tab shows all results from our paper replication analysis.
        Run `python scripts/run_all_analyses.py` to generate all results.
        """)

        # Load all research tables
        research_tables = load_research_tables()

        # Display tables in expanders
        with st.expander("📈 Table 2: Regression Analysis", expanded=True):
            if research_tables['regression'] is not None:
                st.markdown("**Linear Model:** ΔP = α + β × OFI + ε")
                st.dataframe(research_tables['regression'], use_container_width=True, hide_index=True)

                col1, col2, col3 = st.columns(3)
                df = research_tables['regression']
                with col1:
                    st.metric("Avg R²", f"{df['R_squared'].mean():.4f}")
                with col2:
                    st.metric("Avg Correlation", f"{df['Correlation'].mean():.4f}")
                with col3:
                    sig_count = (df['p_Beta'] < 0.05).sum()
                    st.metric("Significant Markets", f"{sig_count}/{len(df)}")
            else:
                st.warning("Run: `python scripts/01_regression_analysis.py`")

        with st.expander("⚖️ Table 5: OFI vs Trade Imbalance"):
            if research_tables['ti_comparison'] is not None:
                st.markdown("**Horse-Race Regressions:** Which predicts price better?")
                st.dataframe(research_tables['ti_comparison'], use_container_width=True, hide_index=True)

                df = research_tables['ti_comparison']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg OFI R²", f"{df['R2_OFI'].mean():.4f}")
                with col2:
                    st.metric("Avg TI R²", f"{df['R2_TI'].mean():.4f}")
                with col3:
                    improvement = df['R2_Improvement'].mean() * 100
                    st.metric("TI Adds", f"{improvement:.2f}%")

                st.success("✅ **Result:** OFI dominates trade imbalance (consistent with paper)")
            else:
                st.warning("Run: `python scripts/06_trade_volume_analysis.py`")

        # Show scatter plot
        st.subheader("OFI vs Price Change")
        scatter_img = load_figure_image("figure_2_combined_comparison.png")
        if scatter_img:
            st.image(scatter_img, caption="Figure 2: OFI vs Price Change Scatter Plots", use_column_width=True)

    # ========================================================================
    # DEPTH & EVENTS TAB
    # ========================================================================

    with tab6:
        st.subheader("📉 Depth Analysis & Event Patterns")

        research_tables = load_research_tables()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Table 3: Depth Analysis")
            if research_tables['depth'] is not None:
                st.markdown("**Power Law:** β = a / AD^λ")
                st.dataframe(research_tables['depth'], use_container_width=True, hide_index=True)

                df = research_tables['depth']
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Avg λ", f"{df['Power_Law_lambda'].mean():.3f}", help="Paper: ~1.0")
                with m2:
                    close_to_one = (np.abs(df['Power_Law_lambda'] - 1.0) < 0.3).sum()
                    st.metric("Close to Paper", f"{close_to_one}/{len(df)}")

                st.info("⚠️ Prediction markets show weaker depth relationship than equities")
            else:
                st.warning("Run: `python scripts/04_depth_analysis.py`")

        with col2:
            st.markdown("### Table 4: Variance Decomposition")
            if research_tables['variance'] is not None:
                st.markdown("**How much price variance is explained by OFI?**")
                st.dataframe(research_tables['variance'], use_container_width=True, hide_index=True)

                df = research_tables['variance']
                avg_explained = df['Pct_Explained'].mean()
                st.metric("Avg Variance Explained", f"{avg_explained:.1f}%")

                st.warning(f"""
                📊 **Key Finding:** OFI explains only **{avg_explained:.1f}%** of price variance
                (vs **65%** in equity markets).

                → Prediction markets are more **news-driven** than orderbook-driven.
                """)
            else:
                st.warning("Run: `python scripts/05_event_analysis.py`")

        # Show depth figure
        st.subheader("Depth Analysis Visualization")
        depth_img = load_figure_image("figure_4_depth_analysis.png")
        if depth_img:
            st.image(depth_img, caption="Figure 4: β vs Average Depth (Power Law)", use_column_width=True)

        # Show event analysis
        st.subheader("Event Pattern Analysis")
        event_img = load_figure_image("figure_5_event_analysis.png")
        if event_img:
            st.image(event_img, caption="Figure 5: Orderbook Event Patterns", use_column_width=True)

    # ========================================================================
    # FIGURES GALLERY TAB
    # ========================================================================

    with tab7:
        st.subheader("🖼️ Publication Figures Gallery")

        st.markdown("""
        All publication-quality figures (300 DPI PNG + PDF vectors).
        Download from `results/figures/` directory.
        """)

        figures = [
            ("figure_2_combined_comparison.png", "Figure 2: OFI vs Price Change"),
            ("figure_3_Fed_residual_diagnostics.png", "Figure 3a: Fed Residual Diagnostics"),
            ("figure_3_NYC_residual_diagnostics.png", "Figure 3b: NYC Residual Diagnostics"),
            ("figure_4_depth_analysis.png", "Figure 4: Depth Analysis"),
            ("figure_5_event_analysis.png", "Figure 5: Event Patterns"),
            ("figure_6_ofi_ti_comparison.png", "Figure 6: OFI vs TI Comparison"),
        ]

        for filename, caption in figures:
            with st.expander(caption, expanded=False):
                img = load_figure_image(filename)
                if img:
                    st.image(img, caption=caption, use_column_width=True)
                    st.caption(f"📁 File: `results/figures/{filename}`")
                else:
                    st.warning(f"Figure not found. Run: `python scripts/run_all_analyses.py`")


if __name__ == "__main__":
    main()
