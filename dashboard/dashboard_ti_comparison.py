"""
dashboard_ti_comparison.py
==========================
Trade Imbalance vs OFI Comparison Dashboard

Following Cont et al. (2011) Section 3.3:
- Compare OFI vs Trade Imbalance (TI) in predicting price changes
- TI = Sum(signed trade volume) = Sum(buy_volume) - Sum(sell_volume)
- OFI captures queue dynamics that TI misses

USAGE:
------
    streamlit run dashboard/dashboard_ti_comparison.py

Then open browser to http://localhost:8506
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

# Time windows to analyze (in minutes)
TIME_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60, 90]

# Market configuration
MARKET_CONFIG = {
    "NYC Mayoral Election 2025": {
        "ofi_file": "nyc_mayor_oct15_nov04_ofi.csv",
        "trades_file": "nyc_mayor_oct15_nov04_trades_processed.csv",
        "description": "Will Zohran Mamdani win the 2025 NYC mayoral election?"
    }
}

TICK_SIZE = 0.01

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_ofi_data(ofi_file):
    """Load OFI data from CSV"""
    file_path = DATA_DIR / ofi_file
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    return df


@st.cache_data
def load_trades_data(trades_file):
    """Load processed trades data from CSV"""
    file_path = DATA_DIR / trades_file
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    return df


def aggregate_ofi_by_time_window(df, time_window_minutes):
    """
    Aggregate OFI data by time window

    Args:
        df: Raw OFI DataFrame
        time_window_minutes: Window size in minutes

    Returns:
        Aggregated DataFrame with OFI sum and price change
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create time bins
    tw_str = f'{time_window_minutes}min'
    df['time_bin'] = df['timestamp'].dt.floor(tw_str)

    # Aggregate
    agg = df.groupby('time_bin').agg({
        'ofi': 'sum',
        'mid_price': ['first', 'last']
    }).reset_index()

    # Flatten columns
    agg.columns = ['timestamp', 'ofi', 'mid_price_first', 'mid_price_last']

    # Calculate price change (between windows)
    agg['mid_price'] = agg['mid_price_last']
    agg['delta_mid_price'] = agg['mid_price'].diff()
    agg['delta_mid_price_ticks'] = agg['delta_mid_price'] / TICK_SIZE

    # Drop first row (no previous for diff)
    agg = agg.iloc[1:].reset_index(drop=True)

    return agg


def aggregate_trades_by_time_window(df, time_window_minutes):
    """
    Aggregate trade data to calculate Trade Imbalance (TI)

    TI = Sum(buy_volume) - Sum(sell_volume)
       = Sum(direction * shares_normalized)

    Args:
        df: Trades DataFrame with direction and shares_normalized
        time_window_minutes: Window size in minutes

    Returns:
        DataFrame with TI and other trade metrics
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create time bins
    tw_str = f'{time_window_minutes}min'
    df['time_bin'] = df['timestamp'].dt.floor(tw_str)

    # Calculate signed volume for each trade
    df['signed_volume'] = df['direction'] * df['shares_normalized']

    # Aggregate
    agg = df.groupby('time_bin').agg({
        'signed_volume': 'sum',  # Trade Imbalance
        'shares_normalized': 'sum',  # Total volume
        'direction': 'count',  # Number of trades
        'price': ['mean', 'first', 'last']
    }).reset_index()

    # Flatten columns
    agg.columns = ['timestamp', 'trade_imbalance', 'total_volume', 'num_trades',
                   'avg_price', 'first_price', 'last_price']

    return agg


def merge_ofi_and_ti(ofi_df, ti_df):
    """
    Merge OFI and Trade Imbalance data on timestamp

    Args:
        ofi_df: Aggregated OFI DataFrame
        ti_df: Aggregated Trade DataFrame with TI

    Returns:
        Merged DataFrame
    """
    if ofi_df is None or ti_df is None:
        return None

    # Merge on timestamp
    merged = pd.merge(ofi_df, ti_df, on='timestamp', how='inner')

    # Drop rows with NaN
    merged = merged.dropna(subset=['ofi', 'trade_imbalance', 'delta_mid_price_ticks'])

    return merged


def run_regression(x, y):
    """
    Run OLS regression and return statistics

    Returns:
        dict with slope, intercept, r_squared, p_value, std_err, n_obs
    """
    if len(x) < 3:
        return None

    # Remove NaN
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
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'n_obs': len(x_clean)
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_scatter(merged_df, time_window):
    """Create side-by-side scatter plots comparing OFI and TI"""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"OFI vs Price Change ({time_window} min)",
            f"Trade Imbalance vs Price Change ({time_window} min)"
        ]
    )

    # OFI scatter
    fig.add_trace(
        go.Scatter(
            x=merged_df['ofi'],
            y=merged_df['delta_mid_price_ticks'],
            mode='markers',
            marker=dict(color='#2E86AB', size=6, opacity=0.6),
            name='OFI'
        ),
        row=1, col=1
    )

    # OFI regression line
    ofi_reg = run_regression(merged_df['ofi'].values, merged_df['delta_mid_price_ticks'].values)
    if ofi_reg:
        x_line = np.array([merged_df['ofi'].min(), merged_df['ofi'].max()])
        y_line = ofi_reg['slope'] * x_line + ofi_reg['intercept']
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', width=2),
                name=f"OFI: R²={ofi_reg['r_squared']:.4f}"
            ),
            row=1, col=1
        )

    # TI scatter
    fig.add_trace(
        go.Scatter(
            x=merged_df['trade_imbalance'],
            y=merged_df['delta_mid_price_ticks'],
            mode='markers',
            marker=dict(color='#06A77D', size=6, opacity=0.6),
            name='TI'
        ),
        row=1, col=2
    )

    # TI regression line
    ti_reg = run_regression(merged_df['trade_imbalance'].values, merged_df['delta_mid_price_ticks'].values)
    if ti_reg:
        x_line = np.array([merged_df['trade_imbalance'].min(), merged_df['trade_imbalance'].max()])
        y_line = ti_reg['slope'] * x_line + ti_reg['intercept']
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='red', width=2),
                name=f"TI: R²={ti_reg['r_squared']:.4f}"
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="OFI", row=1, col=1)
    fig.update_xaxes(title_text="Trade Imbalance", row=1, col=2)
    fig.update_yaxes(title_text="Price Change (ticks)", row=1, col=1)
    fig.update_yaxes(title_text="Price Change (ticks)", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)

    return fig, ofi_reg, ti_reg


def create_r_squared_comparison_chart(results_df):
    """Create bar chart comparing R² values across time windows"""

    fig = go.Figure()

    # OFI bars
    fig.add_trace(go.Bar(
        x=[f"{tw} min" for tw in results_df['time_window']],
        y=results_df['ofi_r_squared'],
        name='OFI',
        marker_color='#2E86AB',
        text=[f"{r:.4f}" for r in results_df['ofi_r_squared']],
        textposition='outside'
    ))

    # TI bars
    fig.add_trace(go.Bar(
        x=[f"{tw} min" for tw in results_df['time_window']],
        y=results_df['ti_r_squared'],
        name='Trade Imbalance',
        marker_color='#06A77D',
        text=[f"{r:.4f}" for r in results_df['ti_r_squared']],
        textposition='outside'
    ))

    fig.update_layout(
        title="R² Comparison: OFI vs Trade Imbalance",
        xaxis_title="Time Window",
        yaxis_title="R²",
        barmode='group',
        height=400
    )

    return fig


def create_improvement_chart(results_df):
    """Create chart showing OFI improvement over TI"""

    # Calculate improvement ratio
    results_df = results_df.copy()
    results_df['improvement'] = (results_df['ofi_r_squared'] - results_df['ti_r_squared']) / results_df['ti_r_squared'] * 100
    results_df['improvement'] = results_df['improvement'].replace([np.inf, -np.inf], np.nan)

    fig = go.Figure()

    colors = ['#06A77D' if x > 0 else '#A23B72' for x in results_df['improvement']]

    fig.add_trace(go.Bar(
        x=[f"{tw} min" for tw in results_df['time_window']],
        y=results_df['improvement'],
        marker_color=colors,
        text=[f"{x:+.1f}%" if not pd.isna(x) else "N/A" for x in results_df['improvement']],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="OFI Improvement over Trade Imbalance (% change in R²)",
        xaxis_title="Time Window",
        yaxis_title="Improvement (%)",
        height=400
    )

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="TI vs OFI Comparison",
        page_icon="📊",
        layout="wide"
    )

    st.title("Trade Imbalance vs OFI Comparison")
    st.markdown("""
    **Cont et al. (2011) Section 3.3**: Comparing Order Flow Imbalance (OFI) with Trade Imbalance (TI)
    in predicting price changes.

    - **OFI** = Order flow imbalance from orderbook changes
    - **TI** = Trade Imbalance = Σ(buy_volume) - Σ(sell_volume)

    The paper found OFI outperforms TI because OFI captures queue dynamics that TI misses.
    """)

    # Load data
    market_key = "NYC Mayoral Election 2025"
    config = MARKET_CONFIG[market_key]

    with st.spinner("Loading data..."):
        ofi_raw = load_ofi_data(config["ofi_file"])
        trades_raw = load_trades_data(config["trades_file"])

    if ofi_raw is None:
        st.error(f"OFI data not found: {config['ofi_file']}")
        return

    if trades_raw is None:
        st.error(f"Trades data not found: {config['trades_file']}")
        return

    st.success(f"Loaded {len(ofi_raw):,} OFI records and {len(trades_raw):,} trades")

    # Sidebar
    st.sidebar.header("Settings")
    st.sidebar.markdown(f"**Market**: {market_key}")
    st.sidebar.markdown(f"**OFI Records**: {len(ofi_raw):,}")
    st.sidebar.markdown(f"**Trades**: {len(trades_raw):,}")

    # Summary stats
    buy_trades = trades_raw[trades_raw['side'] == 'BUY']
    sell_trades = trades_raw[trades_raw['side'] == 'SELL']
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Trade Statistics**")
    st.sidebar.markdown(f"Buy trades: {len(buy_trades):,} ({len(buy_trades)/len(trades_raw)*100:.1f}%)")
    st.sidebar.markdown(f"Sell trades: {len(sell_trades):,} ({len(sell_trades)/len(trades_raw)*100:.1f}%)")

    net_imbalance = trades_raw['shares_normalized'].sum() * (len(buy_trades) - len(sell_trades)) / len(trades_raw)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Time Window Analysis", "Volume Analysis", "Raw Data"])

    with tab1:
        st.header("Summary: OFI vs Trade Imbalance")

        # Calculate results for all time windows
        results = []

        for tw in TIME_WINDOWS:
            ofi_agg = aggregate_ofi_by_time_window(ofi_raw, tw)
            ti_agg = aggregate_trades_by_time_window(trades_raw, tw)
            merged = merge_ofi_and_ti(ofi_agg, ti_agg)

            if merged is not None and len(merged) >= 10:
                ofi_reg = run_regression(merged['ofi'].values, merged['delta_mid_price_ticks'].values)
                ti_reg = run_regression(merged['trade_imbalance'].values, merged['delta_mid_price_ticks'].values)

                if ofi_reg and ti_reg:
                    results.append({
                        'time_window': tw,
                        'n_obs': ofi_reg['n_obs'],
                        'ofi_r_squared': ofi_reg['r_squared'],
                        'ofi_slope': ofi_reg['slope'],
                        'ofi_p_value': ofi_reg['p_value'],
                        'ti_r_squared': ti_reg['r_squared'],
                        'ti_slope': ti_reg['slope'],
                        'ti_p_value': ti_reg['p_value']
                    })

        if not results:
            st.error("Not enough data to run comparison. Need overlapping OFI and trade data.")
            return

        results_df = pd.DataFrame(results)

        # R² comparison chart
        st.subheader("R² Comparison Across Time Windows")
        fig_comparison = create_r_squared_comparison_chart(results_df)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Improvement chart
        st.subheader("OFI Improvement Over Trade Imbalance")
        fig_improvement = create_improvement_chart(results_df)
        st.plotly_chart(fig_improvement, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")

        # Calculate averages
        avg_ofi_r2 = results_df['ofi_r_squared'].mean()
        avg_ti_r2 = results_df['ti_r_squared'].mean()
        ofi_wins = (results_df['ofi_r_squared'] > results_df['ti_r_squared']).sum()
        ti_wins = (results_df['ti_r_squared'] > results_df['ofi_r_squared']).sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg OFI R²", f"{avg_ofi_r2:.4f}")
        with col2:
            st.metric("Avg TI R²", f"{avg_ti_r2:.4f}")
        with col3:
            winner = "OFI" if avg_ofi_r2 > avg_ti_r2 else "TI"
            st.metric("Winner (by avg R²)", winner)
        with col4:
            st.metric("OFI wins / TI wins", f"{ofi_wins} / {ti_wins}")

        # Detailed results table
        st.subheader("Detailed Results")
        display_df = results_df.copy()
        display_df['winner'] = display_df.apply(
            lambda r: 'OFI' if r['ofi_r_squared'] > r['ti_r_squared'] else 'TI', axis=1
        )
        display_df['improvement_%'] = ((display_df['ofi_r_squared'] - display_df['ti_r_squared']) /
                                        display_df['ti_r_squared'] * 100).round(2)

        st.dataframe(
            display_df[[
                'time_window', 'n_obs',
                'ofi_r_squared', 'ti_r_squared',
                'winner', 'improvement_%'
            ]].rename(columns={
                'time_window': 'Window (min)',
                'n_obs': 'Observations',
                'ofi_r_squared': 'OFI R²',
                'ti_r_squared': 'TI R²',
                'winner': 'Winner',
                'improvement_%': 'OFI Improvement %'
            }),
            use_container_width=True
        )

        # Key findings
        st.subheader("Key Findings")

        if avg_ofi_r2 > avg_ti_r2:
            st.success(f"""
            **OFI outperforms Trade Imbalance** (consistent with Cont et al. 2011)

            - Average OFI R²: {avg_ofi_r2:.4f}
            - Average TI R²: {avg_ti_r2:.4f}
            - OFI wins in {ofi_wins}/{len(results_df)} time windows

            This suggests OFI captures market dynamics (queue changes) that simple
            trade imbalance misses.
            """)
        else:
            st.warning(f"""
            **Trade Imbalance outperforms OFI** (contrary to Cont et al. 2011)

            - Average OFI R²: {avg_ofi_r2:.4f}
            - Average TI R²: {avg_ti_r2:.4f}
            - TI wins in {ti_wins}/{len(results_df)} time windows

            This may indicate unique characteristics of this prediction market.
            """)

    with tab2:
        st.header("Time Window Analysis")

        # Time window selector
        selected_tw = st.selectbox(
            "Select Time Window (minutes)",
            TIME_WINDOWS,
            index=TIME_WINDOWS.index(10) if 10 in TIME_WINDOWS else 0
        )

        # Aggregate data
        ofi_agg = aggregate_ofi_by_time_window(ofi_raw, selected_tw)
        ti_agg = aggregate_trades_by_time_window(trades_raw, selected_tw)
        merged = merge_ofi_and_ti(ofi_agg, ti_agg)

        if merged is None or len(merged) < 10:
            st.warning("Not enough overlapping data for this time window")
        else:
            st.success(f"Merged data: {len(merged)} observations")

            # Scatter plots
            fig_scatter, ofi_reg, ti_reg = create_comparison_scatter(merged, selected_tw)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Regression statistics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("OFI Regression")
                if ofi_reg:
                    st.write(f"- **R²**: {ofi_reg['r_squared']:.6f}")
                    st.write(f"- **Slope (β)**: {ofi_reg['slope']:.6e}")
                    st.write(f"- **p-value**: {ofi_reg['p_value']:.6e}")
                    st.write(f"- **Observations**: {ofi_reg['n_obs']}")

            with col2:
                st.subheader("Trade Imbalance Regression")
                if ti_reg:
                    st.write(f"- **R²**: {ti_reg['r_squared']:.6f}")
                    st.write(f"- **Slope (β)**: {ti_reg['slope']:.6e}")
                    st.write(f"- **p-value**: {ti_reg['p_value']:.6e}")
                    st.write(f"- **Observations**: {ti_reg['n_obs']}")

            # Winner
            if ofi_reg and ti_reg:
                if ofi_reg['r_squared'] > ti_reg['r_squared']:
                    improvement = (ofi_reg['r_squared'] - ti_reg['r_squared']) / ti_reg['r_squared'] * 100
                    st.success(f"**OFI wins** with {improvement:.1f}% higher R²")
                else:
                    improvement = (ti_reg['r_squared'] - ofi_reg['r_squared']) / ofi_reg['r_squared'] * 100
                    st.info(f"**TI wins** with {improvement:.1f}% higher R²")

    with tab3:
        st.header("Volume Analysis: OFI vs Raw Volume")
        st.markdown("""
        **Cont et al. (2011) Section 4**: Comparing OFI with raw trading volume.

        - **OFI** = Signed order flow (captures direction)
        - **Volume** = Total unsigned trading volume (ignores direction)
        - **|OFI|** = Absolute OFI (unsigned flow)

        The paper found OFI outperforms volume because volume doesn't capture direction.
        """)

        # Calculate volume analysis results
        vol_results = []

        for tw in TIME_WINDOWS:
            ofi_agg = aggregate_ofi_by_time_window(ofi_raw, tw)
            ti_agg = aggregate_trades_by_time_window(trades_raw, tw)
            merged = merge_ofi_and_ti(ofi_agg, ti_agg)

            if merged is not None and len(merged) >= 10:
                # OFI regression
                ofi_reg = run_regression(merged['ofi'].values, merged['delta_mid_price_ticks'].values)

                # Volume regression (unsigned)
                vol_reg = run_regression(merged['total_volume'].values, merged['delta_mid_price_ticks'].values)

                # |OFI| regression (absolute OFI)
                abs_ofi_reg = run_regression(np.abs(merged['ofi'].values), np.abs(merged['delta_mid_price_ticks'].values))

                if ofi_reg and vol_reg:
                    vol_results.append({
                        'time_window': tw,
                        'n_obs': ofi_reg['n_obs'],
                        'ofi_r_squared': ofi_reg['r_squared'],
                        'volume_r_squared': vol_reg['r_squared'],
                        'abs_ofi_r_squared': abs_ofi_reg['r_squared'] if abs_ofi_reg else 0
                    })

        if not vol_results:
            st.error("Not enough data for volume analysis")
        else:
            vol_results_df = pd.DataFrame(vol_results)

            # Create volume comparison chart
            fig_vol = go.Figure()

            fig_vol.add_trace(go.Bar(
                x=[f"{tw} min" for tw in vol_results_df['time_window']],
                y=vol_results_df['ofi_r_squared'],
                name='OFI',
                marker_color='#2E86AB',
                text=[f"{r:.4f}" for r in vol_results_df['ofi_r_squared']],
                textposition='outside'
            ))

            fig_vol.add_trace(go.Bar(
                x=[f"{tw} min" for tw in vol_results_df['time_window']],
                y=vol_results_df['volume_r_squared'],
                name='Volume',
                marker_color='#A23B72',
                text=[f"{r:.4f}" for r in vol_results_df['volume_r_squared']],
                textposition='outside'
            ))

            fig_vol.add_trace(go.Bar(
                x=[f"{tw} min" for tw in vol_results_df['time_window']],
                y=vol_results_df['abs_ofi_r_squared'],
                name='|OFI|',
                marker_color='#F18F01',
                text=[f"{r:.4f}" for r in vol_results_df['abs_ofi_r_squared']],
                textposition='outside'
            ))

            fig_vol.update_layout(
                title="R² Comparison: OFI vs Volume vs |OFI|",
                xaxis_title="Time Window",
                yaxis_title="R²",
                barmode='group',
                height=400
            )

            st.plotly_chart(fig_vol, use_container_width=True)

            # Summary metrics
            avg_ofi = vol_results_df['ofi_r_squared'].mean()
            avg_vol = vol_results_df['volume_r_squared'].mean()
            avg_abs_ofi = vol_results_df['abs_ofi_r_squared'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg OFI R²", f"{avg_ofi:.4f}")
            with col2:
                st.metric("Avg Volume R²", f"{avg_vol:.4f}")
            with col3:
                st.metric("Avg |OFI| R²", f"{avg_abs_ofi:.4f}")

            # Key findings
            st.subheader("Key Findings")

            if avg_ofi > avg_vol:
                st.success(f"""
                **OFI outperforms raw Volume** (consistent with Cont et al. 2011)

                - OFI captures both magnitude AND direction of order flow
                - Volume ignores direction (buys and sells treated the same)
                - Signed order flow is more predictive of price changes
                """)
            else:
                st.info(f"""
                **Volume performs comparably to OFI**

                This may indicate volume-driven dynamics in this market.
                """)

            # Results table
            st.subheader("Detailed Results")
            st.dataframe(vol_results_df.rename(columns={
                'time_window': 'Window (min)',
                'n_obs': 'Observations',
                'ofi_r_squared': 'OFI R²',
                'volume_r_squared': 'Volume R²',
                'abs_ofi_r_squared': '|OFI| R²'
            }), use_container_width=True)

    with tab4:
        st.header("Raw Data Preview")

        st.subheader("OFI Data (first 100 rows)")
        st.dataframe(ofi_raw.head(100))

        st.subheader("Trades Data (first 100 rows)")
        st.dataframe(trades_raw.head(100))


if __name__ == "__main__":
    main()
