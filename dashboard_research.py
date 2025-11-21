"""
dashboard_research.py
=====================
Cont et al. (2011) Replication Dashboard for Polymarket

Complete replication dashboard with all paper analyses:
- Regression Results (Table 2)
- Depth Analysis (Table 3, Figure 4)
- Event Patterns (Table 4, Figure 5)
- OFI vs TI Comparison (Table 5, Figure 6)
- Multi-Market Comparison

Usage:
    streamlit run dashboard_research.py

Then open browser to http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
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

# Markets configuration
MARKETS = {
    "Fed Rate Decision (Dec 2025)": {
        "ofi_file": "ofi_oct15_nov20_combined.csv",
        "name_short": "Fed",
        "description": "Fed decreases interest rates by 25 bps after December 2025 meeting?"
    },
    "NYC Mayoral Election 2025": {
        "ofi_file": "nyc_mayor_oct15_nov04_ofi.csv",
        "name_short": "NYC",
        "description": "Will Zohran Mamdani win the 2025 NYC mayoral election?"
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_regression_table():
    """Load Table 2 - Regression Statistics"""
    file_path = TABLES_DIR / "table_2_regression_statistics.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_depth_table():
    """Load Table 3 - Depth Analysis"""
    file_path = TABLES_DIR / "table_3_depth_analysis.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_variance_table():
    """Load Table 4 - Variance Decomposition"""
    file_path = TABLES_DIR / "table_4_variance_decomposition.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_ti_comparison_table():
    """Load Table 5 - OFI vs TI Comparison"""
    file_path = TABLES_DIR / "table_5_ofi_ti_comparison.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_event_statistics(market_short):
    """Load event statistics for a market"""
    file_path = ANALYSIS_DIR / f"{market_short}_event_statistics.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_rolling_regression(market_short):
    """Load rolling regression results"""
    file_path = ANALYSIS_DIR / f"{market_short}_rolling_regression.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        df['window_start'] = pd.to_datetime(df['window_start'])
        df['window_end'] = pd.to_datetime(df['window_end'])
        return df
    return None

@st.cache_data
def load_market_ofi_data(market_key):
    """Load OFI data for a market"""
    market_config = MARKETS[market_key]
    ofi_file = DATA_DIR / market_config["ofi_file"]
    if ofi_file.exists():
        df = pd.read_csv(ofi_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        return df
    return None

def load_figure(filename):
    """Load a figure image"""
    file_path = FIGURES_DIR / filename
    if file_path.exists():
        return Image.open(file_path)
    return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_regression_comparison():
    """Create comparison plot of regression statistics"""
    df = load_regression_table()
    if df is None:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Values', 'Beta Coefficients'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # R² comparison
    fig.add_trace(
        go.Bar(
            x=df['Market'],
            y=df['R_squared'],
            name='Linear R²',
            marker_color='steelblue',
            text=df['R_squared'].apply(lambda x: f'{x:.4f}'),
            textposition='outside'
        ),
        row=1, col=1
    )

    if 'R2_Quad' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df['Market'],
                y=df['R2_Quad'],
                name='Quadratic R²',
                marker_color='coral',
                text=df['R2_Quad'].apply(lambda x: f'{x:.4f}'),
                textposition='outside'
            ),
            row=1, col=1
        )

    # Beta comparison
    fig.add_trace(
        go.Bar(
            x=df['Market'],
            y=df['Beta'],
            name='β (slope)',
            marker_color='green',
            text=df['Beta'].apply(lambda x: f'{x:.2e}'),
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Market", row=1, col=1)
    fig.update_xaxes(title_text="Market", row=1, col=2)
    fig.update_yaxes(title_text="R² Value", row=1, col=1)
    fig.update_yaxes(title_text="Beta Coefficient", row=1, col=2)

    fig.update_layout(
        title="Regression Results Comparison",
        height=400,
        showlegend=True
    )

    return fig

def plot_depth_power_law(market_short):
    """Plot beta vs depth from rolling regression"""
    df = load_rolling_regression(market_short)
    if df is None or len(df) == 0:
        return None

    # Filter outliers
    beta_q01 = df['beta'].quantile(0.01)
    beta_q99 = df['beta'].quantile(0.99)
    df_filtered = df[(df['beta'] >= beta_q01) & (df['beta'] <= beta_q99) & (df['beta'] > 0)]

    if len(df_filtered) == 0:
        return None

    # Create log-log plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.log(df_filtered['avg_depth']),
        y=np.log(df_filtered['beta']),
        mode='markers',
        marker=dict(
            size=6,
            color=df_filtered['r_squared'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="R²"),
            opacity=0.6
        ),
        text=df_filtered.apply(lambda row: f"Window: {row['window_start'].strftime('%Y-%m-%d')}<br>β: {row['beta']:.2e}<br>Depth: {row['avg_depth']:,.0f}<br>R²: {row['r_squared']:.4f}", axis=1),
        hovertemplate='%{text}<extra></extra>',
        name='Windows'
    ))

    # Fit power law
    log_depth = np.log(df_filtered['avg_depth'])
    log_beta = np.log(df_filtered['beta'])

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_depth, log_beta)

    x_range = np.linspace(log_depth.min(), log_depth.max(), 100)
    y_pred = slope * x_range + intercept

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f'Power Law Fit (λ={-slope:.3f})',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f'{market_short} Market: β vs Average Depth (Log-Log)',
        xaxis_title='log(Average Depth)',
        yaxis_title='log(β)',
        height=500,
        hovermode='closest'
    )

    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'λ = {-slope:.3f}<br>R² = {r_value**2:.4f}<br>p-value = {p_value:.2e}',
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )

    return fig

def plot_event_impact(market_short):
    """Plot event type impact"""
    df = load_event_statistics(market_short)
    if df is None:
        return None

    # Filter events with valid beta and sort by impact
    df_valid = df[df['beta'].notna()].copy()
    df_sorted = df_valid.nlargest(10, 'beta')

    fig = go.Figure()

    colors = ['green' if b > 0 else 'red' for b in df_sorted['beta']]

    fig.add_trace(go.Bar(
        y=df_sorted['event_type'],
        x=df_sorted['beta'],
        orientation='h',
        marker_color=colors,
        text=df_sorted['beta'].apply(lambda x: f'{x:.2e}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>β: %{x:.2e}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{market_short} Market: Price Impact by Event Type',
        xaxis_title='β (Price Impact Coefficient)',
        yaxis_title='Event Type',
        height=500
    )

    fig.add_vline(x=0, line_dash="dash", line_color="black")

    return fig

def plot_variance_pie():
    """Plot variance decomposition as pie chart"""
    df = load_variance_table()
    if df is None:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[f"{row['Market']} Market" for _, row in df.iterrows()]
    )

    for idx, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(
            go.Pie(
                labels=['Explained by OFI', 'Residual'],
                values=[row['Pct_Explained'], row['Pct_Residual']],
                marker_colors=['#2E86AB', '#E63946'],
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>%{value:.2f}%<extra></extra>'
            ),
            row=1, col=idx+1
        )

    fig.update_layout(
        title="Variance Decomposition: Price Changes Explained by OFI",
        height=400,
        showlegend=False
    )

    return fig

def plot_ti_comparison():
    """Plot OFI vs TI horse-race comparison"""
    df = load_ti_comparison_table()
    if df is None:
        return None

    fig = go.Figure()

    # Create grouped bar chart
    markets = df['Market']
    x = np.arange(len(markets))
    width = 0.25

    fig.add_trace(go.Bar(
        x=x - width,
        y=df['R2_OFI'],
        name='OFI Only',
        marker_color='steelblue',
        text=df['R2_OFI'].apply(lambda x: f'{x:.4f}'),
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=df['R2_TI'],
        name='TI Only',
        marker_color='coral',
        text=df['R2_TI'].apply(lambda x: f'{x:.4f}'),
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=x + width,
        y=df['R2_Both'],
        name='OFI + TI',
        marker_color='green',
        text=df['R2_Both'].apply(lambda x: f'{x:.4f}'),
        textposition='outside'
    ))

    fig.update_layout(
        title="Horse-Race Regression: OFI vs Trade Imbalance",
        xaxis=dict(
            tickmode='array',
            tickvals=x,
            ticktext=markets
        ),
        xaxis_title="Market",
        yaxis_title="R² (Explained Variance)",
        height=500,
        showlegend=True,
        barmode='group'
    )

    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="OFI Research Dashboard - Cont et al. (2011) Replication",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("📊 OFI Research Dashboard")
    st.markdown("### Cont, Kukanov & Stoikov (2011) Replication for Polymarket")

    st.info("""
    **Paper:** "The Price Impact of Order Book Events" (Journal of Financial Econometrics, 2011)
    **Replication:** Polymarket prediction markets (Fed Rate Decision, NYC Mayoral Election)
    """)

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose Analysis:",
            [
                "📈 Overview",
                "📊 Regression Analysis",
                "📉 Depth Analysis",
                "🎯 Event Patterns",
                "⚖️ OFI vs TI Comparison",
                "🔬 Multi-Market Comparison",
                "🖼️ Figures Gallery"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        st.header("Quick Stats")
        reg_table = load_regression_table()
        if reg_table is not None:
            st.metric("Markets Analyzed", len(reg_table))
            st.metric("Avg R²", f"{reg_table['R_squared'].mean():.4f}")
            st.metric("Avg Correlation", f"{reg_table['Correlation'].mean():.4f}")

    # ========================================================================
    # OVERVIEW PAGE
    # ========================================================================

    if page == "📈 Overview":
        st.header("Research Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Study Design")
            st.markdown("""
            **Objective:** Replicate Cont et al. (2011) findings for prediction markets

            **Key Questions:**
            1. Does OFI predict price changes in prediction markets?
            2. Is price impact inversely proportional to depth?
            3. Does OFI dominate trade imbalance?
            4. How does variance decomposition differ from equities?

            **Data:**
            - Fed Rate Decision (26,840 observations)
            - NYC Mayoral Election (101,270 observations)
            - Orderbook snapshots with OFI calculation
            """)

        with col2:
            st.subheader("Key Findings")
            st.markdown("""
            ✅ **OFI is statistically significant** (p < 0.001 for both markets)

            ⚠️ **Lower R² than equities** (5-7% vs 65% in paper)
            - Prediction markets are more news-driven
            - Event resolution structure differs from continuous trading

            ⚠️ **Weak depth relationship** (λ ≈ 0.5 vs 1 in paper)
            - Depth doesn't buffer price impact as strongly

            ✅ **OFI dominates TI** (consistent with paper)
            - OFI provides 90%+ of explanatory power
            """)

        st.divider()

        st.subheader("Summary Tables")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Table 2: Regression",
            "Table 3: Depth",
            "Table 4: Variance",
            "Table 5: OFI vs TI"
        ])

        with tab1:
            df = load_regression_table()
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Table not found. Run `python scripts/01_regression_analysis.py`")

        with tab2:
            df = load_depth_table()
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Table not found. Run `python scripts/04_depth_analysis.py`")

        with tab3:
            df = load_variance_table()
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Table not found. Run `python scripts/05_event_analysis.py`")

        with tab4:
            df = load_ti_comparison_table()
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Table not found. Run `python scripts/06_trade_volume_analysis.py`")

    # ========================================================================
    # REGRESSION ANALYSIS PAGE
    # ========================================================================

    elif page == "📊 Regression Analysis":
        st.header("Regression Analysis (Section 3.2)")

        st.markdown("""
        **Linear Model:** ΔP = α + β × OFI + ε
        **Quadratic Model:** ΔP = α + β₁ × OFI + β₂ × OFI² + ε
        """)

        df = load_regression_table()

        if df is not None:
            # Display table
            st.subheader("Regression Statistics (Table 2)")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Visualization
            st.subheader("Visual Comparison")
            fig = plot_regression_comparison()
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Key insights
            st.subheader("Key Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Average R²",
                    f"{df['R_squared'].mean():.4f}",
                    help="Proportion of price variance explained by OFI"
                )

            with col2:
                st.metric(
                    "Average Correlation",
                    f"{df['Correlation'].mean():.4f}",
                    help="Linear correlation between OFI and price changes"
                )

            with col3:
                significant_count = (df['p_Beta'] < 0.05).sum()
                st.metric(
                    "Significant Markets",
                    f"{significant_count}/{len(df)}",
                    help="Markets where β is significant at 5% level"
                )

            # Scatter plot figure
            st.subheader("OFI vs Price Change Scatter Plots")

            scatter_combined = load_figure("figure_2_combined_comparison.png")
            if scatter_combined:
                st.image(scatter_combined, caption="Figure 2: OFI vs Price Change (Combined)", use_column_width=True)

        else:
            st.error("❌ Regression results not found")
            st.info("Run: `python scripts/01_regression_analysis.py`")

    # ========================================================================
    # DEPTH ANALYSIS PAGE
    # ========================================================================

    elif page == "📉 Depth Analysis":
        st.header("Depth Analysis (Section 3.2)")

        st.markdown("""
        **Power Law Model:** β = a / AD^λ
        **Paper's Finding:** λ ≈ 1 (inverse proportionality)
        **Our Finding:** λ varies significantly (weak relationship)
        """)

        df = load_depth_table()

        if df is not None:
            # Display table
            st.subheader("Depth Statistics (Table 3)")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg λ (Exponent)",
                    f"{df['Power_Law_lambda'].mean():.3f}",
                    help="Power law exponent (paper: ~1.0)"
                )

            with col2:
                st.metric(
                    "Avg R² (Power Law)",
                    f"{df['R_squared'].mean():.4f}",
                    help="Fit quality of power law model"
                )

            with col3:
                close_to_one = (np.abs(df['Power_Law_lambda'] - 1.0) < 0.3).sum()
                st.metric(
                    "Close to Paper (λ≈1)",
                    f"{close_to_one}/{len(df)}",
                    help="Markets with λ within 0.3 of 1.0"
                )

            # Interactive plots
            st.subheader("Interactive β vs Depth Analysis")

            selected_market = st.selectbox(
                "Select Market:",
                options=list(MARKETS.keys())
            )

            market_short = MARKETS[selected_market]["name_short"]

            fig = plot_depth_power_law(market_short)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No rolling regression data for {market_short}")

            # Static figure
            st.subheader("Time Series: β and Depth Evolution")
            depth_fig = load_figure("figure_4_depth_analysis.png")
            if depth_fig:
                st.image(depth_fig, caption="Figure 4: Depth Analysis", use_column_width=True)

        else:
            st.error("❌ Depth analysis results not found")
            st.info("Run: `python scripts/04_depth_analysis.py`")

    # ========================================================================
    # EVENT PATTERNS PAGE
    # ========================================================================

    elif page == "🎯 Event Patterns":
        st.header("Event Pattern Analysis (Section 3)")

        st.markdown("""
        **Orderbook Events:**
        - bid_up/bid_down: Bid-side depth changes
        - ask_up/ask_down: Ask-side depth changes
        - Combined events: Simultaneous changes

        **Variance Decomposition:**
        How much of price variance is explained by OFI?
        """)

        # Variance decomposition
        st.subheader("Variance Decomposition")

        var_df = load_variance_table()
        if var_df is not None:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(var_df, use_container_width=True, hide_index=True)

            with col2:
                fig = plot_variance_pie()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.info(f"""
            **Key Finding:** OFI explains only **{var_df['Pct_Explained'].mean():.1f}%** of price variance on average
            (vs **65%** in equity markets from original paper).

            This suggests prediction markets are driven more by **exogenous news events** than orderbook dynamics.
            """)

        st.divider()

        # Event-specific analysis
        st.subheader("Event-Specific Price Impact")

        selected_market = st.selectbox(
            "Select Market:",
            options=list(MARKETS.keys()),
            key="event_market_select"
        )

        market_short = MARKETS[selected_market]["name_short"]

        event_stats = load_event_statistics(market_short)

        if event_stats is not None:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Event Statistics**")
                st.dataframe(
                    event_stats[['event_type', 'count', 'beta', 'r_squared']].head(10),
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                fig = plot_event_impact(market_short)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # Full figure
        st.subheader("Complete Event Analysis")
        event_fig = load_figure("figure_5_event_analysis.png")
        if event_fig:
            st.image(event_fig, caption="Figure 5: Event Pattern Analysis", use_column_width=True)

    # ========================================================================
    # OFI VS TI COMPARISON PAGE
    # ========================================================================

    elif page == "⚖️ OFI vs TI Comparison":
        st.header("OFI vs Trade Imbalance (Section 4)")

        st.markdown("""
        **Horse-Race Regressions:**
        1. ΔP = α + β_OFI × OFI
        2. ΔP = α + β_TI × TI
        3. ΔP = α + β_OFI × OFI + β_TI × TI

        **Question:** Does OFI dominate trade imbalance in predicting prices?
        """)

        df = load_ti_comparison_table()

        if df is not None:
            # Display table
            st.subheader("Comparison Statistics (Table 5)")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Visualization
            st.subheader("Horse-Race Results")
            fig = plot_ti_comparison()
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Key insights
            st.subheader("Key Insights")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg OFI R²",
                    f"{df['R2_OFI'].mean():.4f}",
                    help="OFI-only model explanatory power"
                )

            with col2:
                st.metric(
                    "Avg TI R²",
                    f"{df['R2_TI'].mean():.4f}",
                    help="TI-only model explanatory power"
                )

            with col3:
                improvement = df['R2_Improvement'].mean() * 100
                st.metric(
                    "Avg R² Gain from TI",
                    f"{improvement:.2f}%",
                    help="Additional variance explained by adding TI"
                )

            st.success("""
            ✅ **Conclusion:** OFI dominates trade imbalance in both markets.
            Adding TI to OFI provides minimal improvement (~0.2-0.4%), consistent with the paper's findings for equities.
            """)

            # Full figure
            st.subheader("Detailed Comparison")
            ti_fig = load_figure("figure_6_ofi_ti_comparison.png")
            if ti_fig:
                st.image(ti_fig, caption="Figure 6: OFI vs TI Comparison", use_column_width=True)

        else:
            st.error("❌ OFI vs TI comparison results not found")
            st.info("Run: `python scripts/06_trade_volume_analysis.py`")

    # ========================================================================
    # MULTI-MARKET COMPARISON PAGE
    # ========================================================================

    elif page == "🔬 Multi-Market Comparison":
        st.header("Multi-Market Comparison")

        st.markdown("""
        Compare findings across both prediction markets and against the original paper's equity market results.
        """)

        # Load all tables
        reg_df = load_regression_table()
        depth_df = load_depth_table()
        var_df = load_variance_table()
        ti_df = load_ti_comparison_table()

        if all(df is not None for df in [reg_df, depth_df, var_df, ti_df]):
            # Create comprehensive comparison
            comparison = pd.DataFrame({
                'Market': reg_df['Market'],
                'Observations': reg_df['N_obs'],
                'R² (Linear)': reg_df['R_squared'],
                'Correlation': reg_df['Correlation'],
                'β (OFI)': reg_df['Beta'],
                'λ (Depth)': depth_df['Power_Law_lambda'],
                'Variance Explained %': var_df['Pct_Explained'],
                'R² Improvement (TI)': ti_df['R2_Improvement']
            })

            st.subheader("Comprehensive Comparison Table")
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            # Comparative visualizations
            st.subheader("Visual Comparisons")

            col1, col2 = st.columns(2)

            with col1:
                # R² comparison
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=comparison['Market'],
                    y=comparison['R² (Linear)'],
                    marker_color=['steelblue', 'coral'],
                    text=comparison['R² (Linear)'].apply(lambda x: f'{x:.4f}'),
                    textposition='outside'
                ))
                fig1.add_hline(y=0.65, line_dash="dash", line_color="red",
                              annotation_text="Paper: ~65% (equities)")
                fig1.update_layout(
                    title="R² Comparison vs Original Paper",
                    yaxis_title="R² Value",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Lambda comparison
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=comparison['Market'],
                    y=comparison['λ (Depth)'],
                    marker_color=['green', 'orange'],
                    text=comparison['λ (Depth)'].apply(lambda x: f'{x:.3f}'),
                    textposition='outside'
                ))
                fig2.add_hline(y=1.0, line_dash="dash", line_color="red",
                              annotation_text="Paper: ~1.0")
                fig2.update_layout(
                    title="Depth Exponent (λ) vs Paper",
                    yaxis_title="λ Value",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Paper comparison summary
            st.divider()

            st.subheader("Differences from Original Paper")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                **R² (Explanatory Power)**
                - Paper: ~65%
                - Our avg: ~{:.1f}%
                - Difference: **-{:.1f}pp**

                Prediction markets are more news-driven than orderbook-driven.
                """.format(
                    comparison['R² (Linear)'].mean() * 100,
                    65 - comparison['R² (Linear)'].mean() * 100
                ))

            with col2:
                st.markdown("""
                **λ (Depth Relationship)**
                - Paper: ~1.0
                - Our avg: ~{:.2f}
                - Difference: **{:.2f}**

                Weaker inverse depth relationship in prediction markets.
                """.format(
                    comparison['λ (Depth)'].mean(),
                    comparison['λ (Depth)'].mean() - 1.0
                ))

            with col3:
                st.markdown("""
                **OFI Dominance**
                - Paper: ✅ OFI >> TI
                - Our finding: ✅ OFI >> TI
                - Result: **Consistent**

                OFI provides 90%+ of explanatory power in both settings.
                """)

        else:
            st.error("❌ Some analysis results missing")
            st.info("Run: `python scripts/run_all_analyses.py`")

    # ========================================================================
    # FIGURES GALLERY PAGE
    # ========================================================================

    elif page == "🖼️ Figures Gallery":
        st.header("Figures Gallery")

        st.markdown("All publication-quality figures generated by the replication study.")

        figures = [
            ("figure_2_combined_comparison.png", "Figure 2: OFI vs Price Change Scatter Plots"),
            ("figure_3_Fed_residual_diagnostics.png", "Figure 3a: Fed Market Residual Diagnostics"),
            ("figure_3_NYC_residual_diagnostics.png", "Figure 3b: NYC Market Residual Diagnostics"),
            ("figure_4_depth_analysis.png", "Figure 4: Depth Analysis (β vs AD)"),
            ("figure_5_event_analysis.png", "Figure 5: Event Pattern Analysis"),
            ("figure_6_ofi_ti_comparison.png", "Figure 6: OFI vs TI Horse-Race Comparison"),
        ]

        for filename, caption in figures:
            with st.expander(caption, expanded=False):
                img = load_figure(filename)
                if img:
                    st.image(img, caption=caption, use_column_width=True)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"📥 [Download PNG]({FIGURES_DIR}/{filename})")
                    with col2:
                        pdf_filename = filename.replace('.png', '.pdf')
                        st.markdown(f"📥 [Download PDF]({FIGURES_DIR}/{pdf_filename})")
                else:
                    st.warning(f"Figure not found: {filename}")

if __name__ == "__main__":
    main()
