"""
dashboard_mlofi.py
==================
Multi-Level OFI (MLOFI) Analysis Dashboard

Displays comprehensive MLOFI evaluation results:
- 2,268 configurations tested
- 7 level configs × 9 time windows × 9 outlier methods × 4 q^a exponents
- Phase analysis (Early/Middle/Late)

USAGE:
------
    streamlit run dashboard/dashboard_mlofi.py

Then open browser to http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "mlofi"

# Expected files
ALL_CONFIGS_FILE = DATA_DIR / "mlofi_all_configs.csv"
PHASE_ANALYSIS_FILE = DATA_DIR / "mlofi_top20_phase_analysis.csv"
CUMULATIVE_MLOFI_FILE = DATA_DIR / "cumulative_mlofi_results.csv"

# Colors
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F77F00',
    'success': '#06A77D',
    'warning': '#E94F37',
    'early': '#2E86AB',
    'middle': '#A23B72',
    'late': '#F77F00',
}

LEVEL_COLORS = {
    'L2': '#1f77b4',
    'L3': '#ff7f0e',
    'L5': '#2ca02c',
    'L10': '#d62728',
    'L20': '#9467bd',
    'L50pct': '#8c564b',
    'ALL': '#e377c2',
}

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_all_configs():
    """Load all MLOFI configuration results"""
    if not ALL_CONFIGS_FILE.exists():
        return None
    df = pd.read_csv(ALL_CONFIGS_FILE)
    return df


@st.cache_data
def load_phase_analysis():
    """Load phase analysis results for top 20 configs"""
    if not PHASE_ANALYSIS_FILE.exists():
        return None
    df = pd.read_csv(PHASE_ANALYSIS_FILE)
    return df


@st.cache_data
def load_cumulative_mlofi():
    """Load cumulative MLOFI results"""
    if not CUMULATIVE_MLOFI_FILE.exists():
        return None
    df = pd.read_csv(CUMULATIVE_MLOFI_FILE)
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_top_configs_bar(df, top_n=20):
    """Bar chart of top N configurations by R²"""
    top_df = df.head(top_n).copy()
    top_df['config_label'] = top_df.apply(
        lambda r: f"{r['level_config']} | {r['time_window']}min | {r['outlier_method'][:8]} | a={r['q_exponent']}",
        axis=1
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_df['r2'] * 100,
        y=top_df['config_label'],
        orientation='h',
        marker=dict(
            color=top_df['r2'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='R²')
        ),
        text=[f"{r*100:.1f}%" for r in top_df['r2']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>R² = %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=f"Top {top_n} MLOFI Configurations by R²",
        xaxis_title="R² (%)",
        yaxis_title="Configuration",
        height=600,
        yaxis=dict(autorange='reversed'),
        showlegend=False
    )

    return fig


def plot_r2_by_level(df):
    """Bar chart showing best R² by level configuration"""
    level_summary = df.groupby('level_config').agg({
        'r2': ['max', 'mean', 'std']
    }).reset_index()
    level_summary.columns = ['level_config', 'max_r2', 'mean_r2', 'std_r2']
    level_summary = level_summary.sort_values('max_r2', ascending=False)

    fig = go.Figure()

    # Max R²
    fig.add_trace(go.Bar(
        name='Max R²',
        x=level_summary['level_config'],
        y=level_summary['max_r2'] * 100,
        marker_color=[LEVEL_COLORS.get(l, '#333') for l in level_summary['level_config']],
        text=[f"{r*100:.1f}%" for r in level_summary['max_r2']],
        textposition='outside',
    ))

    # Mean R² line
    fig.add_trace(go.Scatter(
        name='Mean R²',
        x=level_summary['level_config'],
        y=level_summary['mean_r2'] * 100,
        mode='markers+lines',
        marker=dict(size=10, color='black'),
        line=dict(dash='dash', color='black')
    ))

    fig.update_layout(
        title="R² by Level Configuration",
        xaxis_title="Level Config (Cumulative OFI through Level N)",
        yaxis_title="R² (%)",
        height=400,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    return fig


def plot_r2_by_time_window(df):
    """Line chart showing R² across time windows"""
    tw_summary = df.groupby('time_window').agg({
        'r2': ['max', 'mean', 'min']
    }).reset_index()
    tw_summary.columns = ['time_window', 'max_r2', 'mean_r2', 'min_r2']
    tw_summary = tw_summary.sort_values('time_window')

    fig = go.Figure()

    # Add filled area for range
    fig.add_trace(go.Scatter(
        name='Range',
        x=list(tw_summary['time_window']) + list(tw_summary['time_window'][::-1]),
        y=list(tw_summary['max_r2'] * 100) + list(tw_summary['min_r2'][::-1] * 100),
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        hoverinfo='skip'
    ))

    # Max line
    fig.add_trace(go.Scatter(
        name='Max R²',
        x=tw_summary['time_window'],
        y=tw_summary['max_r2'] * 100,
        mode='lines+markers',
        line=dict(color=COLORS['success'], width=3),
        marker=dict(size=10)
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        name='Mean R²',
        x=tw_summary['time_window'],
        y=tw_summary['mean_r2'] * 100,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="R² by Time Window",
        xaxis_title="Time Window (minutes)",
        yaxis_title="R² (%)",
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    return fig


def plot_r2_by_exponent(df):
    """Bar chart showing R² by q^a exponent"""
    exp_summary = df.groupby('q_exponent').agg({
        'r2': ['max', 'mean']
    }).reset_index()
    exp_summary.columns = ['q_exponent', 'max_r2', 'mean_r2']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Max R²',
        x=[f"a = {e}" for e in exp_summary['q_exponent']],
        y=exp_summary['max_r2'] * 100,
        marker_color=['#06A77D', '#2E86AB', '#F77F00', '#E94F37'],
        text=[f"{r*100:.1f}%" for r in exp_summary['max_r2']],
        textposition='outside',
    ))

    fig.update_layout(
        title="R² by q^a Size Exponent",
        xaxis_title="Exponent (a < 1 reduces large order impact)",
        yaxis_title="Max R² (%)",
        height=350,
        showlegend=False
    )

    # Add annotation explaining exponents
    fig.add_annotation(
        x=0.5, y=-0.15,
        xref='paper', yref='paper',
        text="a=0.3: Heavy compression | a=0.5: Square root | a=0.7: Mild compression | a=1.0: Raw (paper default)",
        showarrow=False,
        font=dict(size=10, color='gray')
    )

    return fig


def plot_phase_comparison(phase_df):
    """Bar chart comparing R² across phases for top configs"""
    if phase_df is None or len(phase_df) == 0:
        return None

    # Calculate averages
    avg_overall = phase_df['overall_r2'].mean()
    avg_early = phase_df['early_r2'].mean()
    avg_middle = phase_df['middle_r2'].mean()
    avg_late = phase_df['late_r2'].mean()

    phases = ['Overall', 'Early', 'Middle', 'Late']
    values = [avg_overall * 100, avg_early * 100, avg_middle * 100, avg_late * 100]
    colors = ['#333', COLORS['early'], COLORS['middle'], COLORS['late']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=phases,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
    ))

    fig.update_layout(
        title="Average R² by Market Phase (Top 20 Configs)",
        xaxis_title="Market Phase",
        yaxis_title="Average R² (%)",
        height=350,
        showlegend=False
    )

    # Add trend annotation
    if avg_late > avg_early:
        fig.add_annotation(
            x=0.5, y=1.05,
            xref='paper', yref='paper',
            text=f"OFI becomes MORE predictive near expiry (+{((avg_late-avg_early)/avg_early)*100:.1f}%)",
            showarrow=False,
            font=dict(size=12, color=COLORS['success']),
            bgcolor='rgba(6, 167, 125, 0.1)'
        )

    return fig


def plot_phase_heatmap(phase_df):
    """Heatmap showing phase R² for each top config"""
    if phase_df is None or len(phase_df) == 0:
        return None

    # Create config labels with all parameters
    phase_df = phase_df.copy()
    phase_df['config'] = phase_df.apply(
        lambda r: f"{r['level_config']} | {r['time_window']}min | {r['outlier_method']} | a={r['q_exponent']}",
        axis=1
    )

    # Get unique configs
    unique_configs = phase_df.drop_duplicates(subset=['level_config', 'time_window', 'outlier_method', 'q_exponent'])

    # Create matrix
    phases = ['early_r2', 'middle_r2', 'late_r2']
    phase_labels = ['Early', 'Middle', 'Late']

    z_data = unique_configs[phases].values * 100

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=phase_labels,
        y=unique_configs['config'],
        colorscale='RdYlGn',
        text=[[f"{v:.1f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='Config: %{y}<br>Phase: %{x}<br>R²: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='R² (%)')
    ))

    fig.update_layout(
        title="R² by Phase for Top Configurations",
        xaxis_title="Market Phase",
        yaxis_title="Configuration",
        height=500
    )

    return fig


def plot_3d_surface(df):
    """3D surface plot of R² by level and time window"""
    # Pivot for 3D
    # Use best q_exponent (0.3) and Raw outlier method
    filtered = df[(df['q_exponent'] == 0.3) & (df['outlier_method'] == 'Raw')]

    if len(filtered) == 0:
        return None

    pivot = filtered.pivot_table(
        index='level_config',
        columns='time_window',
        values='r2'
    )

    # Order levels
    level_order = ['L2', 'L3', 'L5', 'L10', 'L20', 'L50pct', 'ALL']
    pivot = pivot.reindex([l for l in level_order if l in pivot.index])

    fig = go.Figure(data=[go.Surface(
        z=pivot.values * 100,
        x=pivot.columns,
        y=list(range(len(pivot.index))),
        colorscale='RdYlGn',
        hovertemplate='Time: %{x} min<br>Level: %{text}<br>R²: %{z:.1f}%<extra></extra>',
        text=[[l] * len(pivot.columns) for l in pivot.index]
    )])

    fig.update_layout(
        title="R² Surface: Level × Time Window (q^a=0.3, Raw)",
        scene=dict(
            xaxis_title='Time Window (min)',
            yaxis_title='Level Config',
            zaxis_title='R² (%)',
            yaxis=dict(
                ticktext=list(pivot.index),
                tickvals=list(range(len(pivot.index)))
            )
        ),
        height=500
    )

    return fig


def plot_heatmap_level_time(df, exponent=0.3):
    """Heatmap of R² by level and time window for a specific exponent"""
    filtered = df[(df['q_exponent'] == exponent) & (df['outlier_method'] == 'Raw')]

    if len(filtered) == 0:
        return None

    pivot = filtered.pivot_table(
        index='level_config',
        columns='time_window',
        values='r2'
    ) * 100

    # Order levels
    level_order = ['L2', 'L3', 'L5', 'L10', 'L20', 'L50pct', 'ALL']
    pivot = pivot.reindex([l for l in level_order if l in pivot.index])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{t}min" for t in pivot.columns],
        y=pivot.index,
        colorscale='RdYlGn',
        text=[[f"{v:.1f}" for v in row] for row in pivot.values],
        texttemplate="%{text}%",
        textfont={"size": 9},
        hovertemplate='Level: %{y}<br>Window: %{x}<br>R²: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='R² (%)')
    ))

    fig.update_layout(
        title=f"R² Heatmap: Level × Time Window (q^a={exponent}, Raw)",
        xaxis_title="Time Window",
        yaxis_title="Level Config",
        height=400
    )

    return fig


def plot_outlier_comparison(df, level='L2', time_window=60):
    """Compare outlier methods for a specific level and time window"""
    filtered = df[(df['level_config'] == level) &
                  (df['time_window'] == time_window) &
                  (df['q_exponent'] == 0.3)]

    if len(filtered) == 0:
        return None

    filtered = filtered.sort_values('r2', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=filtered['r2'] * 100,
        y=filtered['outlier_method'],
        orientation='h',
        marker=dict(
            color=filtered['r2'],
            colorscale='RdYlGn'
        ),
        text=[f"{r*100:.1f}%" for r in filtered['r2']],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"R² by Outlier Method ({level}, {time_window}min, a=0.3)",
        xaxis_title="R² (%)",
        yaxis_title="Outlier Method",
        height=400,
        showlegend=False
    )

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="MLOFI Analysis Dashboard",
        page_icon=None,
        layout="wide"
    )

    st.title("Multi-Level OFI (MLOFI) Analysis Dashboard")
    st.markdown("Comprehensive evaluation of Order Flow Imbalance across multiple order book levels")

    # Load data
    df = load_all_configs()
    phase_df = load_phase_analysis()

    if df is None:
        st.error(f"Data file not found: {ALL_CONFIGS_FILE}")
        st.info("Please run: `python mlofi/06_comprehensive_evaluation.py` first")
        return

    # Sidebar
    st.sidebar.header("MLOFI Dashboard")
    st.sidebar.markdown("---")

    # Summary stats
    st.sidebar.metric("Total Configurations", f"{len(df):,}")
    st.sidebar.metric("Best R²", f"{df['r2'].max()*100:.2f}%")
    st.sidebar.metric("Mean R²", f"{df['r2'].mean()*100:.2f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configuration Space:**")
    st.sidebar.markdown(f"- {df['level_config'].nunique()} Level Configs")
    st.sidebar.markdown(f"- {df['time_window'].nunique()} Time Windows")
    st.sidebar.markdown(f"- {df['outlier_method'].nunique()} Outlier Methods")
    st.sidebar.markdown(f"- {df['q_exponent'].nunique()} q^a Exponents")

    # Main content - Tabs
    tabs = st.tabs([
        "Overview",
        "Level Analysis",
        "Time Windows",
        "Phase Analysis",
        "Cumulative MLOFI",
        "Detailed Explorer"
    ])

    # =========================================================================
    # TAB 0: OVERVIEW
    # =========================================================================
    with tabs[0]:
        st.header("Overview: Best MLOFI Configurations")

        # Page explanation
        st.markdown("""
        This page presents the overall results of testing 2,266 MLOFI configurations. We evaluated
        combinations of 7 level depths (L2 through ALL), 9 time windows (1-90 minutes), 9 outlier
        filtering methods, and 4 size exponents (q^a). The best configuration achieves 77.9% R²,
        significantly outperforming L1-only OFI. The key finding is that shallow levels (L2-L3)
        combined with the q^a=0.3 size transformation provide the strongest OFI-price relationship.
        """)

        st.markdown("---")

        # Best config highlight
        st.markdown("**Best Configuration Summary:** The single best performing configuration across all tests.")
        best = df.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Level", best['level_config'])
        with col2:
            st.metric("Best Window", f"{best['time_window']} min")
        with col3:
            st.metric("Best q^a", f"a = {best['q_exponent']}")
        with col4:
            st.metric("Best R²", f"{best['r2']*100:.2f}%")

        st.markdown("---")

        # Top 20 bar chart
        st.markdown("**Top 20 Configurations:** Horizontal bar chart showing the 20 highest R² configurations, ranked by explanatory power.")
        fig_top = plot_top_configs_bar(df, 20)
        st.plotly_chart(fig_top, use_container_width=True)

        # Summary by dimension
        st.markdown("### Summary by Dimension")
        st.markdown("Maximum R² achieved within each category, showing which settings perform best.")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**By q^a Exponent**")
            for exp in sorted(df['q_exponent'].unique()):
                max_r2 = df[df['q_exponent'] == exp]['r2'].max()
                st.write(f"a = {exp}: {max_r2*100:.1f}%")

        with col2:
            st.markdown("**By Level Config**")
            for level in ['L2', 'L3', 'L5', 'L10', 'L20']:
                if level in df['level_config'].values:
                    max_r2 = df[df['level_config'] == level]['r2'].max()
                    st.write(f"{level}: {max_r2*100:.1f}%")

        with col3:
            st.markdown("**By Time Window (Top 5)**")
            for tw in [90, 60, 45, 30, 20]:
                if tw in df['time_window'].values:
                    max_r2 = df[df['time_window'] == tw]['r2'].max()
                    st.write(f"{tw} min: {max_r2*100:.1f}%")

        # Top 20 table
        st.markdown("### Top 20 Configurations Table")
        st.markdown("Detailed breakdown of each top configuration with level, time window, outlier method, exponent, and sample size.")
        top20 = df.head(20)[['level_config', 'time_window', 'outlier_method', 'q_exponent', 'r2', 'n_obs']].copy()
        top20['r2_pct'] = (top20['r2'] * 100).round(2)
        top20 = top20.rename(columns={
            'level_config': 'Level',
            'time_window': 'Window (min)',
            'outlier_method': 'Outlier Method',
            'q_exponent': 'q^a',
            'r2_pct': 'R² (%)',
            'n_obs': 'N obs'
        })
        st.dataframe(
            top20[['Level', 'Window (min)', 'Outlier Method', 'q^a', 'R² (%)', 'N obs']],
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 1: LEVEL ANALYSIS
    # =========================================================================
    with tabs[1]:
        st.header("Level Configuration Analysis")

        # Page explanation
        st.markdown("""
        This page analyzes how the number of order book levels included in MLOFI affects predictive
        power. Level configurations range from L2 (cumulative OFI through level 2) to ALL (all 50 levels).
        Contrary to intuition, adding more levels does not improve R². The best results come from
        shallow levels (L2, L3), suggesting that price-relevant order flow concentrates at the top
        of the book. Deeper levels add noise rather than signal.
        """)

        st.markdown("---")

        # Level bar chart
        st.markdown("**R² by Level Configuration:** Bar chart comparing maximum and mean R² for each level depth. Higher is better.")
        fig_level = plot_r2_by_level(df)
        st.plotly_chart(fig_level, use_container_width=True)

        st.markdown("---")

        # Heatmap
        st.subheader("R² Heatmap: Level x Time Window")
        st.markdown("Heatmap showing R² for each combination of level config and time window. Darker green indicates higher R².")
        exponent = st.selectbox("Select q^a exponent:", [0.3, 0.5, 0.7, 1.0], index=0)
        fig_heatmap = plot_heatmap_level_time(df, exponent)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")

        # 3D surface
        st.subheader("3D Surface Plot")
        st.markdown("Interactive 3D visualization of R² across level and time window dimensions. Rotate to explore the surface.")
        fig_3d = plot_3d_surface(df)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)

        # Key finding
        st.markdown("""
        ---
        **Key Finding:** Shallow levels (L2, L3) perform best. Adding more depth beyond L3 does not
        improve R² and can actually decrease performance. This suggests most price-relevant order
        flow occurs at the top 2-3 price levels in this market.
        """)

    # =========================================================================
    # TAB 2: TIME WINDOWS
    # =========================================================================
    with tabs[2]:
        st.header("Time Window Analysis")

        # Page explanation
        st.markdown("""
        This page examines how the choice of aggregation time window affects the OFI-price relationship.
        Time windows range from 1 minute to 90 minutes. Longer windows produce higher R² because they
        aggregate more order flow events and reduce noise. Additionally, the q^a size exponent
        transformation is analyzed here. Using a=0.3 (which compresses large orders) consistently
        outperforms raw sizes (a=1.0), indicating that extreme order sizes add noise to the signal.
        """)

        st.markdown("---")

        # Time window line chart
        st.markdown("**R² by Time Window:** Line chart showing how R² changes with aggregation window length. The shaded area shows the range across all configurations.")
        fig_tw = plot_r2_by_time_window(df)
        st.plotly_chart(fig_tw, use_container_width=True)

        st.markdown("---")

        # q^a exponent comparison
        st.subheader("q^a Size Exponent Analysis")
        st.markdown("Bar chart comparing maximum R² for each size exponent. Lower exponents compress large order sizes, reducing their impact on OFI.")
        fig_exp = plot_r2_by_exponent(df)
        st.plotly_chart(fig_exp, use_container_width=True)

        st.markdown("""
        ---
        **Key Finding:** Longer time windows (60-90 min) produce higher R². The q^a=0.3 transformation
        consistently outperforms raw sizes (a=1.0) by approximately 22 percentage points. The size
        transformation is more important than the choice of level depth.
        """)

    # =========================================================================
    # TAB 3: PHASE ANALYSIS
    # =========================================================================
    with tabs[3]:
        st.header("Market Phase Analysis")

        # Page explanation
        st.markdown("""
        This page analyzes how OFI predictive power changes across different phases of the market
        lifecycle. The data is split into three equal phases: Early (first third), Middle, and Late
        (final third before expiration). For prediction markets, we expect the OFI-price relationship
        to strengthen near expiry as prices converge toward the true outcome. The results confirm this:
        R² increases from 72% in the Early phase to 81% in the Late phase, a 9 percentage point improvement.
        """)

        st.markdown("---")

        if phase_df is not None and len(phase_df) > 0:
            # Phase comparison bar chart
            st.markdown("**Average R² by Phase:** Bar chart comparing mean R² across market phases for the top 20 configurations.")
            fig_phase = plot_phase_comparison(phase_df)
            if fig_phase:
                st.plotly_chart(fig_phase, use_container_width=True)

            st.markdown("---")

            # Phase heatmap
            st.subheader("R² by Phase for Top Configurations")
            st.markdown("Heatmap showing how each top configuration performs in Early, Middle, and Late market phases.")
            fig_phase_heat = plot_phase_heatmap(phase_df)
            if fig_phase_heat:
                st.plotly_chart(fig_phase_heat, use_container_width=True)

            # Phase table
            st.markdown("### Detailed Phase Results")
            st.markdown("Table showing exact R² values for each configuration across all three phases.")
            phase_display = phase_df[['level_config', 'time_window', 'outlier_method', 'q_exponent', 'overall_r2', 'early_r2', 'middle_r2', 'late_r2']].copy()
            phase_display = phase_display.drop_duplicates()
            for col in ['overall_r2', 'early_r2', 'middle_r2', 'late_r2']:
                phase_display[col] = (phase_display[col] * 100).round(2)
            phase_display.columns = ['Level', 'Window', 'Outlier', 'q^a', 'Overall %', 'Early %', 'Middle %', 'Late %']
            st.dataframe(phase_display, use_container_width=True, hide_index=True)

            st.markdown("""
            ---
            **Key Finding:** OFI becomes MORE predictive as the market approaches expiration.
            Early phase averages 72% R², Middle phase 77% R², and Late phase 81% R². This pattern
            is consistent with prediction market theory: near expiry, prices converge toward outcomes
            and order flow becomes more informative about the final resolution.
            """)
        else:
            st.warning("Phase analysis data not available. Run evaluation script first.")

    # =========================================================================
    # TAB 4: CUMULATIVE MLOFI
    # =========================================================================
    with tabs[4]:
        st.header("Cumulative MLOFI Analysis")

        # Page explanation
        st.markdown("""
        This page presents **Cumulative MLOFI** - an alternative approach to multi-level OFI calculation.

        **Standard MLOFI (Sum OFI at each level):**
        ```
        cumulative_ofi_L2 = ofi_l1 + ofi_l2
        ```

        **Cumulative MLOFI (Sum sizes first, then calculate OFI):**
        ```
        cumulative_ofi_L2 = OFI(bid_size_l1 + bid_size_l2, ask_size_l1 + ask_size_l2)
        ```

        This approach first aggregates the liquidity through level N, then calculates OFI on the
        aggregated quantities. This captures the **total order flow imbalance across the top N levels
        as a single unified measure**.

        **Configuration:**
        - **Levels:** L2, L5, L10
        - **Exponent:** a = 0.3 (size transformation)
        - **Time Windows:** 45, 60, 90 minutes
        - **Outlier Methods:** Raw, Z-Score, Winsorized
        """)

        st.markdown("---")

        # Load cumulative MLOFI data
        cumulative_df = load_cumulative_mlofi()

        if cumulative_df is None or len(cumulative_df) == 0:
            st.warning("Cumulative MLOFI data not found. Please run the analysis script first:")
            st.code("python mlofi/07_cumulative_mlofi.py", language="bash")
        else:
            # Summary metrics
            st.subheader("Results Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Configurations Tested", f"{len(cumulative_df)}")
            with col2:
                st.metric("Best R²", f"{cumulative_df['r2'].max()*100:.2f}%")
            with col3:
                best_row = cumulative_df.iloc[0]
                st.metric("Best Level", best_row['level'])
            with col4:
                st.metric("Best Window", f"{best_row['time_window']} min")

            st.markdown("---")

            # Top configurations bar chart
            st.subheader("All Configurations Ranked by R²")

            cumulative_df_sorted = cumulative_df.sort_values('r2', ascending=False).copy()
            cumulative_df_sorted['config_label'] = cumulative_df_sorted.apply(
                lambda r: f"{r['level']} | {r['time_window']}min | {r['outlier_method']}",
                axis=1
            )

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=cumulative_df_sorted['r2'] * 100,
                y=cumulative_df_sorted['config_label'],
                orientation='h',
                marker=dict(
                    color=cumulative_df_sorted['r2'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title='R²')
                ),
                text=[f"{r*100:.1f}%" for r in cumulative_df_sorted['r2']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>R² = %{x:.2f}%<extra></extra>'
            ))

            fig_bar.update_layout(
                title="Cumulative MLOFI: All Configurations by R²",
                xaxis_title="R² (%)",
                yaxis_title="Configuration",
                height=500,
                yaxis=dict(autorange='reversed'),
                showlegend=False
            )

            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # Comparison by Level
            st.subheader("R² by Level Configuration")

            level_summary = cumulative_df.groupby('level').agg({
                'r2': ['max', 'mean']
            }).reset_index()
            level_summary.columns = ['level', 'max_r2', 'mean_r2']
            level_summary = level_summary.sort_values('max_r2', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig_level = go.Figure()
                fig_level.add_trace(go.Bar(
                    x=level_summary['level'],
                    y=level_summary['max_r2'] * 100,
                    name='Max R²',
                    marker_color=['#2ca02c', '#1f77b4', '#d62728'],
                    text=[f"{r*100:.1f}%" for r in level_summary['max_r2']],
                    textposition='outside'
                ))
                fig_level.update_layout(
                    title="Maximum R² by Level",
                    xaxis_title="Level (Cumulative through L)",
                    yaxis_title="R² (%)",
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_level, use_container_width=True)

            with col2:
                fig_level_mean = go.Figure()
                fig_level_mean.add_trace(go.Bar(
                    x=level_summary['level'],
                    y=level_summary['mean_r2'] * 100,
                    name='Mean R²',
                    marker_color=['#2ca02c', '#1f77b4', '#d62728'],
                    text=[f"{r*100:.1f}%" for r in level_summary['mean_r2']],
                    textposition='outside'
                ))
                fig_level_mean.update_layout(
                    title="Mean R² by Level",
                    xaxis_title="Level (Cumulative through L)",
                    yaxis_title="R² (%)",
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_level_mean, use_container_width=True)

            st.markdown("---")

            # Comparison by Time Window
            st.subheader("R² by Time Window")

            tw_summary = cumulative_df.groupby('time_window').agg({
                'r2': ['max', 'mean']
            }).reset_index()
            tw_summary.columns = ['time_window', 'max_r2', 'mean_r2']

            fig_tw = go.Figure()
            fig_tw.add_trace(go.Scatter(
                x=tw_summary['time_window'],
                y=tw_summary['max_r2'] * 100,
                mode='lines+markers',
                name='Max R²',
                line=dict(color=COLORS['success'], width=3),
                marker=dict(size=12)
            ))
            fig_tw.add_trace(go.Scatter(
                x=tw_summary['time_window'],
                y=tw_summary['mean_r2'] * 100,
                mode='lines+markers',
                name='Mean R²',
                line=dict(color=COLORS['primary'], width=2, dash='dash'),
                marker=dict(size=10)
            ))
            fig_tw.update_layout(
                title="R² by Time Window",
                xaxis_title="Time Window (minutes)",
                yaxis_title="R² (%)",
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_tw, use_container_width=True)

            st.markdown("---")

            # Heatmap: Level x Time Window
            st.subheader("R² Heatmap: Level x Time Window")

            # Create pivot for each outlier method
            selected_outlier = st.selectbox(
                "Select Outlier Method:",
                cumulative_df['outlier_method'].unique(),
                key="cumulative_outlier"
            )

            filtered_for_heatmap = cumulative_df[cumulative_df['outlier_method'] == selected_outlier]

            if len(filtered_for_heatmap) > 0:
                pivot = filtered_for_heatmap.pivot_table(
                    index='level',
                    columns='time_window',
                    values='r2'
                ) * 100

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[f"{t}min" for t in pivot.columns],
                    y=pivot.index,
                    colorscale='RdYlGn',
                    text=[[f"{v:.1f}" for v in row] for row in pivot.values],
                    texttemplate="%{text}%",
                    textfont={"size": 12},
                    hovertemplate='Level: %{y}<br>Window: %{x}<br>R²: %{z:.1f}%<extra></extra>',
                    colorbar=dict(title='R² (%)')
                ))

                fig_heatmap.update_layout(
                    title=f"R² Heatmap ({selected_outlier})",
                    xaxis_title="Time Window",
                    yaxis_title="Level",
                    height=300
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("---")

            # Full Results Table
            st.subheader("Full Results Table")
            display_df = cumulative_df.copy()
            display_df['r2_pct'] = (display_df['r2'] * 100).round(2)
            display_df = display_df.sort_values('r2', ascending=False)

            st.dataframe(
                display_df[['level', 'time_window', 'outlier_method', 'r2_pct', 'beta', 'n_obs']].rename(columns={
                    'level': 'Level',
                    'time_window': 'Window (min)',
                    'outlier_method': 'Outlier Method',
                    'r2_pct': 'R² (%)',
                    'beta': 'Beta',
                    'n_obs': 'N obs'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("""
            ---
            **Key Insight:** Cumulative MLOFI aggregates liquidity through level N before calculating OFI.
            This differs from standard MLOFI which sums individual level OFIs. The cumulative approach
            treats the top N levels as a single liquidity pool, which may better capture the overall
            order flow dynamics in markets with varying depth.
            """)

    # =========================================================================
    # TAB 5: DETAILED EXPLORER
    # =========================================================================
    with tabs[5]:
        st.header("Configuration Explorer")

        # Page explanation
        st.markdown("""
        This page allows interactive exploration of specific configuration combinations. Select a
        level configuration and time window to see how different outlier methods and q^a exponents
        perform for that specific setup. This is useful for understanding the sensitivity of results
        to filtering choices and for identifying robust configurations that perform well across
        multiple outlier handling approaches.
        """)

        st.markdown("---")

        st.markdown("**Select Configuration:** Choose a level depth and time window to explore.")
        col1, col2 = st.columns(2)

        with col1:
            selected_level = st.selectbox(
                "Select Level Config:",
                sorted(df['level_config'].unique())
            )

        with col2:
            selected_window = st.selectbox(
                "Select Time Window:",
                sorted(df['time_window'].unique()),
                index=list(sorted(df['time_window'].unique())).index(60) if 60 in df['time_window'].values else 0
            )

        st.markdown("---")

        # Outlier method comparison
        st.markdown("**R² by Outlier Method:** Comparison of how different outlier filtering approaches affect R² for the selected configuration.")
        fig_outlier = plot_outlier_comparison(df, selected_level, selected_window)
        if fig_outlier:
            st.plotly_chart(fig_outlier, use_container_width=True)

        # Filtered results table
        st.markdown("### All Results for Selected Configuration")
        st.markdown("Complete table of all q^a exponent and outlier method combinations for the selected level and time window.")
        filtered = df[(df['level_config'] == selected_level) &
                      (df['time_window'] == selected_window)].copy()
        filtered['r2_pct'] = (filtered['r2'] * 100).round(2)
        filtered = filtered.sort_values('r2', ascending=False)

        st.dataframe(
            filtered[['outlier_method', 'q_exponent', 'r2_pct', 'beta', 'n_obs']].rename(columns={
                'outlier_method': 'Outlier Method',
                'q_exponent': 'q^a',
                'r2_pct': 'R² (%)',
                'beta': 'Beta (slope)',
                'n_obs': 'N obs'
            }),
            use_container_width=True,
            hide_index=True
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    **MLOFI Analysis** | Following Cont, Kukanov & Stoikov (2011) methodology
    - OFI and price change measured in SAME time window (contemporaneous)
    - Full data regression (no train/test split)
    - In-sample R² reported
    """)


if __name__ == "__main__":
    main()
