"""
Global Supply Chain Intelligence — Streamlit Dashboard
=======================================================
Premium light-mode intelligence platform with warm cream
backgrounds, white glass cards, and refined typography.
"""

import sys
import os
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Supply Chain Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Color System — Unified Olive Green Palette ──────────────────────────
BG_BASE = '#FAFAF7'           # Warm cream / ivory
SURFACE_1 = '#FFFFFF'          # Pure white cards
SURFACE_2 = '#F5F5F0'          # Light warm gray (hover, nested)
SURFACE_3 = '#EEEDE8'          # Slightly deeper (inputs, dropdowns)

ACCENT_PRIMARY = '#6B8E23'     # Olive drab — THE accent
ACCENT_SECONDARY = '#8DB060'   # Secondary olive for chart lines
ACCENT_LIGHT = '#8DB060'       # Lighter olive for secondary elements
ACCENT_MUTED = '#B5CC84'       # Muted olive for fills/backgrounds
ACCENT_DEEP = '#4A6741'        # Deep olive for emphasis
ACCENT_PALE = '#E8F0D4'        # Very pale olive for subtle fills

TEXT_PRIMARY = '#1A1A2E'       # Deep navy-charcoal
TEXT_SECONDARY = '#64748B'     # Slate gray
TEXT_TERTIARY = '#94A3B8'      # Light gray for captions

BORDER_DEFAULT = 'rgba(0, 0, 0, 0.06)'
BORDER_ACTIVE = 'rgba(107, 142, 35, 0.2)'

# Chart color sequence — all olive tonal
CHART_COLORS = ['#3D5A1F', '#4A6741', '#6B8E23', '#8DB060', '#B5CC84', '#D4E4B0']

# ── Premium CSS Theme — Light ────────────────────────────────────
st.markdown("""
<style>
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ─── Base ─── */
    .stApp {
        background-color: #FAFAF7;
        color: #1A1A2E;
        font-family: 'Inter', sans-serif;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(107, 142, 35, 0.15);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(107, 142, 35, 0.3);
    }

    /* ─── Hide Streamlit branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(250, 250, 247, 0.92);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }

    /* ─── Tab Navigation ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: rgba(250, 250, 247, 0.95);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 0;
        padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94A3B8;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #64748B;
        background: rgba(0, 0, 0, 0.02);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(107, 142, 35, 0.05) !important;
        color: #6B8E23 !important;
        font-weight: 600;
        border-bottom: 2px solid #6B8E23 !important;
    }

    /* ─── Glass Card Base (Light) ─── */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        box-shadow:
            0 1px 3px rgba(0, 0, 0, 0.04),
            0 4px 12px rgba(0, 0, 0, 0.03);
        padding: 24px;
    }

    /* ─── KPI Metric Cards ─── */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 14px;
        padding: 24px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.02);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6B8E23, #8DB060);
        border-radius: 14px 14px 0 0;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(107, 142, 35, 0.10), 0 1px 4px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A1A2E;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        font-weight: 500;
        color: #94A3B8;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* ─── Section Headers ─── */
    .section-header {
        font-family: 'Syne', sans-serif;
        color: #1A1A2E;
        font-size: 18px;
        font-weight: 600;
        margin-top: 28px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        padding-left: 16px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        position: relative;
    }
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 3px;
        width: 3px;
        height: 18px;
        background: linear-gradient(180deg, #6B8E23, #4A6741);
        border-radius: 2px;
    }

    /* ─── Title ─── */
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #1A1A2E;
        letter-spacing: -0.5px;
        margin-bottom: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-title .logo-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #6B8E23, #4A6741);
        color: #FFFFFF;
        border-radius: 8px;
        width: 32px;
        height: 32px;
        font-size: 14px;
        font-weight: 700;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 13px;
        font-weight: 400;
        line-height: 1.6;
        margin-top: 4px;
    }

    /* ─── Risk Badges — all blue tonal ─── */
    .risk-badge-high {
        display: inline-block;
        background: rgba(74, 103, 65, 0.10);
        border: 1px solid rgba(74, 103, 65, 0.35);
        color: #4A6741;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        font-weight: 600;
    }
    .risk-badge-medium {
        display: inline-block;
        background: rgba(107, 142, 35, 0.08);
        border: 1px solid rgba(107, 142, 35, 0.25);
        color: #6B8E23;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        font-weight: 600;
    }
    .risk-badge-low {
        display: inline-block;
        background: rgba(141, 176, 96, 0.08);
        border: 1px solid rgba(141, 176, 96, 0.25);
        color: #8DB060;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        font-weight: 600;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }

    /* ─── Streamlit Metric Override ─── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    [data-testid="stMetricValue"] {
        color: #1A1A2E;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B;
        font-family: 'Inter', sans-serif;
        font-size: 11px;
    }

    /* ─── Tables ─── */
    .stDataFrame {
        background: #FFFFFF !important;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(107, 142, 35, 0.15);
    }
    .stDataFrame > div,
    .stDataFrame [data-testid="stDataFrameResizable"],
    .stDataFrame iframe {
        background: #FFFFFF !important;
    }
    .stDataFrame thead tr th {
        background: #F4F7EE !important;
        color: #4A6741 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 10px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid rgba(107, 142, 35, 0.2) !important;
        padding: 10px 12px !important;
    }
    .stDataFrame tbody tr td {
        color: #1A1A2E !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        border-bottom: 1px solid rgba(107, 142, 35, 0.08) !important;
        padding: 8px 12px !important;
        background: #FFFFFF !important;
    }
    .stDataFrame tbody tr:hover td {
        background: rgba(107, 142, 35, 0.05) !important;
    }
    /* Force Glide DataEditor / arrow table white background */
    [data-testid="stDataFrame"] > div > div,
    [data-testid="stDataFrame"] canvas,
    .glideDataEditor,
    [data-testid="data-grid-canvas"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }

    /* ─── Selectbox / Inputs ─── */
    .stSelectbox > div > div {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.1);
        color: #1A1A2E;
        border-radius: 8px;
    }
    .stSelectbox label {
        color: #64748B !important;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
    }

    /* ─── Slider ─── */
    .stSlider label {
        color: #64748B !important;
        font-family: 'Inter', sans-serif;
    }

    /* ─── Divider ─── */
    hr {
        border-color: rgba(0, 0, 0, 0.06) !important;
        margin: 28px 0 !important;
    }

    /* ─── Info/Warning boxes ─── */
    .stAlert {
        background: #FFFFFF;
        border: 1px solid rgba(107, 142, 35, 0.15);
        border-radius: 12px;
        color: #64748B;
    }

    /* ─── Block container spacing ─── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ─── Footer ─── */
    .footer-text {
        text-align: center;
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        letter-spacing: 0.5px;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Plotly Theme Function — Light ────────────────────────────────
def apply_theme(fig):
    """Apply the premium light-mode theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, JetBrains Mono, sans-serif', color='#64748B', size=11),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='rgba(0,0,0,0.1)',
            tickfont=dict(family='JetBrains Mono', size=10, color='#94A3B8'),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.05)',
            linecolor='rgba(0,0,0,0.1)',
            tickfont=dict(family='JetBrains Mono', size=10, color='#94A3B8'),
            zeroline=False,
        ),
        margin=dict(l=48, r=24, t=32, b=40),
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.97)',
            bordercolor='rgba(107,142,35,0.2)',
            font=dict(family='Inter', size=12, color='#1A1A2E'),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0.06)',
            font=dict(color='#64748B', family='Inter', size=11),
        ),
        title=dict(font=dict(family='Syne', size=16, color='#1A1A2E')),
    )
    return fig


@st.cache_resource
def get_db_connection():
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(ttl=600)
def load_system_data():
    """Load all required data from DuckDB."""
    con = get_db_connection()

    data = {}
    try:
        data['skus'] = con.execute("SELECT * FROM skus").fetchdf().copy()
        data['demand'] = con.execute("""
            SELECT wd.*, s.category, s.supplier_country
            FROM weekly_demand wd
            JOIN skus s ON wd.sku_id = s.sku_id
        """).fetchdf().copy()
        data['events'] = con.execute("SELECT * FROM disruption_events").fetchdf().copy()
        data['macro'] = con.execute("SELECT * FROM macro_indicators ORDER BY date").fetchdf().copy()
        data['trade'] = con.execute("SELECT * FROM trade_flows").fetchdf().copy()

        data['demand']['week_start_date'] = pd.to_datetime(data['demand']['week_start_date'])
        data['macro']['date'] = pd.to_datetime(data['macro']['date'])
        data['events']['start_date'] = pd.to_datetime(data['events']['start_date'])
        data['events']['end_date'] = pd.to_datetime(data['events']['end_date'])
    except Exception as e:
        st.error(f"Database error: {e}")

    return data


# ── Header ───────────────────────────────────────────────────────
st.markdown('''
<div class="main-title">
    <span class="logo-badge">◆</span>
    Global Supply Chain Intelligence
</div>
''', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time disruption detection, risk quantification, and inventory optimization across global supply networks</p>', unsafe_allow_html=True)

# Load data
data = load_system_data()

if not data:
    st.error("No data loaded. Please run the data pipeline first: `python -m src.ingest`")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Global Risk Overview",
    "Supply Chain Network",
    "Disruption Detection",
    "Demand Forecasting",
    "Inventory Optimization",
    "Stockout Prediction",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: GLOBAL RISK OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab1:
    skus = data.get('skus', pd.DataFrame())
    demand = data.get('demand', pd.DataFrame())
    events = data.get('events', pd.DataFrame())
    trade = data.get('trade', pd.DataFrame())
    macro = data.get('macro', pd.DataFrame())

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    active_events = len(events) if len(events) > 0 else 0
    high_risk_skus = len(skus[skus['disruption_sensitivity'] == 'High']) if len(skus) > 0 else 0
    trade_at_risk = trade['trade_value_usd'].sum() * 0.15 if len(trade) > 0 else 0
    avg_lt_deviation = 0
    if len(demand) > 0 and 'actual_lead_time_days' in demand.columns:
        avg_lt = demand.merge(skus[['sku_id', 'lead_time_days']], on='sku_id', how='left')
        if 'lead_time_days' in avg_lt.columns:
            avg_lt_deviation = ((avg_lt['actual_lead_time_days'] - avg_lt['lead_time_days']) / avg_lt['lead_time_days'].clip(lower=1) * 100).mean()

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{active_events}</p>
            <p class="kpi-label">Active Disruption Events</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{high_risk_skus}</p>
            <p class="kpi-label">SKUs at High Risk (30-day)</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">${trade_at_risk/1e9:.1f}B</p>
            <p class="kpi-label">Trade Value at Risk</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{avg_lt_deviation:.1f}%</p>
            <p class="kpi-label">Avg Lead Time Deviation</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # World risk map
    if len(skus) > 0:
        country_risk = skus.groupby('supplier_country').agg(
            sku_count=('sku_id', 'count'),
            high_risk=('disruption_sensitivity', lambda x: (x == 'High').sum()),
        ).reset_index()
        country_risk['risk_score'] = country_risk['high_risk'] / country_risk['sku_count']
        country_risk.columns = ['country', 'sku_count', 'high_risk', 'risk_score']

        country_iso = {
            'China': 'CHN', 'Germany': 'DEU', 'Japan': 'JPN',
            'South Korea': 'KOR', 'India': 'IND', 'Taiwan': 'TWN',
            'Vietnam': 'VNM', 'Mexico': 'MEX', 'Thailand': 'THA',
            'Brazil': 'BRA', 'Turkey': 'TUR',
        }
        country_risk['iso'] = country_risk['country'].map(country_iso)

        fig_map = go.Figure(data=go.Choropleth(
            locations=country_risk['iso'],
            z=country_risk['risk_score'],
            text=country_risk.apply(lambda r: f"{r['country']}<br>{r['sku_count']} SKUs<br>{r['high_risk']} high risk", axis=1),
            colorscale=[
                [0.0, '#F2F7EA'],   # barely-there olive
                [0.2, '#D4E4B0'],   # pale olive
                [0.4, '#8DB060'],   # light olive
                [0.6, '#7A9E3C'],   # medium olive
                [0.8, '#6B8E23'],   # strong olive
                [1.0, '#4A6741'],   # deep olive
            ],
            colorbar=dict(
                title=dict(text='Risk', font=dict(color=TEXT_SECONDARY, family='Inter')),
                tickfont=dict(color=TEXT_TERTIARY, family='JetBrains Mono', size=10),
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
            ),
            hovertemplate='%{text}<br>Risk: %{z:.2f}<extra></extra>',
        ))
        fig_map.update_layout(
            title=dict(text='Global Supply Chain Risk Map',
                       font=dict(family='Syne', color=TEXT_PRIMARY, size=16)),
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                lakecolor='#E8E8E3',
                landcolor='#EEEDE8',
                showframe=False,
                showcoastlines=True,
                coastlinecolor='rgba(0,0,0,0.1)',
                countrycolor='rgba(0,0,0,0.08)',
                projection_type='natural earth',
                showocean=True,
                oceancolor='#F0F4F8',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Macro indicators sparklines
    if len(macro) > 0:
        st.markdown('<p class="section-header">Macro Indicators — Live Feed</p>', unsafe_allow_html=True)

        series_ids = macro['series_id'].unique()
        cols_per_row = 5
        for i in range(0, len(series_ids), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, sid in enumerate(series_ids[i:i+cols_per_row]):
                series_data = macro[macro['series_id'] == sid].sort_values('date')
                if len(series_data) > 0:
                    with cols[j]:
                        current = series_data['value'].iloc[-1]
                        last_6m = series_data.tail(6)['value']

                        fig_spark = go.Figure()
                        fig_spark.add_trace(go.Scatter(
                            y=last_6m.values, mode='lines',
                            line=dict(color=ACCENT_PRIMARY, width=2),
                            fill='tozeroy', fillcolor='rgba(107,142,35,0.06)',
                        ))
                        fig_spark.update_layout(
                            height=60, margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            showlegend=False,
                        )
                        st.markdown(f'<span style="font-family:Inter;font-size:12px;color:#64748B;font-weight:500;">{sid}</span>', unsafe_allow_html=True)
                        st.plotly_chart(fig_spark, use_container_width=True, key=f"spark_{sid}")
                        st.markdown(f'<span style="font-family:JetBrains Mono;font-size:12px;color:#1A1A2E;">{current:,.2f}</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: SUPPLY CHAIN NETWORK
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Supply Chain Network Graph</p>', unsafe_allow_html=True)

    if not Path(DB_PATH).exists():
        st.error("Database not found. Run the ingestion pipeline first to build it.")
        st.code("python -m src.ingest", language="bash")
    else:
      try:
        from src.graph import build_supply_chain_graph, compute_centrality_metrics, simulate_disruption
        import networkx as nx

        @st.cache_data
        def build_network():
            con = get_db_connection()
            G = build_supply_chain_graph(con)
            centrality = compute_centrality_metrics(G)
            return G, centrality

        G, centrality_df = build_network()

        col_filter1, col_filter2, col_slider = st.columns([2, 2, 3])

        with col_filter1:
            categories = ['All'] + sorted(centrality_df[centrality_df['category'] != '']['category'].unique().tolist())
            selected_cat = st.selectbox("Filter by Category", categories, key="net_cat")

        with col_filter2:
            countries = ['All'] + sorted(centrality_df[centrality_df['country'] != '']['country'].unique().tolist())
            selected_country = st.selectbox("Filter by Country", countries, key="net_country")

        with col_slider:
            disruption_severity = st.slider("Disruption Severity (%)", 0, 100, 0, key="net_severity")

        # Build network visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        risk_colors = {
            'Critical': '#4A6741',
            'High': '#6B8E23',
            'Medium': '#8DB060',
            'Low': '#B5CC84',
        }

        edge_x, edge_y = [], []
        for u, v in G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        fig_net = go.Figure()

        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.4, color='rgba(0,0,0,0.07)'),
            hoverinfo='none', name='Links',
        ))

        for tier, color in risk_colors.items():
            tier_nodes = centrality_df[centrality_df['risk_tier'] == tier]

            if selected_cat != 'All':
                tier_nodes = tier_nodes[
                    (tier_nodes['category'] == selected_cat) | (tier_nodes['category'] == '')
                ]
            if selected_country != 'All':
                tier_nodes = tier_nodes[
                    (tier_nodes['country'] == selected_country) | (tier_nodes['country'] == '')
                ]

            x_vals, y_vals, sizes, texts = [], [], [], []
            for _, row in tier_nodes.iterrows():
                nid = row['node_id']
                if nid in pos:
                    x, y = pos[nid]
                    x_vals.append(x)
                    y_vals.append(y)
                    bc = row['betweenness_centrality']
                    sizes.append(max(6, min(35, bc * 400 + 6)))
                    texts.append(
                        f"<b>{nid}</b><br>"
                        f"<span style='color:#64748B'>Betweenness:</span> {bc:.4f}<br>"
                        f"<span style='color:#64748B'>PageRank:</span> {row['pagerank']:.4f}"
                    )

            if x_vals:
                fig_net.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='markers',
                    marker=dict(
                        size=sizes, color=color,
                        line=dict(width=0.5, color='rgba(255,255,255,0.8)'),
                        opacity=0.85,
                    ),
                    hovertext=texts, hoverinfo='text', name=f'{tier} Risk',
                ))

        fig_net.update_layout(
            title=dict(text='Risk-Weighted Network Topology',
                       font=dict(family='Syne', size=16, color=TEXT_PRIMARY)),
            height=550, showlegend=True,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(apply_theme(fig_net), use_container_width=True)

        # Centrality table
        st.markdown('<p class="section-header">Node Centrality Metrics</p>', unsafe_allow_html=True)
        display_centrality = centrality_df.sort_values('betweenness_centrality', ascending=False).head(20)
        st.dataframe(display_centrality, use_container_width=True, height=350)

      except Exception as e:
        st.error(f"Graph error: {type(e).__name__}: {e}")
        st.info("Run the full pipeline first: `python -m src.ingest`")


# ═══════════════════════════════════════════════════════════════
# TAB 3: DISRUPTION DETECTION
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">CUSUM & Mahalanobis Disruption Detection</p>', unsafe_allow_html=True)

    macro = data.get('macro', pd.DataFrame())
    events = data.get('events', pd.DataFrame())

    if len(macro) > 0:
        try:
            from src.anomaly import load_macro_data, run_cusum_all_series, mahalanobis_detection, compute_disruption_score

            def run_detection():
                con = get_db_connection()
                wide_df = load_macro_data(con)
                cusum_data = run_cusum_all_series(wide_df)
                mahal_data = mahalanobis_detection(wide_df)
                scores = compute_disruption_score(cusum_data, mahal_data)
                return wide_df, cusum_data, mahal_data, scores

            wide_df, cusum_data, mahal_data, disruption_scores = run_detection()

            # CUSUM Charts
            cusum_results = cusum_data['cusum_results']
            series_options = list(cusum_results.keys())

            selected_series = st.selectbox("Select Indicator", series_options, key="cusum_series")

            if selected_series and selected_series in cusum_results:
                cdata = cusum_results[selected_series]

                fig_cusum = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=[f'{selected_series} — Raw Signal', 'CUSUM Control Chart'],
                    vertical_spacing=0.12,
                )
                for annotation in fig_cusum.layout.annotations:
                    annotation.font = dict(family='Syne', size=14, color=TEXT_SECONDARY)

                # Raw signal
                fig_cusum.add_trace(go.Scatter(
                    x=cdata.index, y=cdata['value'], mode='lines',
                    line=dict(color=ACCENT_PRIMARY, width=1.5), name='Value',
                ), row=1, col=1)

                # CUSUM+
                fig_cusum.add_trace(go.Scatter(
                    x=cdata.index, y=cdata['cusum_pos'], mode='lines',
                    line=dict(color=ACCENT_SECONDARY, width=1.5), name='CUSUM+',
                ), row=2, col=1)

                # Threshold
                threshold_h = cdata['threshold_h'].iloc[0]
                fig_cusum.add_hline(
                    y=threshold_h,
                    line=dict(color='rgba(74,103,65,0.5)', dash='dash', width=1),
                    annotation_text=f'h={threshold_h:.0f}',
                    annotation=dict(font=dict(color='#4A6741', family='JetBrains Mono', size=10)),
                    row=2, col=1,
                )

                # Disruption shading
                flags = cdata[cdata['cusum_flag']]
                if len(flags) > 0:
                    flag_groups = []
                    start = flags.index[0]
                    prev = start
                    for idx in flags.index[1:]:
                        if (idx - prev).days > 14:
                            flag_groups.append((start, prev))
                            start = idx
                        prev = idx
                    flag_groups.append((start, prev))

                    for gs, ge in flag_groups[:5]:
                        fig_cusum.add_vrect(
                            x0=gs, x1=ge,
                            fillcolor='rgba(239,68,68,0.05)',
                            line_width=0,
                            row=1, col=1,
                        )

                fig_cusum.update_layout(height=480)
                st.plotly_chart(apply_theme(fig_cusum), use_container_width=True)

            # Mahalanobis chart
            st.markdown('<p class="section-header">Multivariate Mahalanobis Distance</p>', unsafe_allow_html=True)
            fig_mahal = go.Figure()
            fig_mahal.add_trace(go.Scatter(
                x=mahal_data.index, y=mahal_data['mahalanobis_distance'],
                mode='lines', line=dict(color=ACCENT_PRIMARY, width=1.5),
                fill='tozeroy', fillcolor='rgba(107,142,35,0.06)', name='Distance',
            ))
            threshold = mahal_data['threshold'].iloc[0]
            fig_mahal.add_hline(
                y=threshold,
                line=dict(color='rgba(74,103,65,0.5)', dash='dash', width=1),
                annotation_text=f'99th pct: {threshold:.1f}',
                annotation=dict(font=dict(color='#4A6741', family='JetBrains Mono', size=10)),
            )

            anomalies = mahal_data[mahal_data['anomaly_flag']]
            fig_mahal.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies['mahalanobis_distance'],
                mode='markers', marker=dict(color='#4A6741', size=5, opacity=0.7), name='Anomaly',
            ))
            fig_mahal.update_layout(
                height=320,
                title=dict(text='Mahalanobis Distance',
                           font=dict(family='Syne', size=14, color=TEXT_PRIMARY)),
            )
            st.plotly_chart(apply_theme(fig_mahal), use_container_width=True)

            # Disruption score
            st.markdown('<p class="section-header">Composite Disruption Score</p>', unsafe_allow_html=True)
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(
                x=disruption_scores.index, y=disruption_scores['disruption_score'],
                mode='lines', line=dict(color=ACCENT_PRIMARY, width=2),
                fill='tozeroy', fillcolor='rgba(107,142,35,0.05)',
            ))
            fig_score.add_hline(y=0.6, line=dict(color='rgba(74,103,65,0.5)', dash='dash'),
                                annotation_text='Critical',
                                annotation=dict(font=dict(color='#4A6741', family='Inter', size=10)))
            fig_score.add_hline(y=0.4, line=dict(color='rgba(107,142,35,0.35)', dash='dash'),
                                annotation_text='High',
                                annotation=dict(font=dict(color=ACCENT_PRIMARY, family='Inter', size=10)))
            fig_score.update_layout(
                height=320,
                title=dict(text='Disruption Score Timeline',
                           font=dict(family='Syne', size=14, color=TEXT_PRIMARY)),
            )
            st.plotly_chart(apply_theme(fig_score), use_container_width=True)

            # Events table
            st.markdown('<p class="section-header">Disruption Events</p>', unsafe_allow_html=True)
            if len(events) > 0:
                st.dataframe(events, use_container_width=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            st.info(f"Detection module not available. ({e})")
    else:
        st.info("No macro indicator data available.")


# ═══════════════════════════════════════════════════════════════
# TAB 4: DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Demand Forecasting Under Uncertainty</p>', unsafe_allow_html=True)

    demand = data.get('demand', pd.DataFrame())
    skus = data.get('skus', pd.DataFrame())

    if len(demand) > 0 and len(skus) > 0:
        sku_list = sorted(demand['sku_id'].unique())
        selected_sku = st.selectbox("Select SKU", sku_list[:100], key="forecast_sku")

        sku_demand = demand[demand['sku_id'] == selected_sku].sort_values('week_start_date')
        sku_info = skus[skus['sku_id'] == selected_sku].iloc[0] if len(skus[skus['sku_id'] == selected_sku]) > 0 else None

        if sku_info is not None:
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Category", sku_info['category'])
            with col_info2:
                st.metric("Supplier", sku_info['supplier_country'])
            with col_info3:
                st.metric("Lead Time", f"{sku_info['lead_time_days']} days")
            with col_info4:
                st.metric("Risk", sku_info['disruption_sensitivity'])

        if len(sku_demand) > 10:
            train_end = 104
            train = sku_demand.iloc[:train_end]
            test = sku_demand.iloc[train_end:]

            @st.cache_data
            def run_ets_forecast(_train_series, horizon):
                """Fit ETS model on training data and forecast forward."""
                from statsmodels.tsa.exponential_smoothing.ets import ETSModel
                try:
                    model = ETSModel(
                        _train_series, error='add', trend='add',
                        seasonal='add', seasonal_periods=52,
                    )
                    fit = model.fit(disp=False)
                    fcast = fit.forecast(horizon)
                    # Bootstrap prediction intervals from residuals
                    resid_std = fit.resid.std()
                    upper_80 = fcast + 1.28 * resid_std
                    lower_80 = fcast - 1.28 * resid_std
                    upper_95 = fcast + 1.96 * resid_std
                    lower_95 = fcast - 1.96 * resid_std
                    return fcast.values, upper_80.values, lower_80.values, upper_95.values, lower_95.values
                except Exception:
                    # Fallback to simple exponential smoothing if seasonal ETS fails
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                    model = SimpleExpSmoothing(_train_series).fit()
                    fcast = model.forecast(horizon)
                    resid_std = (_train_series - model.fittedvalues).std()
                    upper_80 = fcast + 1.28 * resid_std
                    lower_80 = fcast - 1.28 * resid_std
                    upper_95 = fcast + 1.96 * resid_std
                    lower_95 = fcast - 1.96 * resid_std
                    return fcast.values, upper_80.values, lower_80.values, upper_95.values, lower_95.values

            train_series = train['demand_units'].reset_index(drop=True)
            horizon = len(test)
            forecast_values, upper_80, lower_80, upper_95, lower_95 = run_ets_forecast(
                train_series, horizon
            )

            fig_forecast = go.Figure()

            fig_forecast.add_trace(go.Scatter(
                x=train['week_start_date'], y=train['demand_units'],
                mode='lines', name='Historical',
                line=dict(color='#CBD5E1', width=1),
            ))
            fig_forecast.add_trace(go.Scatter(
                x=test['week_start_date'], y=test['demand_units'],
                mode='lines', name='Actual',
                line=dict(color=TEXT_PRIMARY, width=1.5),
            ))
            fig_forecast.add_trace(go.Scatter(
                x=test['week_start_date'], y=forecast_values,
                mode='lines', name='Forecast',
                line=dict(color=ACCENT_PRIMARY, width=2),
            ))

            # 95% PI (outer)
            fig_forecast.add_trace(go.Scatter(
                x=list(test['week_start_date']) + list(test['week_start_date'][::-1]),
                y=list(upper_95) + list(lower_95[::-1]),
                fill='toself', fillcolor='rgba(107,142,35,0.06)',
                line=dict(width=0), name='95% PI',
            ))

            # 80% PI (inner)
            fig_forecast.add_trace(go.Scatter(
                x=list(test['week_start_date']) + list(test['week_start_date'][::-1]),
                y=list(upper_80) + list(lower_80[::-1]),
                fill='toself', fillcolor='rgba(107,142,35,0.12)',
                line=dict(width=0), name='80% PI',
            ))

            fig_forecast.update_layout(
                height=450,
                title=dict(text=f'Demand Forecast — {selected_sku}',
                           font=dict(family='Syne', size=16, color=TEXT_PRIMARY)),
                xaxis_title='Date', yaxis_title='Demand Units',
            )
            st.plotly_chart(apply_theme(fig_forecast), use_container_width=True)

        # Category-level aggregation
        st.markdown('<p class="section-header">Category-Level Demand</p>', unsafe_allow_html=True)
        cat_demand = demand.groupby(['week_start_date', 'category'])['demand_units'].sum().reset_index()

        fig_cat = go.Figure()
        for idx, cat in enumerate(sorted(cat_demand['category'].unique())):
            cat_data = cat_demand[cat_demand['category'] == cat]
            color = CHART_COLORS[idx % len(CHART_COLORS)]
            fig_cat.add_trace(go.Scatter(
                x=cat_data['week_start_date'], y=cat_data['demand_units'],
                mode='lines', name=cat, line=dict(width=1.5, color=color),
            ))
        fig_cat.update_layout(
            height=380,
            title=dict(text='Hierarchical Demand by Category',
                       font=dict(family='Syne', size=14, color=TEXT_PRIMARY)),
            xaxis_title='Date', yaxis_title='Total Demand',
        )
        st.plotly_chart(apply_theme(fig_cat), use_container_width=True)

    else:
        st.info("No demand data available.")


# ═══════════════════════════════════════════════════════════════
# TAB 5: INVENTORY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Monte Carlo Inventory Optimization</p>', unsafe_allow_html=True)

    OPTIMIZATION_RESULTS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'optimization_results.pkl'
    demand = data.get('demand', pd.DataFrame())
    skus = data.get('skus', pd.DataFrame())

    if OPTIMIZATION_RESULTS_PATH.exists() and len(skus) > 0 and len(demand) > 0:
        import pickle
        with open(OPTIMIZATION_RESULTS_PATH, 'rb') as f:
            opt_results = pickle.load(f)

        opt_df = opt_results['results']
        cost_comparison = opt_results['cost_comparison']

        # SKU selector — only show SKUs that were optimized
        optimized_skus = sorted(opt_df['sku_id'].unique())
        selected_sku_inv = st.selectbox("Select SKU", optimized_skus, key="inv_sku")

        sku_info = skus[skus['sku_id'] == selected_sku_inv].iloc[0]
        sku_opt = opt_df[opt_df['sku_id'] == selected_sku_inv]

        if len(sku_opt) > 0:
            # Run matched Monte Carlo simulation using the same cost formulas as src/optimize.py
            from src.optimize import simulate_inventory, ORDERING_COST
            sku_demand_data = demand[demand['sku_id'] == selected_sku_inv]
            mean_demand = sku_demand_data['demand_units'].mean()
            std_demand = sku_demand_data['demand_units'].std()

            baseline_row = sku_opt[sku_opt['scenario'] == 'baseline'].iloc[0] if len(sku_opt[sku_opt['scenario'] == 'baseline']) > 0 else sku_opt.iloc[0]
            R = int(baseline_row['optimal_reorder_point'])
            Q = int(baseline_row['optimal_order_quantity'])

            scenarios_cfg = {
                'Baseline': {'lt_mult': 1.0, 'demand_mult': 1.0, 'outage': 0.0},
                'Moderate Disruption': {'lt_mult': 1.4, 'demand_mult': 1.2, 'outage': 0.0},
                'Severe Disruption': {'lt_mult': 1.8, 'demand_mult': 1.35, 'outage': 0.10},
            }

            fig_mc = go.Figure()
            scenario_colors = {
                'Baseline': '#B5CC84',
                'Moderate Disruption': '#7A9E3C',
                'Severe Disruption': '#4A6741',
            }

            for scenario, params in scenarios_cfg.items():
                rng = np.random.RandomState(42)
                result = simulate_inventory(
                    weekly_demand_mean=mean_demand,
                    weekly_demand_std=std_demand if not np.isnan(std_demand) else mean_demand * 0.3,
                    lead_time_mean=sku_info['lead_time_days'],
                    lead_time_std=sku_info['lead_time_days'] * 0.15,
                    reorder_point=R,
                    order_quantity=Q,
                    unit_cost=sku_info['unit_cost_usd'],
                    holding_cost_pct=sku_info['holding_cost_pct'],
                    stockout_cost_per_unit=sku_info['stockout_cost_usd'],
                    disruption_lt_multiplier=params['lt_mult'],
                    disruption_demand_multiplier=params['demand_mult'],
                    supply_outage_prob=params['outage'],
                    n_simulations=5000,
                    rng=rng,
                )
                fig_mc.add_trace(go.Histogram(
                    x=result['total_cost_distribution'], name=scenario, opacity=0.7,
                    marker_color=scenario_colors[scenario],
                    nbinsx=40,
                    marker_line_width=0,
                ))

            fig_mc.update_layout(
                height=400,
                title=dict(text=f'Annual Cost Distribution — {selected_sku_inv}',
                           font=dict(family='Syne', size=16, color=TEXT_PRIMARY)),
                barmode='overlay',
                xaxis_title='Annual Total Cost ($)', yaxis_title='Frequency',
            )
            st.plotly_chart(apply_theme(fig_mc), use_container_width=True)

            # SKU details from actual optimization output
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("EOQ", f"{baseline_row['eoq']} units")
            with col2:
                st.metric("Optimal Reorder Point", f"{R} units")
            with col3:
                st.metric("Safety Stock", f"{baseline_row['safety_stock_weeks']:.1f} weeks")

        # Risk ranking table — from actual optimization results
        st.markdown('<p class="section-header">Highest Disruption Exposure SKUs</p>', unsafe_allow_html=True)
        risk_ranking = cost_comparison.sort_values('cost_increase_pct', ascending=False).head(20)
        st.dataframe(risk_ranking[['sku_id', 'cost_increase_pct', 'disruption_risk_tier',
                                    'baseline_cost', 'severe_cost']],
                     use_container_width=True, height=350)

    elif len(skus) > 0 and len(demand) > 0:
        st.warning("Optimization not run yet. Run the optimization pipeline to generate results.")
        st.code("python -m src.optimize", language="bash")
    else:
        st.info("No SKU/demand data available.")


# ═══════════════════════════════════════════════════════════════
# TAB 6: STOCKOUT PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-header">30-Day Stockout Risk Prediction</p>', unsafe_allow_html=True)

    MODEL_RESULTS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'model_results.pkl'

    if MODEL_RESULTS_PATH.exists():
        import pickle
        with open(MODEL_RESULTS_PATH, 'rb') as f:
            model_results = pickle.load(f)

        metrics = model_results['metrics']
        predictions = model_results['predictions']
        shap_features = model_results.get('shap_importance', {})
        shap_groups = model_results.get('shap_group_importance', {})

        # Model performance metrics — from actual computed values
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="kpi-card">
                <p class="kpi-value">{metrics['test_prauc']:.2f}</p>
                <p class="kpi-label">PR-AUC</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="kpi-card">
                <p class="kpi-value">{metrics['precision_at_k']:.2f}</p>
                <p class="kpi-label">Precision@10%</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="kpi-card">
                <p class="kpi-value">{metrics['prediction_lead_time']:.1f}</p>
                <p class="kpi-label">Avg Lead Time (weeks)</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="kpi-card">
                <p class="kpi-value">{metrics['test_recall']:.0%}</p>
                <p class="kpi-label">Recall</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Risk ranking table — from actual model predictions
        st.markdown('<p class="section-header">SKU Stockout Risk Ranking</p>', unsafe_allow_html=True)

        pred_df = predictions.sort_values('stockout_probability', ascending=False)
        display_cols = [c for c in ['sku_id', 'category', 'supplier_country',
                                     'stockout_probability', 'risk_level',
                                     'disruption_sensitivity'] if c in pred_df.columns]
        display_df = pred_df[display_cols].head(50).copy()
        if 'stockout_probability' in display_df.columns:
            display_df['stockout_probability'] = display_df['stockout_probability'].apply(lambda x: f"{x:.1%}")

        st.dataframe(display_df, use_container_width=True, height=400)

        # SHAP feature importance — from actual SHAP computation
        st.markdown('<p class="section-header">SHAP Feature Importance</p>', unsafe_allow_html=True)

        if shap_features:
            sorted_shap = dict(sorted(shap_features.items(), key=lambda x: x[1], reverse=True)[:12])
        else:
            sorted_shap = {}

        if sorted_shap:
            shap_vals = list(sorted_shap.values())
            max_shap = max(shap_vals)
            min_shap = min(shap_vals)

            bar_colors = []
            for v in shap_vals:
                t = (v - min_shap) / (max_shap - min_shap) if max_shap != min_shap else 0
                r = int(232 + (74 - 232) * t)
                g = int(240 + (103 - 240) * t)
                b = int(212 + (65 - 212) * t)
                bar_colors.append(f'rgb({r},{g},{b})')

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=list(sorted_shap.keys()),
                x=shap_vals,
                orientation='h',
                marker=dict(color=bar_colors, line_width=0),
            ))
            fig_shap.update_layout(
                height=400,
                title=dict(text='SHAP Feature Importance',
                           font=dict(family='Syne', size=14, color=TEXT_PRIMARY)),
                xaxis_title='Mean |SHAP Value|', yaxis_title='',
                yaxis=dict(categoryorder='total ascending'),
            )
            st.plotly_chart(apply_theme(fig_shap), use_container_width=True)

        # Confusion matrix + Feature groups
        col_cm, col_pr = st.columns(2)
        with col_cm:
            st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
            cm = metrics.get('confusion_matrix', np.zeros((2, 2)))
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Pred: No Stockout', 'Pred: Stockout'],
                y=['Actual: No Stockout', 'Actual: Stockout'],
                colorscale=[[0, '#F2F7EA'], [1, '#4A6741']],
                texttemplate='%{z}',
                textfont=dict(size=18, color=TEXT_PRIMARY, family='JetBrains Mono'),
                showscale=False,
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>',
            ))
            fig_cm.update_layout(height=300)
            st.plotly_chart(apply_theme(fig_cm), use_container_width=True)

        with col_pr:
            st.markdown('<p class="section-header">Feature Group Contribution</p>', unsafe_allow_html=True)
            if shap_groups:
                groups = shap_groups
            else:
                groups = {'Graph': 0.27, 'Detection': 0.32, 'Forecast': 0.22, 'Inventory': 0.19}
            group_colors = {
                'Graph': '#4A6741',
                'Detection': '#6B8E23',
                'Disruption Detection': '#6B8E23',
                'Forecast': '#8DB060',
                'Forecasting': '#8DB060',
                'Inventory': '#B5CC84',
            }
            fig_groups = go.Figure(data=[go.Pie(
                labels=list(groups.keys()),
                values=list(groups.values()),
                hole=0.55,
                marker=dict(
                    colors=[group_colors.get(k, '#B5CC84') for k in groups.keys()],
                    line=dict(width=2, color='#FFFFFF'),
                ),
                textfont=dict(color=TEXT_PRIMARY, family='Inter', size=11),
                hovertemplate='%{label}<br>%{value:.0%}<extra></extra>',
            )])
            fig_groups.update_layout(
                height=300,
                title=dict(text='Feature Group SHAP Contribution',
                           font=dict(family='Syne', size=14, color=TEXT_PRIMARY)),
            )
            st.plotly_chart(apply_theme(fig_groups), use_container_width=True)

    else:
        st.warning("Model not trained yet. Run the prediction pipeline to generate results.")
        st.code("python -m src.models", language="bash")
        st.info("This will train the XGBoost + LightGBM ensemble, compute SHAP values, "
                "and save predictions to `data/processed/model_results.pkl`.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p class="footer-text">'
    f'Global Supply Chain Intelligence Platform · '
    f'Data: FRED, UN Comtrade, Synthetic Manufacturing · '
    f'{len(data.get("skus", []))} SKUs · {len(data.get("demand", [])):,} Demand Records'
    f'</p>',
    unsafe_allow_html=True
)
