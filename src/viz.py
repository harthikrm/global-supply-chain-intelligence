"""
Visualization Functions — Plotly Chart Factory
================================================
Shared visualization functions used by both notebooks
and the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Theme ────────────────────────────────────────────────────────
DARK_BG = '#0D0D0D'
CARD_BG = '#1A1A1A'
AMBER = '#F59E0B'
AMBER_LIGHT = '#FBBF24'
RED = '#EF4444'
GREEN = '#10B981'
BLUE = '#3B82F6'
PURPLE = '#8B5CF6'
GRAY = '#6B7280'
WHITE = '#F9FAFB'

CHART_TEMPLATE = dict(
    layout=dict(
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=WHITE, family='Inter, sans-serif'),
        title=dict(font=dict(size=18, color=WHITE)),
        xaxis=dict(gridcolor='#333333', zerolinecolor='#333333'),
        yaxis=dict(gridcolor='#333333', zerolinecolor='#333333'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#333'),
        margin=dict(l=60, r=30, t=60, b=40),
    )
)


def apply_dark_theme(fig):
    """Apply the dark industrial theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=WHITE, family='Inter, sans-serif'),
        xaxis=dict(gridcolor='#222222', zerolinecolor='#333333'),
        yaxis=dict(gridcolor='#222222', zerolinecolor='#333333'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#333',
                    font=dict(color=WHITE)),
        margin=dict(l=60, r=30, t=60, b=40),
    )
    return fig


def create_kpi_card(value, label, delta=None, delta_color=None, prefix='', suffix=''):
    """Create a KPI indicator figure."""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+delta" if delta else "number",
        value=value,
        title=dict(text=label, font=dict(size=14, color=GRAY)),
        number=dict(
            prefix=prefix, suffix=suffix,
            font=dict(size=36, color=AMBER),
        ),
        delta=dict(
            reference=value - delta if delta else None,
            relative=False,
            increasing=dict(color=RED if delta_color == 'inverse' else GREEN),
            decreasing=dict(color=GREEN if delta_color == 'inverse' else RED),
        ) if delta else None,
    ))

    fig.update_layout(
        height=140,
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def create_world_risk_map(risk_data: pd.DataFrame) -> go.Figure:
    """Create a choropleth map colored by disruption risk per country."""
    country_iso = {
        'China': 'CHN', 'Germany': 'DEU', 'Japan': 'JPN',
        'South Korea': 'KOR', 'India': 'IND', 'Taiwan': 'TWN',
        'Vietnam': 'VNM', 'Mexico': 'MEX', 'Thailand': 'THA',
        'Brazil': 'BRA', 'Turkey': 'TUR', 'USA': 'USA',
        'Switzerland': 'CHE', 'Luxembourg': 'LUX', 'Spain': 'ESP',
    }

    if 'iso_code' not in risk_data.columns:
        risk_data = risk_data.copy()
        risk_data['iso_code'] = risk_data['country'].map(country_iso)

    fig = go.Figure(data=go.Choropleth(
        locations=risk_data['iso_code'],
        z=risk_data['risk_score'],
        text=risk_data['country'],
        colorscale=[
            [0, '#10B981'],
            [0.5, '#F59E0B'],
            [1.0, '#EF4444'],
        ],
        colorbar=dict(
            title='Risk Score',
            bgcolor=CARD_BG,
            tickfont=dict(color=WHITE),
            titlefont=dict(color=WHITE),
        ),
        hovertemplate='<b>%{text}</b><br>Risk Score: %{z:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='Global Supply Chain Risk Map', font=dict(size=18, color=WHITE)),
        geo=dict(
            bgcolor=DARK_BG,
            lakecolor=DARK_BG,
            landcolor='#1A1A1A',
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#333333',
            countrycolor='#333333',
            projection_type='natural earth',
        ),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def create_cusum_chart(cusum_results: dict, series_id: str = None) -> go.Figure:
    """Create CUSUM time-series chart with detection flags."""
    if series_id and series_id in cusum_results:
        data = cusum_results[series_id]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[f'{series_id} — Raw Values', 'CUSUM Chart'],
                            vertical_spacing=0.12)

        # Raw values
        fig.add_trace(go.Scatter(
            x=data.index, y=data['value'],
            mode='lines', name='Value',
            line=dict(color=BLUE, width=1.5),
        ), row=1, col=1)

        # CUSUM positive
        fig.add_trace(go.Scatter(
            x=data.index, y=data['cusum_pos'],
            mode='lines', name='CUSUM+',
            line=dict(color=AMBER, width=1.5),
        ), row=2, col=1)

        # CUSUM negative
        fig.add_trace(go.Scatter(
            x=data.index, y=data['cusum_neg'],
            mode='lines', name='CUSUM−',
            line=dict(color=PURPLE, width=1.5),
        ), row=2, col=1)

        # Threshold line
        threshold = data['threshold_h'].iloc[0]
        fig.add_hline(y=threshold, line=dict(color=RED, dash='dash', width=1),
                       annotation_text=f'Threshold (h={threshold:.0f})',
                       row=2, col=1)

        # Flag verticals
        flags = data[data['cusum_flag']]
        for idx in flags.index:
            fig.add_vline(x=idx, line=dict(color=RED, width=0.5, dash='dot'),
                           row=1, col=1)
            fig.add_vline(x=idx, line=dict(color=RED, width=0.5, dash='dot'),
                           row=2, col=1)

        fig.update_layout(height=500, title=f'CUSUM Detection — {series_id}')
        return apply_dark_theme(fig)

    # Multi-series view
    series_list = list(cusum_results.keys())[:6]
    fig = make_subplots(rows=len(series_list), cols=1, shared_xaxes=True,
                        subplot_titles=series_list, vertical_spacing=0.05)

    for i, sid in enumerate(series_list):
        data = cusum_results[sid]
        fig.add_trace(go.Scatter(
            x=data.index, y=data['cusum_pos'],
            mode='lines', name=f'{sid} CUSUM+',
            line=dict(width=1.5),
        ), row=i+1, col=1)

        flags = data[data['cusum_flag']]
        for idx in flags.index:
            fig.add_vline(x=idx, line=dict(color=RED, width=0.3, dash='dot'),
                           row=i+1, col=1)

    fig.update_layout(height=150 * len(series_list), showlegend=False,
                       title='CUSUM Detection — All Signals')
    return apply_dark_theme(fig)


def create_mahalanobis_chart(mahal_data: pd.DataFrame) -> go.Figure:
    """Create Mahalanobis distance chart with threshold."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=mahal_data.index, y=mahal_data['mahalanobis_distance'],
        mode='lines', name='Mahalanobis Distance',
        line=dict(color=AMBER, width=1.5),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
    ))

    threshold = mahal_data['threshold'].iloc[0]
    fig.add_hline(y=threshold, line=dict(color=RED, dash='dash', width=2),
                   annotation_text=f'99th Percentile Threshold ({threshold:.1f})')

    # Highlight anomalies
    anomalies = mahal_data[mahal_data['anomaly_flag']]
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies['mahalanobis_distance'],
        mode='markers', name='Anomaly Detected',
        marker=dict(color=RED, size=8, symbol='x'),
    ))

    fig.update_layout(
        title='Multivariate Mahalanobis Distance — Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Mahalanobis Distance',
        height=350,
    )

    return apply_dark_theme(fig)


def create_forecast_fan_chart(train_series: pd.Series, test_series: pd.Series,
                                forecast: np.ndarray, intervals: dict,
                                sku_id: str = '', category: str = '') -> go.Figure:
    """Create demand forecast fan chart with prediction intervals."""
    fig = go.Figure()

    # Historical demand
    fig.add_trace(go.Scatter(
        x=train_series.index, y=train_series.values,
        mode='lines', name='Historical',
        line=dict(color=BLUE, width=1.5),
    ))

    # Actual test values
    fig.add_trace(go.Scatter(
        x=test_series.index, y=test_series.values,
        mode='lines', name='Actual',
        line=dict(color=WHITE, width=1.5),
    ))

    # Forecast
    forecast_dates = test_series.index[:len(forecast)]
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast,
        mode='lines', name='Forecast',
        line=dict(color=AMBER, width=2),
    ))

    # Prediction intervals
    for level, interval in sorted(intervals.items()):
        lower = interval['lower'][:len(forecast_dates)]
        upper = interval['upper'][:len(forecast_dates)]
        alpha = 0.15 if level == 0.95 else 0.25

        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor=f'rgba(245, 158, 11, {alpha})',
            line=dict(width=0),
            name=f'{int(level*100)}% PI',
            showlegend=True,
        ))

    fig.update_layout(
        title=f'Demand Forecast — {sku_id} ({category})' if sku_id else 'Demand Forecast',
        xaxis_title='Date',
        yaxis_title='Demand Units',
        height=400,
    )

    return apply_dark_theme(fig)


def create_monte_carlo_distribution(cost_distribution: np.ndarray,
                                      optimal_cost: float,
                                      sku_id: str = '') -> go.Figure:
    """Create histogram of Monte Carlo simulation cost distribution."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=cost_distribution,
        nbinsx=50,
        marker_color=AMBER,
        opacity=0.7,
        name='Cost Distribution',
    ))

    fig.add_vline(x=optimal_cost, line=dict(color=GREEN, dash='dash', width=2),
                   annotation_text=f'Optimal: ${optimal_cost:,.0f}')

    mean_cost = np.mean(cost_distribution)
    p95_cost = np.percentile(cost_distribution, 95)

    fig.add_vline(x=mean_cost, line=dict(color=BLUE, dash='dot', width=1.5),
                   annotation_text=f'Mean: ${mean_cost:,.0f}')
    fig.add_vline(x=p95_cost, line=dict(color=RED, dash='dot', width=1.5),
                   annotation_text=f'P95: ${p95_cost:,.0f}')

    fig.update_layout(
        title=f'Monte Carlo Cost Distribution — {sku_id}' if sku_id else 'Annual Cost Distribution',
        xaxis_title='Annual Total Cost ($)',
        yaxis_title='Frequency',
        height=350,
    )

    return apply_dark_theme(fig)


def create_shap_summary(shap_values: np.ndarray, feature_names: list,
                          X_sample: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create a SHAP summary plot using Plotly."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[-top_n:]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[feature_names[i] for i in sorted_idx],
        x=mean_abs_shap[sorted_idx],
        orientation='h',
        marker=dict(
            color=mean_abs_shap[sorted_idx],
            colorscale=[[0, BLUE], [0.5, AMBER], [1.0, RED]],
        ),
    ))

    fig.update_layout(
        title='SHAP Feature Importance — Stockout Prediction',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        height=max(350, top_n * 30),
    )

    return apply_dark_theme(fig)


def create_confusion_matrix_chart(cm: np.ndarray, labels=['No Stockout', 'Stockout']) -> go.Figure:
    """Create a confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Predicted<br>{l}' for l in labels],
        y=[f'Actual<br>{l}' for l in labels],
        colorscale=[[0, '#1A1A1A'], [1, AMBER]],
        texttemplate='%{z:,}',
        textfont=dict(size=16, color=WHITE),
        showscale=False,
    ))

    fig.update_layout(
        title='Confusion Matrix',
        height=350,
        width=400,
    )

    return apply_dark_theme(fig)


def create_disruption_score_chart(scores: pd.DataFrame) -> go.Figure:
    """Create disruption score time series with alert levels."""
    fig = go.Figure()

    # Score line
    fig.add_trace(go.Scatter(
        x=scores.index, y=scores['disruption_score'],
        mode='lines', name='Disruption Score',
        line=dict(color=AMBER, width=2),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
    ))

    # Alert thresholds
    fig.add_hline(y=0.6, line=dict(color=RED, dash='dash', width=1),
                   annotation_text='Critical')
    fig.add_hline(y=0.4, line=dict(color=AMBER, dash='dash', width=1),
                   annotation_text='High')
    fig.add_hline(y=0.2, line=dict(color=BLUE, dash='dash', width=1),
                   annotation_text='Elevated')

    fig.update_layout(
        title='Composite Disruption Score',
        xaxis_title='Date',
        yaxis_title='Score',
        height=350,
    )

    return apply_dark_theme(fig)


def create_sparkline(series: pd.Series, color: str = AMBER, height: int = 60) -> go.Figure:
    """Create a small sparkline chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(series))),
        y=series.values,
        mode='lines',
        line=dict(color=color, width=1.5),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    return fig


def create_risk_table_colors(prob: float) -> str:
    """Return risk color based on probability."""
    if prob > 0.6:
        return RED
    elif prob > 0.3:
        return AMBER
    return GREEN
