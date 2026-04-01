"""
Feature Engineering — Feature Matrix Builder
==============================================
Consolidates features from all modules (graph, detection,
forecasting, inventory) into a single feature matrix
for the ML prediction model.
"""

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')


def build_feature_matrix(graph_centrality: pd.DataFrame = None,
                          disruption_scores: pd.DataFrame = None,
                          forecast_results: dict = None,
                          optimization_results: pd.DataFrame = None,
                          con: duckdb.DuckDBPyConnection = None) -> pd.DataFrame:
    """
    Build the complete feature matrix for stockout prediction.

    Features span 4 analytical layers:
    1. Graph features (betweenness, pagerank, disruption impact)
    2. Disruption detection features (CUSUM flags, Mahalanobis)
    3. Forecasting features (demand forecast, uncertainty, trend)
    4. Inventory features (weeks of cover, safety stock adequacy)

    Target: stockout_risk_30d = 1 if stockout in next 4 weeks
    """
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)

    print("\n  ── Building Feature Matrix ──")

    # ── Load base data ──
    demand_df = con.execute("""
        SELECT wd.*, s.category, s.supplier_country, s.lead_time_days,
               s.unit_cost_usd, s.disruption_sensitivity,
               s.holding_cost_pct, s.stockout_cost_usd, s.reorder_quantity
        FROM weekly_demand wd
        JOIN skus s ON wd.sku_id = s.sku_id
        ORDER BY wd.sku_id, wd.week_start_date
    """).fetchdf()

    demand_df['week_start_date'] = pd.to_datetime(demand_df['week_start_date'])

    # Assign week number (1-based)
    min_date = demand_df['week_start_date'].min()
    demand_df['week_num'] = ((demand_df['week_start_date'] - min_date).dt.days // 7) + 1

    # ── Target Variable ──
    # stockout_risk_30d = 1 if any stockout in next 4 weeks
    print("    Computing target variable (stockout_risk_30d)...")
    demand_df = demand_df.sort_values(['sku_id', 'week_start_date'])

    demand_df['stockout_next_4w'] = demand_df.groupby('sku_id')['stockout_flag'].transform(
        lambda x: x.rolling(window=4, min_periods=1).max().shift(-4)
    ).fillna(0).astype(int)

    # ── Feature Group 1: Graph Features ──
    print("    Adding graph features...")
    if graph_centrality is not None and len(graph_centrality) > 0:
        # Map supplier nodes to SKU's country-category
        graph_features = graph_centrality[
            graph_centrality['node_type'] == 'supplier'
        ][['node_id', 'betweenness_centrality', 'pagerank',
           'degree_centrality', 'clustering_coefficient', 'risk_tier']]

        # Create mapping: country-category -> metrics
        graph_lookup = {}
        for _, row in graph_features.iterrows():
            node_id = row['node_id']
            graph_lookup[node_id] = {
                'betweenness_centrality': row['betweenness_centrality'],
                'pagerank': row['pagerank'],
                'degree_centrality': row['degree_centrality'],
                'clustering_coefficient': row['clustering_coefficient'],
            }

        demand_df['graph_node_id'] = demand_df['supplier_country'] + '-' + demand_df['category']

        for col in ['betweenness_centrality', 'pagerank', 'degree_centrality', 'clustering_coefficient']:
            demand_df[col] = demand_df['graph_node_id'].map(
                lambda x: graph_lookup.get(x, {}).get(col, 0)
            )

        # Risk tier encoding
        risk_tier_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
        demand_df['supplier_country_risk_tier'] = demand_df['graph_node_id'].map(
            lambda x: risk_tier_map.get(
                graph_features[graph_features['node_id'] == x]['risk_tier'].values[0]
                if len(graph_features[graph_features['node_id'] == x]) > 0
                else 'Low',
                0
            )
        )
    else:
        # Generate synthetic graph features
        rng = np.random.RandomState(42)
        demand_df['betweenness_centrality'] = rng.uniform(0, 0.08, len(demand_df))
        demand_df['pagerank'] = rng.uniform(0.005, 0.05, len(demand_df))
        demand_df['degree_centrality'] = rng.uniform(0.02, 0.15, len(demand_df))
        demand_df['clustering_coefficient'] = rng.uniform(0, 0.5, len(demand_df))
        risk_map = {'High': 3, 'Medium': 2, 'Low': 1}
        demand_df['supplier_country_risk_tier'] = demand_df['disruption_sensitivity'].map(
            risk_map
        ).fillna(1)

    # ── Feature Group 2: Disruption Detection Features ──
    print("    Adding disruption detection features...")
    if disruption_scores is not None and len(disruption_scores) > 0:
        # Map weekly disruption scores to demand records
        ds = disruption_scores.copy()
        ds.index = pd.to_datetime(ds.index)

        # Create rolling CUSUM flags (4-week)
        demand_df = demand_df.sort_values('week_start_date')

        ds_aligned = ds.reindex(demand_df['week_start_date'].unique())
        ds_aligned = ds_aligned.fillna(method='ffill').fillna(0)

        score_map = ds_aligned['disruption_score'].to_dict()
        cusum_map = ds_aligned['cusum_ratio'].to_dict() if 'cusum_ratio' in ds_aligned.columns else {}
        mahal_map = ds_aligned['mahal_ratio'].to_dict() if 'mahal_ratio' in ds_aligned.columns else {}

        demand_df['disruption_score_current'] = demand_df['week_start_date'].map(score_map).fillna(0)
        demand_df['cusum_flag_rolling_4w'] = demand_df['week_start_date'].map(cusum_map).fillna(0)
        demand_df['mahalanobis_distance_current'] = demand_df['week_start_date'].map(mahal_map).fillna(0)
    else:
        # Synthetic disruption features
        demand_df['disruption_score_current'] = 0.0
        demand_df['cusum_flag_rolling_4w'] = 0.0
        demand_df['mahalanobis_distance_current'] = 0.0

        # Increase during disruption windows
        for evt_id, weeks_range in [('EVT001', (8, 16)), ('EVT002', (92, 104)), ('EVT003', (124, 132))]:
            mask = demand_df['disruption_event_id'] == evt_id
            demand_df.loc[mask, 'disruption_score_current'] = np.random.uniform(0.4, 0.9, mask.sum())
            demand_df.loc[mask, 'cusum_flag_rolling_4w'] = np.random.uniform(0.3, 0.8, mask.sum())
            demand_df.loc[mask, 'mahalanobis_distance_current'] = np.random.uniform(0.5, 1.0, mask.sum())

    # Weeks since last disruption flag
    demand_df['is_disrupted'] = (demand_df['disruption_event_id'].notna()).astype(int)
    demand_df['weeks_since_last_disruption'] = demand_df.groupby('sku_id')['is_disrupted'].transform(
        lambda x: x.cumsum().diff().fillna(0).abs()
    ).clip(0, 52)

    # ── Feature Group 3: Forecasting Features ──
    print("    Adding forecasting features...")
    # Rolling demand statistics as proxy for forecast
    demand_df['demand_rolling_4w'] = demand_df.groupby('sku_id')['demand_units'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    demand_df['demand_rolling_8w'] = demand_df.groupby('sku_id')['demand_units'].transform(
        lambda x: x.rolling(window=8, min_periods=1).mean()
    )
    demand_df['demand_std_4w'] = demand_df.groupby('sku_id')['demand_units'].transform(
        lambda x: x.rolling(window=4, min_periods=1).std()
    ).fillna(0)

    # Forecast uncertainty width (proxy: std / mean)
    demand_df['forecast_uncertainty_width'] = (
        demand_df['demand_std_4w'] / demand_df['demand_rolling_4w'].clip(lower=1)
    ).clip(0, 5)

    # Demand trend slope (8-week linear regression slope)
    def compute_slope(x):
        if len(x) < 3:
            return 0
        t = np.arange(len(x))
        try:
            slope = np.polyfit(t, x, 1)[0]
        except Exception:
            slope = 0
        return slope

    demand_df['demand_trend_slope'] = demand_df.groupby('sku_id')['demand_units'].transform(
        lambda x: x.rolling(window=8, min_periods=3).apply(compute_slope, raw=True)
    ).fillna(0)

    # Disruption-adjusted forecast flag
    demand_df['disruption_adjusted_forecast_flag'] = (
        demand_df['disruption_score_current'] > 0.3
    ).astype(int)

    # ── Feature Group 4: Inventory Features ──
    print("    Adding inventory features...")
    # Simulated current inventory (based on reorder quantity decay)
    demand_df['current_inventory_weeks_of_cover'] = (
        demand_df['reorder_quantity'] / demand_df['demand_rolling_4w'].clip(lower=1)
    ).clip(0, 20)

    # Days to reorder point
    demand_df['days_to_reorder_point'] = (
        demand_df['current_inventory_weeks_of_cover'] * 7
    ).clip(0, 140)

    # Safety stock adequacy ratio
    demand_df['safety_stock_adequacy_ratio'] = (
        demand_df['current_inventory_weeks_of_cover'] / 4.0  # Assume 4 weeks optimal
    ).clip(0, 5)

    # Lead time deviation from normal
    demand_df['lead_time_deviation_from_normal'] = (
        demand_df['actual_lead_time_days'] - demand_df['lead_time_days']
    ) / demand_df['lead_time_days'].clip(lower=1)

    # ── Categorical Encodings ──
    print("    Encoding categorical features...")
    category_map = {cat: i for i, cat in enumerate(demand_df['category'].unique())}
    country_map = {c: i for i, c in enumerate(demand_df['supplier_country'].unique())}

    demand_df['sku_category_encoded'] = demand_df['category'].map(category_map)
    demand_df['supplier_country_encoded'] = demand_df['supplier_country'].map(country_map)

    # In disruption window flag
    demand_df['in_disruption_window'] = demand_df['disruption_event_id'].notna().astype(int)

    # ── Select Final Features ──
    feature_columns = [
        # Graph features
        'betweenness_centrality', 'pagerank', 'degree_centrality',
        'clustering_coefficient', 'supplier_country_risk_tier',
        # Disruption detection features
        'cusum_flag_rolling_4w', 'mahalanobis_distance_current',
        'disruption_score_current', 'weeks_since_last_disruption',
        # Forecasting features
        'demand_rolling_4w', 'demand_rolling_8w', 'demand_std_4w',
        'forecast_uncertainty_width', 'demand_trend_slope',
        'disruption_adjusted_forecast_flag',
        # Inventory features
        'current_inventory_weeks_of_cover', 'days_to_reorder_point',
        'safety_stock_adequacy_ratio', 'lead_time_deviation_from_normal',
        # Raw features
        'sku_category_encoded', 'supplier_country_encoded',
        'unit_cost_usd', 'in_disruption_window',
    ]

    # Metadata columns to keep
    meta_columns = ['week_start_date', 'sku_id', 'week_num', 'category',
                     'supplier_country', 'demand_units', 'stockout_flag']

    # Target
    target_column = 'stockout_next_4w'

    # Build final matrix
    all_columns = meta_columns + feature_columns + [target_column]
    feature_matrix = demand_df[all_columns].copy()

    # Clean up NaN/inf
    for col in feature_columns:
        feature_matrix[col] = feature_matrix[col].replace([np.inf, -np.inf], np.nan)
        feature_matrix[col] = feature_matrix[col].fillna(0)

    positive_rate = feature_matrix[target_column].mean() * 100
    print(f"\n    Feature matrix shape: {feature_matrix.shape}")
    print(f"    Feature count: {len(feature_columns)}")
    print(f"    Positive class rate: {positive_rate:.1f}%")

    return feature_matrix


def split_chronological(feature_matrix: pd.DataFrame,
                         train_end_week: int = 104,
                         val_end_week: int = 130) -> tuple:
    """
    Chronological train/validation/test split.
    Train: weeks 1-104 (2022-2023)
    Validate: weeks 105-130 (first half 2024)
    Test: weeks 131-156 (second half 2024)
    """
    feature_columns = [
        'betweenness_centrality', 'pagerank', 'degree_centrality',
        'clustering_coefficient', 'supplier_country_risk_tier',
        'cusum_flag_rolling_4w', 'mahalanobis_distance_current',
        'disruption_score_current', 'weeks_since_last_disruption',
        'demand_rolling_4w', 'demand_rolling_8w', 'demand_std_4w',
        'forecast_uncertainty_width', 'demand_trend_slope',
        'disruption_adjusted_forecast_flag',
        'current_inventory_weeks_of_cover', 'days_to_reorder_point',
        'safety_stock_adequacy_ratio', 'lead_time_deviation_from_normal',
        'sku_category_encoded', 'supplier_country_encoded',
        'unit_cost_usd', 'in_disruption_window',
    ]

    target = 'stockout_next_4w'

    train = feature_matrix[feature_matrix['week_num'] <= train_end_week]
    val = feature_matrix[(feature_matrix['week_num'] > train_end_week) &
                          (feature_matrix['week_num'] <= val_end_week)]
    test = feature_matrix[feature_matrix['week_num'] > val_end_week]

    X_train = train[feature_columns]
    y_train = train[target]
    X_val = val[feature_columns]
    y_val = val[target]
    X_test = test[feature_columns]
    y_test = test[target]

    print(f"\n    Train: {len(X_train):,} samples (pos rate: {y_train.mean()*100:.1f}%)")
    print(f"    Val:   {len(X_val):,} samples (pos rate: {y_val.mean()*100:.1f}%)")
    print(f"    Test:  {len(X_test):,} samples (pos rate: {y_test.mean()*100:.1f}%)")

    return (X_train, y_train, X_val, y_val, X_test, y_test, feature_columns)
