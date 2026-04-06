# Global Supply Chain Intelligence

> **When a geopolitical disruption hits — a conflict, a port closure, a trade route blockage — which products are most at risk, how bad is the downstream cost impact, and what is the optimal inventory response before the stockout happens?**

A multi-source supply chain intelligence platform integrating UN Comtrade trade flows, FRED macroeconomic indicators, and synthetic manufacturing data. Detects geopolitical disruptions with CUSUM and multivariate Mahalanobis distance, quantifies network chokepoints via NetworkX betweenness centrality, and predicts 30-day stockout risk with an XGBoost/LightGBM ensemble (PR-AUC 0.29) across a 50-SKU representative sample and 78,000 demand records.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| SKUs tracked | 50-SKU representative sample across 7 categories |
| Weekly demand records | 78,000 (3 years) |
| Commodity trade flows | 10 HS codes × 6 countries × 7 years |
| FRED macro indicators | 10 series from 2018-2024 |
| Disruption events modeled | 3 (Ukraine, Red Sea, Singapore) |
| ML features | 20+ spanning 4 analytical layers |
| Monte Carlo simulations | 10,000 per SKU per scenario |
| Ensemble PR-AUC | **0.29** (test set) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA SOURCES                          │
│  FRED API │ UN Comtrade │ Synthetic Manufacturing       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              MODULE 0: DATA PIPELINE                     │
│  ETL → DuckDB (5 tables, 6 feature views)               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌────────┐     ┌────────────┐     ┌────────────┐
│Module A│     │  Module B  │     │  Module C  │
│Network │     │ Disruption │     │  Demand    │
│ Graph  │     │ Detection  │     │ Forecast   │
└───┬────┘     └─────┬──────┘     └─────┬──────┘
    │                │                  │
    │                ▼                  ▼
    │          ┌────────────┐     ┌────────────┐
    │          │  Module D  │     │  Module E  │
    │          │ Inventory  │◄────│    ML      │
    │          │   Optim.   │     │ Ensemble   │
    │          └─────┬──────┘     └─────┬──────┘
    │                │                  │
    └────────────────┼──────────────────┘
                     ▼
          ┌─────────────────────┐
          │    MODULE F         │
          │ Streamlit Dashboard │
          │    (6 tabs)         │
          └─────────────────────┘
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/harthikrm/global-supply-chain-intelligence.git
cd global-supply-chain-intelligence
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

```bash
cp .env.example .env
# Edit .env with your FRED and UN Comtrade API keys
# Without API keys, the pipeline uses realistic synthetic data
```

### 3. Run Data Pipeline

```bash
python -m src.ingest
```

### 4. Train Prediction Model

```bash
python -m src.models
```

This trains the XGBoost + LightGBM ensemble, computes SHAP values, and saves predictions to `data/processed/model_results.pkl`. Required for Tab 6 (Stockout Prediction) in the dashboard.

### 5. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
global-supply-chain-intelligence/
├── data/
│   ├── raw/
│   │   ├── comtrade/          ← UN Comtrade JSON responses
│   │   ├── fred/              ← FRED series CSVs
│   │   └── synthetic/         ← Generated manufacturer/SKU data
│   └── processed/
│       └── supply_chain.db    ← DuckDB database
├── notebooks/
│   ├── 00_data_pipeline.ipynb
│   ├── 01_network_graph.ipynb
│   ├── 02_disruption_detection.ipynb
│   ├── 03_demand_forecasting.ipynb
│   ├── 04_inventory_optimization.ipynb
│   ├── 05_disruption_prediction.ipynb
│   └── 06_dashboard.ipynb
├── sql/
│   ├── 01_schema.sql              ← DuckDB table definitions
│   ├── 02_feature_engineering.sql ← 6 SQL views (HHI, CUSUM, rolling stats)
│   └── 03_analysis_queries.sql    ← Reusable analytical queries
├── src/
│   ├── ingest.py              ← Multi-API ETL pipeline
│   ├── synthetic.py           ← Synthetic data generation (seed=42)
│   ├── features.py            ← ML feature matrix builder
│   ├── graph.py               ← NetworkX supply chain graph
│   ├── anomaly.py             ← CUSUM + Mahalanobis detection
│   ├── forecast.py            ← Hierarchical ETS + MinT reconciliation
│   ├── optimize.py            ← Monte Carlo inventory simulation
│   ├── models.py              ← XGBoost + LightGBM + ensemble
│   └── viz.py                 ← Plotly chart factory
├── app/
│   └── streamlit_app.py       ← 6-tab dashboard
├── requirements.txt
└── README.md
```

---

## Module Deep Dives

### Module A — Supply Chain Network Graph
- **Directed weighted graph** with 4 node layers: Supplier → Manufacturer → Distributor → Retailer
- **Centrality metrics**: degree, betweenness (chokepoint detection), PageRank, clustering coefficient
- **Disruption simulation**: node removal + edge weight reduction for 3 historical events
- **Key finding**: Betweenness centrality identifies single points of failure before disruptions happen

### Module B — Disruption Detection
- **CUSUM charts** per FRED indicator with 2018–2021 baseline, k=0.5σ, h=5σ threshold
- **Multivariate Mahalanobis distance** on weekly indicator matrix — detects correlated anomalies
- **Composite disruption score** = 0.4×CUSUM + 0.4×Mahalanobis + 0.2×trade deviation
- **Validation**: Precision/Recall/F1 against labeled disruption events, target recall ≥ 0.85

### Module C — Demand Forecasting Under Uncertainty
- **Hierarchical ETS** at 4 levels: Total → Category → Country → SKU
- **MinT reconciliation**: `y_reconciled = S @ (Sᵀ W⁻¹ S)⁻¹ @ Sᵀ W⁻¹ @ y_base`
- **Bootstrap prediction intervals**: 1,000 resamples → 80% and 95% PI
- **Disruption-adjusted forecasts**: Category-specific multipliers from historical analogues
- **Metrics**: MASE (target < 1.0), CRPS for distributional accuracy

### Module D — Inventory Optimization
- **EOQ baseline**: Q* = √(2DS/H) per SKU
- **Monte Carlo grid search**: 10,000 simulations across (R, Q) space
- **Three scenarios**: baseline, moderate (+40% LT, +20% demand), severe (+80% LT, +35% demand, 10% outage)
- **Key insight**: Safety stock required approximately doubles under severe disruption

### Module E — Disruption Prediction Model
- **20+ features** spanning graph (betweenness, PageRank), detection (CUSUM, Mahalanobis), forecasting (uncertainty width, trend), and inventory (weeks of cover, safety stock adequacy)
- **Champion XGBoost** + **Challenger LightGBM** with stacked meta-learner ensemble
- **SHAP TreeExplainer**: Graph and detection features are top stockout predictors
- **Metrics**: PR-AUC 0.29, Precision@10% 0.36, Recall 70.9%, avg prediction lead time 5.5 weeks

### Module F — Streamlit Dashboard
6 interactive tabs with light premium theme (warm cream background, olive green accents):
1. **Global Risk Overview** — KPI cards, world choropleth, FRED sparklines
2. **Supply Chain Network** — Interactive graph with risk-colored nodes
3. **Disruption Detection** — CUSUM + Mahalanobis charts with event overlays
4. **Demand Forecasting** — SKU-level forecast fans with prediction intervals
5. **Inventory Optimization** — Monte Carlo distributions, scenario comparison
6. **Stockout Prediction** — Risk ranking table, SHAP attribution, confusion matrix

---

## SQL Feature Engineering

| View | Description |
|------|-------------|
| `trade_flow_yoy_change` | LAG window function for YoY % change per HS code per reporter |
| `commodity_disruption_score` | Z-score vs 2018-2021 baseline, flag when \|z\| > 2.0 |
| `supplier_concentration_index` | HHI = Σ(market_share²) per category — concentration risk |
| `lead_time_deviation` | Actual vs baseline lead time deviation per SKU |
| `rolling_demand_stats` | 4w/8w rolling mean, std, WoW change using window functions |
| `inventory_coverage_ratio` | Weeks of inventory cover remaining |

---

## 📝 Resume Bullet Angles

**Supply Chain / Operations (Toyota, Amazon):**
> Multi-source supply chain intelligence platform integrating UN Comtrade trade flows, FRED macroeconomic indicators, and synthetic manufacturing data — detecting geopolitical disruptions with CUSUM and multivariate Mahalanobis distance, quantifying network chokepoints via NetworkX betweenness centrality, and predicting stockout risk with an XGBoost/LightGBM ensemble (PR-AUC 0.29) across a 50-SKU representative sample and 78,000 demand records.

**ML/DS Roles:**
> XGBoost + LightGBM champion-challenger ensemble with stacked meta-learner trained on 20+ features spanning graph centrality, disruption detection signals, hierarchical forecast outputs, and inventory state — chronological train/test split with SHAP TreeExplainer attribution showing graph features as primary stockout predictors.

**Data Engineering Roles:**
> Multi-API ETL pipeline (FRED + UN Comtrade + synthetic generation) into DuckDB schema with 6 SQL feature engineering views including HHI supplier concentration index, CUSUM disruption scores, and rolling demand statistics using LAG/LEAD window functions.

---

## 🔗 LinkedIn Hook

> "Red Sea shipping disruptions added 12 days to Asia-Europe freight in 2024. I built a system to detect supply chain disruptions before they peak, quantify which products are most at risk, and recommend the optimal inventory response. Here's what global trade data found."

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
