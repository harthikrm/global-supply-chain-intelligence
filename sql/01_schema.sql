-- ============================================================
-- Global Supply Chain Intelligence — DuckDB Schema
-- ============================================================
-- Database: data/processed/supply_chain.db
-- ============================================================

-- 1. Macroeconomic Indicators (from FRED API)
CREATE TABLE IF NOT EXISTS macro_indicators (
    date           DATE NOT NULL,
    series_id      VARCHAR NOT NULL,
    value          DOUBLE,
    series_name    VARCHAR,
    yoy_change     DOUBLE,   -- Year-over-year % change
    mom_change     DOUBLE,   -- Month-over-month % change
    PRIMARY KEY (date, series_id)
);

-- 2. International Trade Flows (from UN Comtrade API)
CREATE TABLE IF NOT EXISTS trade_flows (
    year              INTEGER NOT NULL,
    reporter_code     INTEGER NOT NULL,
    reporter_name     VARCHAR,
    partner_code      INTEGER,
    partner_name      VARCHAR,
    hs_code           VARCHAR NOT NULL,
    commodity_name    VARCHAR,
    trade_value_usd   DOUBLE,
    net_weight_kg     DOUBLE,
    flow_type         VARCHAR,   -- 'Import' or 'Export'
);

-- 3. SKU Master Data (from synthetic generation)
CREATE TABLE IF NOT EXISTS skus (
    sku_id                 VARCHAR PRIMARY KEY,
    category               VARCHAR NOT NULL,
    supplier_country       VARCHAR NOT NULL,
    lead_time_days         INTEGER,
    unit_cost_usd          DOUBLE,
    holding_cost_pct       DOUBLE,   -- Monthly holding cost as % of unit cost
    stockout_cost_usd      DOUBLE,   -- Per-unit stockout penalty
    reorder_quantity       INTEGER,
    disruption_sensitivity VARCHAR   -- 'High', 'Medium', 'Low'
);

-- 4. Weekly Demand Records (from synthetic generation)
CREATE TABLE IF NOT EXISTS weekly_demand (
    week_start_date      DATE NOT NULL,
    sku_id               VARCHAR NOT NULL,
    demand_units         INTEGER,
    actual_lead_time_days INTEGER,
    stockout_flag        BOOLEAN DEFAULT FALSE,
    disruption_event_id  VARCHAR,
    PRIMARY KEY (week_start_date, sku_id)
);

-- 5. Disruption Events (labeled ground truth)
CREATE TABLE IF NOT EXISTS disruption_events (
    event_id            VARCHAR PRIMARY KEY,
    event_name          VARCHAR NOT NULL,
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    affected_countries  VARCHAR,   -- Comma-separated list
    affected_hs_codes   VARCHAR,   -- Comma-separated list
    severity_score      DOUBLE     -- 0.0 to 1.0
);
