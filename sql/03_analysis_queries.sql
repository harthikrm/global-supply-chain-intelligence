-- ============================================================
-- Global Supply Chain Intelligence — Analysis Queries
-- ============================================================
-- Reusable analytical queries for the dashboard and notebooks.
-- ============================================================

-- 1. Top Disrupted Commodities by Year
-- Identifies which commodities experienced the largest trade value
-- deviations from their historical baseline.
SELECT
    year,
    hs_code,
    commodity_name,
    reporter_name,
    flow_type,
    trade_value_usd,
    z_score,
    disruption_flag
FROM commodity_disruption_score
WHERE ABS(z_score) > 1.5
ORDER BY ABS(z_score) DESC;


-- 2. Supply Concentration Risk by Category
-- Which product categories are most concentrated (vulnerable)?
SELECT
    sci.category,
    sci.hhi_index,
    sci.concentration_level,
    sci.num_supplier_countries,
    COUNT(DISTINCT s.sku_id) AS num_skus,
    AVG(s.lead_time_days) AS avg_lead_time,
    SUM(CASE WHEN s.disruption_sensitivity = 'High' THEN 1 ELSE 0 END) AS high_risk_skus
FROM supplier_concentration_index sci
JOIN skus s ON sci.category = s.category
GROUP BY sci.category, sci.hhi_index, sci.concentration_level, sci.num_supplier_countries
ORDER BY sci.hhi_index DESC;


-- 3. Disruption Impact Timeline
-- Weekly view of demand patterns during disruption events.
SELECT
    de.event_name,
    de.severity_score,
    wd.week_start_date,
    COUNT(DISTINCT wd.sku_id) AS affected_skus,
    SUM(wd.demand_units) AS total_demand,
    AVG(wd.actual_lead_time_days) AS avg_lead_time,
    SUM(CASE WHEN wd.stockout_flag THEN 1 ELSE 0 END) AS stockout_count,
    SUM(CASE WHEN wd.stockout_flag THEN 1 ELSE 0 END)::DOUBLE 
        / COUNT(*) * 100 AS stockout_rate_pct
FROM weekly_demand wd
JOIN disruption_events de ON wd.disruption_event_id = de.event_id
GROUP BY de.event_name, de.severity_score, wd.week_start_date
ORDER BY wd.week_start_date;


-- 4. Baltic Dry Index vs Trade Flow Correlation
-- Macro indicator trends alongside trade flow changes.
SELECT
    mi.date,
    mi.value AS bdi_value,
    mi.yoy_change AS bdi_yoy_change
FROM macro_indicators mi
WHERE mi.series_id = 'BDIY'
ORDER BY mi.date;


-- 5. Lead Time Anomalies During Disruptions
-- SKUs with the largest lead time deviations.
SELECT
    ltd.sku_id,
    s.category,
    s.supplier_country,
    ltd.week_start_date,
    ltd.actual_lead_time_days,
    ltd.baseline_avg_lead_time,
    ltd.lead_time_deviation_days,
    ltd.lead_time_z_score,
    de.event_name
FROM lead_time_deviation ltd
JOIN skus s ON ltd.sku_id = s.sku_id
LEFT JOIN disruption_events de ON ltd.disruption_event_id = de.event_id
WHERE ABS(ltd.lead_time_z_score) > 2.0
ORDER BY ABS(ltd.lead_time_z_score) DESC
LIMIT 100;


-- 6. SKU Risk Summary Dashboard
-- Consolidated risk view per SKU for the dashboard.
SELECT
    s.sku_id,
    s.category,
    s.supplier_country,
    s.disruption_sensitivity,
    s.unit_cost_usd,
    s.lead_time_days AS base_lead_time,
    -- Demand stats
    AVG(wd.demand_units) AS avg_weekly_demand,
    STDDEV(wd.demand_units) AS demand_volatility,
    -- Stockout history
    SUM(CASE WHEN wd.stockout_flag THEN 1 ELSE 0 END) AS total_stockout_weeks,
    SUM(CASE WHEN wd.stockout_flag THEN 1 ELSE 0 END)::DOUBLE
        / COUNT(*) * 100 AS stockout_rate_pct,
    -- Lead time stats
    AVG(wd.actual_lead_time_days) AS avg_actual_lead_time,
    MAX(wd.actual_lead_time_days) AS max_lead_time,
    -- Cost exposure
    SUM(wd.demand_units) * s.unit_cost_usd AS total_cost_exposure,
    SUM(CASE WHEN wd.stockout_flag THEN wd.demand_units ELSE 0 END) 
        * s.stockout_cost_usd AS total_stockout_cost
FROM skus s
JOIN weekly_demand wd ON s.sku_id = wd.sku_id
GROUP BY s.sku_id, s.category, s.supplier_country, s.disruption_sensitivity,
         s.unit_cost_usd, s.lead_time_days, s.stockout_cost_usd
ORDER BY total_stockout_cost DESC;


-- 7. Trade Flow by Country and Commodity Heatmap Data
SELECT
    year,
    reporter_name,
    hs_code,
    commodity_name,
    SUM(CASE WHEN flow_type = 'Import' THEN trade_value_usd ELSE 0 END) AS import_value,
    SUM(CASE WHEN flow_type = 'Export' THEN trade_value_usd ELSE 0 END) AS export_value,
    SUM(trade_value_usd) AS total_trade_value
FROM trade_flows
GROUP BY year, reporter_name, hs_code, commodity_name
ORDER BY year, reporter_name, total_trade_value DESC;


-- 8. Macro Indicator Correlation Matrix Data
-- Pivot macro indicators for correlation analysis.
PIVOT (
    SELECT
        date,
        series_id,
        value
    FROM macro_indicators
    WHERE date >= '2020-01-01'
) ON series_id USING FIRST(value)
ORDER BY date;
