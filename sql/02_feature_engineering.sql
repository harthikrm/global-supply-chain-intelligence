-- ============================================================
-- Global Supply Chain Intelligence — Feature Engineering Views
-- ============================================================
-- These views sit on top of the base tables and compute derived
-- features used by the ML pipeline, dashboard, and notebooks.
-- ============================================================

-- 1. Trade Flow Year-over-Year Change
-- Uses LAG window function to compute YoY % change in trade value
-- per HS code per reporting country.
CREATE OR REPLACE VIEW trade_flow_yoy_change AS
SELECT
    year,
    reporter_code,
    reporter_name,
    hs_code,
    commodity_name,
    flow_type,
    trade_value_usd,
    LAG(trade_value_usd, 1) OVER (
        PARTITION BY reporter_code, hs_code, flow_type
        ORDER BY year
    ) AS prev_year_value,
    CASE
        WHEN LAG(trade_value_usd, 1) OVER (
            PARTITION BY reporter_code, hs_code, flow_type
            ORDER BY year
        ) > 0
        THEN (trade_value_usd - LAG(trade_value_usd, 1) OVER (
            PARTITION BY reporter_code, hs_code, flow_type
            ORDER BY year
        )) / LAG(trade_value_usd, 1) OVER (
            PARTITION BY reporter_code, hs_code, flow_type
            ORDER BY year
        ) * 100
        ELSE NULL
    END AS yoy_change_pct
FROM trade_flows;


-- 2. Commodity Disruption Score
-- Z-score of trade value vs 2018–2021 baseline mean and std.
-- Score above 2.0 indicates a disruption signal.
CREATE OR REPLACE VIEW commodity_disruption_score AS
WITH baseline_stats AS (
    SELECT
        hs_code,
        reporter_code,
        flow_type,
        AVG(trade_value_usd) AS baseline_mean,
        STDDEV(trade_value_usd) AS baseline_std
    FROM trade_flows
    WHERE year BETWEEN 2018 AND 2021
    GROUP BY hs_code, reporter_code, flow_type
)
SELECT
    tf.year,
    tf.reporter_code,
    tf.reporter_name,
    tf.hs_code,
    tf.commodity_name,
    tf.flow_type,
    tf.trade_value_usd,
    bs.baseline_mean,
    bs.baseline_std,
    CASE
        WHEN bs.baseline_std > 0
        THEN (tf.trade_value_usd - bs.baseline_mean) / bs.baseline_std
        ELSE 0
    END AS z_score,
    CASE
        WHEN bs.baseline_std > 0
            AND ABS((tf.trade_value_usd - bs.baseline_mean) / bs.baseline_std) > 2.0
        THEN TRUE
        ELSE FALSE
    END AS disruption_flag
FROM trade_flows tf
LEFT JOIN baseline_stats bs
    ON tf.hs_code = bs.hs_code
    AND tf.reporter_code = bs.reporter_code
    AND tf.flow_type = bs.flow_type;


-- 3. Supplier Concentration Index (HHI)
-- Herfindahl-Hirschman Index of trade value by supplier country
-- per SKU category. High HHI = concentrated supply = higher risk.
CREATE OR REPLACE VIEW supplier_concentration_index AS
WITH category_country_trade AS (
    SELECT
        s.category,
        s.supplier_country,
        SUM(wd.demand_units * s.unit_cost_usd) AS total_value
    FROM skus s
    JOIN weekly_demand wd ON s.sku_id = wd.sku_id
    GROUP BY s.category, s.supplier_country
),
category_totals AS (
    SELECT
        category,
        SUM(total_value) AS category_total
    FROM category_country_trade
    GROUP BY category
),
market_shares AS (
    SELECT
        cct.category,
        cct.supplier_country,
        cct.total_value,
        ct.category_total,
        (cct.total_value / ct.category_total * 100) AS market_share_pct
    FROM category_country_trade cct
    JOIN category_totals ct ON cct.category = ct.category
)
SELECT
    category,
    SUM(market_share_pct * market_share_pct) AS hhi_index,
    CASE
        WHEN SUM(market_share_pct * market_share_pct) > 2500 THEN 'High Concentration'
        WHEN SUM(market_share_pct * market_share_pct) > 1500 THEN 'Moderate Concentration'
        ELSE 'Low Concentration'
    END AS concentration_level,
    COUNT(DISTINCT supplier_country) AS num_supplier_countries
FROM market_shares
GROUP BY category;


-- 4. Lead Time Deviation
-- Deviation of actual lead time from the baseline (non-disruption) average
-- per SKU.
CREATE OR REPLACE VIEW lead_time_deviation AS
WITH baseline_lead_times AS (
    SELECT
        sku_id,
        AVG(actual_lead_time_days) AS baseline_avg_lead_time,
        STDDEV(actual_lead_time_days) AS baseline_std_lead_time
    FROM weekly_demand
    WHERE disruption_event_id IS NULL
    GROUP BY sku_id
)
SELECT
    wd.week_start_date,
    wd.sku_id,
    wd.actual_lead_time_days,
    blt.baseline_avg_lead_time,
    blt.baseline_std_lead_time,
    wd.actual_lead_time_days - blt.baseline_avg_lead_time AS lead_time_deviation_days,
    CASE
        WHEN blt.baseline_std_lead_time > 0
        THEN (wd.actual_lead_time_days - blt.baseline_avg_lead_time) / blt.baseline_std_lead_time
        ELSE 0
    END AS lead_time_z_score,
    wd.disruption_event_id
FROM weekly_demand wd
LEFT JOIN baseline_lead_times blt ON wd.sku_id = blt.sku_id;


-- 5. Rolling Demand Statistics
-- 4-week and 8-week rolling mean, std, and week-over-week demand change
-- per SKU using window functions.
CREATE OR REPLACE VIEW rolling_demand_stats AS
SELECT
    week_start_date,
    sku_id,
    demand_units,
    -- 4-week rolling statistics
    AVG(demand_units) OVER (
        PARTITION BY sku_id
        ORDER BY week_start_date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_4w,
    STDDEV(demand_units) OVER (
        PARTITION BY sku_id
        ORDER BY week_start_date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS rolling_std_4w,
    -- 8-week rolling statistics
    AVG(demand_units) OVER (
        PARTITION BY sku_id
        ORDER BY week_start_date
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_8w,
    STDDEV(demand_units) OVER (
        PARTITION BY sku_id
        ORDER BY week_start_date
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS rolling_std_8w,
    -- Week-over-week demand change
    demand_units - LAG(demand_units, 1) OVER (
        PARTITION BY sku_id
        ORDER BY week_start_date
    ) AS wow_demand_change,
    -- Week-over-week demand change percentage
    CASE
        WHEN LAG(demand_units, 1) OVER (
            PARTITION BY sku_id
            ORDER BY week_start_date
        ) > 0
        THEN (demand_units - LAG(demand_units, 1) OVER (
            PARTITION BY sku_id
            ORDER BY week_start_date
        ))::DOUBLE / LAG(demand_units, 1) OVER (
            PARTITION BY sku_id
            ORDER BY week_start_date
        ) * 100
        ELSE NULL
    END AS wow_change_pct,
    disruption_event_id
FROM weekly_demand;


-- 6. Inventory Coverage Ratio
-- Weeks of inventory cover remaining based on forecast demand.
-- Uses rolling 4-week average demand as a proxy for forecast.
CREATE OR REPLACE VIEW inventory_coverage_ratio AS
WITH demand_forecast AS (
    SELECT
        sku_id,
        week_start_date,
        AVG(demand_units) OVER (
            PARTITION BY sku_id
            ORDER BY week_start_date
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS avg_weekly_demand
    FROM weekly_demand
),
latest_demand AS (
    SELECT
        sku_id,
        MAX(week_start_date) AS latest_week,
        LAST(avg_weekly_demand ORDER BY week_start_date) AS current_avg_demand
    FROM demand_forecast
    GROUP BY sku_id
)
SELECT
    s.sku_id,
    s.category,
    s.supplier_country,
    s.reorder_quantity,
    ld.current_avg_demand,
    CASE
        WHEN ld.current_avg_demand > 0
        THEN s.reorder_quantity::DOUBLE / ld.current_avg_demand
        ELSE NULL
    END AS weeks_of_cover,
    CASE
        WHEN ld.current_avg_demand > 0
            AND s.reorder_quantity::DOUBLE / ld.current_avg_demand < 2.0
        THEN 'Critical'
        WHEN ld.current_avg_demand > 0
            AND s.reorder_quantity::DOUBLE / ld.current_avg_demand < 4.0
        THEN 'Low'
        ELSE 'Adequate'
    END AS coverage_status
FROM skus s
LEFT JOIN latest_demand ld ON s.sku_id = ld.sku_id;
