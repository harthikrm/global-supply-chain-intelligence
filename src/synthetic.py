"""
Synthetic Data Generation for Global Supply Chain Intelligence
==============================================================
Generates realistic manufacturing/SKU data, weekly demand with
disruption events, and labeled ground truth for model validation.

Uses numpy.random with seed=42 for reproducibility.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Constants ────────────────────────────────────────────────────
SEED = 42
NUM_WEEKS = 156  # 3 years: 2022-2024

CATEGORIES = {
    'Electronics': 80,
    'Auto Parts': 100,
    'Food & Beverage': 80,
    'Chemicals': 60,
    'Metals': 60,
    'Pharmaceuticals': 70,
    'Textiles': 50,
}

SUPPLIER_COUNTRIES = {
    'China': 0.35,
    'Germany': 0.15,
    'Japan': 0.12,
    'South Korea': 0.10,
    'India': 0.08,
    'Taiwan': 0.05,
    'Vietnam': 0.04,
    'Mexico': 0.03,
    'Thailand': 0.03,
    'Brazil': 0.03,
    'Turkey': 0.02,
}

COUNTRY_LEAD_TIMES = {
    'China': (45, 10),
    'Germany': (21, 5),
    'Japan': (28, 7),
    'South Korea': (30, 7),
    'India': (40, 12),
    'Taiwan': (32, 8),
    'Vietnam': (42, 10),
    'Mexico': (14, 4),
    'Thailand': (38, 9),
    'Brazil': (35, 10),
    'Turkey': (25, 6),
}

CATEGORY_BASE_DEMAND = {
    'Electronics': (120, 40),
    'Auto Parts': (80, 25),
    'Food & Beverage': (200, 60),
    'Chemicals': (60, 20),
    'Metals': (50, 15),
    'Pharmaceuticals': (90, 30),
    'Textiles': (70, 25),
}

CATEGORY_UNIT_COST = {
    'Electronics': (50, 500),
    'Auto Parts': (20, 300),
    'Food & Beverage': (5, 80),
    'Chemicals': (30, 200),
    'Metals': (40, 250),
    'Pharmaceuticals': (100, 800),
    'Textiles': (10, 100),
}

# Disruption events: (event_id, name, start_week, end_week, affected_countries, affected_categories, severity)
DISRUPTION_EVENTS = [
    {
        'event_id': 'EVT001',
        'event_name': 'Ukraine Conflict Onset',
        'start_week': 8,   # Week 8 of 2022
        'end_week': 16,    # Week 16 of 2022
        'affected_countries': ['Turkey', 'Germany'],
        'affected_categories': ['Food & Beverage', 'Metals', 'Chemicals', 'Electronics'],
        'affected_hs_codes': '1001,1512,2814,2804,7601,7502',
        'severity': 0.85,
        'lead_time_multiplier': (1.4, 1.8),
        'demand_spike': (1.20, 1.35),
        'stockout_prob': 0.15,
    },
    {
        'event_id': 'EVT002',
        'event_name': 'Red Sea Shipping Disruption',
        'start_week': 92,  # Week 40 of 2023 (40 + 52 = 92)
        'end_week': 104,   # Week 52 of 2023 (52 + 52 = 104)
        'affected_countries': ['China', 'India', 'Vietnam', 'Thailand', 'Taiwan', 'South Korea', 'Japan'],
        'affected_categories': ['Electronics', 'Auto Parts', 'Textiles', 'Chemicals', 'Metals'],
        'affected_hs_codes': '8541,8703,8708,2709,7601,7502',
        'severity': 0.75,
        'lead_time_multiplier': (1.4, 1.7),
        'demand_spike': (1.15, 1.30),
        'stockout_prob': 0.10,
    },
    {
        'event_id': 'EVT003',
        'event_name': 'Port of Singapore Congestion',
        'start_week': 124, # Week 20 of 2024 (20 + 104 = 124)
        'end_week': 132,   # Week 28 of 2024 (28 + 104 = 132)
        'affected_countries': ['China', 'Vietnam', 'Thailand', 'Taiwan', 'South Korea', 'Japan', 'India'],
        'affected_categories': ['Electronics', 'Auto Parts', 'Pharmaceuticals', 'Chemicals'],
        'affected_hs_codes': '8541,8703,8708,2804',
        'severity': 0.65,
        'lead_time_multiplier': (1.5, 1.8),
        'demand_spike': (1.20, 1.35),
        'stockout_prob': 0.12,
    },
]


def generate_skus(rng: np.random.RandomState = None) -> pd.DataFrame:
    """
    Generate 500 SKUs across 7 product categories with realistic
    supplier country distribution, lead times, and cost parameters.
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    countries = list(SUPPLIER_COUNTRIES.keys())
    country_probs = list(SUPPLIER_COUNTRIES.values())

    rows = []
    sku_counter = 0

    for category, count in CATEGORIES.items():
        cost_low, cost_high = CATEGORY_UNIT_COST[category]

        for i in range(count):
            sku_counter += 1
            sku_id = f"SKU-{sku_counter:04d}"

            # Assign supplier country
            supplier_country = rng.choice(countries, p=country_probs)

            # Lead time based on country
            lt_mean, lt_std = COUNTRY_LEAD_TIMES[supplier_country]
            lead_time = max(7, int(rng.normal(lt_mean, lt_std)))

            # Unit cost
            unit_cost = round(rng.uniform(cost_low, cost_high), 2)

            # Holding cost: 2-5% per month
            holding_cost_pct = round(rng.uniform(0.02, 0.05), 4)

            # Stockout cost: 2-5x unit cost
            stockout_multiplier = rng.uniform(2.0, 5.0)
            stockout_cost = round(unit_cost * stockout_multiplier, 2)

            # Reorder quantity
            reorder_qty = int(rng.uniform(50, 500))

            # Disruption sensitivity based on country + category
            high_risk_countries = {'China', 'India', 'Vietnam', 'Thailand', 'Taiwan'}
            high_risk_categories = {'Electronics', 'Auto Parts', 'Metals', 'Chemicals'}

            if supplier_country in high_risk_countries and category in high_risk_categories:
                sensitivity = 'High'
            elif supplier_country in high_risk_countries or category in high_risk_categories:
                sensitivity = 'Medium'
            else:
                sensitivity = 'Low'

            rows.append({
                'sku_id': sku_id,
                'category': category,
                'supplier_country': supplier_country,
                'lead_time_days': lead_time,
                'unit_cost_usd': unit_cost,
                'holding_cost_pct': holding_cost_pct,
                'stockout_cost_usd': stockout_cost,
                'reorder_quantity': reorder_qty,
                'disruption_sensitivity': sensitivity,
            })

    df = pd.DataFrame(rows)
    print(f"  ✓ Generated {len(df)} SKUs across {len(CATEGORIES)} categories")
    return df


def generate_disruption_events() -> pd.DataFrame:
    """
    Generate labeled disruption events as ground truth for validation.
    """
    base_date = datetime(2022, 1, 3)  # First Monday of 2022

    rows = []
    for evt in DISRUPTION_EVENTS:
        start_date = base_date + timedelta(weeks=evt['start_week'] - 1)
        end_date = base_date + timedelta(weeks=evt['end_week'] - 1)
        rows.append({
            'event_id': evt['event_id'],
            'event_name': evt['event_name'],
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'affected_countries': ','.join(evt['affected_countries']),
            'affected_hs_codes': evt['affected_hs_codes'],
            'severity_score': evt['severity'],
        })

    df = pd.DataFrame(rows)
    print(f"  ✓ Generated {len(df)} disruption events")
    return df


def generate_weekly_demand(skus_df: pd.DataFrame, rng: np.random.RandomState = None) -> pd.DataFrame:
    """
    Generate 156 weeks of weekly demand per SKU (78,000 total rows).
    Includes Poisson base demand with seasonality and 3 disruption events.
    """
    if rng is None:
        rng = np.random.RandomState(SEED + 1)

    base_date = datetime(2022, 1, 3)  # First Monday of 2022
    weeks = [base_date + timedelta(weeks=w) for w in range(NUM_WEEKS)]

    rows = []
    total_skus = len(skus_df)

    for idx, sku in skus_df.iterrows():
        sku_id = sku['sku_id']
        category = sku['category']
        supplier_country = sku['supplier_country']
        base_lt = sku['lead_time_days']

        # Base demand parameters
        demand_mean, demand_std = CATEGORY_BASE_DEMAND[category]
        # Add per-SKU variation
        sku_demand_mean = max(10, int(rng.normal(demand_mean, demand_std * 0.3)))

        for week_idx in range(NUM_WEEKS):
            week_date = weeks[week_idx]

            # Seasonality: 1 + 0.3 * sin(2π * week/52)
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (week_idx % 52) / 52)

            # Base demand (Poisson with seasonal adjustment)
            adjusted_mean = max(1, sku_demand_mean * seasonal_factor)
            demand = rng.poisson(adjusted_mean)

            # Normal lead time with some variation
            actual_lt = max(1, int(rng.normal(base_lt, base_lt * 0.1)))

            stockout = False
            event_id = None

            # Check disruption events
            for evt in DISRUPTION_EVENTS:
                if evt['start_week'] <= week_idx + 1 <= evt['end_week']:
                    # Check if this SKU is affected
                    country_affected = supplier_country in evt['affected_countries']
                    category_affected = category in evt['affected_categories']

                    if country_affected or category_affected:
                        event_id = evt['event_id']

                        # Severity scales with how affected the SKU is
                        severity_mult = 1.0
                        if country_affected and category_affected:
                            severity_mult = 1.0
                        elif country_affected:
                            severity_mult = 0.7
                        else:
                            severity_mult = 0.5

                        # Lead time increase
                        lt_low, lt_high = evt['lead_time_multiplier']
                        lt_mult = rng.uniform(lt_low, lt_high)
                        actual_lt = int(actual_lt * (1 + (lt_mult - 1) * severity_mult))

                        # Demand spike
                        d_low, d_high = evt['demand_spike']
                        d_mult = rng.uniform(d_low, d_high)
                        demand = int(demand * (1 + (d_mult - 1) * severity_mult))

                        # Stockout probability
                        if rng.random() < evt['stockout_prob'] * severity_mult:
                            stockout = True

                        break  # Only apply one disruption event per week

            rows.append({
                'week_start_date': week_date.strftime('%Y-%m-%d'),
                'sku_id': sku_id,
                'demand_units': max(0, demand),
                'actual_lead_time_days': actual_lt,
                'stockout_flag': stockout,
                'disruption_event_id': event_id,
            })

        if (idx + 1) % 100 == 0:
            print(f"    ... generated demand for {idx + 1}/{total_skus} SKUs")

    df = pd.DataFrame(rows)
    print(f"  ✓ Generated {len(df):,} weekly demand records")
    return df


def generate_synthetic_fred_data(rng: np.random.RandomState = None) -> pd.DataFrame:
    """
    Generate synthetic FRED macro indicator data when API is not available.
    Produces realistic time series that mimic the actual FRED series behavior.
    """
    if rng is None:
        rng = np.random.RandomState(SEED + 2)

    series_config = {
        'BDIY': {'name': 'Baltic Dry Index', 'base': 1500, 'std': 400,
                 'disruption_2022': 1.4, 'disruption_2023': 1.3},
        'WTISPLC': {'name': 'WTI Crude Oil Spot Price', 'base': 65, 'std': 15,
                    'disruption_2022': 1.6, 'disruption_2023': 1.2},
        'PNGASEUUSDM': {'name': 'EU Natural Gas Price', 'base': 8, 'std': 3,
                        'disruption_2022': 3.0, 'disruption_2023': 1.5},
        'PWHEAMTUSDM': {'name': 'Global Wheat Price', 'base': 220, 'std': 40,
                        'disruption_2022': 1.8, 'disruption_2023': 1.1},
        'PSUNOUSDM': {'name': 'Sunflower Oil Price', 'base': 800, 'std': 150,
                      'disruption_2022': 2.0, 'disruption_2023': 1.1},
        'PALUMUSDM': {'name': 'Aluminum Price', 'base': 2000, 'std': 400,
                      'disruption_2022': 1.5, 'disruption_2023': 1.2},
        'PNICKUSDM': {'name': 'Nickel Price', 'base': 15000, 'std': 3000,
                      'disruption_2022': 1.8, 'disruption_2023': 1.3},
        'CPIAUCSL': {'name': 'US CPI', 'base': 260, 'std': 5,
                     'disruption_2022': 1.05, 'disruption_2023': 1.02},
        'UNRATE': {'name': 'US Unemployment Rate', 'base': 3.7, 'std': 0.5,
                   'disruption_2022': 1.0, 'disruption_2023': 1.0},
        'MRTSSM44X72USS': {'name': 'US Retail Sales', 'base': 550000, 'std': 30000,
                           'disruption_2022': 0.98, 'disruption_2023': 1.02},
    }

    dates = pd.date_range('2018-01-01', '2024-12-31', freq='MS')
    rows = []

    for series_id, cfg in series_config.items():
        values = []
        for i, date in enumerate(dates):
            # Base value with trend
            trend = 1.0 + 0.002 * i  # Slight upward trend
            seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * date.month / 12)

            # Disruption effects
            disruption = 1.0
            if date.year == 2022 and date.month >= 2 and date.month <= 6:
                disruption = cfg['disruption_2022']
            elif date.year == 2022 and date.month > 6:
                # Gradual decay
                disruption = 1.0 + (cfg['disruption_2022'] - 1.0) * 0.5
            elif date.year == 2023 and date.month >= 10:
                disruption = cfg['disruption_2023']

            value = cfg['base'] * trend * seasonal * disruption
            value += rng.normal(0, cfg['std'] * 0.3)

            # CPI and retail sales should be monotonically increasing
            if series_id == 'CPIAUCSL':
                value = cfg['base'] + i * 0.8 + rng.normal(0, 0.5)
            elif series_id == 'UNRATE':
                value = max(2.5, cfg['base'] + rng.normal(0, cfg['std'] * 0.3))

            values.append(max(0.01, value))

        for date, value in zip(dates, values):
            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'series_id': series_id,
                'value': round(value, 4),
                'series_name': cfg['name'],
            })

    df = pd.DataFrame(rows)
    print(f"  ✓ Generated {len(df):,} synthetic FRED records ({len(series_config)} series)")
    return df


def generate_synthetic_comtrade_data(rng: np.random.RandomState = None) -> pd.DataFrame:
    """
    Generate synthetic UN Comtrade trade flow data when API is not available.
    Produces realistic bilateral trade flows reflecting disruption patterns.
    """
    if rng is None:
        rng = np.random.RandomState(SEED + 3)

    hs_codes = {
        '1001': ('Wheat and meslin', 1e9),
        '1512': ('Sunflower-seed, safflower oil', 5e8),
        '2814': ('Ammonia, anhydrous or solution', 3e8),
        '2804': ('Hydrogen, noble gases, other nonmetals', 2e8),
        '8541': ('Semiconductor devices', 5e9),
        '7601': ('Unwrought aluminium', 2e9),
        '7502': ('Unwrought nickel', 1e9),
        '8703': ('Motor cars and vehicles', 1e10),
        '8708': ('Parts for motor vehicles', 8e9),
        '2709': ('Petroleum oils, crude', 2e10),
    }

    reporters = {
        842: 'USA',
        276: 'Germany',
        392: 'Japan',
        156: 'China',
        699: 'India',
        410: 'South Korea',
    }

    # Major trade partners
    partners = {
        0: 'World',
        156: 'China',
        276: 'Germany',
        392: 'Japan',
        842: 'USA',
        804: 'Ukraine',
        643: 'Russian Federation',
    }

    years = list(range(2018, 2025))
    rows = []

    for hs_code, (commodity_name, base_value) in hs_codes.items():
        for reporter_code, reporter_name in reporters.items():
            for year in years:
                for flow_type in ['Import', 'Export']:
                    # Scale by reporter size
                    reporter_scale = {842: 1.0, 276: 0.6, 392: 0.7, 156: 1.2, 699: 0.4, 410: 0.5}
                    scale = reporter_scale.get(reporter_code, 0.5)

                    # Apply disruption effects
                    disruption_mult = 1.0
                    if year == 2022:
                        if hs_code in ['1001', '1512', '2814']:
                            disruption_mult = 0.6 if flow_type == 'Import' else 1.3
                        elif hs_code in ['2804', '7601', '7502']:
                            disruption_mult = 0.75
                    elif year == 2023:
                        if hs_code in ['8541', '8703', '8708']:
                            disruption_mult = 0.85
                    elif year == 2024:
                        if hs_code in ['8541', '8703', '8708', '2709']:
                            disruption_mult = 0.90

                    # Trend growth
                    trend = 1.0 + 0.03 * (year - 2018)

                    trade_value = base_value * scale * trend * disruption_mult
                    trade_value *= rng.uniform(0.7, 1.3)

                    # Weight proportional to value
                    weight_per_dollar = rng.uniform(0.1, 5.0)
                    net_weight = trade_value * weight_per_dollar / 1000

                    # Select a partner
                    partner_code = rng.choice(list(partners.keys()))
                    partner_name = partners[partner_code]

                    rows.append({
                        'year': year,
                        'reporter_code': reporter_code,
                        'reporter_name': reporter_name,
                        'partner_code': partner_code,
                        'partner_name': partner_name,
                        'hs_code': hs_code,
                        'commodity_name': commodity_name,
                        'trade_value_usd': round(trade_value, 2),
                        'net_weight_kg': round(net_weight, 2),
                        'flow_type': flow_type,
                    })

    df = pd.DataFrame(rows)
    print(f"  ✓ Generated {len(df):,} synthetic trade flow records")
    return df


def save_all(output_dir: str = 'data/raw/synthetic') -> dict:
    """
    Generate and save all synthetic data to CSV files.
    Returns dict of DataFrames.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(SEED)
    print("\n═══ Generating Synthetic Data ═══")

    # 1. SKUs
    print("\n[1/5] Generating SKU master data...")
    skus_df = generate_skus(rng)
    skus_path = os.path.join(output_dir, 'skus.csv')
    skus_df.to_csv(skus_path, index=False)
    print(f"      Saved to {skus_path}")

    # 2. Disruption events
    print("\n[2/5] Generating disruption events...")
    events_df = generate_disruption_events()
    events_path = os.path.join(output_dir, 'disruption_events.csv')
    events_df.to_csv(events_path, index=False)
    print(f"      Saved to {events_path}")

    # 3. Weekly demand
    print("\n[3/5] Generating weekly demand (this may take a moment)...")
    demand_df = generate_weekly_demand(skus_df, rng)
    demand_path = os.path.join(output_dir, 'weekly_demand.csv')
    demand_df.to_csv(demand_path, index=False)
    print(f"      Saved to {demand_path}")

    # 4. Synthetic FRED data
    print("\n[4/5] Generating synthetic FRED indicators...")
    fred_df = generate_synthetic_fred_data(rng)
    fred_path = os.path.join(output_dir, 'fred_synthetic.csv')
    fred_df.to_csv(fred_path, index=False)
    print(f"      Saved to {fred_path}")

    # 5. Synthetic Comtrade data
    print("\n[5/5] Generating synthetic trade flow data...")
    comtrade_df = generate_synthetic_comtrade_data(rng)
    comtrade_path = os.path.join(output_dir, 'comtrade_synthetic.csv')
    comtrade_df.to_csv(comtrade_path, index=False)
    print(f"      Saved to {comtrade_path}")

    print("\n═══ Synthetic Data Generation Complete ═══\n")

    return {
        'skus': skus_df,
        'disruption_events': events_df,
        'weekly_demand': demand_df,
        'fred': fred_df,
        'comtrade': comtrade_df,
    }


if __name__ == '__main__':
    save_all()
