"""
Data Ingestion Pipeline for Global Supply Chain Intelligence
=============================================================
Handles data collection from FRED API, UN Comtrade API, and
synthetic data generation. Loads all data into DuckDB.

Falls back to synthetic data when API keys are unavailable.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic import save_all as generate_synthetic_data

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / '.env')

FRED_API_KEY = os.getenv('FRED_API_KEY', '')
COMTRADE_API_KEY = os.getenv('COMTRADE_API_KEY', '')

DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')
SCHEMA_PATH = str(PROJECT_ROOT / 'sql' / '01_schema.sql')
FEATURES_PATH = str(PROJECT_ROOT / 'sql' / '02_feature_engineering.sql')

FRED_SERIES = {
    'BDIY': 'Baltic Dry Index',
    'WTISPLC': 'WTI Crude Oil Spot Price',
    'PNGASEUUSDM': 'European Natural Gas Price',
    'PWHEAMTUSDM': 'Global Wheat Price Index',
    'PSUNOUSDM': 'Sunflower Oil Price',
    'PALUMUSDM': 'Aluminum Price',
    'PNICKUSDM': 'Nickel Price',
    'CPIAUCSL': 'US Consumer Price Index',
    'UNRATE': 'US Unemployment Rate',
    'MRTSSM44X72USS': 'US Retail Sales',
}

COMTRADE_HS_CODES = ['1001', '1512', '2814', '2804', '8541',
                      '7601', '7502', '8703', '8708', '2709']

COMTRADE_REPORTERS = {842: 'USA', 276: 'Germany', 392: 'Japan',
                       156: 'China', 699: 'India', 410: 'South Korea'}

COMTRADE_YEARS = list(range(2018, 2025))


def init_database() -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB database and create schema."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Remove old DB to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)

    # Read and execute schema
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()

    # Strip comment-only lines, then split by semicolons
    cleaned_lines = []
    for line in schema_sql.split('\n'):
        stripped = line.strip()
        if stripped.startswith('--') or stripped == '':
            continue
        cleaned_lines.append(line)

    cleaned_sql = '\n'.join(cleaned_lines)

    for stmt in cleaned_sql.split(';'):
        stmt = stmt.strip()
        if stmt:
            try:
                con.execute(stmt)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    print(f"  ⚠ Schema warning: {e}")

    print("  ✓ Database initialized")
    return con


def ingest_fred(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pull FRED macroeconomic indicator data.
    Falls back to synthetic data if API key is unavailable.
    """
    print("\n── FRED Macro Indicators ──")

    fred_data = None

    # Try real API first
    if FRED_API_KEY and FRED_API_KEY != 'your_fred_api_key_here':
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            print("  ✓ FRED API key detected, pulling live data...")

            all_rows = []
            for series_id, series_name in FRED_SERIES.items():
                try:
                    data = fred.get_series(series_id, observation_start='2018-01-01')
                    if data is not None and len(data) > 0:
                        series_df = pd.DataFrame({
                            'date': data.index,
                            'series_id': series_id,
                            'value': data.values,
                            'series_name': series_name,
                        })
                        all_rows.append(series_df)
                        print(f"    ✓ {series_id}: {len(data)} observations")

                        # Save individual CSV
                        fred_dir = PROJECT_ROOT / 'data' / 'raw' / 'fred'
                        os.makedirs(fred_dir, exist_ok=True)
                        series_df.to_csv(fred_dir / f'{series_id}.csv', index=False)

                    time.sleep(0.3)
                except Exception as e:
                    print(f"    ⚠ {series_id}: {e}")

            if all_rows:
                fred_data = pd.concat(all_rows, ignore_index=True)
                print(f"  ✓ Pulled {len(fred_data)} total observations from FRED API")

        except ImportError:
            print("  ⚠ fredapi not installed, falling back to synthetic data")
        except Exception as e:
            print(f"  ⚠ FRED API error: {e}, falling back to synthetic data")

    # Fallback to synthetic
    if fred_data is None:
        print("  → Using synthetic FRED data (no API key or API unavailable)")
        synthetic_path = PROJECT_ROOT / 'data' / 'raw' / 'synthetic' / 'fred_synthetic.csv'
        if synthetic_path.exists():
            fred_data = pd.read_csv(synthetic_path)
        else:
            from src.synthetic import generate_synthetic_fred_data
            fred_data = generate_synthetic_fred_data()

    # Compute YoY and MoM changes
    fred_data['date'] = pd.to_datetime(fred_data['date'])
    fred_data = fred_data.sort_values(['series_id', 'date'])

    fred_data['yoy_change'] = fred_data.groupby('series_id')['value'].pct_change(12) * 100
    fred_data['mom_change'] = fred_data.groupby('series_id')['value'].pct_change(1) * 100

    # Handle infinities and NaN for computed columns
    fred_data['yoy_change'] = fred_data['yoy_change'].replace([np.inf, -np.inf], np.nan)
    fred_data['mom_change'] = fred_data['mom_change'].replace([np.inf, -np.inf], np.nan)

    # Load into DuckDB
    con.execute("DELETE FROM macro_indicators")
    con.execute("""
        INSERT INTO macro_indicators
        SELECT date, series_id, value, series_name, yoy_change, mom_change
        FROM fred_data
    """)

    count = con.execute("SELECT COUNT(*) FROM macro_indicators").fetchone()[0]
    print(f"  ✓ Loaded {count:,} rows into macro_indicators table")

    return fred_data


def ingest_comtrade(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pull UN Comtrade trade flow data.
    Falls back to synthetic data if API key is unavailable.
    """
    print("\n── UN Comtrade Trade Flows ──")

    trade_data = None

    if COMTRADE_API_KEY and COMTRADE_API_KEY != 'your_comtrade_api_key_here':
        try:
            import comtradeapicall
            print("  ✓ Comtrade API key detected, pulling live data...")

            all_rows = []
            call_count = 0

            for hs_code in COMTRADE_HS_CODES:
                for year in COMTRADE_YEARS:
                    try:
                        reporter_codes = ','.join(str(c) for c in COMTRADE_REPORTERS.keys())
                        df = comtradeapicall.getFinalData(
                            subscription_key=COMTRADE_API_KEY,
                            typeCode='C',
                            freqCode='A',
                            clCode='HS',
                            period=str(year),
                            reporterCode=reporter_codes,
                            cmdCode=hs_code,
                            flowCode='M,X',
                            partnerCode=None,
                            maxRecords=2500,
                        )

                        if df is not None and len(df) > 0:
                            # Standardize columns
                            parsed = pd.DataFrame({
                                'year': year,
                                'reporter_code': df.get('reporterCode', df.get('reportercode', 0)),
                                'reporter_name': df.get('reporterDesc', df.get('reporter', '')),
                                'partner_code': df.get('partnerCode', df.get('partnercode', 0)),
                                'partner_name': df.get('partnerDesc', df.get('partner', '')),
                                'hs_code': hs_code,
                                'commodity_name': df.get('cmdDesc', df.get('cmdDescription', '')),
                                'trade_value_usd': df.get('primaryValue', df.get('TradeValue', 0)),
                                'net_weight_kg': df.get('netWgt', df.get('NetWeight', 0)),
                                'flow_type': df.get('flowDesc', df.get('flowDescription', '')),
                            })
                            all_rows.append(parsed)

                            # Save JSON
                            comtrade_dir = PROJECT_ROOT / 'data' / 'raw' / 'comtrade'
                            os.makedirs(comtrade_dir, exist_ok=True)
                            df.to_json(comtrade_dir / f'{hs_code}_{year}.json', orient='records')

                        call_count += 1
                        print(f"    ✓ HS {hs_code}, {year}: {len(df) if df is not None else 0} records "
                              f"(call {call_count})")

                        time.sleep(1)  # Rate limiting: 1 second between calls

                    except Exception as e:
                        print(f"    ⚠ HS {hs_code}, {year}: {e}")
                        time.sleep(2)

            if all_rows:
                trade_data = pd.concat(all_rows, ignore_index=True)
                # Normalize flow type
                trade_data['flow_type'] = trade_data['flow_type'].str.strip()
                trade_data['flow_type'] = trade_data['flow_type'].replace({
                    'Import': 'Import', 'Imports': 'Import',
                    'Export': 'Export', 'Exports': 'Export',
                    'Re-imports': 'Import', 'Re-exports': 'Export',
                    'M': 'Import', 'X': 'Export',
                })
                print(f"  ✓ Pulled {len(trade_data)} total records from Comtrade API")

        except ImportError:
            print("  ⚠ comtradeapicall not installed, falling back to synthetic data")
        except Exception as e:
            print(f"  ⚠ Comtrade API error: {e}, falling back to synthetic data")

    # Fallback to synthetic
    if trade_data is None:
        print("  → Using synthetic trade flow data (no API key or API unavailable)")
        synthetic_path = PROJECT_ROOT / 'data' / 'raw' / 'synthetic' / 'comtrade_synthetic.csv'
        if synthetic_path.exists():
            trade_data = pd.read_csv(synthetic_path)
        else:
            from src.synthetic import generate_synthetic_comtrade_data
            trade_data = generate_synthetic_comtrade_data()

    # Load into DuckDB
    con.execute("DELETE FROM trade_flows")
    con.execute("""
        INSERT INTO trade_flows
        SELECT year, reporter_code, reporter_name, partner_code, partner_name,
               hs_code, commodity_name, trade_value_usd, net_weight_kg, flow_type
        FROM trade_data
    """)

    count = con.execute("SELECT COUNT(*) FROM trade_flows").fetchone()[0]
    print(f"  ✓ Loaded {count:,} rows into trade_flows table")

    return trade_data


def ingest_synthetic(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Load synthetic manufacturing data (SKUs, demand, events) into DuckDB.
    """
    print("\n── Synthetic Manufacturing Data ──")

    synthetic_dir = PROJECT_ROOT / 'data' / 'raw' / 'synthetic'

    # Check if synthetic data already exists
    skus_path = synthetic_dir / 'skus.csv'
    demand_path = synthetic_dir / 'weekly_demand.csv'
    events_path = synthetic_dir / 'disruption_events.csv'

    if skus_path.exists() and demand_path.exists() and events_path.exists():
        print("  → Loading existing synthetic data from CSV...")
        skus_df = pd.read_csv(skus_path)
        demand_df = pd.read_csv(demand_path)
        events_df = pd.read_csv(events_path)
    else:
        print("  → Generating new synthetic data...")
        data = generate_synthetic_data(str(synthetic_dir))
        skus_df = data['skus']
        demand_df = data['weekly_demand']
        events_df = data['disruption_events']

    # Load SKUs
    con.execute("DELETE FROM skus")
    con.execute("INSERT INTO skus SELECT * FROM skus_df")
    sku_count = con.execute("SELECT COUNT(*) FROM skus").fetchone()[0]
    print(f"  ✓ Loaded {sku_count} SKUs")

    # Load disruption events
    con.execute("DELETE FROM disruption_events")
    con.execute("INSERT INTO disruption_events SELECT * FROM events_df")
    evt_count = con.execute("SELECT COUNT(*) FROM disruption_events").fetchone()[0]
    print(f"  ✓ Loaded {evt_count} disruption events")

    # Load weekly demand
    con.execute("DELETE FROM weekly_demand")
    con.execute("INSERT INTO weekly_demand SELECT * FROM demand_df")
    demand_count = con.execute("SELECT COUNT(*) FROM weekly_demand").fetchone()[0]
    print(f"  ✓ Loaded {demand_count:,} weekly demand records")

    return {'skus': skus_df, 'demand': demand_df, 'events': events_df}


def create_feature_views(con: duckdb.DuckDBPyConnection):
    """Create feature engineering SQL views."""
    print("\n── Creating Feature Engineering Views ──")

    with open(FEATURES_PATH, 'r') as f:
        sql_content = f.read()

    # Split by CREATE and execute each view
    statements = sql_content.split('CREATE OR REPLACE VIEW')
    for stmt in statements[1:]:  # Skip the header comments
        stmt = 'CREATE OR REPLACE VIEW' + stmt
        # Split at the closing semicolon
        stmt = stmt.split(';')[0] + ';'
        try:
            con.execute(stmt)
            view_name = stmt.split('VIEW')[1].split('AS')[0].strip()
            print(f"  ✓ Created view: {view_name}")
        except Exception as e:
            print(f"  ⚠ View creation warning: {e}")

    print("  ✓ All feature engineering views created")


def validate_database(con: duckdb.DuckDBPyConnection):
    """Run validation checks on the loaded data."""
    print("\n── Data Validation ──")

    checks = [
        ("macro_indicators", 500),
        ("trade_flows", 500),
        ("skus", 500),
        ("weekly_demand", 70000),
        ("disruption_events", 3),
    ]

    all_passed = True
    for table, min_rows in checks:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        status = "✓" if count >= min_rows else "✗"
        if count < min_rows:
            all_passed = False
        print(f"  {status} {table}: {count:,} rows (min: {min_rows:,})")

    # Check feature views
    views = ['trade_flow_yoy_change', 'commodity_disruption_score',
             'supplier_concentration_index', 'lead_time_deviation',
             'rolling_demand_stats', 'inventory_coverage_ratio']

    for view in views:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            print(f"  ✓ View {view}: {count:,} rows")
        except Exception as e:
            print(f"  ✗ View {view}: {e}")
            all_passed = False

    return all_passed


def run_pipeline():
    """Execute the full data ingestion pipeline."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Global Supply Chain Intelligence — Data Pipeline    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Initialize database
    print("\n── Initializing Database ──")
    con = init_database()

    # Generate synthetic data first (always needed)
    synthetic_dir = PROJECT_ROOT / 'data' / 'raw' / 'synthetic'
    if not (synthetic_dir / 'skus.csv').exists():
        generate_synthetic_data(str(synthetic_dir))

    # Ingest all data sources
    ingest_fred(con)
    ingest_comtrade(con)
    ingest_synthetic(con)

    # Create feature views
    create_feature_views(con)

    # Validate
    all_good = validate_database(con)

    # Summary
    print("\n╔══════════════════════════════════════════════════════╗")
    if all_good:
        print("║  ✓ Pipeline Complete — All checks passed!           ║")
    else:
        print("║  ⚠ Pipeline Complete — Some checks need attention   ║")
    print("╚══════════════════════════════════════════════════════╝")

    con.close()
    return all_good


if __name__ == '__main__':
    run_pipeline()
