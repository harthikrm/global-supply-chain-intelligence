#!/usr/bin/env python
# coding: utf-8

# # 00 — Data Pipeline
# **Global Supply Chain Intelligence**
# 
# This notebook walks through the complete data ingestion pipeline:
# 1. FRED API macro indicators (or synthetic fallback)
# 2. UN Comtrade trade flow data (or synthetic fallback)
# 3. Synthetic manufacturing data (SKUs, demand, disruption events)
# 4. DuckDB schema creation and data loading
# 5. Feature engineering SQL views

# In[1]:


import sys
sys.path.insert(0, '..')

import pandas as pd
import duckdb
from src.ingest import run_pipeline
from src.synthetic import save_all


# ## Run Full Pipeline

# In[2]:


# Run the full data pipeline
# NOTE: This rebuilds the DB from scratch. Skip if DB already exists.
from pathlib import Path
db_path = Path('../data/processed/supply_chain.db')

if not db_path.exists():
    all_good = run_pipeline()
else:
    print(f"Database already exists at {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")
    print("To rebuild, delete the DB file and re-run this cell.")
    all_good = True


# ## Inspect DuckDB Tables

# In[3]:


con = duckdb.connect('../data/processed/supply_chain.db', read_only=True)

# Table counts
tables = ['macro_indicators', 'trade_flows', 'skus', 'weekly_demand', 'disruption_events']
for t in tables:
    count = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {count:,} rows')


# In[4]:


# Sample macro indicators
print(con.execute("SELECT * FROM macro_indicators WHERE series_id = 'BDIY' LIMIT 10").fetchdf())


# In[5]:


# SKU distribution by category
print(con.execute("""
    SELECT category, COUNT(*) as count, 
           AVG(lead_time_days) as avg_lead_time,
           AVG(unit_cost_usd) as avg_cost
    FROM skus 
    WHERE category != 'Unknown'
    GROUP BY category 
    ORDER BY count DESC
""").fetchdf())


# In[6]:


# Disruption events
print(con.execute('SELECT * FROM disruption_events WHERE severity_score > 0.6').fetchdf())


# In[7]:


# Feature engineering views
print('\n--- Supplier Concentration Index (HHI) ---')
print(con.execute('SELECT * FROM supplier_concentration_index WHERE hhi_index > 2500 LIMIT 10').fetchdf())

print('\n--- Top Disrupted Commodities ---')
print(con.execute("""
    SELECT * FROM commodity_disruption_score 
    WHERE ABS(z_score) > 1.5 ORDER BY ABS(z_score) DESC LIMIT 10
""").fetchdf())


# In[8]:


con.close()
