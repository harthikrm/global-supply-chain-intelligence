# Data Sources

## fred_data.csv
Synthetic data matching FRED API schema (series: BDIY, WTISPLC, PNGASEUUSDM, PWHEAMTUSDM, PSUNOUSDM, PALUMUSDM, PNICKUSDM, CPIAUCSL, UNRATE, MRTSSM44X72USS).
Generated via `src/synthetic.py` with `seed=42`. Replace with real API data by running:
```bash
python -m src.ingest --fred
```

## comtrade_data.csv
Synthetic data matching UN Comtrade API schema (HS codes: 1001, 1512, 2814, 2804, 8541, 7601, 7502, 8703, 8708, 2709).
Generated via `src/synthetic.py` with `seed=42`. Replace with real API data by running:
```bash
python -m src.ingest --comtrade
```

## skus.csv, weekly_demand.csv, disruption_events.csv
Synthetic manufacturing data generated via `src/synthetic.py`.
- 50-SKU representative sample across 7 categories and 11 supplier countries
- 156 weeks (2022-2024) of weekly demand with disruption effects
- 3 labeled disruption events (Ukraine Conflict, Red Sea Shipping, Port of Singapore Congestion)
