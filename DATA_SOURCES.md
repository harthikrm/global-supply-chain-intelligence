# Global Supply Chain Intelligence - Data Sources

This project uses synthetic data to simulate a global supply chain intelligence platform. While the structure and variations mimic real-world phenomena (like structural breaks, demand spikes, and lead-time decay), the underlying data points are programmatically generated.

## Federal Reserve Economic Data (FRED) Approximation

The macroeconomic indicators generated in `fred_data.csv` are designed to test the model's structural break detection (CUSUM, Mahalanobis distances) rather than precisely replicate historical data. 

**Important Note on the 2020 COVID-19 Shock:**
The synthetic series approximate the general volatility of disruptions but **do not perfectly reflect the exact historical magnitudes of the COVID-19 shock in 2020**. 
- **Baltic Dry Index (BDIY):** In the real world, BDIY crashed to roughly 400 at the onset of the pandemic before its massive 2021 spike. The synthetic BDIY generator currently models a more generic baseline through 2020 before modeling the 2021 surge and 2023 Red Sea disruptions.
- **US Unemployment Rate (UNRATE):** In reality, [UNRATE spiked to 14.7% in April 2020](https://fred.stlouisfed.org/series/UNRATE). The synthetic data intentionally maintains a steadier baseline (~3.5–3.9%) to prevent the CUSUM detector from overly weighting a single historic singularity and overshadowing the latter supply chain-specific disruptions (like the Red Sea). 
- **WTI Crude Oil (WTISPLC):** The synthetic data brings WTISPLC down to ~$26 for Mar/Apr 2020. This is directionally consistent with the real-world crash (which saw spot prices drop near and below $20, [see FRED WTISPLC](https://fred.stlouisfed.org/series/WTISPLC)), serving as an acceptable directional approximation.

For production or backtesting scenarios, researchers should replace the `fred_data.csv` with a pull from the actual FRED API using the `fredapi` library to get true historic baselines.
