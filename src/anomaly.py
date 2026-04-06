"""
Disruption Detection System — Module B
========================================
Multivariate anomaly detection using CUSUM charts and
Mahalanobis distance to detect supply chain disruptions.
Validates against labeled ground truth events.
"""

import numpy as np
import pandas as pd
import duckdb
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')


def load_macro_data(con: duckdb.DuckDBPyConnection = None) -> pd.DataFrame:
    """Load and pivot macro indicators into a wide-format time series."""
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)

    df = con.execute("""
        SELECT date, series_id, value
        FROM macro_indicators
        ORDER BY date
    """).fetchdf().copy()

    df['date'] = pd.to_datetime(df['date'])

    # Pivot to wide format
    wide = df.pivot_table(index='date', columns='series_id', values='value', aggfunc='first')
    wide = wide.sort_index()

    # Resample to weekly frequency (forward fill)
    wide = wide.resample('W-MON').ffill()

    # Drop rows with too many NaN
    wide = wide.dropna(thresh=len(wide.columns) - 2)

    # Fill remaining NaN with forward fill then backward fill
    wide = wide.ffill().bfill()

    return wide


def cusum_detection(series: pd.Series, baseline_end: str = '2021-12-31',
                    k_factor: float = 0.5, h_factor: float = 5.0) -> pd.DataFrame:
    """
    CUSUM (Cumulative Sum Control Chart) for detecting persistent
    shifts in a time series.

    S_t = max(0, S_{t-1} + (x_t - μ₀ - k))
    Alert when S_t > h

    Parameters:
    -----------
    series : pd.Series - Time series with datetime index
    baseline_end : str - End of baseline period for computing in-control mean
    k_factor : float - Allowable slack in standard deviations
    h_factor : float - Detection threshold in standard deviations
    """
    baseline = series[series.index <= baseline_end]

    if len(baseline) < 10:
        # Not enough baseline data — use first 50% of series
        baseline = series.iloc[:len(series) // 2]

    mu_0 = baseline.mean()
    sigma = baseline.std()

    if sigma == 0 or np.isnan(sigma):
        sigma = series.std()
    if sigma == 0:
        sigma = 1.0

    k = k_factor * sigma
    h = h_factor * sigma

    # Compute CUSUM
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    flags = np.zeros(len(series), dtype=bool)

    for i in range(1, len(series)):
        x = series.iloc[i]
        if np.isnan(x):
            cusum_pos[i] = cusum_pos[i-1]
            cusum_neg[i] = cusum_neg[i-1]
            continue

        cusum_pos[i] = max(0, cusum_pos[i-1] + (x - mu_0 - k))
        cusum_neg[i] = max(0, cusum_neg[i-1] + (mu_0 - x - k))

        if cusum_pos[i] > h or cusum_neg[i] > h:
            flags[i] = True

    result = pd.DataFrame({
        'date': series.index,
        'value': series.values,
        'cusum_pos': cusum_pos,
        'cusum_neg': cusum_neg,
        'cusum_flag': flags,
        'baseline_mean': mu_0,
        'threshold_h': h,
    })

    return result


def run_cusum_all_series(wide_df: pd.DataFrame,
                          baseline_end: str = '2021-12-31') -> dict:
    """
    Run CUSUM detection on all macro indicator series.
    Returns dict of DataFrames per series and a combined flag matrix.
    """
    print("\n  ── CUSUM Detection ──")

    # Key signals to monitor
    key_series = ['BDIY', 'WTISPLC', 'PWHEAMTUSDM', 'PALUMUSDM',
                  'PNICKUSDM', 'PNGASEUUSDM', 'PSUNOUSDM']

    results = {}
    flag_matrix = pd.DataFrame(index=wide_df.index)

    for series_id in key_series:
        if series_id in wide_df.columns:
            cusum_result = cusum_detection(wide_df[series_id], baseline_end)
            cusum_result.set_index('date', inplace=True)
            results[series_id] = cusum_result
            flag_matrix[f'{series_id}_flag'] = cusum_result['cusum_flag'].reindex(wide_df.index).fillna(False)

            flag_count = cusum_result['cusum_flag'].sum()
            print(f"    {series_id}: {flag_count} CUSUM flags detected")

    # Count total flags per week
    flag_matrix['total_flags'] = flag_matrix.sum(axis=1)
    flag_matrix['flag_ratio'] = flag_matrix['total_flags'] / len(key_series)

    return {
        'cusum_results': results,
        'flag_matrix': flag_matrix,
    }


def mahalanobis_detection(wide_df: pd.DataFrame,
                           baseline_end: str = '2021-12-31',
                           threshold_pct: float = 99.0) -> pd.DataFrame:
    """
    Multivariate Mahalanobis distance anomaly detection.
    Detects when multiple signals move together in unusual combinations.

    mahal_distance = sqrt((x - μ)ᵀ @ Σ⁻¹ @ (x - μ))
    """
    print("\n  ── Mahalanobis Distance Detection ──")

    # Select numeric columns with sufficient data
    numeric_cols = [c for c in wide_df.columns if wide_df[c].dtype in ['float64', 'int64']]
    data = wide_df[numeric_cols].copy()

    # Normalize to prevent scale issues
    data_normalized = (data - data.mean()) / data.std()
    data_normalized = data_normalized.fillna(0)

    # Baseline period
    baseline = data_normalized[data_normalized.index <= baseline_end]

    if len(baseline) < len(numeric_cols) + 5:
        print(f"  ⚠ Insufficient baseline data ({len(baseline)} rows). Using first 60%.")
        split_idx = int(len(data_normalized) * 0.6)
        baseline = data_normalized.iloc[:split_idx]

    # Compute baseline statistics
    mu = np.asarray(baseline.mean(), dtype=np.float64)
    cov = np.asarray(baseline.cov(), dtype=np.float64)
    
    # Regularize covariance matrix
    cov = cov + np.eye(len(numeric_cols)) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Compute Mahalanobis distance for each week
    distances = []
    for i in range(len(data_normalized)):
        x = data_normalized.iloc[i].values
        try:
            dist = mahalanobis(x, mu, cov_inv)
        except Exception:
            dist = 0.0
        distances.append(dist)

    distances = np.array(distances)

    # Threshold: percentile of baseline distribution
    baseline_distances = distances[:len(baseline)]
    threshold = np.percentile(baseline_distances, threshold_pct)

    # Flag anomalies
    anomaly_flags = distances > threshold

    result = pd.DataFrame({
        'date': data_normalized.index,
        'mahalanobis_distance': distances,
        'threshold': threshold,
        'anomaly_flag': anomaly_flags,
    })
    result.set_index('date', inplace=True)

    total_anomalies = anomaly_flags.sum()
    print(f"    Threshold (p{threshold_pct}): {threshold:.2f}")
    print(f"    Anomalies detected: {total_anomalies}")

    return result


def compute_disruption_score(cusum_data: dict, mahal_data: pd.DataFrame,
                              trade_deviation: pd.Series = None) -> pd.DataFrame:
    """
    Composite disruption score per week:
    score = 0.4 * CUSUM_ratio + 0.4 * mahal_ratio + 0.2 * trade_deviation
    """
    print("\n  ── Computing Composite Disruption Score ──")

    flag_matrix = cusum_data['flag_matrix']

    # Align indices
    common_index = flag_matrix.index.intersection(mahal_data.index)

    df = pd.DataFrame(index=common_index)

    # CUSUM component: ratio of flagged signals
    df['cusum_ratio'] = flag_matrix.loc[common_index, 'flag_ratio'].values

    # Mahalanobis component: distance / threshold (capped at 1)
    threshold = mahal_data['threshold'].iloc[0]
    df['mahal_ratio'] = np.minimum(
        mahal_data.loc[common_index, 'mahalanobis_distance'].values / threshold, 2.0
    ) / 2.0

    # Trade deviation component
    if trade_deviation is not None:
        td = trade_deviation.reindex(common_index).fillna(0)
        df['trade_deviation'] = td.values
    else:
        # Synthesize from Mahalanobis as proxy
        df['trade_deviation'] = (df['mahal_ratio'] * 0.7 +
                                  df['cusum_ratio'] * 0.3)

    # Composite score
    df['disruption_score'] = (
        0.4 * df['cusum_ratio'] +
        0.4 * df['mahal_ratio'] +
        0.2 * df['trade_deviation']
    )

    # Classification
    df['alert_level'] = pd.cut(
        df['disruption_score'],
        bins=[-0.01, 0.2, 0.4, 0.6, 1.01],
        labels=['Normal', 'Elevated', 'High', 'Critical']
    )

    print(f"    Score range: [{df['disruption_score'].min():.3f}, {df['disruption_score'].max():.3f}]")
    print(f"    Critical weeks: {(df['alert_level'] == 'Critical').sum()}")
    print(f"    High weeks: {(df['alert_level'] == 'High').sum()}")

    return df


def validate_detection(disruption_scores: pd.DataFrame,
                        con: duckdb.DuckDBPyConnection = None) -> dict:
    """
    Validate disruption detection against labeled ground truth events.
    Compute precision, recall, F1, and detection lead time.
    """
    print("\n  ── Validation Against Ground Truth ──")

    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)

    events = con.execute("SELECT * FROM disruption_events").fetchdf().copy()
    events['start_date'] = pd.to_datetime(events['start_date'])
    events['end_date'] = pd.to_datetime(events['end_date'])

    # Create ground truth labels
    true_labels = pd.Series(False, index=disruption_scores.index)
    for _, evt in events.iterrows():
        mask = (disruption_scores.index >= evt['start_date']) & \
               (disruption_scores.index <= evt['end_date'])
        true_labels[mask] = True

    # Predicted labels (score > 0.3 threshold)
    threshold = 0.3
    pred_labels = disruption_scores['disruption_score'] > threshold

    # Align
    common_idx = true_labels.index.intersection(pred_labels.index)
    y_true = true_labels[common_idx].astype(int)
    y_pred = pred_labels[common_idx].astype(int)

    if y_true.sum() == 0:
        print("  ⚠ No ground truth disruption weeks in the detection period")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'detection_lead_weeks': 0}

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Detection lead time: how many weeks before event start does system first flag
    lead_times = []
    for _, evt in events.iterrows():
        # Look in 8-week window before event start
        pre_window = disruption_scores[
            (disruption_scores.index >= evt['start_date'] - pd.Timedelta(weeks=8)) &
            (disruption_scores.index < evt['start_date'])
        ]
        early_flags = pre_window[pre_window['disruption_score'] > threshold * 0.7]

        if len(early_flags) > 0:
            first_flag = early_flags.index[0]
            lead_weeks = (evt['start_date'] - first_flag).days / 7
            lead_times.append(lead_weeks)
        else:
            lead_times.append(0)

    avg_lead_time = np.mean(lead_times) if lead_times else 0

    results = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'detection_lead_weeks': round(avg_lead_time, 1),
        'threshold': threshold,
        'true_events': int(y_true.sum()),
        'predicted_events': int(y_pred.sum()),
    }

    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    F1 Score:  {results['f1']:.4f}")
    print(f"    Avg Detection Lead Time: {results['detection_lead_weeks']:.1f} weeks")

    return results


def run_module_b() -> dict:
    """Execute the full Module B disruption detection pipeline."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Module B — Disruption Detection                     ║")
    print("╚══════════════════════════════════════════════════════╝")

    con = duckdb.connect(DB_PATH, read_only=True)

    # Load macro data
    print("\n  Loading macro indicator data...")
    wide_df = load_macro_data(con)
    print(f"  ✓ Loaded {len(wide_df)} weeks × {len(wide_df.columns)} indicators")

    # CUSUM detection
    cusum_data = run_cusum_all_series(wide_df)

    # Mahalanobis detection
    mahal_data = mahalanobis_detection(wide_df)

    # Composite disruption score
    disruption_scores = compute_disruption_score(cusum_data, mahal_data)

    # Validation
    validation = validate_detection(disruption_scores, con)

    con.close()

    return {
        'wide_df': wide_df,
        'cusum_data': cusum_data,
        'mahal_data': mahal_data,
        'disruption_scores': disruption_scores,
        'validation': validation,
    }


if __name__ == '__main__':
    results = run_module_b()
    print("\n── Disruption Score Summary ──")
    scores = results['disruption_scores']
    print(scores[scores['alert_level'].isin(['High', 'Critical'])].head(20).to_string())
