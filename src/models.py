"""
Disruption Prediction Model — Module E
========================================
Champion-challenger ensemble (XGBoost + LightGBM) with stacked
meta-learner for predicting 30-day stockout risk.
Includes SHAP explainability.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    recall_score
)
import pickle
from sklearn.model_selection import cross_val_predict
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / 'data' / 'processed' / 'supply_chain.db')


def train_xgboost(X_train, y_train, X_val, y_val, feature_names=None) -> dict:
    """
    Train XGBoost champion model.
    """
    print("\n  ── Training XGBoost Champion ──")

    # Compute class weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(1, pos_count)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_prauc = average_precision_score(y_train, train_proba)
    val_prauc = average_precision_score(y_val, val_proba)

    print(f"    Train PR-AUC: {train_prauc:.4f}")
    print(f"    Val PR-AUC:   {val_prauc:.4f}")
    print(f"    Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    # Feature importance
    if feature_names:
        importance = dict(zip(feature_names, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("    Top 5 features:")
        for feat, imp in sorted_imp[:5]:
            print(f"      {feat}: {imp:.4f}")

    return {
        'model': model,
        'train_proba': train_proba,
        'val_proba': val_proba,
        'train_prauc': train_prauc,
        'val_prauc': val_prauc,
        'feature_importance': importance if feature_names else {},
    }


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names=None) -> dict:
    """
    Train LightGBM challenger model.
    """
    print("\n  ── Training LightGBM Challenger ──")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=63,
        class_weight='balanced',
        metric='average_precision',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_prauc = average_precision_score(y_train, train_proba)
    val_prauc = average_precision_score(y_val, val_proba)

    print(f"    Train PR-AUC: {train_prauc:.4f}")
    print(f"    Val PR-AUC:   {val_prauc:.4f}")

    # Feature importance
    if feature_names:
        importance = dict(zip(feature_names, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("    Top 5 features:")
        for feat, imp in sorted_imp[:5]:
            print(f"      {feat}: {imp:.4f}")

    return {
        'model': model,
        'train_proba': train_proba,
        'val_proba': val_proba,
        'train_prauc': train_prauc,
        'val_prauc': val_prauc,
        'feature_importance': importance if feature_names else {},
    }


def build_ensemble(xgb_results: dict, lgb_results: dict,
                    X_val, y_val, X_test=None, y_test=None,
                    xgb_weight: float = 0.6, lgb_weight: float = 0.4,
                    y_train=None) -> dict:
    """
    Ensemble: weighted average + stacked logistic regression meta-learner.
    """
    print("\n  ── Building Ensemble ──")

    # 1. Weighted average ensemble
    val_proba_weighted = (xgb_weight * xgb_results['val_proba'] +
                           lgb_weight * lgb_results['val_proba'])
    weighted_prauc = average_precision_score(y_val, val_proba_weighted)
    print(f"    Weighted Ensemble Val PR-AUC: {weighted_prauc:.4f}")

    # 2. Stacked meta-learner
    meta_features_train = np.column_stack([
        xgb_results['train_proba'],
        lgb_results['train_proba']
    ])
    meta_features_val = np.column_stack([
        xgb_results['val_proba'],
        lgb_results['val_proba']
    ])

    # Use actual training labels if available, else reconstruct from probabilities
    if y_train is not None:
        train_labels = np.array(y_train).astype(int)
    else:
        # Use consensus of both models at optimized threshold
        xgb_pred = (xgb_results['train_proba'] > 0.3).astype(int)
        lgb_pred = (lgb_results['train_proba'] > 0.3).astype(int)
        train_labels = ((xgb_pred + lgb_pred) >= 1).astype(int)

    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(meta_features_train, train_labels)

    val_proba_stacked = meta_model.predict_proba(meta_features_val)[:, 1]
    stacked_prauc = average_precision_score(y_val, val_proba_stacked)
    print(f"    Stacked Ensemble Val PR-AUC:  {stacked_prauc:.4f}")

    # Use whichever ensemble is better
    if stacked_prauc > weighted_prauc:
        best_method = 'stacked'
        best_val_proba = val_proba_stacked
        best_prauc = stacked_prauc
    else:
        best_method = 'weighted'
        best_val_proba = val_proba_weighted
        best_prauc = weighted_prauc

    print(f"    Best ensemble: {best_method} (PR-AUC: {best_prauc:.4f})")

    # Test set evaluation
    test_results = {}
    if X_test is not None and y_test is not None:
        xgb_test_proba = xgb_results['model'].predict_proba(X_test)[:, 1]
        lgb_test_proba = lgb_results['model'].predict_proba(X_test)[:, 1]

        if best_method == 'stacked':
            test_proba = meta_model.predict_proba(
                np.column_stack([xgb_test_proba, lgb_test_proba])
            )[:, 1]
        else:
            test_proba = xgb_weight * xgb_test_proba + lgb_weight * lgb_test_proba

        test_prauc = average_precision_score(y_test, test_proba)
        print(f"\n    Test PR-AUC: {test_prauc:.4f}")

        # Precision@K (top 10%)
        k = max(1, int(len(test_proba) * 0.10))
        top_k_indices = np.argsort(test_proba)[-k:]
        precision_at_k = y_test.iloc[top_k_indices].mean()
        print(f"    Precision@10%: {precision_at_k:.4f}")

        # Confusion matrix and recall at threshold 0.3 (matches disruption_score threshold)
        test_pred = (test_proba >= 0.3).astype(int)
        cm = confusion_matrix(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        print(f"    Recall@0.3: {test_recall:.4f}")
        print(f"    Confusion Matrix:\n{cm}")

        test_results = {
            'test_proba': test_proba,
            'test_prauc': test_prauc,
            'precision_at_k': precision_at_k,
            'test_recall': test_recall,
            'confusion_matrix': cm,
            'xgb_test_proba': xgb_test_proba,
            'lgb_test_proba': lgb_test_proba,
        }

    return {
        'method': best_method,
        'val_prauc': best_prauc,
        'val_proba': best_val_proba,
        'meta_model': meta_model if best_method == 'stacked' else None,
        'xgb_weight': xgb_weight,
        'lgb_weight': lgb_weight,
        **test_results,
    }


def compute_shap_values(model, X, feature_names=None, max_samples=5000) -> dict:
    """
    Compute SHAP values using TreeExplainer.
    """
    print("\n  ── Computing SHAP Values ──")

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use positive class

        # Global feature importance (mean |SHAP|)
        mean_shap = np.abs(shap_values).mean(axis=0)

        if feature_names:
            importance = dict(zip(feature_names, mean_shap))
        else:
            importance = {f'feature_{i}': v for i, v in enumerate(mean_shap)}

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        print("    Top 10 SHAP features:")
        for feat, imp in sorted_imp[:10]:
            print(f"      {feat}: {imp:.4f}")

        # Feature group contributions
        feature_groups = {
            'Graph': ['betweenness_centrality', 'pagerank', 'degree_centrality',
                       'clustering_coefficient', 'supplier_country_risk_tier'],
            'Disruption Detection': ['cusum_flag_rolling_4w', 'mahalanobis_distance_current',
                                      'disruption_score_current', 'weeks_since_last_disruption'],
            'Forecasting': ['demand_rolling_4w', 'demand_rolling_8w', 'demand_std_4w',
                             'forecast_uncertainty_width', 'demand_trend_slope',
                             'disruption_adjusted_forecast_flag'],
            'Inventory': ['current_inventory_weeks_of_cover', 'days_to_reorder_point',
                           'safety_stock_adequacy_ratio', 'lead_time_deviation_from_normal'],
        }

        group_importance = {}
        for group, features in feature_groups.items():
            group_total = sum(importance.get(f, 0) for f in features)
            group_importance[group] = round(group_total, 4)

        total_importance = sum(group_importance.values())
        if total_importance > 0:
            print("\n    Feature Group Contributions:")
            for group, imp in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
                pct = imp / total_importance * 100
                print(f"      {group}: {pct:.1f}%")

        return {
            'shap_values': shap_values,
            'feature_importance': importance,
            'group_importance': group_importance,
            'explainer': explainer,
            'X_sample': X_sample,
        }

    except Exception as e:
        print(f"    ⚠ SHAP computation error: {e}")
        return {
            'shap_values': None,
            'feature_importance': {},
            'group_importance': {},
            'explainer': None,
            'X_sample': X_sample,
        }


def compute_prediction_lead_time(feature_matrix: pd.DataFrame,
                                   test_proba: np.ndarray,
                                   threshold: float = 0.5) -> float:
    """
    Compute average prediction lead time: how many weeks before
    actual stockout does the model first flag a SKU.
    """
    test_data = feature_matrix[feature_matrix['week_start_date'] >= '2023-10-01'].copy()

    if len(test_data) != len(test_proba):
        return 0.0

    test_data['pred_risk'] = test_proba

    lead_times = []

    for sku_id in test_data['sku_id'].unique():
        sku_data = test_data[test_data['sku_id'] == sku_id].sort_values('week_start_date')

        # Find actual stockout weeks
        stockout_weeks = sku_data[sku_data['stockout_flag'] == True]['week_start_date']

        for stockout_date in stockout_weeks:
            # Look back for first prediction flag
            pre_stockout = sku_data[
                (sku_data['week_start_date'] < stockout_date) &
                (sku_data['week_start_date'] >= stockout_date - pd.Timedelta(weeks=8))
            ]

            flags = pre_stockout[pre_stockout['pred_risk'] > threshold]

            if len(flags) > 0:
                first_flag = flags.iloc[0]['week_start_date']
                lead_weeks = (stockout_date - first_flag).days / 7
                lead_times.append(lead_weeks)

    return round(np.mean(lead_times), 1) if lead_times else 0.0


def run_module_e(feature_matrix: pd.DataFrame = None) -> dict:
    """Execute the full Module E prediction pipeline."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Module E — Disruption Prediction Model              ║")
    print("╚══════════════════════════════════════════════════════╝")

    from src.features import build_feature_matrix, split_chronological

    # Build feature matrix if not provided
    if feature_matrix is None:
        feature_matrix = build_feature_matrix()

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
        split_chronological(feature_matrix)

    # Train models
    xgb_results = train_xgboost(X_train, y_train, X_val, y_val, feature_names)
    lgb_results = train_lightgbm(X_train, y_train, X_val, y_val, feature_names)

    # Ensemble
    ensemble_results = build_ensemble(
        xgb_results, lgb_results,
        X_val, y_val, X_test, y_test,
        y_train=y_train,
    )

    # SHAP
    shap_results = compute_shap_values(
        xgb_results['model'], X_test, feature_names
    )

    # Prediction lead time
    if 'test_proba' in ensemble_results:
        lead_time = compute_prediction_lead_time(
            feature_matrix, ensemble_results['test_proba']
        )
        print(f"\n    Avg Prediction Lead Time: {lead_time:.1f} weeks")
    else:
        lead_time = 0

    # Summary
    print("\n  ── Model Performance Summary ──")
    print(f"     XGBoost Val PR-AUC:    {xgb_results['val_prauc']:.4f}")
    print(f"     LightGBM Val PR-AUC:   {lgb_results['val_prauc']:.4f}")
    print(f"     Ensemble Val PR-AUC:   {ensemble_results['val_prauc']:.4f}")
    if 'test_prauc' in ensemble_results:
        print(f"     Ensemble Test PR-AUC:  {ensemble_results['test_prauc']:.4f}")
        print(f"     Precision@10%:         {ensemble_results.get('precision_at_k', 0):.4f}")
        print(f"     Recall@0.3:            {ensemble_results.get('test_recall', 0):.4f}")
    print(f"     Prediction Lead Time:  {lead_time:.1f} weeks")

    # ── Serialize results for dashboard consumption ──
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'model_results.pkl'

    # Build predictions DataFrame with SKU-level probabilities
    test_data = feature_matrix[feature_matrix['week_start_date'] >= '2023-10-01'].copy()
    if 'test_proba' in ensemble_results:
        test_data['stockout_probability'] = ensemble_results['test_proba']
        test_data['risk_level'] = pd.cut(
            test_data['stockout_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )

    serializable = {
        'predictions': test_data,
        'metrics': {
            'test_prauc': ensemble_results.get('test_prauc', 0),
            'precision_at_k': ensemble_results.get('precision_at_k', 0),
            'test_recall': ensemble_results.get('test_recall', 0),
            'prediction_lead_time': lead_time,
            'confusion_matrix': ensemble_results.get('confusion_matrix', np.zeros((2, 2))),
        },
        'shap_importance': shap_results.get('feature_importance', {}),
        'shap_group_importance': shap_results.get('group_importance', {}),
        'feature_names': feature_names,
    }

    with open(results_path, 'wb') as f:
        pickle.dump(serializable, f)
    print(f"\n  ✓ Model results saved to {results_path}")

    return {
        'feature_matrix': feature_matrix,
        'xgb_results': xgb_results,
        'lgb_results': lgb_results,
        'ensemble': ensemble_results,
        'shap': shap_results,
        'feature_names': feature_names,
        'prediction_lead_time': lead_time,
    }


if __name__ == '__main__':
    results = run_module_e()
