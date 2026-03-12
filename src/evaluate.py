"""
Model Evaluation for FPL Points Prediction

Evaluates both baseline and tuned models and compares their performance.
Outputs separate metrics files for each and a comparison summary.
"""

import pandas as pd
import numpy as np
import json
import os
import yaml
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# Load Configuration from params.yaml
# =============================================================================

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

PARAMS = load_params()

# Paths from params.yaml
TEST_PATH = PARAMS['paths']['test_data']
BASELINE_PREDICTIONS_PATH = PARAMS['paths']['baseline_predictions']
TUNED_PREDICTIONS_PATH = PARAMS['paths']['tuned_predictions']

METRICS_DIR = PARAMS['paths']['metrics_dir']
BASELINE_METRICS_PATH = f"{METRICS_DIR}/baseline_metrics.json"
TUNED_METRICS_PATH = f"{METRICS_DIR}/tuned_metrics.json"
COMPARISON_METRICS_PATH = f"{METRICS_DIR}/comparison_metrics.json"
# Legacy path for backwards compatibility
RESULTS_PATH = f"{METRICS_DIR}/results.json"

TARGET = PARAMS['features']['target']


def compute_metrics(y_true, y_pred, model_name):
    """Calculate MAE, RMSE, and R² for predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}


def evaluate_predictions(predictions_path, test_df, model_type="baseline"):
    """Evaluate predictions and return metrics dictionary."""
    
    if not os.path.exists(predictions_path):
        print(f"  ⚠️  Predictions file not found: {predictions_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} Models")
    print('='*60)
    
    predictions_df = pd.read_csv(predictions_path)
    y_true = predictions_df[TARGET]
    
    metrics = {}
    
    # Naive baselines (same for both)
    print("\n  --- Naive Baselines ---")
    metrics['naive_lag1'] = compute_metrics(
        y_true, test_df['total_points_lag1'].fillna(0), "Naive (Lag1)"
    )
    metrics['naive_ma5'] = compute_metrics(
        y_true, test_df['total_points_ma5'].fillna(0), "Naive (MA5)"
    )
    
    # Model predictions
    print(f"\n  --- {model_type.title()} Models ---")
    metrics['ridge_regression'] = compute_metrics(
        y_true, predictions_df['pred_ridge'], "Ridge Regression"
    )
    metrics['random_forest'] = compute_metrics(
        y_true, predictions_df['pred_rf'], "Random Forest"
    )
    metrics['gradient_boosting'] = compute_metrics(
        y_true, predictions_df['pred_gb'], "XGBoost"
    )
    
    return metrics


def compare_baseline_vs_tuned(baseline_metrics, tuned_metrics):
    """
    Compare baseline and tuned model performance.
    Calculate improvement for each model.
    """
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Tuned")
    print("="*60)
    
    comparison = {}
    
    models = ['ridge_regression', 'random_forest', 'gradient_boosting']
    
    for model in models:
        if model in baseline_metrics and model in tuned_metrics:
            base = baseline_metrics[model]
            tuned = tuned_metrics[model]
            
            mae_diff = base['mae'] - tuned['mae']
            rmse_diff = base['rmse'] - tuned['rmse']
            r2_diff = tuned['r2'] - base['r2']
            
            mae_pct = (mae_diff / base['mae']) * 100 if base['mae'] != 0 else 0
            
            comparison[model] = {
                'baseline': base,
                'tuned': tuned,
                'mae_improvement': round(mae_diff, 4),
                'mae_improvement_pct': round(mae_pct, 2),
                'rmse_improvement': round(rmse_diff, 4),
                'r2_improvement': round(r2_diff, 4),
                'tuned_is_better': mae_diff > 0
            }
            
            model_display = model.replace('_', ' ').title()
            print(f"\n  {model_display}:")
            print(f"    MAE:  {base['mae']:.4f} → {tuned['mae']:.4f} "
                  f"({'↓' if mae_diff > 0 else '↑'} {abs(mae_diff):.4f}, {abs(mae_pct):.1f}%)")
            print(f"    RMSE: {base['rmse']:.4f} → {tuned['rmse']:.4f} "
                  f"({'↓' if rmse_diff > 0 else '↑'} {abs(rmse_diff):.4f})")
            print(f"    R²:   {base['r2']:.4f} → {tuned['r2']:.4f} "
                  f"({'↑' if r2_diff > 0 else '↓'} {abs(r2_diff):.4f})")
            
            if mae_diff > 0:
                print(f"    ✅ Tuning IMPROVED {model_display} by {mae_pct:.1f}%")
            else:
                print(f"    ❌ Tuning did not improve {model_display}")
    
    # Summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    
    improved = sum(1 for m in comparison.values() if m['tuned_is_better'])
    total = len(comparison)
    
    print(f"  Models improved: {improved}/{total}")
    
    if improved > 0:
        best_improvement = max(comparison.items(), 
                               key=lambda x: x[1]['mae_improvement_pct'])
        print(f"  Best improvement: {best_improvement[0].replace('_', ' ').title()} "
              f"({best_improvement[1]['mae_improvement_pct']:.1f}% lower MAE)")
    
    return comparison


if __name__ == "__main__":
    print("="*70)
    print("FPL POINTS PREDICTOR - MODEL EVALUATION")
    print("="*70)
    
    # Load test data
    print("\n📂 Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    
    # Ensure metrics directory exists
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Evaluate models with progress bar
    evaluation_tasks = [
        ("Baseline models", BASELINE_PREDICTIONS_PATH, "baseline", BASELINE_METRICS_PATH),
        ("Tuned models", TUNED_PREDICTIONS_PATH, "tuned", TUNED_METRICS_PATH),
    ]
    
    all_metrics = {}
    pbar = tqdm(evaluation_tasks, desc="⏳ Evaluating models", unit="model")
    for task_name, pred_path, model_type, metrics_path in pbar:
        pbar.set_postfix_str(task_name)
        metrics = evaluate_predictions(pred_path, test_df, model_type)
        if metrics:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            all_metrics[model_type] = metrics
    
    baseline_metrics = all_metrics.get("baseline")
    tuned_metrics = all_metrics.get("tuned")
    
    # Also save to legacy results.json for backwards compatibility
    if baseline_metrics:
        with open(RESULTS_PATH, 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
    
    # Compare if both exist
    if baseline_metrics and tuned_metrics:
        comparison = compare_baseline_vs_tuned(baseline_metrics, tuned_metrics)
        with open(COMPARISON_METRICS_PATH, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n💾 Comparison saved to {COMPARISON_METRICS_PATH}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
