import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_PATH = "data/test.csv"
PREDICTIONS_BASELINE_PATH = "data/predictions.csv"
PREDICTIONS_TUNED_PATH = "data/predictions_tuned.csv"
METRICS_PATH = "metrics/results.json"

TARGET = 'total_points'


def compute_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}


def evaluate_prediction_file(test_df, predictions_path, label):
    predictions_df = pd.read_csv(predictions_path)
    y_true = predictions_df[TARGET]
    print(f"\nEvaluating {label} predictions: {predictions_path}")
    return {
        'naive_lag1':        compute_metrics(y_true, test_df['total_points_lag1'].fillna(0), f"{label} | Naive (Lag1)"),
        'naive_ma5':         compute_metrics(y_true, test_df['total_points_ma5'].fillna(0),  f"{label} | Naive (MA5)"),
        'ridge_regression':  compute_metrics(y_true, predictions_df['pred_ridge'],            f"{label} | Ridge Regression"),
        'random_forest':     compute_metrics(y_true, predictions_df['pred_rf'],               f"{label} | Random Forest"),
        'gradient_boosting': compute_metrics(y_true, predictions_df['pred_gb'],               f"{label} | XGBoost"),
    }


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_PATH)
    baseline_metrics = evaluate_prediction_file(test_df, PREDICTIONS_BASELINE_PATH, "Baseline")
    tuned_metrics = evaluate_prediction_file(test_df, PREDICTIONS_TUNED_PATH, "Tuned")

    deltas = {}
    for model_name in ["ridge_regression", "random_forest", "gradient_boosting"]:
        deltas[model_name] = {
            "mae_delta": round(tuned_metrics[model_name]["mae"] - baseline_metrics[model_name]["mae"], 4),
            "rmse_delta": round(tuned_metrics[model_name]["rmse"] - baseline_metrics[model_name]["rmse"], 4),
            "r2_delta": round(tuned_metrics[model_name]["r2"] - baseline_metrics[model_name]["r2"], 4),
        }

    metrics = {
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
        "tuned_minus_baseline": deltas,
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")
