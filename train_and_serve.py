#!/usr/bin/env python3
"""
Train a model to predict MLPerf Inference throughput from system specs.

Covers all MLflow lab requirements:
  1. mlflow.autolog()           — auto-log sklearn model params/metrics
  2. mlflow.start_run()         — manual run context
  3. mlflow.log_param/metric()  — manual param/metric logging
  4. infer_signature()          — model input/output schema
  5. mlflow.sklearn.log_model() — log model with signature
  6. registered_model_name      — register in Model Registry
  7. mlflow.pyfunc.load_model() — reload and predict
  8. mlflow models serve        — REST API serving (instructions printed)

Usage:
    # Step 1: Make sure you've already run run_tracker.py to populate data/
    python train_and_serve.py

    # Step 2: Serve the best model
    mlflow models serve --env-manager=local -m models:/mlperf-throughput-predictor/1 -h 127.0.0.1 -p 5002

    # Step 3: Send a prediction request
    python predict_client.py
"""

import json
import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKING_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT_NAME = "mlperf-throughput-prediction"

# Target benchmark to predict
TARGET_BENCHMARK = "llama2-70b-99"
TARGET_SCENARIO = "Offline"


def eval_metrics(actual, pred):
    """Evaluate model performance — same pattern as the original lab."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_and_prepare_data():
    """
    Load raw MLPerf JSON results and build a feature matrix for training.

    Features:
      - accelerator_encoded: label-encoded accelerator name
      - accelerator_count: number of GPUs
      - nodes: number of nodes
      - is_nvidia, is_amd: binary flags
      - has_tensorrt, has_vllm: software stack flags

    Target:
      - Performance_Result for Llama2-70B Offline
    """
    data_dir = os.path.join(PROJECT_ROOT, "data")
    all_rows = []

    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            all_rows.extend(raw)

    # Filter to target benchmark + scenario
    filtered = []
    for row in all_rows:
        if (row.get("Model") == TARGET_BENCHMARK and
            row.get("Scenario") == TARGET_SCENARIO):
            score = row.get("Performance_Result")
            if score is not None and score != "":
                try:
                    score = float(str(score).replace(",", ""))
                except (ValueError, TypeError):
                    continue
                filtered.append({
                    "accelerator": row.get("Accelerator", "unknown"),
                    "accelerator_count": int(row.get("a#", 1) or 1),
                    "nodes": int(row.get("Nodes", 1) or 1),
                    "software": str(row.get("Software", "")),
                    "submitter": row.get("Submitter", "unknown"),
                    "weight_data_types": str(row.get("weight_data_types", "")),
                    "throughput": score,
                })

    if not filtered:
        print("ERROR: No data found. Run 'python run_tracker.py' first.")
        sys.exit(1)

    df = pd.DataFrame(filtered)
    print(f"Loaded {len(df)} samples for {TARGET_BENCHMARK} {TARGET_SCENARIO}")

    # ---- Feature engineering ----

    # Accelerator name encoding
    le_accel = LabelEncoder()
    df["accelerator_encoded"] = le_accel.fit_transform(df["accelerator"])

    # Binary hardware vendor flags
    accel_lower = df["accelerator"].str.lower()
    df["is_nvidia"] = accel_lower.str.contains("nvidia|h100|h200|b200|gb200|l40|a100").astype(int)
    df["is_amd"] = accel_lower.str.contains("amd|mi300|mi325|mi355").astype(int)

    # Software stack flags
    sw_lower = df["software"].str.lower()
    df["has_tensorrt"] = sw_lower.str.contains("tensorrt").astype(int)
    df["has_vllm"] = sw_lower.str.contains("vllm").astype(int)

    # Weight format flags
    wdt_lower = df["weight_data_types"].str.lower()
    df["uses_fp8"] = wdt_lower.str.contains("fp8").astype(int)
    df["uses_int4"] = wdt_lower.str.contains("int4|fp4").astype(int)

    # Total GPU count
    df["total_gpus"] = df["accelerator_count"] * df["nodes"]

    feature_cols = [
        "accelerator_encoded", "accelerator_count", "nodes", "total_gpus",
        "is_nvidia", "is_amd", "has_tensorrt", "has_vllm",
        "uses_fp8", "uses_int4",
    ]

    X = df[feature_cols]
    y = df["throughput"]

    return X, y, le_accel, feature_cols


def main():
    np.random.seed(42)

    # ---- Load data ----
    X, y, le_accel, feature_cols = load_and_prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ================================================================
    # Part A: Autologging demo (same pattern as starter.ipynb)
    # ================================================================
    print("\n--- Part A: Autologging with RandomForest ---")
    mlflow.autolog()

    rf_auto = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3, random_state=42)
    rf_auto.fit(X_train, y_train)
    auto_preds = rf_auto.predict(X_test)
    print(f"  Autolog RF — R2: {r2_score(y_test, auto_preds):.4f}")

    mlflow.autolog(disable=True)  # Turn off for manual logging below

    # ================================================================
    # Part B: Manual logging — multiple models (same as lab course_notes)
    # ================================================================
    models = {
        "Linear Regression": LinearRegression(),
        "ElasticNet alpha=0.5 l1=0.5": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
        "ElasticNet alpha=0.2 l1=0.7": ElasticNet(alpha=0.2, l1_ratio=0.7, random_state=42),
        "ElasticNet alpha=0.8 l1=0.3": ElasticNet(alpha=0.8, l1_ratio=0.3, random_state=42),
        "Random Forest n=50": RandomForestRegressor(n_estimators=50, random_state=42),
        "Random Forest n=100": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting n=100": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    best_r2 = -999
    best_run_id = None
    best_model_name = None

    for model_name, model in models.items():
        print(f"\n--- Training: {model_name} ---")

        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Evaluate
            (rmse, mae, r2) = eval_metrics(y_test, predictions)
            print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2:.4f}")

            # Log parameters
            mlflow.log_param("model_type", model_name)
            if hasattr(model, "alpha"):
                mlflow.log_param("alpha", model.alpha)
            if hasattr(model, "l1_ratio"):
                mlflow.log_param("l1_ratio", model.l1_ratio)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Infer signature
            train_preds = model.predict(X_train)
            signature = infer_signature(X_train, train_preds)

            # Log model with signature — check tracking store type
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                # Register the best model in Model Registry
                if r2 > best_r2:
                    mlflow.sklearn.log_model(
                        model, "model",
                        registered_model_name="mlperf-throughput-predictor",
                        signature=signature,
                    )
                    best_r2 = r2
                    best_run_id = mlflow.active_run().info.run_id
                    best_model_name = model_name
                else:
                    mlflow.sklearn.log_model(model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)

    # ================================================================
    # Part C: Load model back and predict (same as lab course_notes)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Best model: {best_model_name} (R2={best_r2:.4f})")
    print(f"  Run ID: {best_run_id}")
    print(f"{'='*60}")

    print("\n--- Part C: Loading model back with mlflow.pyfunc.load_model ---")
    model_uri = f"runs:/{best_run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Predict on a sample
    sample = X_test.head(3)
    sample_preds = loaded_model.predict(sample)
    print(f"  Sample predictions: {sample_preds}")
    print(f"  Actual values:      {y_test.head(3).values}")

    # ================================================================
    # Part D: Serving instructions
    # ================================================================
    print(f"\n{'='*60}")
    print("  SERVING THE MODEL")
    print(f"{'='*60}")
    print(f"""
To serve the registered model via REST API:

  mlflow models serve \\
    --env-manager=local \\
    -m runs:/{best_run_id}/model \\
    -h 127.0.0.1 -p 5002

Then send a prediction request:

  python predict_client.py

Or via curl:

  curl -X POST http://127.0.0.1:5002/invocations \\
    -H "Content-Type: application/json" \\
    -d '{{"columns": {feature_cols}, "instances": [[0, 8, 1, 8, 1, 0, 1, 0, 1, 0]]}}'
""")

    print("Done. Launch MLflow UI with:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001\n")


if __name__ == "__main__":
    main()