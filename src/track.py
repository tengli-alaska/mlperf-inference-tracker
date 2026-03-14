"""
Log parsed MLPerf Inference results into MLflow.

Each unique system becomes one MLflow run with:
  - Parameters: system metadata
  - Metrics: benchmark scores + per-accelerator normalized scores
  - Tags: accelerator family for filtering
"""

import mlflow
import os
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKING_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT_NAME = "mlperf-inference-tracker"

PARAM_KEYS = {
    "mlperf_version", "submitter", "system", "platform", "accelerator",
    "accelerator_count", "processor", "suite", "category", "availability",
    "software", "weight_data_types", "nodes", "os",
}


def init_tracking():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_all_results(records: List[Dict[str, Any]]):
    """Log every system record as an MLflow run."""
    init_tracking()

    total = len(records)
    logged = 0
    skipped = 0

    for i, rec in enumerate(records, 1):
        try:
            _log_one(rec)
            logged += 1
            if i % 50 == 0 or i == total:
                print(f"  Logged {i}/{total} runs...")
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [WARN] Skipped record {i}: {e}")

    print(f"\nLogging complete: {logged} runs logged, {skipped} skipped")


def _log_one(rec: Dict[str, Any]):
    """Log a single system record."""
    submitter = rec.get("submitter", "unknown")
    platform = rec.get("platform", "unknown")
    ver = rec.get("mlperf_version", "?")
    run_name = f"{submitter}_{platform}_{ver}".replace(" ", "_")[:250]

    with mlflow.start_run(run_name=run_name):
        # ---- Parameters ----
        params = {}
        for k in PARAM_KEYS:
            val = rec.get(k)
            if val is not None:
                params[k] = str(val)[:250]
        if params:
            mlflow.log_params(params)

        # ---- Metrics ----
        metrics = {}
        accel_count = int(rec.get("accelerator_count", 1) or 1)

        for k, v in rec.items():
            if k in PARAM_KEYS:
                continue
            if isinstance(v, (int, float)) and v > 0:
                clean_key = k[:250]
                metrics[clean_key] = v

                # Per-accelerator normalized score
                if accel_count > 1:
                    metrics[f"{clean_key}_per_accel"] = round(v / accel_count, 2)

        if metrics:
            mlflow.log_metrics(metrics)

        # ---- Tags ----
        tags = {"source": "mlcommons"}
        if rec.get("mlperf_version"):
            tags["mlperf_version"] = rec["mlperf_version"]
        if rec.get("category"):
            tags["category"] = rec["category"]
        if rec.get("suite"):
            tags["suite"] = rec["suite"]

        # Accelerator family tag
        accel_str = str(rec.get("accelerator", "")).lower()
        family = _classify_accel(accel_str)
        if family:
            tags["accel_family"] = family

        mlflow.set_tags(tags)


def _classify_accel(accel_lower: str) -> str:
    """Map accelerator string to a family for filtering."""
    families = [
        ("gb300", "GB300"),
        ("gb200", "GB200"),
        ("b300", "B300"),
        ("b200", "B200"),
        ("h200", "H200"),
        ("h100", "H100"),
        ("a100", "A100"),
        ("l40s", "L40S"),
        ("l40", "L40"),
        ("l4", "L4"),
        ("mi325", "MI325X"),
        ("mi300x", "MI300X"),
        ("mi300a", "MI300A"),
        ("gaudi3", "Gaudi3"),
        ("gaudi2", "Gaudi2"),
    ]
    for pattern, label in families:
        if pattern in accel_lower:
            return label
    return "Other"