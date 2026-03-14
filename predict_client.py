#!/usr/bin/env python3
"""
Send prediction requests to the served MLflow model.

Prerequisites:
  1. Run train_and_serve.py first
  2. Start the model server:
     mlflow models serve --env-manager=local \
       -m runs:/<RUN_ID>/model -h 127.0.0.1 -p 5002

Usage:
  python predict_client.py
"""

import requests
import json

URL = "http://127.0.0.1:5002/invocations"

# Feature columns (must match training order):
# accelerator_encoded, accelerator_count, nodes, total_gpus,
# is_nvidia, is_amd, has_tensorrt, has_vllm, uses_fp8, uses_int4

# Example: 8x NVIDIA H200 with TensorRT, FP8
sample_systems = [
    {
        "description": "8x NVIDIA H200 + TensorRT + FP8",
        "features": [10, 8, 1, 8, 1, 0, 1, 0, 1, 0],
    },
    {
        "description": "8x AMD MI325X + vLLM + FP8",
        "features": [3, 8, 1, 8, 0, 1, 0, 1, 1, 0],
    },
    {
        "description": "1x NVIDIA H100 + TensorRT + FP8",
        "features": [8, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    },
    {
        "description": "4x NVIDIA B200 + TensorRT + FP8",
        "features": [5, 4, 1, 4, 1, 0, 1, 0, 1, 0],
    },
]

COLUMNS = [
    "accelerator_encoded", "accelerator_count", "nodes", "total_gpus",
    "is_nvidia", "is_amd", "has_tensorrt", "has_vllm",
    "uses_fp8", "uses_int4",
]


def main():
    print("Sending prediction requests to model server...\n")

    data = {
        "inputs": {col: [s["features"][i] for s in sample_systems] for i, col in enumerate(COLUMNS)},
    }

    try:
        response = requests.post(URL, json=data, timeout=10)
        response.raise_for_status()
        predictions = response.json()

        print(f"{'System Description':<45} {'Predicted Throughput (tokens/s)':>30}")
        print("-" * 78)

        preds = predictions.get("predictions", predictions)
        for system, pred in zip(sample_systems, preds):
            print(f"{system['description']:<45} {pred:>25,.0f}")

    except requests.ConnectionError:
        print("ERROR: Could not connect to model server at", URL)
        print("Make sure the server is running:")
        print("  mlflow models serve --env-manager=local -m runs:/<RUN_ID>/model -h 127.0.0.1 -p 5002")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
