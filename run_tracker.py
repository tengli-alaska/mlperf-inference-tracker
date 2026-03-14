#!/usr/bin/env python3
"""
Main entry point: fetch MLPerf results, parse, and log to MLflow.

Usage:
    python run_tracker.py                  # All versions
    python run_tracker.py --versions v5.1  # Single version
"""

import argparse
import time
from src.fetch_results import fetch_and_parse
from src.track import log_all_results


def main():
    parser = argparse.ArgumentParser(description="MLPerf Inference Results Tracker")
    parser.add_argument(
        "--versions", nargs="+", default=None,
        help="MLPerf versions to fetch (e.g., v4.1 v5.0 v5.1). Default: all",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MLPerf Inference Results Tracker")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Fetch and parse
    records = fetch_and_parse(args.versions)

    if not records:
        print("\nNo records found. Check network connection or data format.")
        print("You can manually download summary_results.json from:")
        print("  https://github.com/mlcommons/inference_results_v5.1")
        print("  Place it in data/summary_results_v5.1.json")
        return

    # Step 2: Log to MLflow
    print("\nLogging to MLflow...")
    log_all_results(records)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print("Launch MLflow UI with:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")


if __name__ == "__main__":
    main()