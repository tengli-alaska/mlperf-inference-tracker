"""
Download and parse official MLPerf Inference results from MLCommons GitHub repos.

Each version publishes a summary_results.json where every row is one
(system × model × scenario) result. We pivot these into one record per
unique system, with benchmark scores as columns.
"""

import json
import os
import re
import requests
from collections import defaultdict
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

RESULT_SOURCES = {
    "v4.1": "https://raw.githubusercontent.com/mlcommons/inference_results_v4.1/main/summary_results.json",
    "v5.0": "https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/summary_results.json",
    "v5.1": "https://raw.githubusercontent.com/mlcommons/inference_results_v5.1/main/summary_results.json",
}


def download_results(versions: List[str] = None) -> Dict[str, str]:
    """Download summary_results.json for each version. Returns {version: filepath}."""
    os.makedirs(DATA_DIR, exist_ok=True)
    versions = versions or list(RESULT_SOURCES.keys())
    paths = {}

    for ver in versions:
        url = RESULT_SOURCES.get(ver)
        if not url:
            print(f"  [SKIP] Unknown version: {ver}")
            continue

        dest = os.path.join(DATA_DIR, f"summary_results_{ver}.json")
        if os.path.exists(dest):
            print(f"  [CACHED] {ver}: {dest}")
            paths[ver] = dest
            continue

        print(f"  [DOWNLOAD] {ver} from {url}...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(dest, "w") as f:
                f.write(resp.text)
            size_mb = len(resp.text) / (1024 * 1024)
            print(f"    Saved ({size_mb:.1f} MB)")
            paths[ver] = dest
        except Exception as e:
            print(f"    FAILED: {e}")

    return paths


def parse_results(filepath: str, version: str) -> List[Dict[str, Any]]:
    """
    Parse summary_results.json into one record per unique system.

    Raw data: each row = (system, model, scenario, score).
    Output:  each record = one system with all its benchmark scores as keys.

    Grouping key: (Submitter, Platform, version) to keep systems unique
    even if the same org submits multiple platforms.
    """
    with open(filepath) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        print(f"  Unexpected format (expected list, got {type(raw)})")
        return []

    # Group rows by unique system
    systems = defaultdict(lambda: {"scores": {}, "metadata": {}})

    for row in raw:
        if not isinstance(row, dict):
            continue

        score = row.get("Performance_Result")
        if score is None or score == "":
            continue
        try:
            score = float(str(score).replace(",", ""))
        except (ValueError, TypeError):
            continue

        submitter = row.get("Submitter", "unknown")
        platform = row.get("Platform", row.get("System", "unknown"))
        model = row.get("Model", "unknown")
        scenario = row.get("Scenario", "unknown")

        # Unique system key
        sys_key = (submitter, platform, version)

        # Store metadata (first row wins for shared fields)
        if not systems[sys_key]["metadata"]:
            systems[sys_key]["metadata"] = {
                "mlperf_version": version,
                "submitter": submitter,
                "system": row.get("System", ""),
                "platform": platform,
                "accelerator": row.get("Accelerator", ""),
                "accelerator_count": row.get("a#", 1),
                "processor": row.get("Processor", ""),
                "suite": row.get("Suite", ""),
                "category": row.get("Category", ""),
                "availability": row.get("Availability", ""),
                "software": row.get("Software", ""),
                "weight_data_types": row.get("weight_data_types", ""),
                "nodes": row.get("Nodes", 1),
                "os": row.get("operating_system", ""),
            }

        # Store score: metric key = model_scenario
        metric_key = _metric_key(model, scenario)
        systems[sys_key]["scores"][metric_key] = score

        # Also store units for context
        units = row.get("Performance_Units", "")
        if units:
            systems[sys_key]["scores"][f"{metric_key}_units"] = units

    # Flatten into records
    records = []
    for sys_key, data in systems.items():
        rec = dict(data["metadata"])
        # Add numeric scores (skip unit strings)
        for k, v in data["scores"].items():
            if isinstance(v, (int, float)):
                rec[k] = v
        if any(isinstance(v, (int, float)) for v in data["scores"].values()):
            records.append(rec)

    return records


def _metric_key(model: str, scenario: str) -> str:
    """Normalize model+scenario into a clean metric name."""
    m = model.lower().replace("-", "_").replace(".", "_")
    s = scenario.lower().replace(" ", "_")
    return f"{m}_{s}"


def _parse_accel_count(accel_str: str) -> int:
    """Extract count from strings like 'NVIDIA H100 x 8' or '8x H100'."""
    match = re.search(r'x\s*(\d+)', accel_str)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\s*x\s', accel_str)
    if match:
        return int(match.group(1))
    return 1


def fetch_and_parse(versions: List[str] = None) -> List[Dict[str, Any]]:
    """Full pipeline: download + parse all versions."""
    print("Fetching MLPerf Inference results...")
    paths = download_results(versions)

    all_records = []
    for ver, path in paths.items():
        print(f"\nParsing {ver}...")
        records = parse_results(path, ver)
        print(f"  Found {len(records)} unique systems with benchmark scores")
        all_records.extend(records)

    print(f"\nTotal systems across all versions: {len(all_records)}")
    return all_records