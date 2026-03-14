#!/usr/bin/env python3
"""
Query MLflow and generate cross-generation analysis of MLPerf Inference results.

Produces:
  1. Accelerator family comparison (bar chart)
  2. Version-over-version improvement (grouped bars)
  3. Per-accelerator efficiency chart
  4. Markdown summary report
"""

import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKING_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT_NAME = "mlperf-inference-tracker"

# Focus benchmarks for analysis (most widely submitted)
# Keys match the metric names: model_scenario (lowercased, dashes->underscores)
FOCUS_BENCHMARKS = {
    "llama2_70b_99_offline": "Llama2-70B Offline",
    "llama2_70b_99_server": "Llama2-70B Server",
    "llama2_70b_99_interactive": "Llama2-70B Interactive",
    "resnet50_offline": "ResNet50 Offline",
    "resnet50_server": "ResNet50 Server",
    "bert_99_offline": "BERT Offline",
    "stable_diffusion_xl_offline": "SDXL Offline",
    "gptj_99_offline": "GPT-J Offline",
    "retinanet_offline": "RetinaNet Offline",
}


def load_runs():
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("No experiment found. Run run_tracker.py first.")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=50000,
    )
    print(f"Loaded {len(runs)} runs from MLflow")
    return runs


def plot_accel_comparison(df):
    """Bar chart: best throughput per accelerator family for key benchmarks."""
    if df.empty or "tags.accel_family" not in df.columns:
        print("  Skipping accelerator comparison (no accel_family tags)")
        return

    families = df["tags.accel_family"].dropna().unique()
    if len(families) == 0:
        return

    # Pick the benchmark with the most data
    best_bench = None
    best_count = 0
    for metric_col in FOCUS_BENCHMARKS:
        col = f"metrics.{metric_col}"
        if col in df.columns:
            count = df[col].dropna().shape[0]
            if count > best_count:
                best_count = count
                best_bench = metric_col

    if best_bench is None:
        print("  No focus benchmark data found")
        return

    col = f"metrics.{best_bench}"
    label = FOCUS_BENCHMARKS[best_bench]

    # Best score per family
    grouped = (df.dropna(subset=[col, "tags.accel_family"])
               .groupby("tags.accel_family")[col]
               .max()
               .sort_values(ascending=True))

    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(grouped) * 0.6)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(grouped)))
    bars = ax.barh(grouped.index, grouped.values, color=colors, edgecolor="white", height=0.6)

    # Value labels
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_width() + grouped.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=10)

    ax.set_xlabel("Throughput (queries/s or tokens/s)", fontsize=12)
    ax.set_title(f"Best {label} Score by Accelerator Family", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PROJECT_ROOT, "analysis_accel_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_version_progression(df):
    """Grouped bars: how top scores evolve across MLPerf versions."""
    if df.empty or "params.mlperf_version" not in df.columns:
        return

    versions = sorted(df["params.mlperf_version"].dropna().unique())
    if len(versions) < 2:
        print("  Need 2+ versions for progression chart")
        return

    # Collect best score per version for each benchmark
    data = {}
    for metric_key, label in FOCUS_BENCHMARKS.items():
        col = f"metrics.{metric_key}"
        if col not in df.columns:
            continue
        ver_scores = {}
        for ver in versions:
            subset = df[df["params.mlperf_version"] == ver][col].dropna()
            if not subset.empty:
                ver_scores[ver] = subset.max()
        if len(ver_scores) >= 2:
            data[label] = ver_scores

    if not data:
        print("  No multi-version data for progression chart")
        return

    # Normalize to earliest version as 1.0x baseline
    baseline_ver = versions[0]
    norm_data = {}
    raw_baselines = {}
    for bk, ver_scores in data.items():
        baseline = ver_scores.get(baseline_ver)
        if baseline and baseline > 0:
            norm_data[bk] = {v: s / baseline for v, s in ver_scores.items()}
            raw_baselines[bk] = baseline

    if not norm_data:
        print("  Could not normalize — no baseline data")
        return

    data = norm_data

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    n_benchmarks = len(data)
    n_versions = len(versions)
    width = 0.8 / n_versions
    x = np.arange(n_benchmarks)

    colors = plt.cm.Set2(np.linspace(0, 1, n_versions))
    for i, ver in enumerate(versions):
        vals = [data[bk].get(ver, 0) for bk in data.keys()]
        bars = ax.bar(x + (i - n_versions / 2 + 0.5) * width, vals, width,
                      label=ver, color=colors[i], edgecolor="white")
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.1f}x", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys(), rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(f"Relative to {baseline_ver} (1.0x = baseline)", fontsize=12)
    ax.set_title("MLPerf Inference: Generation-over-Generation Improvement", fontsize=14, fontweight="bold")
    ax.legend(title="Version", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PROJECT_ROOT, "analysis_version_progression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_accel_efficiency(df):
    """Scatter: total score vs accelerator count, showing scaling efficiency."""
    if df.empty:
        return

    # Find the best benchmark column available
    for metric_key in ["llama2_70b_99_offline", "resnet50_offline", "bert_99_offline"]:
        col = f"metrics.{metric_key}"
        per_col = f"metrics.{metric_key}_per_accel"
        if col in df.columns and per_col in df.columns:
            break
    else:
        print("  No per-accel data for efficiency chart")
        return

    label = FOCUS_BENCHMARKS.get(metric_key, metric_key)
    subset = df.dropna(subset=[col, "params.accelerator_count"]).copy()
    subset["accel_count"] = pd.to_numeric(subset["params.accelerator_count"], errors="coerce")
    subset = subset.dropna(subset=["accel_count"])
    subset = subset[subset["accel_count"] > 0]

    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if "tags.accel_family" in subset.columns:
        families = subset["tags.accel_family"].fillna("Other").unique()
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(families), 1)))
        for i, fam in enumerate(sorted(families)):
            fam_data = subset[subset["tags.accel_family"] == fam]
            ax.scatter(fam_data["accel_count"], fam_data[col],
                      label=fam, alpha=0.6, s=50, color=cmap[i % len(cmap)])
    else:
        ax.scatter(subset["accel_count"], subset[col], alpha=0.6, s=50)

    ax.set_xlabel("Number of Accelerators", fontsize=12)
    ax.set_ylabel(f"{label} Score", fontsize=12)
    ax.set_title(f"Scaling: {label} vs Accelerator Count", fontsize=14, fontweight="bold")
    if "tags.accel_family" in subset.columns:
        ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PROJECT_ROOT, "analysis_scaling_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def generate_report(df):
    """Write a Markdown summary report."""
    if df.empty:
        return

    n_versions = df["params.mlperf_version"].nunique() if "params.mlperf_version" in df.columns else 0
    n_submitters = df["params.submitter"].nunique() if "params.submitter" in df.columns else 0

    report = f"""# MLPerf Inference — Cross-Generation Analysis Report

## Summary

- **Total submissions tracked**: {len(df)}
- **MLPerf versions**: {n_versions}
- **Unique submitters**: {n_submitters}

## Accelerator Families Represented
"""
    if "tags.accel_family" in df.columns:
        family_counts = df["tags.accel_family"].value_counts()
        for fam, count in family_counts.items():
            report += f"- **{fam}**: {count} submissions\n"

    report += "\n## Best Scores by Benchmark\n"
    for metric_key, label in FOCUS_BENCHMARKS.items():
        col = f"metrics.{metric_key}"
        if col in df.columns:
            best = df[col].max()
            if pd.notna(best):
                best_row = df.loc[df[col].idxmax()]
                submitter = best_row.get("params.submitter", "?")
                accel = best_row.get("params.accelerator_clean",
                                     best_row.get("params.accelerator", "?"))
                report += f"- **{label}**: {best:,.0f} — {submitter} ({accel})\n"

    report += """
## Artifacts

- `analysis_accel_comparison.png` — Best score by accelerator family
- `analysis_version_progression.png` — Score progression across MLPerf rounds
- `analysis_scaling_efficiency.png` — Throughput vs accelerator count
- MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`
"""

    path = os.path.join(PROJECT_ROOT, "EXPERIMENT_REPORT.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved {path}")


def main():
    print("=" * 60)
    print("  MLPerf Inference — Cross-Generation Analysis")
    print("=" * 60)

    df = load_runs()
    if df.empty:
        return

    print("\nGenerating analysis...")
    plot_accel_comparison(df)
    plot_version_progression(df)
    plot_per_accel_efficiency(df)
    generate_report(df)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()