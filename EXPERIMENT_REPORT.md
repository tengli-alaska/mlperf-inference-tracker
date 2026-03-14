# MLPerf Inference — Cross-Generation Analysis Report

## Summary

- **Total submissions tracked**: 320
- **MLPerf versions**: 2
- **Unique submitters**: 37

## Accelerator Families Represented
- **Other**: 165 submissions
- **H200**: 55 submissions
- **H100**: 25 submissions
- **B200**: 24 submissions
- **MI325X**: 19 submissions
- **MI300X**: 11 submissions
- **GB200**: 8 submissions
- **L40S**: 8 submissions
- **L4**: 3 submissions
- **GB300**: 2 submissions

## Best Scores by Benchmark
- **Llama2-70B Offline**: 648,248 — AMD_MangoBoost (AMD Instinct MI355X 288GB HBM3e)
- **Llama2-70B Server**: 153,076 — MangoBoost (AMD Instinct MI300X 192GB HBM3 (x32), AMD Instinct MI325X 256GB HBM3e (x16))
- **Llama2-70B Interactive**: 62,851 — GigaComputing (NVIDIA B200-SXM-180GB)
- **BERT Offline**: 8,266 — GATEOverflow (NVIDIA GeForce RTX 4090)
- **SDXL Offline**: 33 — Lambda (NVIDIA B200-SXM-180GB)
- **GPT-J Offline**: 21,626 — Lenovo (NVIDIA H200-SXM-141GB)
- **RetinaNet Offline**: 15,200 — Supermicro (NVIDIA H200-SXM-141GB)

## Artifacts

- `analysis_accel_comparison.png` — Best score by accelerator family
- `analysis_version_progression.png` — Score progression across MLPerf rounds
- `analysis_scaling_efficiency.png` — Throughput vs accelerator count
- MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`
