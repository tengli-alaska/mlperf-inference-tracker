"""Utility helpers for accelerator parsing and data normalization."""

# Canonical accelerator family mapping
ACCEL_FAMILIES = {
    "H100": ["h100"],
    "H200": ["h200"],
    "Blackwell": ["b200", "b300", "gb200", "gb300"],
    "A100": ["a100"],
    "MI300X": ["mi300x"],
    "MI325X": ["mi325x"],
    "Gaudi2": ["gaudi2"],
    "Gaudi3": ["gaudi3"],
    "L40S": ["l40s"],
    "L4": ["l4"],
}


def classify_accelerator(accel_str: str) -> str:
    """Map an accelerator string to a canonical family name."""
    lower = accel_str.lower()
    for family, patterns in ACCEL_FAMILIES.items():
        if any(p in lower for p in patterns):
            return family
    return "Other"


def normalize_metric_name(raw: str) -> str:
    """Clean up metric names for display."""
    return (raw
            .replace("llama2_70b_99", "Llama2-70B")
            .replace("llama2_70b_99_9", "Llama2-70B-99.9")
            .replace("resnet50", "ResNet50")
            .replace("bert_99", "BERT")
            .replace("gptj_99", "GPT-J")
            .replace("stable_diffusion_xl", "SDXL")
            .replace("dlrm_v2_99", "DLRM-v2")
            .replace("retinanet", "RetinaNet")
            .replace("_server", " (Server)")
            .replace("_offline", " (Offline)")
            .replace("_per_accel", "/accel"))