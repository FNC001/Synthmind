"""External artifact-layout checks for Synthmind V1.0."""

from __future__ import annotations

from pathlib import Path
from typing import Any


COMMON_PATHS = (
    "00_OVERVIEW_AND_MANIFEST",
    "02_RAW_DATA",
    "03_CLEANED_AND_MERGED_DATA",
    "04_SPLITS",
    "05_FEATURES_AND_EMBEDDINGS",
    "06_TRAIN_READY_DATA",
    "07_BEST_MODELS",
    "08_GENERATED_OUTPUTS",
    "09_ACCURACY_EVALUATION",
    "11_TESTS_AND_AUDITS",
    "06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/"
    "stage2_full_cation_family_canonical_v1/train.npz",
    "06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/"
    "stage3_full_cation_family_v1/train.npz",
)

MODEL_PATHS = (
    "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/"
    "stage2_factorized_h2048_b6_top20_g1_s9140/best_factorized_generator.pt",
    "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/"
    "stage2_factorized_h1536_b4_top20_g2_s9151/best_factorized_generator.pt",
    "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/"
    "stage2_factorized_h2048_b6_top20_g1_s9152/best_factorized_generator.pt",
    "07_BEST_MODELS/04_STAGE3_NF/"
    "stage3_conditional_flow_h1024_s8060/best_model.pt",
    "07_BEST_MODELS/05_STAGE3_CVAE/"
    "stage3_hybrid_cvae_h1024_s8040/best_model.pt",
    "07_BEST_MODELS/06_STAGE3_DIFFUSION/"
    "stage3_conditional_diffusion_h1536_s8320/best_model.pt",
)

METRIC_PATHS = (
    "09_ACCURACY_EVALUATION/03_THREE_METRICS/final_three_metrics.json",
    "09_ACCURACY_EVALUATION/04_RECOMPUTE_TOOLS/"
    "evaluate_final_three_metrics.py",
)


def required_paths(profile: str) -> tuple[str, ...]:
    paths = list(COMMON_PATHS)
    if profile in {"validate-full", "validate-fast"}:
        paths.extend(MODEL_PATHS)
    if profile == "validate-full":
        paths.extend(METRIC_PATHS)
    return tuple(dict.fromkeys(paths))


def check_data_root(root: Path | str, profile: str = "validate-full") -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    checks = []
    for relative in required_paths(profile):
        path = resolved / relative
        checks.append({"path": relative, "exists": path.exists()})
    present = sum(int(item["exists"]) for item in checks)
    return {
        "data_root": str(resolved),
        "profile": profile,
        "required": len(checks),
        "present": present,
        "passed": present == len(checks),
        "checks": checks,
    }
