#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated family-routed V1 artifact registry.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code_revision", default="working-tree")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    artifact_paths = [
        root / "data/stage2_full_cation_family_v1/split_manifest.json",
        root / "data/stage2_full_cation_family_v1/summary.json",
        root / "data/stage3_full_cation_family_v1/split_manifest.json",
        root / "data/stage3_full_cation_family_v1/schema.json",
        root / "runs/stage2_film_wide768_v1/best_model.pt",
        root / "runs/stage2_film_wide768_v1/metrics.json",
        root / "runs/stage2_film_wide768_frozen_rerank_v1/best_reranker.pt",
        root / "runs/stage2_film_wide768_frozen_rerank_v1/metrics.json",
        root / "runs/stage2_no_family_wide768_v1/best_model.pt",
        root / "runs/stage2_no_family_wide768_v1/metrics.json",
        root / "runs/stage2_family_knn_v1/family_knn_metrics.json",
        root / "runs/stage3_family_lgbm_v1/stage3_family_lgbm.joblib",
        root / "runs/stage3_family_lgbm_v1/metrics.json",
        root / "runs/stage3_no_family_ablation_v1/stage3_family_lgbm.joblib",
        root / "runs/stage3_no_family_ablation_v1/metrics.json",
        root / "runs/stage3_gpu_multitask_family_v1/best_model.pt",
        root / "runs/stage3_gpu_multitask_family_v1/metrics.json",
        root / "runs/stage3_gpu_multitask_no_family_v1/best_model.pt",
        root / "runs/stage3_gpu_multitask_no_family_v1/metrics.json",
    ]
    code_paths = sorted((root / "code/synthmind/chemistry").glob("*")) + sorted(
        (root / "code/training/family").glob("*.py")
    )
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required registry artifacts missing: {missing}")
    registry = {
        "registry_version": "family_routed_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision,
        "family_schema_version": "target_cation_family_v1",
        "isolated_root": str(root),
        "production_default_changed": False,
        "artifacts": [file_record(path) for path in artifact_paths],
        "code_files": [file_record(path) for path in code_paths if path.is_file()],
        "routing": {
            "primary_rule": "periodic groups of target cation/metalloid backbone; anions and stoichiometry do not split families",
            "required_equivalence": ["LiCl", "NaCl", "NaBr", "Li2O", "Na2S"],
            "required_family": "G01",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "n_artifacts": len(registry["artifacts"]), "n_code_files": len(registry["code_files"])}, indent=2))


if __name__ == "__main__":
    main()
