#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


DIR_KEYS = {"stage2_npz_dir", "route_out_dir"}


def _is_valid_output(key: str, path: Path) -> bool:
    try:
        if key in DIR_KEYS:
            return path.is_dir() and any(path.iterdir())
        if path.suffix in (".csv", ".jsonl", ".tsv"):
            if path.stat().st_size == 0:
                return False
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline()
                second = f.readline()
            return bool(first.strip() and second.strip())
        return path.stat().st_size > 0
    except OSError:
        return False


class PipelineRunner:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.project_root = Path(cfg["project_root"])
        self.device = cfg.get("device", "cpu")
        self.infer_name = cfg.get("infer_name", "demo_poscar_test")
        self.paths = cfg.get("paths", {})
        self.work_dir = Path(self.paths["work_dir"])
        self.out_dir = Path(self.paths["out_dir"])
        self.route_scripts_dir = self.project_root
        self.outputs: Dict[str, str] = {}
        self.degraded_steps: List[Dict[str, str]] = []
        self.step_timings: List[Dict[str, float]] = []
        self._step_t0: Optional[float] = None
        self._step_name: Optional[str] = None

    def log(self, msg: str):
        print(msg, flush=True)

    def begin_step(self, name: str):
        self._step_name = name
        self._step_t0 = time.time()

    def end_step(self):
        if self._step_t0 is not None:
            elapsed = time.time() - self._step_t0
            self.step_timings.append({"step": self._step_name, "seconds": round(elapsed, 2)})
            self.log(f"[TIME] {self._step_name}: {elapsed:.1f}s")
            self._step_t0 = None
            self._step_name = None

    def run(self, cmd: List[str], required: bool = True):
        self.log("[RUN] " + " ".join(map(str, cmd)))
        try:
            subprocess.run(list(map(str, cmd)), check=True)
        except subprocess.CalledProcessError:
            if required:
                raise
            self.log("[WARN] optional command failed")

    def require_file(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
        self.log(f"[OK] {p}")

    def require_dir(self, path: str | Path):
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Missing required directory: {p}")
        self.log(f"[OK] {p}")

    def step_enabled(self, name: str) -> bool:
        return bool(self.cfg.get("steps", {}).get(name, False))

    def record_degradation(self, step_name: str, reason: str):
        self.degraded_steps.append({"step": step_name, "reason": reason})
        self.log(f"[DEGRADED] {step_name}: {reason}")

    def ensure_dirs(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def restore_existing_outputs(self):
        """
        Restore standard output paths from previous runs.
        This makes --start_from usable.
        Files are validated: CSVs must have header + at least 1 data row;
        directories must be non-empty.
        """
        candidates = {
            "infer_jsonl": self.work_dir / "split/infer.jsonl",
            "infer_structdesc_csv": self.work_dir / "infer_structdesc.csv",
            "chgnet_embed_csv": self.work_dir / "chgnet/graph_embed/infer_graph_embed.csv",
            "final_graph_embed_csv": self.work_dir / "graph_embed/infer_graph_embed.csv",
            "stage2_hybrid_csv": self.work_dir / "stage2_hybrid_csv/stage2_train_hybrid.csv",
            "stage2_npz_dir": self.work_dir / "stage2_hybrid",
            "stage2_sample_csv": self.work_dir / "stage2_gflownet_candidates/test_samples.csv",
            "stage2_sample_csv_composition_biased": self.work_dir / "stage2_gflownet_candidates_composition_decoding/test_samples.csv",
            "stage2_constrained_sample_csv": self.work_dir / "stage2_gflownet_candidates/test_samples_composition_constrained.csv",
            "stage2_unique_csv": self.work_dir / "stage2_summary/unique_sets_ranked.csv",
            "stage2_fallback_csv": self.work_dir / "stage2_summary/unique_sets_ranked_with_fallback.csv",
            "stage2_retrieval_csv": self.work_dir / "stage2_summary/retrieval_npz_candidates.csv",
            "stage2_baseline_csv": self.work_dir / "stage2_summary/extratrees_baseline_candidates.csv",
            "stage2_merged_csv": self.work_dir / "stage2_summary/unique_sets_ranked_with_fallback_retrieval_baseline.csv",
            "stage2_final_csv": self.work_dir / "stage2_summary/unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv",
            "stage3_hybrid_csv": self.work_dir / "stage3_hybrid/stage3_train_hybrid.csv",
            "stage3_conditioned_x": self.work_dir / "stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv",
            "flow_flat_csv": self.out_dir / "stage3_condition_predictions_flow_fallback_retrieval_baseline_element_reranked/test_candidates_flat.csv",
            "route_out_dir": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked",
            "route_csv": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_readable.csv",
            "route_md": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_readable.md",
            "display_csv": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_display_filtered.csv",
            "display_md": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_display_filtered.md",
            "stage35_v21_csv": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_stage35_v21_hybrid_reranked.csv",
            "stage35_v21_md": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_stage35_v21_hybrid_reranked.md",
            "best_per_precursor_csv": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_stage35_v21_best_per_precursor.csv",
            "best_per_precursor_md": self.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked/synthesis_routes_stage35_v21_best_per_precursor.md",
        }

        restored = {}
        skipped = {}
        for key, path in candidates.items():
            p = Path(path)
            if not p.exists():
                continue
            if _is_valid_output(key, p):
                self.outputs[key] = str(p)
                restored[key] = str(p)
            else:
                skipped[key] = str(p)

        if restored:
            self.log("[RESTORE] existing outputs:")
            for k, v in restored.items():
                self.log(f"  {k}: {v}")

        if skipped:
            self.log("[WARN] skipped invalid/empty outputs:")
            for k, v in skipped.items():
                self.log(f"  {k}: {v}")

    def save_manifest(self):
        p = self.out_dir / "pipeline_v3_manifest.json"
        obj = {
            "pipeline_name": self.cfg.get("pipeline_name"),
            "infer_name": self.infer_name,
            "config_path": self.cfg.get("_config_path"),
            "work_dir": str(self.work_dir),
            "out_dir": str(self.out_dir),
            "outputs": self.outputs,
            "degraded_steps": self.degraded_steps,
            "step_timings": self.step_timings,
        }
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"[SAVE] {p}")
