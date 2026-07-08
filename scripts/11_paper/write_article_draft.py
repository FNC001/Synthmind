#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_DIR = SCRIPT_DIR.parent / "08_auto_improve"
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from metrics_registry import build_registry, pct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Write SynPred paper Results/Methods/Figure caption drafts.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/autorun/24h_optimization_20260613/09_article_draft")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    reg = build_registry(root, out.parent, include_experiments=True).records["baselines"]
    s2 = reg["stage2_v5_all_test"]["metrics"]
    s2c = reg["stage2_core_calibrated_test"]["metrics"]
    s3m = reg["stage3_v3_missing_aware_test"]["metrics"]
    r3m = reg["stage35_v3_final_missing_aware_test"]["metrics"]
    r3s = reg["stage35_v3_final_strict_comparable_test"]["metrics"]

    results = f"""# PAPER RESULTS DRAFT

## Title draft

SynPred: Structure-Guided Precursor and Synthesis-Route Prediction for Inorganic Materials

## Abstract draft

Predicting inorganic synthesis routes remains difficult because precursor choices, reaction conditions, and route-level ranking are tightly coupled. SynPred addresses this task from a structure or composition input using a staged pipeline: Stage2 generates chemistry-checked precursor sets, Stage3 proposes distributional synthesis conditions, and Stage35 ranks complete precursor-condition routes.

Across the current method-stratified benchmark, Stage2 v5 reaches {pct(s2.get('top500_exact'))} all-method top500 exact precursor recall and {pct(s2c.get('top500_exact'))} core-method top500 exact recall. Stage3 v3 reaches {pct(s3m.get('top10_relaxed_condition'))} missing-aware top10 relaxed condition success. The current Stage35 v3 final blend achieves {pct(r3m.get('top1_relaxed_route'))} missing-aware and {pct(r3s.get('top1_relaxed_route'))} strict-comparable top1 relaxed route success.

These results show that compositional precursor candidate generation and distributional condition modeling can recover substantial route coverage, while OOV precursors, missing atmosphere/solvent labels, and top1 route ranking remain the main bottlenecks.

## Results

### Dataset and reaction-method distribution

The benchmark uses reaction-method stratified train/validation/test splits and reports both all-method and core-method slices.

### Stage2 precursor prediction performance

Stage2 v5 improves the chemistry-checked precursor candidate pool and reaches {pct(s2.get('top1_exact'))} top1, {pct(s2.get('top10_exact'))} top10, and {pct(s2.get('top500_exact'))} top500 exact recall on the all-method test split.

### Stage3 condition prediction performance

Stage3 v3 is retained as the best-current condition model because v4 alignment did not improve the primary top1 relaxed condition metrics under both protocols.

### Complete synthesis-route ranking

Stage35 v3 final blend remains the default because it preserves the best top1 route performance; v4 alignment is retained as coverage evidence rather than as the default ranking mode.

### Error decomposition and bottlenecks

The auto-improvement diagnostics decompose failures into Stage2 precursor, Stage3 condition, Stage35 ranking, and distribution mismatch buckets.
"""

    methods = """# PAPER METHODS DRAFT

## Dataset construction and reaction-method stratified split

The dataset is built from curated synthesis-structure records and split by reaction method to preserve held-out evaluation structure.

## Precursor canonicalization and ontology

Precursor labels are canonicalized through a chemistry ontology, alias patches, and conservative parsing rules.

## Stage2 precursor candidate generation

Stage2 combines neural, retrieval, template, and repair-generated precursor candidates, followed by validation-selected score calibration.

## Chemistry-constrained precursor checking

Candidates are filtered and annotated for elemental coverage, open-generated status, and repair provenance.

## Stage3 distributional condition modeling

Stage3 predicts temperature, time, atmosphere, and solvent candidates using distributional and retrieval/template-informed features.

## Missing-aware atmosphere and solvent modeling

The evaluation reports both missing-aware and strict-comparable protocols to separate label missingness from strict field recovery.

## Stage35 complete-route ranking

Stage35 ranks full precursor-condition pairs using route scores and a final blend reranker selected on validation data.

## Evaluation protocols

Precursor exact recall, condition strict/relaxed hits, and route strict/relaxed hits are evaluated at multiple K values.

## Auto-improvement framework

The autorun framework reproduces baselines, diagnoses bottlenecks, runs gated experiments, and only promotes models that clear validation and test thresholds.
"""

    captions = """# PAPER FIGURE CAPTIONS

**Figure 1.** SynPred pipeline from structure/composition input through Stage2 precursor generation, chemistry checking, Stage3 condition generation, and Stage35 route ranking.

**Figure 2.** Dataset distribution across train/validation/test splits and reaction methods.

**Figure 3.** Stage2 topK exact precursor recall curves for all-method and core-method settings.

**Figure 4.** Stage3 condition prediction metrics under missing-aware and strict-comparable protocols.

**Figure 5.** Stage35 complete-route success for current final and alignment candidates.

**Figure 6.** Error decomposition into precursor, condition, route-ranking, and mismatch buckets.

**Figure 7.** Core-method and all-method performance comparison across reaction methods.
"""

    (out / "PAPER_RESULTS_DRAFT.md").write_text(results, encoding="utf-8")
    (out / "PAPER_METHODS_DRAFT.md").write_text(methods, encoding="utf-8")
    (out / "PAPER_FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    print(json.dumps({"article_draft_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

