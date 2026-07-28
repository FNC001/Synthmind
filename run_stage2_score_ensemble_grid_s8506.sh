#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
DATA="$ROOT/data/stage2_full_cation_family_canonical_v1"
RUN="$ROOT/runs/canonical_v1/stage2_matscibert_qwen3_ensemble_s8506"
BASE="$ROOT/runs/canonical_v1/stage2_s8502_top10_s8494_depth_s8504/val_candidates.jsonl"
SCORE_A="$ROOT/runs/canonical_v1/stage2_matscibert_global_union13_s8502/val_scores.npz"
SCORE_B="$ROOT/runs/canonical_v1/stage2_qwen3_1p7b_exact_s8505/val_scores.npz"

mkdir -p "$RUN"
cd "$ROOT/code"

python3 training/family/evaluate_stage2_score_ensemble_grid.py \
  --input_dir "$DATA" \
  --base_val_candidates "$BASE" \
  --score_npz_a "$SCORE_A" \
  --score_npz_b "$SCORE_B" \
  --mix_grid "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1" \
  --candidate_limit 800 \
  --output_json "$RUN/metrics.json" \
  --output_candidates_jsonl "$RUN/val_candidates.jsonl" \
  2>&1 | tee "$RUN/run.log"
