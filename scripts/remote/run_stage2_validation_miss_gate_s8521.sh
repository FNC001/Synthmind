#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
DATA="$ROOT/data/stage2_full_cation_family_canonical_v1"
RUN="$ROOT/runs/canonical_v1/stage2_validation_miss_gate_s8521"

mkdir -p "$RUN"
cd "$ROOT/code"
export PYTHONPATH=.

/root/miniconda3/bin/python3 training/family/train_stage2_validation_miss_gate.py \
  --input_dir "$DATA" \
  --matsci_embeddings "$DATA/matscibert_embeddings_canonical_v1.npz" \
  --matsci_components 64 \
  --base_candidates "$ROOT/runs/canonical_v1/stage2_s8502_top10_s8494_depth_s8504/val_candidates.jsonl" \
  --expert_manifest "$ROOT/runs/canonical_v1/stage2_all13_template_top10_oracle_s8513/metrics.json" \
  --reranker_scores_npz "$ROOT/runs/canonical_v1/stage2_validation_meta_missonly_s8520/val_scores.npz" \
  --base_limit 100 \
  --expert_limit 10 \
  --folds 5 \
  --num_boost_round 300 \
  --learning_rate 0.03 \
  --num_leaves 15 \
  --min_data_in_leaf 20 \
  --feature_fraction 0.90 \
  --bagging_fraction 0.90 \
  --miss_weight 1.0 \
  --num_threads 64 \
  --seed 8521 \
  --output_json "$RUN/metrics.json" \
  --output_candidates_jsonl "$RUN/val_candidates.jsonl" \
  --output_model "$RUN/model.txt" \
  --output_probabilities_npz "$RUN/val_gate_probabilities.npz" \
  2>&1 | tee "$RUN/run.log"
