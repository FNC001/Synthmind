#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"

DATA="${STAGE2_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/stage2_full_cation_family_canonical_v1}"
EMBED="$DATA/matscibert_embeddings_canonical_v1.npz"
IMPORTED_BASE="${STAGE2_IMPORTED_BASE:-$DATA_ROOT/07_BEST_MODELS/01_STAGE2_IMPORTED_UPSTREAM/stage2_validation_meta_missonly_stability_s8720/val_candidates.jsonl}"
OUT="${1:-$WORK_ROOT/06_train_stage2/full_selected_chain}"
THREADS="${NUM_THREADS:-16}"
mkdir -p "$OUT"

train_expert() {
  local run_dir="$1" hidden="$2" blocks="$3" dropout="$4" batch="$5" epochs="$6" patience="$7" lr="$8" gamma="$9" seed="${10}"
  "$PYTHON" "$CORE/training/family/train_stage2_factorized_generator.py" \
    --input_dir "$DATA" --run_dir "$run_dir" --hidden "$hidden" --blocks "$blocks" \
    --dropout "$dropout" --batch_size "$batch" --epochs "$epochs" --patience "$patience" \
    --lr "$lr" --weight_decay 0.0001 --length_loss_weight 1 --gamma_neg "$gamma" --gamma_pos 0 \
    --top_labels 20 --candidate_limit 500 --max_enumerated_length 4 \
    --length_score_weights 0.25,0.5,1,2,4 --seed "$seed" --device "${DEVICE:-cuda}"
}

A="$OUT/01_expert_A_s9140"
B="$OUT/02_expert_B_s9151"
C="$OUT/03_expert_C_s9152"
train_expert "$A" 2048 6 0.12 128 160 20 0.00015 1 9140
train_expert "$B" 1536 4 0.10 256 140 18 0.00025 2 9151
train_expert "$C" 2048 6 0.10 256 160 20 0.00010 1 9152

META1="$OUT/04_meta_s9141"; mkdir -p "$META1"
"$PYTHON" "$CORE/training/family/train_stage2_validation_meta_lambdarank.py" \
  --input_dir "$DATA" --matsci_embeddings "$EMBED" --matsci_components 64 \
  --base_candidates "$IMPORTED_BASE" --expert_source "factorized=$A/val_candidates.jsonl" \
  --base_limit 100 --expert_limit 200 --folds 5 --num_boost_round 120 --learning_rate 0.04 \
  --num_leaves 15 --min_data_in_leaf 80 --feature_fraction 0.9 --bagging_fraction 0.9 \
  --lambda_l1 0.01 --lambda_l2 0.2 --group_balance_power 0.25 --family_balance_power 0 \
  --base_miss_weight 1 --exclude_base_rank_features --train_accessible_base_misses_only \
  --num_threads "$THREADS" --seed 9141 --output_json "$META1/metrics.json" \
  --output_candidates_jsonl "$META1/val_candidates.jsonl" --output_model "$META1/model.txt" \
  --output_scores_npz "$META1/val_scores.npz"

GATE1="$OUT/05_missgate_s9144"; mkdir -p "$GATE1"
"$PYTHON" "$CORE/training/family/train_stage2_validation_miss_gate.py" \
  --input_dir "$DATA" --matsci_embeddings "$EMBED" --matsci_components 64 \
  --base_candidates "$META1/val_candidates.jsonl" --expert_source "factorized=$A/val_candidates.jsonl" \
  --reranker_scores_npz "$META1/val_scores.npz" --base_limit 100 --expert_limit 200 --folds 5 \
  --num_boost_round 180 --learning_rate 0.03 --num_leaves 7 --min_data_in_leaf 40 \
  --feature_fraction 0.9 --bagging_fraction 0.9 --miss_weight 1 --num_threads "$THREADS" --seed 9144 \
  --output_json "$GATE1/metrics.json" --output_candidates_jsonl "$GATE1/val_candidates.jsonl" \
  --output_model "$GATE1/model.txt" --output_probabilities_npz "$GATE1/val_gate_probabilities.npz"

META2="$OUT/06_meta_s9156"; mkdir -p "$META2"
"$PYTHON" "$CORE/training/family/train_stage2_validation_meta_lambdarank.py" \
  --input_dir "$DATA" --matsci_embeddings "$EMBED" --matsci_components 64 \
  --base_candidates "$GATE1/val_candidates.jsonl" \
  --expert_source "factorized_a=$A/val_candidates.jsonl" \
  --expert_source "factorized_b=$B/val_candidates.jsonl" \
  --expert_source "factorized_c=$C/val_candidates.jsonl" \
  --base_limit 100 --expert_limit 100 --folds 5 --num_boost_round 450 --learning_rate 0.03 \
  --num_leaves 63 --min_data_in_leaf 40 --feature_fraction 0.9 --bagging_fraction 0.9 \
  --lambda_l1 0.01 --lambda_l2 0.2 --group_balance_power 1 --family_balance_power 0.25 \
  --base_miss_weight 1 --exclude_base_rank_features --train_accessible_base_misses_only \
  --num_threads "$THREADS" --seed 9156 --output_json "$META2/metrics.json" \
  --output_candidates_jsonl "$META2/val_candidates.jsonl" --output_model "$META2/model.txt" \
  --output_scores_npz "$META2/val_scores.npz"

FINAL="$OUT/07_final_missgate_s9161"; mkdir -p "$FINAL"
"$PYTHON" "$CORE/training/family/train_stage2_validation_miss_gate.py" \
  --input_dir "$DATA" --matsci_embeddings "$EMBED" --matsci_components 64 \
  --base_candidates "$GATE1/val_candidates.jsonl" \
  --expert_source "factorized_a=$A/val_candidates.jsonl" \
  --expert_source "factorized_b=$B/val_candidates.jsonl" \
  --expert_source "factorized_c=$C/val_candidates.jsonl" \
  --reranker_scores_npz "$META2/val_scores.npz" --base_limit 100 --expert_limit 100 --folds 5 \
  --num_boost_round 300 --learning_rate 0.03 --num_leaves 15 --min_data_in_leaf 30 \
  --feature_fraction 0.9 --bagging_fraction 0.9 --miss_weight 2 --num_threads "$THREADS" --seed 9161 \
  --output_json "$FINAL/metrics.json" --output_candidates_jsonl "$FINAL/val_candidates.jsonl" \
  --output_model "$FINAL/model.txt" --output_probabilities_npz "$FINAL/val_gate_probabilities.npz"

announce "Stage2 selected chain trained under $OUT"
