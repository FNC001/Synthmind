#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
WORK="${1:-$WORK_ROOT/09_evaluate_three_metrics}"
mkdir -p "$WORK/package_view/data"
DATA2="${STAGE2_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/stage2_full_cation_family_canonical_v1}"
DATA3="${STAGE3_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/stage3_full_cation_family_v1}"
BASE="$DATA_ROOT/07_BEST_MODELS/03_STAGE2_META_AND_GATE/stage2_factorized200_missgate_s9144/val_candidates.jsonl"
EXPERTS="$DATA_ROOT/07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS"
META="$DATA_ROOT/07_BEST_MODELS/03_STAGE2_META_AND_GATE/stage2_meta_factorized3_100_s9156/val_scores.npz"

"$PYTHON" "$CORE/training/family/train_stage2_validation_miss_gate.py" --input_dir "$DATA2" --matsci_embeddings "$DATA2/matscibert_embeddings_canonical_v1.npz" --matsci_components 64 --base_candidates "$BASE" --expert_source "factorized_a=$EXPERTS/stage2_factorized_h2048_b6_top20_g1_s9140/val_candidates.jsonl" --expert_source "factorized_b=$EXPERTS/stage2_factorized_h1536_b4_top20_g2_s9151/val_candidates.jsonl" --expert_source "factorized_c=$EXPERTS/stage2_factorized_h2048_b6_top20_g1_s9152/val_candidates.jsonl" --reranker_scores_npz "$META" --base_limit 100 --expert_limit 100 --folds 5 --num_boost_round 300 --learning_rate 0.03 --num_leaves 15 --min_data_in_leaf 30 --feature_fraction 0.9 --bagging_fraction 0.9 --miss_weight 2 --num_threads 16 --seed 9161 --output_json "$WORK/stage2_metrics.json" --output_candidates_jsonl "$WORK/stage2_val_candidates.jsonl" --output_model "$WORK/stage2_model.txt" --output_probabilities_npz "$WORK/stage2_probabilities.npz" >/dev/null

EXPECTED_SHA=ad7ddc7c92d78cc33cef1d1216e5ef812fac5f18d3de33caf747435b9e5565ec
ACTUAL_SHA="$(sha256sum "$WORK/stage2_val_candidates.jsonl" | awk '{print $1}')"
test "$ACTUAL_SHA" = "$EXPECTED_SHA" || { echo "Stage2 candidate hash mismatch: $ACTUAL_SHA" >&2; exit 3; }

bash "$WORKFLOW_ROOT/08_TRAIN_STAGE3/run_rebuild_ensemble.sh" "$WORK/stage3_ensemble.npz" >/dev/null
"$PYTHON" "$CORE/training/family/evaluate_stage3_sample_topk.py" --input_dir "$DATA3" --split val --predictions_npz "$WORK/stage3_ensemble.npz" --temperature_bin 100 --time_bin 24 --temperature_tolerance 200 --time_tolerance 48 --output_json "$WORK/stage3_internal_metrics.json" --output_candidates_jsonl "$WORK/stage3_condition_candidates.jsonl" >/dev/null

ln -sfn "$CORE" "$WORK/package_view/code"
ln -sfn "$DATA2" "$WORK/package_view/data/stage2_full_cation_family_canonical_v1"
ln -sfn "$DATA3" "$WORK/package_view/data/stage3_full_cation_family_v1"
"$PYTHON" "$DATA_ROOT/09_ACCURACY_EVALUATION/04_RECOMPUTE_TOOLS/evaluate_final_three_metrics.py" --package-root "$WORK/package_view" --stage2-candidates "$WORK/stage2_val_candidates.jsonl" --condition-candidates "$WORK/stage3_condition_candidates.jsonl" --expected-json "$DATA_ROOT/09_ACCURACY_EVALUATION/03_THREE_METRICS/final_three_metrics.json" --output-json "$WORK/final_three_metrics_recomputed.json" >/dev/null
announce "PASS: three metrics reproduced exactly; report=$WORK/final_three_metrics_recomputed.json"
