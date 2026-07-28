#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
DATA="${STAGE2_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/stage2_full_cation_family_canonical_v1}"
if test "${MODE:-validate}" = train_full; then
  bash "$STEP_DIR/train_full_selected_chain.sh" "${OUTPUT_DIR:-$WORK_ROOT/06_train_stage2/full_selected_chain}"
elif test "${MODE:-validate}" = train_experts; then
  OUT="$WORK_ROOT/06_train_stage2"; mkdir -p "$OUT"
  "$PYTHON" "$CORE/training/family/train_stage2_factorized_generator.py" --input_dir "$DATA" --run_dir "$OUT/expert_A_s9140" --hidden 2048 --blocks 6 --dropout 0.12 --batch_size 128 --epochs 160 --patience 20 --lr 0.00015 --gamma_neg 1 --top_labels 20 --candidate_limit 500 --length_score_weights 0.25,0.5,1,2,4 --seed 9140 --device cuda
  "$PYTHON" "$CORE/training/family/train_stage2_factorized_generator.py" --input_dir "$DATA" --run_dir "$OUT/expert_B_s9151" --hidden 1536 --blocks 4 --dropout 0.10 --batch_size 256 --epochs 140 --patience 18 --lr 0.00025 --gamma_neg 2 --top_labels 20 --candidate_limit 500 --length_score_weights 0.25,0.5,1,2,4 --seed 9151 --device cuda
  "$PYTHON" "$CORE/training/family/train_stage2_factorized_generator.py" --input_dir "$DATA" --run_dir "$OUT/expert_C_s9152" --hidden 2048 --blocks 6 --dropout 0.10 --batch_size 256 --epochs 160 --patience 20 --lr 0.0001 --gamma_neg 1 --top_labels 20 --candidate_limit 500 --length_score_weights 0.25,0.5,1,2,4 --seed 9152 --device cuda
else
  require_file "$DATA_ROOT/07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/stage2_factorized_h2048_b6_top20_g1_s9140/best_factorized_generator.pt"
  require_file "$DATA_ROOT/07_BEST_MODELS/03_STAGE2_META_AND_GATE/stage2_factorized3_missgate_w2_s9161/model.txt"
  announce "Frozen Stage2 experts and final gate are present"
fi
