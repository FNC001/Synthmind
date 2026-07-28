#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
DATA="${STAGE3_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/stage3_full_cation_family_v1}"
if test "${MODE:-validate}" = train_all; then
  OUT="$WORK_ROOT/08_train_stage3"; mkdir -p "$OUT"
  "$PYTHON" "$CORE/training/family/train_stage3_conditional_flow.py" --input_dir "$DATA" --run_dir "$OUT/NF_s8060" --hidden 1024 --precursor_hidden 512 --context_blocks 4 --flow_layers 12 --coupling_hidden 512 --dropout 0.08 --partial_weight 0.2 --batch_size 256 --epochs 240 --patience 30 --lr 0.0002 --samples 256 --base_scale 1 --categorical_temperature 1 --seed 8060 --device cuda
  "$PYTHON" "$CORE/training/family/train_stage3_hybrid_cvae.py" --input_dir "$DATA" --run_dir "$OUT/CVAE_s8040" --hidden 1024 --precursor_hidden 512 --latent 128 --blocks 4 --dropout 0.12 --batch_size 256 --epochs 200 --patience 25 --lr 0.0003 --kl_beta 0.02 --kl_warmup_epochs 30 --samples 256 --seed 8040 --device cuda
  "$PYTHON" "$CORE/training/family/train_stage3_conditional_diffusion.py" --input_dir "$DATA" --run_dir "$OUT/Diffusion_s8320" --hidden 1536 --precursor_hidden 768 --blocks 8 --time_dim 256 --dropout 0.08 --timesteps 1000 --sampling_steps 64 --ddim_eta 0.8 --categorical_temperature 1 --categorical_weight 1 --batch_size 256 --epochs 300 --patience 35 --lr 0.0002 --samples 256 --seed 8320 --device cuda
  NF_SAMPLES="$OUT/NF_s8060/val_samples.npz" \
  CVAE_SAMPLES="$OUT/CVAE_s8040/val_samples.npz" \
  DIFFUSION_SAMPLES="$OUT/Diffusion_s8320/val_samples.npz" \
    bash "$STEP_DIR/run_rebuild_ensemble.sh" "$OUT/stage3_ensemble.npz"
  "$PYTHON" "$CORE/training/family/evaluate_stage3_sample_topk.py" \
    --input_dir "$DATA" --split val --predictions_npz "$OUT/stage3_ensemble.npz" \
    --temperature_bin 100 --time_bin 24 --temperature_tolerance 200 --time_tolerance 48 \
    --output_json "$OUT/condition_metrics.json" --output_candidates_jsonl "$OUT/condition_candidates.jsonl"
  announce "Trained Stage3 models and generated ensemble/candidates: $OUT"
else
  require_file "$DATA_ROOT/07_BEST_MODELS/04_STAGE3_NF/stage3_conditional_flow_h1024_s8060/best_model.pt"
  require_file "$DATA_ROOT/07_BEST_MODELS/05_STAGE3_CVAE/stage3_hybrid_cvae_h1024_s8040/best_model.pt"
  require_file "$DATA_ROOT/07_BEST_MODELS/06_STAGE3_DIFFUSION/stage3_conditional_diffusion_h1536_s8320/best_model.pt"
  require_file "$DATA_ROOT/08_GENERATED_OUTPUTS/04_STAGE3_ENSEMBLE_SAMPLES/nf_cvae_diffusion_best.npz"
  announce "Frozen Stage3 models and ensemble are present"
fi
