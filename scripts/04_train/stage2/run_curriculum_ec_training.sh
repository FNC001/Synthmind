#!/bin/bash
# Train Stage2 GFlowNet with curriculum learning + element-constrained decoding
# Phase 1: Pretrain on relaxed data (5914 samples)
# Phase 2: Finetune on gold data (3265 samples)
# Phase 3: Element-constrained inference

set -e

PROJECT_ROOT="/Users/wyc/SynPred"
SCRIPT_DIR="$PROJECT_ROOT/scripts/04_train/stage2"
RUN_DIR="$PROJECT_ROOT/runs/stage2/gflownet_curriculum_ec_v1"

echo "============================================"
echo "Stage2 GFlowNet Curriculum + EC Training"
echo "============================================"

# Phase 1: Pretrain on relaxed data
echo ""
echo "[Phase 1] Pretraining on relaxed data (5914 samples)..."
python "$SCRIPT_DIR/train_gflownet_rerank.py" \
    --input_dir "$PROJECT_ROOT/data/interim/generative/stage2_gflownet_dataset/hybrid/relaxed_only" \
    --run_dir "$RUN_DIR/phase1_relaxed" \
    --hidden_dim 256 \
    --x_mlp_hidden_dims "512,256" \
    --dropout 0.1 \
    --batch_size 128 \
    --epochs 60 \
    --patience 12 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --metric_name "micro_f1" \
    --warmup_epochs 5 \
    --rl_weight 0.1 \
    --sample_temperature 1.0 \
    --exact_bonus 4.0 \
    --max_traj_len_override 7 \
    --length_penalty 0.05

echo ""
echo "[Phase 2] Finetuning on gold data (3265 samples)..."
python "$SCRIPT_DIR/train_gflownet_rerank.py" \
    --input_dir "$PROJECT_ROOT/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only" \
    --run_dir "$RUN_DIR/phase2_gold" \
    --pretrained_model "$RUN_DIR/phase1_relaxed/best_model.pt" \
    --hidden_dim 256 \
    --x_mlp_hidden_dims "512,256" \
    --dropout 0.1 \
    --batch_size 128 \
    --epochs 80 \
    --patience 12 \
    --lr 3e-4 \
    --weight_decay 1e-5 \
    --metric_name "micro_f1" \
    --warmup_epochs 3 \
    --rl_weight 0.15 \
    --sample_temperature 1.0 \
    --exact_bonus 4.0 \
    --length_penalty 0.05 \
    --rerank_enabled \
    --rerank_num_samples_train 64 \
    --rerank_num_samples_eval 128 \
    --rerank_sample_temperatures "0.8,1.0,1.2" \
    --rerank_hidden_dims "256,128" \
    --rerank_dropout 0.1 \
    --rerank_lr 1e-3 \
    --rerank_weight_decay 1e-5 \
    --rerank_batch_size 256 \
    --rerank_epochs 30 \
    --max_traj_len_override 7 \
    --rerank_patience 8

echo ""
echo "[Phase 3] Element-constrained inference..."
python "$SCRIPT_DIR/element_constrained_decode.py" \
    --model_path "$RUN_DIR/phase2_gold/best_model.pt" \
    --input_dir "$PROJECT_ROOT/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only" \
    --output_dir "$RUN_DIR/inference_ec" \
    --num_samples 256 \
    --sample_temperatures "0.8,1.0,1.2"

echo ""
echo "============================================"
echo "Training complete. Results in: $RUN_DIR"
echo "============================================"
