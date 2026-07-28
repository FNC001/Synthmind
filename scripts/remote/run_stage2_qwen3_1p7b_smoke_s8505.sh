#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/synthmind_family_routed_v1
CODE_ROOT="$ROOT/code"
RUN_ROOT="$ROOT/runs/canonical_v1/stage2_qwen3_1p7b_smoke_s8505"
SOURCE_ROOT="$ROOT/metrics/canonical_v1/sources"
DATA_ROOT="$ROOT/data/stage2_full_cation_family_canonical_v1"

mkdir -p "$RUN_ROOT"
cd "$CODE_ROOT"
export PYTHONPATH=.

# This is a non-reportable engineering smoke test.  It exercises the exact
# production code path on the shared G02 family while keeping training and
# validation small enough to catch loader, memory, and tokenizer failures
# before committing to the full 1.7B-parameter run.
exec /root/miniconda3/bin/python3 \
  training/family/train_stage2_matscibert_cross_encoder.py \
  --input_dir "$DATA_ROOT" \
  --model_path /root/autodl-tmp/models/qwen3-1.7b-base \
  --matsci_embeddings "$DATA_ROOT/matscibert_embeddings_canonical_v1.npz" \
  --matsci_components 64 \
  --base_val_candidates "$ROOT/runs/canonical_v1/stage2_label_free_union13_s8501/val_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_oof5_fixedrrf10_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_oof_familypool_train1000_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_autoregressive_oof_fixed38_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_gflownet_film_oof_fixed31_beam100_train_candidates.jsonl" \
  --candidate_limit 16 \
  --candidate_windows 16 \
  --source_union_limit 16 \
  --train_pool_limit 16 \
  --cross_family_negatives 4 \
  --multi_positive_route_kind none \
  --multi_positive_max_candidates 1 \
  --multi_positive_weight 0.0 \
  --train_families G02 \
  --val_families G02 \
  --train_row_limit 4 \
  --max_length 96 \
  --text_pair_mode explicit \
  --freeze_bottom_layers 20 \
  --freeze_embeddings \
  --pooling last \
  --attention_implementation sdpa \
  --dropout 0.10 \
  --epochs 1 \
  --batch_queries 1 \
  --eval_batch_size 32 \
  --encoder_learning_rate 1e-6 \
  --head_learning_rate 1e-5 \
  --pairwise_weight 0.50 \
  --gradient_accumulation 1 \
  --num_workers 0 \
  --log_every_steps 1 \
  --seed 8505 \
  --output_json "$RUN_ROOT/metrics.json" \
  --output_candidates_jsonl "$RUN_ROOT/val_candidates.jsonl" \
  --output_model "$RUN_ROOT/model.pt" \
  --output_scores_npz "$RUN_ROOT/val_scores.npz"
