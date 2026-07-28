#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/synthmind_family_routed_v1
CODE_ROOT="$ROOT/code"
RUN_ROOT="$ROOT/runs/canonical_v1/stage2_matscibert_exact_resume_s8503"
SOURCE_ROOT="$ROOT/metrics/canonical_v1/sources"
DATA_ROOT="$ROOT/data/stage2_full_cation_family_canonical_v1"

mkdir -p "$RUN_ROOT"
cd "$CODE_ROOT"
export PYTHONPATH=.

exec /root/miniconda3/bin/python3 \
  training/family/train_stage2_matscibert_cross_encoder.py \
  --input_dir "$DATA_ROOT" \
  --model_path /root/autodl-tmp/models/matscibert \
  --matsci_embeddings "$DATA_ROOT/matscibert_embeddings_canonical_v1.npz" \
  --matsci_components 64 \
  --base_val_candidates "$ROOT/runs/canonical_v1/stage2_label_free_union13_s8501/val_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_oof5_fixedrrf10_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_oof_familypool_train1000_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_autoregressive_oof_fixed38_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_gflownet_film_oof_fixed31_beam100_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_substitution_oof_allgroups_exact1p0_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_substitution_oof_periodic_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_fulltrain_oof6_aligned_matsci32_g0p25_s8005_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_fulltrain_oof6_aligned_fast_g0p25_s8007_train_candidates.jsonl" \
  --train_candidate_source "$SOURCE_ROOT/stage2_oof_element_cartesian_zeroshot_train2000_candidates.jsonl" \
  --candidate_limit 800 \
  --candidate_windows 20,50,100,200,400 \
  --source_union_limit 100 \
  --train_pool_limit 96 \
  --cross_family_negatives 24 \
  --multi_positive_route_kind none \
  --multi_positive_min_count 2 \
  --multi_positive_max_candidates 1 \
  --multi_positive_weight 0.0 \
  --max_length 96 \
  --text_pair_mode tokenizer_pair \
  --freeze_bottom_layers 2 \
  --attention_implementation sdpa \
  --dropout 0.10 \
  --epochs 4 \
  --early_stopping_patience 2 \
  --batch_queries 3 \
  --eval_batch_size 2048 \
  --encoder_learning_rate 3e-6 \
  --head_learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --pairwise_weight 0.50 \
  --margin 0.5 \
  --gradient_clip 1.0 \
  --gradient_accumulation 4 \
  --num_workers 4 \
  --log_every_steps 100 \
  --seed 8503 \
  --resume_model "$ROOT/runs/canonical_v1/stage2_matscibert_global_union13_s8502/model.pt" \
  --output_json "$RUN_ROOT/metrics.json" \
  --output_candidates_jsonl "$RUN_ROOT/val_candidates.jsonl" \
  --output_model "$RUN_ROOT/model.pt" \
  --output_scores_npz "$RUN_ROOT/val_scores.npz"
