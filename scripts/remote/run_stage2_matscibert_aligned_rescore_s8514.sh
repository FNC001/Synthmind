#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
DATA="$ROOT/data/stage2_full_cation_family_canonical_v1"
SRC="$ROOT/metrics/canonical_v1/sources"
RUN="$ROOT/runs/canonical_v1/stage2_matscibert_s8502_aligned_s8514"
CHECKPOINT="$ROOT/runs/canonical_v1/stage2_matscibert_global_union13_s8502/model.pt"

mkdir -p "$RUN"
cd "$ROOT/code"
export PYTHONPATH=.

/root/miniconda3/bin/python3 training/family/train_stage2_matscibert_cross_encoder.py \
  --input_dir "$DATA" \
  --model_path /root/autodl-tmp/models/matscibert \
  --matsci_embeddings "$DATA/matscibert_embeddings_canonical_v1.npz" \
  --matsci_components 64 \
  --base_val_candidates "$ROOT/runs/canonical_v1/stage2_s8502_top10_s8494_depth_s8504/val_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_oof5_fixedrrf10_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_oof_familypool_train1000_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_autoregressive_oof_fixed38_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_gflownet_film_oof_fixed31_beam100_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_substitution_oof_allgroups_exact1p0_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_substitution_oof_periodic_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_fulltrain_oof6_aligned_matsci32_g0p25_s8005_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_fulltrain_oof6_aligned_fast_g0p25_s8007_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_oof_element_cartesian_zeroshot_train2000_candidates.jsonl" \
  --candidate_limit 800 \
  --candidate_windows 20,50,100,200,400 \
  --source_union_limit 100 \
  --train_pool_limit 64 \
  --cross_family_negatives 16 \
  --multi_positive_route_kind anion_synthesis_nofamily_factorized \
  --multi_positive_min_count 2 \
  --multi_positive_max_candidates 8 \
  --multi_positive_weight 0.15 \
  --max_length 96 \
  --text_pair_mode tokenizer_pair \
  --freeze_bottom_layers 2 \
  --attention_implementation sdpa \
  --dropout 0.12 \
  --eval_batch_size 2048 \
  --pairwise_weight 0.30 \
  --seed 8502 \
  --resume_model "$CHECKPOINT" \
  --eval_only \
  --output_json "$RUN/metrics.json" \
  --output_candidates_jsonl "$RUN/val_candidates.jsonl" \
  --output_model "$RUN/model.pt" \
  --output_scores_npz "$RUN/val_scores.npz" \
  2>&1 | tee "$RUN/run.log"
