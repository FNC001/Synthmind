#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
DATA="$ROOT/data/stage2_full_cation_family_canonical_v1"
SRC="$ROOT/metrics/canonical_v1/sources"
RUN="$ROOT/runs/canonical_v1/stage2_matscibert_rankaware_s8509"

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
  --train_pool_limit 96 \
  --cross_family_negatives 24 \
  --include_rank_features \
  --train_aux_rank_source "$ROOT/runs/canonical_v1/stage2_family_template_oof5_s8511/train_candidates.jsonl" \
  --val_aux_rank_source "$ROOT/runs/canonical_v1/stage2_family_template_canonical_s8508/val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_oof_familypool_train1000_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_union_familyfirst_h1024_s6262_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_autoregressive_oof_fixed38_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_autoregressive_cardinality_beam_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_gflownet_film_oof_fixed31_beam100_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_gflow_relativechem_s8006_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_substitution_oof_allgroups_exact1p0_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_family_substitution_allgroups_exact0p1_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_substitution_oof_periodic_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_family_substitution_periodic_knn1000_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_oof_element_cartesian_zeroshot_train2000_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_element_cartesian_deep_p120_z200_a50_val_candidates.jsonl" \
  --aux_rank_limit 100 \
  --multi_positive_route_kind none \
  --multi_positive_max_candidates 1 \
  --multi_positive_weight 0 \
  --max_length 96 \
  --text_pair_mode tokenizer_pair \
  --freeze_bottom_layers 2 \
  --pooling auto \
  --attention_implementation sdpa \
  --dropout 0.10 \
  --epochs 6 \
  --early_stopping_patience 2 \
  --batch_queries 4 \
  --eval_batch_size 2048 \
  --encoder_learning_rate 5e-6 \
  --head_learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.08 \
  --pairwise_weight 0.50 \
  --margin 0.5 \
  --gradient_clip 1.0 \
  --gradient_accumulation 4 \
  --num_workers 4 \
  --log_every_steps 100 \
  --seed 8509 \
  --output_json "$RUN/metrics.json" \
  --output_candidates_jsonl "$RUN/val_candidates.jsonl" \
  --output_model "$RUN/model.pt" \
  --output_scores_npz "$RUN/val_scores.npz" \
  2>&1 | tee "$RUN/run.log"
