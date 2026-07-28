#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
DATA="$ROOT/data/stage2_full_cation_family_canonical_v1"
SRC="$ROOT/metrics/canonical_v1/sources"
RUN="${RUN:-$ROOT/runs/canonical_v1/stage2_validation_meta_lambdarank_s8518}"

mkdir -p "$RUN"
cd "$ROOT/code"
export PYTHONPATH=.

/root/miniconda3/bin/python3 training/family/train_stage2_validation_meta_lambdarank.py \
  --input_dir "$DATA" \
  --matsci_embeddings "$DATA/matscibert_embeddings_canonical_v1.npz" \
  --matsci_components 64 \
  --base_candidates "$ROOT/runs/canonical_v1/stage2_s8502_top10_s8494_depth_s8504/val_candidates.jsonl" \
  --expert_source "family_template=$ROOT/runs/canonical_v1/stage2_family_template_canonical_s8508/val_candidates.jsonl" \
  --expert_source "groupbal05=$SRC/stage2_zs_groupbal_g1_f05_s7902_val_candidates.jsonl" \
  --expert_source "zerochem=$SRC/stage2_zeroshot_chemonly_h1024_s6701_scalegrid_val_candidates.jsonl" \
  --expert_source "groupbal0=$SRC/stage2_zs_groupbal_g1_f0_s7901_val_candidates.jsonl" \
  --expert_source "familyfirst=$SRC/stage2_union_familyfirst_h1024_s6262_val_candidates.jsonl" \
  --expert_source "elementfirst=$SRC/stage2_union_elementfirst_h1024_s6363_val_candidates.jsonl" \
  --expert_source "autoregressive=$SRC/stage2_autoregressive_cardinality_beam_val_candidates.jsonl" \
  --expert_source "gflow=$SRC/stage2_gflow_relativechem_s8006_val_candidates.jsonl" \
  --expert_source "substitution_all=$SRC/stage2_family_substitution_allgroups_exact0p1_val_candidates.jsonl" \
  --expert_source "substitution_periodic=$SRC/stage2_family_substitution_periodic_knn1000_val_candidates.jsonl" \
  --expert_source "element_cartesian=$SRC/stage2_element_cartesian_deep_p120_z200_a50_val_candidates.jsonl" \
  --expert_source "family_g10=$SRC/stage2_zs_family_G10_G08G10_h1024_s6801_val_candidates.jsonl" \
  --expert_source "family_g11soft=$SRC/stage2_zs_family_G11_G16_soft_h768_s6804_val_candidates.jsonl" \
  --expert_source "family_g06g07=$SRC/stage2_zs_family_G06_G07_h768_s6803_val_candidates.jsonl" \
  --base_limit 100 \
  --expert_limit 10 \
  --folds 5 \
  --num_boost_round 450 \
  --learning_rate 0.03 \
  --num_leaves 63 \
  --min_data_in_leaf 80 \
  --feature_fraction 0.90 \
  --bagging_fraction 0.90 \
  --lambda_l1 0.01 \
  --lambda_l2 0.20 \
  --group_balance_power 1.0 \
  --family_balance_power 0.25 \
  --num_threads 64 \
  --seed "${SEED:-8518}" \
  --output_json "$RUN/metrics.json" \
  --output_candidates_jsonl "$RUN/val_candidates.jsonl" \
  --output_model "$RUN/model.txt" \
  --output_scores_npz "$RUN/val_scores.npz" \
  "$@" \
  2>&1 | tee "$RUN/run.log"
