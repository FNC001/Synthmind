# Stage2 v3 Precision Report

Date: 2026-06-10

Scope: Stage2 only. Stage3 synthesis-condition prediction was not modified.

## 1. v2 Baseline

Fair method-stratified test baseline from Stage2 v2:

| Metric | v2 best |
|---|---:|
| top1 exact | 33.95% |
| top10 exact | 60.81% |
| top200 exact | 70.41% |
| top500 exact | 73.06% |
| top500 best Jaccard | 82.93% |

v2 showed that canonicalization and hybrid candidate generation help, but closed-vocabulary candidate coverage and OOV rows remain hard bottlenecks.

## 2. Ontology v3

Ontology artifacts:
- `data/interim/ontology/precursor_ontology_v3_20260610/precursor_ontology.csv`
- `data/interim/ontology/precursor_ontology_v3_20260610/precursor_ontology.json`
- `data/interim/ontology/precursor_ontology_v3_20260610/precursor_ontology_report.md`

Statistics:

| Item | Value |
|---|---:|
| precursor labels | 4440 |
| parse failed | 261 |
| family classes | 12 |

Family distribution:

| Family | Count |
|---|---:|
| oxide | 1247 |
| halide | 986 |
| other_salt | 787 |
| organic | 297 |
| nitrate | 288 |
| hydroxide | 184 |
| acetate | 162 |
| sulfate | 156 |
| phosphate | 145 |
| carbonate | 106 |
| elemental | 75 |
| unknown | 7 |

The ontology keeps failed parses instead of dropping them. `pymatgen.Composition` is used for formula keys and diagnostics, not for destructive label canonicalization.

## 3. Element-Family Dataset

Dataset:
- `data/interim/generative/stage2_family_dataset/route_method_stratified_family_v3_20260610`

Rows:

| Split | Samples | Element rows | Mean target elements |
|---|---:|---:|---:|
| train | 29208 | 61077 | 2.09 |
| val | 3658 | 7806 | 2.13 |
| test | 3664 | 7883 | 2.15 |

## 4. Family Predictor v3

Model:
- LightGBM one-vs-rest family classifiers
- Features: structure/descriptor vector + target-element one-hot + reaction-method one-hot
- Run: `runs/stage2/family_predictor_route_method_stratified_v3_20260610`

Test metrics:

| Metric | Result |
|---|---:|
| per-element family top1 exact | 73.44% |
| per-element family top1 recall | 76.47% |
| per-element family top3 recall | 94.03% |
| threshold family F1 | 73.67% |

Selected reaction-method top3 recall:

| Method | top3 recall |
|---|---:|
| solid_state | 96.20% |
| solution | 89.95% |
| hydro_solvothermal | 92.29% |
| precipitation | 90.80% |
| melt_arc | 99.48% |
| flux_molten_salt | 94.71% |

Family prediction is good enough to serve as a candidate prior, especially with top3 families.

## 5. Candidate Pool v3

Candidate sources:
- v2 MLP/retrieval/template candidate sets
- ontology-derived seen precursors by `(element, family)`
- ontology-open labels with zero train frequency
- rule-generated formulas for oxide, carbonate, nitrate, hydroxide, acetate, sulfate, phosphate, halide, elemental
- family-predictor scores

Test results:

| Metric | v2 | v3 | Delta |
|---|---:|---:|---:|
| top1 exact | 33.95% | 33.08% | -0.87 |
| top10 exact | 60.81% | 54.72% | -6.09 |
| top50 exact | 66.59% | 64.33% | -2.26 |
| top100 exact | 68.45% | 68.29% | -0.16 |
| top200 exact | 70.41% | 72.08% | +1.67 |
| top500 exact | 73.06% | 76.12% | +3.06 |
| top500 best F1 | 86.00% | 89.40% | +3.40 |
| top500 best Jaccard | 82.93% | 86.21% | +3.28 |

v3 improves candidate coverage at deeper K. It does not improve early ranking by itself; top10 drops because the open-vocab expansion adds many plausible but hard-to-rank alternatives.

Validation set was stronger:
- val top200 exact: 75.15%
- val top500 exact: 79.03%

Test did not reach the requested 80% top500 target, but moved from 73.06% to 76.12%.

## 6. Open-Vocab / OOV Contribution

OOV analysis:
- `outputs/evaluation/stage2_candidate_pool_v3_20260610/oov_analysis.csv`
- `outputs/evaluation/stage2_candidate_pool_v3_20260610/oov_analysis_report.md`

OOV subset:

| Metric | Result |
|---|---:|
| OOV test rows | 312 |
| OOV rows with any true OOV label in v3 pool | 61.22% |
| OOV rows with generated OOV label | 53.85% |
| OOV rows with ontology-open OOV label | 61.22% |
| OOV exact top10 | 0.64% |
| OOV exact top50 | 5.77% |
| OOV exact top100 | 9.29% |
| OOV exact top200 | 11.54% |
| OOV exact top500 | 18.59% |

This is an important improvement over v2, where OOV top1/top10 exact was 0%. However, OOV rows are still the main blocker for 80%+ exact recall.

OOV by family in v3 ontology:

| Family | Count |
|---|---:|
| oxide | 120 |
| halide | 59 |
| other_salt | 52 |
| organic | 19 |
| hydroxide | 18 |
| nitrate | 14 |
| phosphate | 13 |
| acetate | 13 |
| carbonate | 10 |
| sulfate | 7 |

## 7. Reranker v3

Reranker:
- LightGBM LambdaRank
- LightGBM dense-target regressor
- Hard negatives come from v3 candidate pool: correct element coverage but wrong family/formula, high-Jaccard variants, generated/open-vocab candidates, and high-scoring non-exact candidates.
- Run: `runs/stage2/ranker_stage2_candidate_set_lgbm_v3_20260610`

Pure reranker again underperformed original candidate score. Best result came from blending:

`final_score = 0.9 * ranker_score + 0.1 * v3_total_score`

Test metrics:

| Metric | v2 best | v3 best blend |
|---|---:|---:|
| top1 exact | 33.95% | 35.94% |
| top1 F1 | n/a | 52.90% |
| top1 Jaccard | n/a | 48.15% |
| top3 exact | n/a | 47.49% |
| top5 exact | n/a | 53.82% |
| top10 exact | 60.81% | 60.04% |
| top50 exact | 66.59% | 67.39% |
| top100 exact | 68.45% | 70.50% |
| top200 exact | 70.41% | 73.42% |
| top500 exact | 73.06% | 76.12% |

Subset metrics for best blend:

| Subset | top1 exact | top10 exact | top200 exact | top500 exact |
|---|---:|---:|---:|---:|
| all | 35.94% | 60.04% | 73.42% | 76.12% |
| OOV rows | 0.00% | 0.64% | 10.90% | 18.59% |
| non-OOV rows | 39.29% | 65.57% | 79.24% | 81.47% |

By reaction method, the strongest groups are melt_arc and solid_state. The weakest groups remain flux_molten_salt, hydro_solvothermal, precipitation, and other.

## 8. Upper Bound And Bottlenecks

The v3 candidate pool moved the deeper candidate upper bound:

- top200 exact: 70.41% -> 73.42% after reranker blend
- top500 exact: 73.06% -> 76.12%
- non-OOV top500 exact: 81.47%

So the non-OOV subset has crossed 80% at top500. The full test set is held below 80% mainly by OOV rows and difficult non-solid methods.

Remaining blockers:

1. OOV exact generation is still incomplete. v3 gets some true OOV labels into the pool, but exact full-set recall for OOV rows is only 18.59%@500.
2. Ranking is still weak. The best top1 exact is 35.94%, below the 45% target.
3. Open-vocab expansion increases deep recall but hurts early ranking unless blended with a learned ranker.
4. Some labels are likely still aliases or malformed formulas rather than genuinely new chemistry.

## 9. Next Recommendations

Most useful next steps:

1. Manually clean high-frequency OOV/failed-parse labels, especially oxide/halide/other_salt OOVs.
2. Add graph/structure embeddings to Stage2 candidate scoring; current descriptor vector is likely too weak for method-conditioned precursor choice.
3. Add synthesis-text route descriptions if available; precursor choice is often protocol/history driven, not determined by structure alone.
4. Improve open-vocab formula generation with oxidation-state-aware templates and alias variants.
5. Train a better set model: autoregressive or set-transformer decoder optimized for exact-set/F1, not independent BCE labels.
6. Rebuild reranker training with validation-like negatives and calibration; the current ranker helps only after blending.

Stage2 v3 achieved meaningful deeper candidate coverage gains, but top500 exact is still 76.12%, not 80%. The next push should focus on OOV cleaning/generation and ranking calibration, not Stage3.
