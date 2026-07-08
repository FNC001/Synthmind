# STAGE2 V4 OOV/RANKING REPORT

## Scope

Stage3 was not changed. This run only modifies Stage2 precursor-set prediction data cleanup, ontology parsing, candidate generation, OOV analysis, and score calibration.

## Artifacts

- canonical v4 dataset: `/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/generative/stage2_setpred_dataset/descriptor/route_method_stratified_canonical_v4_20260610_relaxed_only`
- ontology v4: `/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/ontology/precursor_ontology_v4_20260610`
- alias patch v4: `/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/ontology/precursor_alias_patch_v4_20260610/precursor_alias_patch.csv`
- candidate pool v4: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_candidate_pool_v4_20260610`
- calibrated candidates: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_candidate_pool_v4_20260610/test_candidate_sets_calibrated.csv`
- error analysis: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_candidate_pool_v4_20260610/error_analysis_v4.md`

## Baseline To Beat

| model | top1 exact | top10 exact | top200 exact | top500 exact | best Jaccard@500 |
|---|---:|---:|---:|---:|---:|
| v2 baseline | 33.95% | 60.81% | 70.41% | 73.06% | 82.93% |
| v3 best blend | 35.94% | 60.04% | 73.42% | 76.12% | n/a |

## Canonical/Ontology V4

- labels: v2 4440 -> v4 4414
- conservative alias patches: 26 ({'hydrate_normalize': 20, 'merge_alias': 6})
- audit labels: 477; OOV labels: 242; failed-parse labels audited: 261
- parse failed: v3 261 -> v4 182
- oxidation-state parse ok: 3319 / 4414
- remaining OOV labels after v4 patch: 216

Family distribution:

- oxide: 1242
- halide: 983
- other_salt: 785
- organic: 299
- nitrate: 281
- hydroxide: 182
- acetate: 158
- sulfate: 155
- phosphate: 143
- carbonate: 103
- elemental: 76
- unknown: 7

## Family Predictor

- test family top1 exact: 73.44%
- test family top1 recall: 76.47%
- test family top3 recall: 94.03%
- threshold family F1: 73.67%

## Candidate Pool V4

| split/scoring | top1 exact | top10 exact | top50 exact | top100 exact | top200 exact | top500 exact | best F1@500 | best Jaccard@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val raw v4 | 34.83% | 54.13% | 64.52% | 68.56% | 72.85% | 79.06% | 90.67% | 87.97% |
| test raw v4 | 33.60% | 51.80% | 63.05% | 66.89% | 70.41% | 76.83% | 89.83% | 86.71% |
| test calibrated + v3 reranker signal | 34.99% | 58.57% | 66.62% | 70.44% | 73.36% | 76.83% | 89.83% | 86.71% |

Compared with v2 baseline:

- top500 exact recall: 76.83% vs 73.06%, +3.77 points.
- top200 exact recall: 73.36% vs 70.41%, +2.95 points.
- top1 exact did not improve beyond v3 best blend: v4 calibrated 34.99% vs v3 best 35.94%.

## OOV/Open-Vocab Contribution

- OOV rows: 312
- OOV rows with any OOV label appearing in v4 pool: 57.05%
- OOV rows with generated OOV label: 50.32%
- OOV top1/top10/top200/top500 exact: 1.28% / 3.85% / 17.95% / 24.68%
- non-OOV calibrated top500 exact: 81.68%

OOV by reaction method:

- other: 105
- solution: 95
- solid_state: 60
- precipitation: 22
- hydro_solvothermal: 18
- flux_molten_salt: 10
- melt_arc: 10
- thermal_decomposition: 4
- mechanochemical: 1

OOV by family:

- oxide: 114
- halide: 55
- other_salt: 50
- unknown: 36
- organic: 19
- acetate: 12
- hydroxide: 12
- phosphate: 11
- carbonate: 7
- sulfate: 5
- nitrate: 4

## Ranking Calibration

- best val objective: 0.5657
- best weights favored the external v3 reranker score plus v3 total score, with modest penalties for generated/open-vocab candidates.
- test top1 exact after calibration: 34.99%
- test NDCG@10 exact: 46.45%
- OOV top1 remains very low: 1.60%

## By Reaction Method (calibrated)

| method | n | top1 | top10 | top200 | top500 |
|---|---:|---:|---:|---:|---:|
| solid_state | 1142 | 48.42% | 65.24% | 78.72% | 81.79% |
| solution | 1102 | 25.32% | 61.89% | 75.77% | 79.40% |
| other | 737 | 25.37% | 48.03% | 65.81% | 70.56% |
| melt_arc | 224 | 75.00% | 85.71% | 91.07% | 91.96% |
| hydro_solvothermal | 191 | 17.80% | 38.74% | 57.07% | 58.64% |
| precipitation | 128 | 17.19% | 38.28% | 55.47% | 62.50% |
| flux_molten_salt | 68 | 25.00% | 27.94% | 48.53% | 48.53% |
| thermal_decomposition | 32 | 15.62% | 31.25% | 78.12% | 84.38% |
| mechanochemical | 28 | 42.86% | 46.43% | 64.29% | 64.29% |
| sol_gel | 7 | 42.86% | 57.14% | 71.43% | 85.71% |
| combustion | 5 | 40.00% | 80.00% | 80.00% | 80.00% |

## Conclusion

Stage2 v4 improves the candidate-pool ceiling and OOV coverage, but it does not achieve the requested 80% top500 target on method-stratified test and does not reach 45% top1. The best test top500 exact recall is 76.83%; the best v4 top1 exact is 34.99%, while the older v3 best blend remains slightly better at 35.94% top1.

The main blocker is exact concrete precursor identity for OOV and complex salts. Non-OOV top500 is already 81.68%, but OOV top500 is only 24.68%. The next meaningful path is not another generic reranker; it is targeted OOV/alias/template expansion, especially oxide/halide/other_salt labels and weak methods such as hydro/solvothermal, precipitation, flux/molten-salt, and other.

To push toward 80% top500 and then improve top1, prioritize: manual cleanup of high-frequency failed/OOV labels, method-specific precursor templates, charge-balanced mixed-salt generation, and a reranker trained with explicit OOV/generated hard negatives plus text route descriptors if available.
