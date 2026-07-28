# Stage3 v3 Distributional Condition Report

Date: 2026-06-12

## 1. Why Stage3 v3

上一版 Stage3 / Stage35 的主要瓶颈不是普通 LGBM 单点模型没调好，而是数据和任务形式本身不适合单点回归：

- train 使用 true precursor fallback，val/test 使用 Stage2 predicted precursor，存在明显 train/test precursor input mismatch。
- 同一 formula-method-precursor 下存在多峰合成工艺，单个温度/时间标签不能代表所有可行路线。
- atmosphere 缺失约三分之一，solvent 缺失超过一半；把 missing 当作普通错误会低估模型能提供的工艺建议。
- Stage35 reranker v2 没有超过 raw route score；当时 route 上限主要被 Stage3 condition top10/top200 限制。

v3 的核心改动是：predicted-precursor training input、distributional condition generation、missing-aware atmosphere/solvent labels、Stage35 route score calibration，以及满足 gating 后训练诊断版 reranker v3。

## 2. Predicted-Precursor Training Dataset

输出目录：`data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610`

| split | rows | precursor input mode | chemistry ok | mean precursor F1 | mean Jaccard | open/generated rows | repair rows |
|---|---:|---|---:|---:|---:|---:|---:|
| train | 29,208 | pseudo_predicted | 29,208 | 45.32% | 40.92% | 9,340 | 9,340 |
| val | 3,658 | val_predicted | 3,658 | 43.63% | 39.52% | 0 | 0 |
| test | 3,664 | test_predicted | 3,664 | 46.06% | 41.72% | 0 | 0 |

相比上一版 train true fallback 的约 93% F1 / 90% Jaccard，v3 train precursor 输入已经更接近真实推理时的 val/test 噪声水平。

## 3. Condition Targets v3

输出目录：`data/interim/generative/stage3_condition_targets_v3_20260610`

| split | rows | atmosphere known / missing | solvent known / missing | multimodal rows | temp outliers | time outliers |
|---|---:|---:|---:|---:|---:|---:|
| train | 29,208 | 19,734 / 9,474 | 12,886 / 16,322 | 8,476 | 253 | 168 |
| val | 3,658 | 2,346 / 1,312 | 1,561 / 2,097 | 356 | 35 | 37 |
| test | 3,664 | 2,312 / 1,352 | 1,496 / 2,168 | 355 | 37 | 42 |

新增标签包括：

- continuous: `temperature_clipped`, `log_time_clipped`
- bins: method-specific `temperature_bin`, global `time_bin`
- distribution: p10/p25/p50/p75/p90 temperature/time group statistics
- missing-aware: `atmosphere_known_mask`, `solvent_known_mask`, masked target classes

## 4. Stage3 Distributional Model

输出目录：`runs/stage3/distributional_condition_v3_20260610`

Test single-candidate metrics，missing-aware 口径：atmosphere 缺失样本不参与 atmosphere 错误惩罚。

| scope | temp MAE | temp within100 | time MAE | time within24 | atm acc known | strict | relaxed |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-method | 210.80 C | 34.96% | 31.67 h | 62.77% | 53.46% | 16.54% | 36.49% |
| core-method | 214.11 C | 35.53% | 28.44 h | 66.45% | 56.01% | 18.07% | 37.84% |
| solid_state | 222.92 C | 34.59% | 31.75 h | 62.35% | 63.32% | 17.95% | 42.73% |
| solution | 199.67 C | 38.02% | 19.85 h | 74.68% | 45.02% | 19.69% | 34.48% |
| melt_arc | 240.30 C | 28.12% | 53.86 h | 46.88% | 76.92% | 10.71% | 29.46% |
| non-core | 203.96 C | 33.78% | 38.33 h | 55.18% | 47.15% | 13.38% | 33.70% |

Bin/classification metrics on test:

- temperature bin top1/top3: 39.74% / 80.59%
- time bin top1/top3: 34.83% / 70.66%
- atmosphere known-label top1/top3: 53.46% / 89.66%

## 5. Condition Candidate v3

输出目录：`outputs/evaluation/stage3_condition_calibration_v3_20260610`

| setting | top1 strict | top1 relaxed | top5 relaxed | top10 relaxed | top20 relaxed | oracle relaxed |
|---|---:|---:|---:|---:|---:|---:|
| v2 calibrated, original stricter missing treatment | 9.50% | 14.49% | 21.40% | 24.70% | n/a | 24.70% |
| v3 calibrated, missing-aware | 16.54% | 36.49% | 54.86% | 69.27% | 76.26% | 76.83% |

候选来源包括 point model、quantile p10/p50/p90、bin center、multimodal group template、method template、nearest-neighbor condition。v3 明显提高了 condition topK 上限，但需要注意：v3 与 v2 的 atmosphere missing 评价口径不同，不能简单当作纯模型结构提升。

## 6. Stage35 Route v3

Route candidates: Stage2 v5 top20 × Stage3 v3 top20，每个样本最多 400 条。

输出目录：

- `outputs/evaluation/stage35_route_candidates_v3_20260610`
- `outputs/evaluation/stage35_route_score_calibration_v3_20260610`

Test raw/calibrated route score:

| setting | top1 strict route | top1 relaxed route | top10 relaxed route | top200 relaxed route | usable relaxed top200 |
|---|---:|---:|---:|---:|---:|
| v2 raw calibrated route | 3.93% | 6.11% | 12.83% | 18.59% | 21.42% |
| v3 raw/calibrated route | 7.10% | 15.26% | 26.12% | 50.46% | 62.06% |

v3 已超过本轮目标：

- top1 exact+relaxed route > 7%：实际 15.26%
- top10 exact+relaxed route > 15%：实际 26.12%
- top200 exact+relaxed route > 22%：实际 50.46%

## 7. Reranker v3

触发条件满足：

- Stage3 v3 top10 relaxed condition = 69.27% > 30%
- Stage35 v3 raw top10 exact+relaxed route = 26.12% > 15%

因此训练了诊断版 reranker v3：

- script: `scripts/04_train/stage35/train_stage35_route_reranker_v3.py`
- run dir: `runs/stage35/route_reranker_v3_distributional_20260610`

重要限制：完整 train route candidates v3 未生成，reranker v3 使用 val route candidates 训练、test 最终评估。因此它可作为当前 best-current 诊断/候选部署模型，但后续最好补全 train route candidates 后重训。

| model | top1 strict | top1 relaxed | top10 relaxed | top200 relaxed |
|---|---:|---:|---:|---:|
| v3 calibrated raw route | 7.10% | 15.26% | 26.12% | 50.46% |
| v3 reranker diagnostic | 19.41% | 21.32% | 33.08% | 50.22% |

reranker v3 提升了 top1/top10，但 top200 上限基本不变。当前 selector 已把 reranker v3 作为默认 ranking，同时保留 `route_calibrated_score_v3` 为 fallback。

## 8. Current Best-Current Recommendation

当前推荐组合：

| component | artifact |
|---|---|
| Stage2 | `outputs/evaluation/stage2_score_calibration_v5_20260610` |
| Stage2 repaired candidates | `outputs/evaluation/stage2_candidate_pool_v5_20260610` |
| Stage3 v3 model | `runs/stage3/distributional_condition_v3_20260610/stage3_distributional_condition_v3.joblib` |
| Stage3 v3 condition candidates | `outputs/evaluation/stage3_condition_calibration_v3_20260610` |
| Stage35 v3 route candidates | `outputs/evaluation/stage35_route_candidates_v3_20260610` |
| Stage35 raw score calibration | `outputs/evaluation/stage35_route_score_calibration_v3_20260610` |
| Stage35 reranker v3 | `runs/stage35/route_reranker_v3_distributional_20260610/stage35_route_reranker_v3.joblib` |
| inference selector | `scripts/07_infer/best_current_route_predictor/predict_best_current_route.py` |

Inference output requirements:

- Must use chemistry-checked precursors, not raw `precursors_text`.
- Temperature/time should be emitted as intervals.
- Missing or low-confidence atmosphere/solvent should be marked low confidence.
- Open-generated or repair precursor routes should have lower confidence.
- If reaction_method is unknown, output solid_state, solution, and melt_arc core routes.

## 9. Next Steps

Recommended follow-up:

1. Generate train route candidates v3 and retrain reranker v3 with real train/val/test separation.
2. Report two parallel Stage3 metrics: missing-aware and strict-known-only, so comparison to older v2 is fully transparent.
3. Add method-specific distributional experts for melt_arc and non-core classes where time/temperature remain weak.
4. Use route text / synthesis paragraph embeddings to disambiguate atmosphere and solvent.

