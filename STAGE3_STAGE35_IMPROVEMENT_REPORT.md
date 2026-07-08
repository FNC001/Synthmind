# Stage3 / Stage35 Improvement Report

Date: 2026-06-11

## 1. Scope

本轮只推进 Stage3 合成工艺预测与 Stage35 完整路线排序，核心修正是：Stage3/Stage35 不再依赖未校验的 raw Stage2 `precursors_text`，而是使用 Stage2 v5 chemistry-checked precursor candidates。

新增或更新的主要产物：

- Stage3 chemistry-checked dataset: `data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610`
- Stage3 label/unit audit: `outputs/evaluation/stage3_condition_label_audit_v2_20260610`
- Stage3 condition candidates v2: `outputs/evaluation/stage3_condition_candidates_v2_20260610`
- Stage3 condition score calibration v2: `outputs/evaluation/stage3_condition_calibration_v2_20260610`
- Stage35 route candidates v2: `outputs/evaluation/stage35_route_candidates_v2_20260610`
- Stage35 reranker v2: `runs/stage35/route_reranker_v2_chem_checked_20260610`
- Updated artifact selector: `scripts/07_infer/best_current_route_predictor/predict_best_current_route.py`

## 2. Chemistry-Checked Stage3 Dataset

| split | rows | core rows | precursor check ok | missing source element | extra forbidden element | mean precursor F1 | mean precursor Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 29,208 | 19,722 | 29,208 | 0 | 0 | 93.41% | 90.40% |
| val | 3,658 | 2,441 | 3,658 | 0 | 0 | 43.63% | 39.52% |
| test | 3,664 | 2,468 | 3,664 | 0 | 0 | 46.06% | 41.72% |

说明：

- train 暂用 true precursor fallback，因为当前没有 no-leakage 的 train Stage2 v5 prediction 文件。
- val/test 使用 Stage2 v5 repaired/calibrated top1 precursor set。
- chemistry check 后，目标元素来源缺失与明显 forbidden extra element 均为 0。
- val/test 的 precursor F1/Jaccard 明显低于 train，说明真实推理时 Stage3 输入噪声仍然很大。

## 3. Condition Label / Unit Audit

| item | value |
|---|---:|
| total rows | 36,530 |
| train / val / test | 29,208 / 3,658 / 3,664 |
| core / non-core rows | 24,631 / 11,899 |
| temperature outlier rows | 325 |
| time outlier rows | 355 |
| multimodal formula-method-precursor groups | 2,228 |
| suspected noisy rows | 1,191 |

Temperature 已统一为 Celsius，time 已统一为 hour。整体分布：

- temperature median 595.88 C，p95 1300 C，max 2000 C。
- time median 27.0 h，p95 176.55 h，max 500 h。
- atmosphere 缺失比例 33.23%；solvent 缺失比例 56.36%。

这说明 Stage3 的低成功率并不只是模型问题，标签本身存在明显缺失、多峰和噪声。

## 4. Stage3 A/B/C Single-Point Evaluation

测试集 all rows：

| experiment | temp MAE (C) | time MAE (h) | atmosphere acc | strict success | relaxed success |
|---|---:|---:|---:|---:|---:|
| A old all-method on chem-checked input | 203.08 | 34.35 | 52.72% | 8.95% | 14.14% |
| B core-only retrain | 211.20 | 35.28 | 51.56% | 8.73% | 13.78% |
| C all-train + core-val calibration | 203.38 | 34.45 | 52.21% | 9.58% | 14.57% |

Core method test rows only:

| experiment | temp MAE (C) | time MAE (h) | atmosphere acc | strict success | relaxed success |
|---|---:|---:|---:|---:|---:|
| A old all-method on chem-checked input | 203.56 | 29.65 | 54.62% | 10.41% | 15.88% |
| B core-only retrain | 204.30 | 29.72 | 54.62% | 11.30% | 16.65% |
| C all-train + core-val calibration | 204.41 | 29.49 | 54.68% | 11.51% | 16.65% |

结论：C 版是当前更稳的 Stage3 单点模型，core strict 从 10.41% 提升到 11.51%，但提升有限。主要瓶颈是条件标签噪声、同一体系多峰工艺路线、气氛/溶剂缺失，而不是单纯换训练 split。

## 5. Stage3 Condition Candidate Pool v2

候选来源：

- model point
- method template
- model quantile low/high
- calibrated blend
- train-only nearest-neighbor retrieval template

Raw candidate order 的 test 指标较差，因为 template/retrieval 初始分数压过了 model point；校准后结果如下：

| metric | val | test |
|---|---:|---:|
| top1 strict condition | 8.83% | 9.50% |
| top1 relaxed condition | 15.01% | 14.49% |
| top3 relaxed condition | 19.68% | 17.79% |
| top5 relaxed condition | 24.00% | 21.40% |
| top10 strict condition | 19.57% | 17.30% |
| top10 relaxed condition | 27.12% | 24.70% |

Interpretation:

- 校准恢复了合理 top1，基本等价于选择 model point 优先。
- top10 relaxed condition 从 top1 14.49% 提升到 24.70%，说明条件候选池有一定上限收益。
- 但 top10 上限仍不高，完整路线成功率会被 Stage3 condition oracle 明显限制。

## 6. Stage2 v5 Precursor Side Reference

当前 method-stratified test 最好的 Stage2 v5 calibrated metrics：

| metric | value |
|---|---:|
| top1 exact | 39.47% |
| top10 exact | 63.35% |
| top50 exact | 71.34% |
| top100 exact | 74.51% |
| top200 exact | 77.02% |
| top500 exact | 80.24% |
| top500 best Jaccard | 89.77% |

Stage2 已达到 top500 exact recall 超过 80%，但 top1 exact 仍只有约 39.5%。因此完整路线 top1 会同时受到 Stage2 top1 与 Stage3 top1 的乘法限制。

## 7. Stage35 Route Candidate v2

构建方式：Stage2 v5 precursor top20 × Stage3 condition top10，每个样本最多 200 条完整 route。

Test raw calibrated route score:

| metric | top1 | top3 | top5 | top10 | top50 | top100 | top200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| route exact + strict condition | 3.93% | 5.79% | 6.88% | 9.31% | 12.66% | 13.21% | 13.43% |
| route exact + relaxed condition | 6.11% | 8.05% | 9.72% | 12.83% | 17.63% | 18.26% | 18.59% |
| route usable relaxed (Jaccard >= 0.5 + relaxed condition) | 7.67% | 9.72% | 11.52% | 15.12% | 20.14% | 20.99% | 21.42% |
| precursor exact recall inside route list | 39.47% | 47.52% | 50.76% | 56.52% | 66.38% | 67.52% | 67.52% |
| condition relaxed recall inside route list | 14.49% | 16.43% | 18.70% | 22.13% | 24.56% | 24.62% | 24.70% |

结论：

- 完整路线 top1 exact+relaxed 为 6.11%，top10 exact+relaxed 为 12.83%。
- top200 exact+relaxed 上限只有 18.59%，核心限制来自 Stage3 condition top10/top200 上限，而不是 route 组合不够多。
- 如果把 precursor 放宽到 Jaccard >= 0.5，top200 usable relaxed 可到 21.42%，说明部分候选已经接近但前驱体集合未完全 exact。

## 8. Stage35 Reranker v2

模型：LightGBM LambdaRank，训练集使用 val route candidates，特征只使用推理时可得字段：

- precursor rank/score/source/element coverage
- condition rank/score/source/plausibility
- reaction method / atmosphere / solvent one-hot
- rank interaction and source flags

未使用 `exact`、`F1`、`Jaccard` 等真值作为特征。

Test result:

| model | top1 exact+strict | top1 exact+relaxed | top10 exact+relaxed | top200 exact+relaxed |
|---|---:|---:|---:|---:|
| raw calibrated route score | 3.93% | 6.11% | 12.83% | 18.59% |
| Stage35 reranker v2 | 3.03% | 4.53% | 11.52% | 18.59% |

结论：Stage35 reranker v2 未超过 raw calibrated route score。当前部署/报告推荐继续使用 calibrated raw route score；reranker v2 保留为诊断产物，不作为默认排序器。

## 9. Why Stage3 Success Is Lower Than Earlier 24%

之前看到的约 24% 更接近 topK/relaxed/oracle 或较小范围条件候选的成功率，而不是严格完整路线 top1。当前拆开看：

- Stage3 calibrated test top1 relaxed condition: 14.49%。
- Stage3 calibrated test top10 relaxed condition: 24.70%。
- Stage35 test top1 exact precursor + relaxed condition: 6.11%。
- Stage35 test top10 exact precursor + relaxed condition: 12.83%。

所以“24%”仍然存在，但它对应的是 Stage3 条件 top10 relaxed oracle，不是完整路线 top1 exact success。完整路线需要前驱体集合 exact 且条件命中，因此数值自然更低。

## 10. Current Best Recommendation

当前最佳可用路线：

1. Stage2: use v5 calibrated/repaired chemistry-checked precursor candidates.
2. Stage3: use all-train/core-val chemistry-checked LGBM method experts.
3. Stage3 topK: use calibrated condition candidates v2.
4. Stage35: rank by calibrated raw route score, not by current v2 reranker.
5. GNoME output: use `_chem_checked.csv` files, not raw `precursors_text`.

For GNoME selected outputs:

- `outputs/inference/genome_selected_best_current_20260611/genome_selected_recommended_top1_chem_checked.csv`
- `outputs/inference/genome_selected_best_current_20260611/genome_selected_predictions_chem_checked.csv`

## 11. Next Work

短期最有效方向不是继续换普通 GBDT/reranker，而是提高 Stage3 条件候选上限：

1. 按 reaction_method 建立更细的条件模板库，尤其区分 solid-state calcination/sintering、solution drying/annealing、melt/arc。
2. 将 temperature/time 从单点回归改成 mixture/quantile/distribution prediction，直接优化 topK interval hit。
3. 对 atmosphere/solvent 缺失做显式 missing-label learning，不把 unknown 当成普通类别。
4. 清洗高频 formula-method-precursor 多峰组，保留多条可行工艺而不是强行单标签。
5. 在 Stage35 中加入 synthesis route text 或论文工艺摘要，单靠结构/组成/前驱体特征对条件预测仍不够。

