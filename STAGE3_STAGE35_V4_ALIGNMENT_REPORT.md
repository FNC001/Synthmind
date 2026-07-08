# SynPred Stage3 / Stage35 v4 Alignment Report

生成日期：2026-06-12

本轮 v4 的目标不是重新优化 Stage2 前驱体，而是解决 Stage3/Stage35 训练和推理分布不一致的问题：用 Stage2 train OOF top20 前驱体候选构造 predprec OOF 数据集，再训练 missing-aware atmosphere/solvent、method distributional experts，并把 top20 前驱体与 top20 条件组合成 v4 Stage35 route candidates。

最终结论：v4 的 route candidate 上限和 top10 覆盖有局部提升，但 v4 reranker blend 的 top1 relaxed 明显低于 v3 final blend。因此当前默认推理入口仍保留 v3 final，不切换到 v4。

## 1. Stage2 Train OOF Top20 Precursors v4

脚本：`scripts/06_eval/generate_stage2_train_oof_top20_candidates_v4.py`

输出目录：`outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612`

| item | samples | candidates | top1 exact | top10 exact | top20 exact | mean top1 F1 | mean best F1@20 | open generated rate | repair rate | chemistry ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-rerank | 29,208 | 369,523 | 13.17% | 50.91% | 54.80% | 0.2893 | 0.7542 | 61.51% | 60.99% | 100.00% |
| OOF reranked | 29,208 | 369,523 | 27.68% | 53.28% | 54.80% | 0.4351 | 0.7542 | 61.51% | 60.99% | 100.00% |

说明：这是 fold-safe approximation，retrieval/template 候选对每个 train fold 排除了当前 fold，reranker 也用 OOF 方式预测当前 fold；但 Stage2 神经模型概率没有重新做完整 OOF 训练。因此它比 v3 final 的 train pseudo precursor 更接近 val/test 分布，但不是完全重新训练的 Stage2 OOF 神经模型。

## 2. Stage3 Predprec OOF Dataset v4

脚本：`scripts/03_data/58_build_stage3_predprec_oof_dataset_v4.py`

输出目录：`data/interim/generative/stage3_condition_dataset_predprec_oof_v4_20260612`

| split | rows | v3 mean F1 | v4 mean F1 | v3 mean Jaccard | v4 mean Jaccard | v4 mode |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 29,208 | 0.4532 | 0.4351 | 0.4092 | 0.3884 | stage2_v4_oof_top1:29208 |
| val | 3,658 | 0.4363 | 0.4520 | 0.3952 | 0.4097 | stage2_v5_val_top1:3658 |
| test | 3,664 | 0.4606 | 0.4692 | 0.4172 | 0.4261 | stage2_v5_test_top1:3664 |

解读：train v4 因为使用真正的 OOF top1 候选，均值 F1/Jaccard 比 v3 pseudo train 略低，但这反而更贴近 held-out val/test 的真实预测分布；val/test 使用 Stage2 v5 repaired/calibrated top1 后略高于 v3。

## 3. Stage3 Missing-Aware Labels and Method Experts v4

脚本：

- `scripts/04_train/stage3/train_stage3_atmosphere_solvent_missing_v4.py`
- `scripts/04_train/stage3/train_stage3_method_distributional_experts_v4.py`

输出目录：

- `runs/stage3/atmosphere_solvent_missing_v4_20260612`
- `runs/stage3/method_distributional_experts_v4_20260612`

val missing-aware label metrics：atmosphere known-mask acc 65.91%，atmosphere known-label top1 52.39% / top3 89.98%；solvent known-mask acc 88.44%，solvent known-label top1 78.48% / top3 95.64%。
test missing-aware label metrics：atmosphere known-mask acc 63.78%，atmosphere known-label top1 54.54% / top3 88.15%；solvent known-mask acc 89.71%，solvent known-label top1 76.27% / top3 95.32%。

method distributional global expert test：strict condition 15.72%，relaxed condition 35.64%，temperature MAE 213.28 C，time MAE 31.76 h。

## 4. Stage3 Condition Candidates v4

脚本：

- `scripts/06_eval/generate_stage3_condition_candidates_v4.py`
- `scripts/06_eval/calibrate_stage3_condition_scores_v4.py`

输出目录：`outputs/evaluation/stage3_condition_calibration_v4_20260612`

| protocol | version | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top20 relaxed | oracle relaxed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| missing-aware | v3 final | 16.54% | 36.49% | 66.40% | 69.27% | 76.26% | 76.83% |
| missing-aware | v4 | 15.72% | 35.64% | 66.10% | 69.10% | 75.46% | 76.01% |
| strict-comparable | v3 final | 8.65% | 17.96% | 31.80% | 33.19% | 39.85% | 40.42% |
| strict-comparable | v4 | 7.48% | 17.79% | 31.33% | 32.97% | 39.08% | 39.57% |

结论：v4 condition candidate 的分布对齐更规范，但 test 指标没有超过 v3 final。校准搜索选择了 raw score 权重，说明新增 missing-aware/method expert 字段对 Stage3 排序没有形成稳定增益。

## 5. Stage35 Route Candidates v4

脚本：`scripts/06_eval/build_stage35_route_candidates_v4.py`

输出目录：`outputs/evaluation/stage35_route_candidates_v4_20260612`

| split | samples | route candidates | top1 relaxed | top10 relaxed | top200 relaxed | top400 relaxed | top200 strict-comparable relaxed | top400 strict-comparable relaxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 29,208 | 7,258,197 | 13.78% | 29.62% | 48.08% | 49.10% | 32.19% | 33.18% |
| val | 3,658 | 1,449,100 | 14.79% | 29.83% | 49.95% | 51.42% | 27.58% | 29.06% |
| test | 3,664 | 1,450,440 | 14.82% | 32.59% | 50.27% | 51.47% | 28.30% | 29.45% |

与 v3 final raw test 对比：

| protocol | metric | v3 final raw | v4 raw | delta |
| --- | --- | ---: | ---: | ---: |
| missing-aware | top1 relaxed | 15.26% | 14.82% | -0.44 pp |
| missing-aware | top10 relaxed | 26.12% | 32.59% | +6.47 pp |
| missing-aware | top200 relaxed | 50.46% | 50.27% | -0.19 pp |
| missing-aware | top400 relaxed | 51.94% | 51.47% | -0.47 pp |
| strict-comparable | top1 relaxed | 8.13% | 8.05% | -0.08 pp |
| strict-comparable | top10 relaxed | 14.33% | 17.88% | +3.55 pp |
| strict-comparable | top200 relaxed | 28.52% | 28.30% | -0.22 pp |
| strict-comparable | top400 relaxed | 29.89% | 29.45% | -0.44 pp |

解读：v4 raw route top10 明显高于 v3 raw，说明 train OOF top20 与 v4 score 对 route candidate 排序有帮助；但 top1 仍低，Stage3 v4 condition 本身略低于 v3 final，抵消了部分收益。

## 6. Stage35 Reranker v4

脚本：`scripts/04_train/stage35/train_stage35_route_reranker_v4.py`

输出目录：`runs/stage35/route_reranker_v4_20260612`

训练使用 train rows：2,326,984；val 选择 blend 权重：raw 0.70，LambdaRank 0.30，binary 0.00。

| model | protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top200 relaxed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v4 raw | missing-aware | 6.58% | 14.82% | 21.94% | 32.59% | 50.27% |
| v4 raw | strict-comparable | 3.44% | 8.05% | 11.63% | 17.88% | 28.30% |
| v4 LambdaRank | missing-aware | 0.30% | 0.41% | 1.86% | 2.67% | 29.64% |
| v4 LambdaRank | strict-comparable | 0.16% | 0.16% | 0.93% | 1.12% | 14.74% |
| v4 binary | missing-aware | 0.44% | 1.06% | 3.58% | 6.66% | 32.83% |
| v4 binary | strict-comparable | 0.27% | 0.55% | 1.83% | 3.36% | 16.65% |
| v4 blend | missing-aware | 6.58% | 14.82% | 25.60% | 33.98% | 50.55% |
| v4 blend | strict-comparable | 3.44% | 8.05% | 13.46% | 18.40% | 28.52% |

与当前 v3 final blend 对比：

| protocol | metric | v3 final blend | v4 blend | decision |
| --- | --- | ---: | ---: | --- |
| missing-aware | top1 relaxed | 20.72% | 14.82% | v3 better |
| missing-aware | top10 relaxed | 34.55% | 33.98% | v3 better |
| missing-aware | top200 relaxed | 48.85% | 50.55% | v4 better |
| strict-comparable | top1 relaxed | 10.45% | 8.05% | v3 better |
| strict-comparable | top10 relaxed | 18.04% | 18.40% | v4 better |
| strict-comparable | top200 relaxed | 26.97% | 28.52% | v4 better |

结论：v4 blend 在 top10/top200 上接近或局部超过 v3 final，但 top1 relaxed 在两套协议下都低于 v3 final。因此不满足“只有 v4 同时超过 v3 final 才更新 inference selector”的条件。

## 7. 部署选择

当前 `scripts/07_infer/best_current_route_predictor/predict_best_current_route.py` 继续指向 v3 final：Stage2 v5 + Stage3 v3 + Stage35 reranker v3 final blend。v4 artifacts 保留为实验产物，不作为默认推理入口。

不切换的原因：

1. v4 Stage3 condition candidate 没有超过 v3 final。
2. v4 Stage35 blend top1 relaxed：missing-aware 14.82%，strict-comparable 8.05%，均低于 v3 final 的 20.72% 和 10.45%。
3. v4 的优势主要在 top10/top200 覆盖和训练分布对齐，对“第一推荐路线”的排序还不够稳定。

## 8. 文件与检查

关键输出：

- `outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612/train_oof_top20_precursor_candidates.csv`
- `data/interim/generative/stage3_condition_dataset_predprec_oof_v4_20260612`
- `runs/stage3/atmosphere_solvent_missing_v4_20260612`
- `runs/stage3/method_distributional_experts_v4_20260612`
- `outputs/evaluation/stage3_condition_calibration_v4_20260612`
- `outputs/evaluation/stage35_route_candidates_v4_20260612`
- `runs/stage35/route_reranker_v4_20260612`

新增脚本均保留 `--help`，并通过最终 `py_compile` 检查。

## 9. 下一步建议

1. 如果要让 v4 真正超过 v3 final top1，需要继续提高 Stage3 condition top1，而不是只做 route-level rerank。
2. 对 v4 LambdaRank 单独排序 top1 很低，说明当前 label/relevance 与推理目标仍有错配；下一轮应把训练目标改成 pairwise hard-negative loss 或直接优化 top1 relaxed route。
3. OOF train candidate 分布对齐是正确方向，但 Stage2 neural score 仍不是完全 OOF；后续可以做真正的 Stage2 K-fold neural OOF。
4. 对 atmosphere missing/unknown 的严格口径仍是主要扣分来源，可以引入 route text 或原文 synthesis paragraph 来判断气氛/溶剂是否真的缺失。
5. 当前 v4 候选池大幅增加磁盘占用，后续建议将 train route candidates 改成 parquet/zstd 或 chunked binary cache。

