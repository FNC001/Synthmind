# Stage3 条件成功率口径与 Top10 评估

## 为什么这次看起来比之前低

之前看到的约 24% 成功率，对应的是旧的 old/core route 候选评估中的 `top1 relaxed condition`，即温度误差 ≤200 °C、时间误差 ≤48 h、气氛正确，数值为 24.54%；Stage35 rerank 后对应值为 23.98%。这个评估集只有 538 条，属于较早的 old/core 口径，样本更少、更偏容易。

本轮 core-only Stage3 retrain 报告中的 16.61%，对应的是新的 core-method test 全部 2,468 条样本，以所有样本为分母计算 `top1 relaxed condition`。这个分母更大，且包含更多 solution / solid_state / melt_arc 样本；其中 melt_arc 的温度/时间字段噪声较大，会拉低整体条件成功率。

如果只在温度、时间、气氛都存在的可评估样本上计算，本轮 core-only Stage3 的结果为：

| 口径 | strict condition | relaxed condition |
|---|---:|---:|
| 所有 core test 样本为分母 | 11.02% | 16.61% |
| 仅可评估样本为分母 | 20.86% | 31.44% |

因此，本轮结果并不是简单退化，而是评估口径更严格、测试集更大；若按可评估样本分母，relaxed condition 已超过 30%。

## Top10 条件命中率

`top10 condition` 表示前 10 条候选工艺中，只要有一条满足温度、时间和气氛条件就算命中。

| 评估设置 | n | top1 strict cond | top1 relaxed cond | top10 strict cond | top10 relaxed cond |
|---|---:|---:|---:|---:|---:|
| method-stratified top10，全方法 | 3,664 | 9.01% | 14.63% | 11.46% | 17.36% |
| method-stratified top10，core-filter | 2,378 | 12.78% | 19.22% | 16.06% | 22.96% |
| Stage35 rerank，全方法 | 3,664 | 9.09% | 14.74% | 11.46% | 17.36% |
| Stage35 rerank，core-filter | 2,378 | 12.91% | 19.39% | 16.06% | 22.96% |
| topK condition MoE + retrieval，全方法 | 3,664 | 5.84% | 10.21% | 21.29% | 28.85% |
| topK condition MoE + retrieval，core-filter | 2,378 | 7.74% | 12.83% | 28.93% | 37.59% |
| old/core top10 | 538 | 14.50% | 24.54% | 19.14% | 29.55% |
| old/core Stage35 | 538 | 14.13% | 23.98% | 19.14% | 29.55% |

## Top10 完整路线命中率

完整路线还要求前驱体也满足约束，因此数值低于单独 Stage3 条件命中率。

| 评估设置 | n | top1 strict route | top1 relaxed route | top10 strict route | top10 relaxed route |
|---|---:|---:|---:|---:|---:|
| method-stratified top10，全方法 | 3,664 | 3.11% | 6.39% | 5.90% | 11.35% |
| method-stratified top10，core-filter | 2,378 | 4.58% | 8.24% | 8.70% | 15.18% |
| Stage35 rerank，全方法 | 3,664 | 3.49% | 6.85% | 5.90% | 11.35% |
| Stage35 rerank，core-filter | 2,378 | 5.09% | 8.92% | 8.70% | 15.18% |
| topK condition MoE + retrieval，全方法 | 3,664 | 2.05% | 4.69% | 8.43% | 14.06% |
| topK condition MoE + retrieval，core-filter | 2,378 | 2.69% | 5.85% | 11.73% | 18.00% |
| old/core top10 | 538 | 7.99% | 15.61% | 10.97% | 21.75% |
| old/core Stage35 | 538 | 8.18% | 16.73% | 10.97% | 21.75% |

## 结论

1. 之前的 24% 是旧 538 条 old/core 测试集上的 top1 relaxed condition，不是当前 2,468 条 core-only Stage3 retrain 的同口径结果。
2. 当前 core-filter top10 relaxed condition 为 22.96%，接近旧口径 29.55%，但仍低一些。
3. 使用 topK condition MoE + retrieval 后，core-filter top10 relaxed condition 可到 37.59%，说明候选池里有更好的工艺条件，但 top1 排序还没有把它们选上来。
4. 下一步 Stage3 的重点不是单点回归，而是训练 condition reranker，把 top10 中已存在的好工艺条件排到第一。
