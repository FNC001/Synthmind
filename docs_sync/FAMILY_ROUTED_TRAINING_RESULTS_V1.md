# 同族元素路由训练结果 V1

初版日期：2026-07-13；最近更新：2026-07-15  
代码分支：`codex/family-routed-training-v1`  
远程隔离目录：`/root/autodl-tmp/synthmind_family_routed_v1`

## 1. 结论

V1 已按完整数据库重新划分和训练。固定 validation 上的高成本 Top-K 优化已越过目标线，但独立 test 显示明显的族分布与未见标签偏移，因此本文件仍是阶段报告，不是生产验收报告。主分族只看目标材料中金属/骨架阳离子的周期表族，不用阴离子或化学计量拆分，因此以下目标均为 `G01`：

```text
LiCl = NaCl = NaBr = Li2O = Na2S = G01
```

Stage 2 当前研究方案由同族/全元素族候选生成、零样本化学排序器、交叉注意力排序器、五折 OOF 候选堆叠、RRF 和支持度受限的族＋阴离子族路由组成。阴离子族只用于选择专家，O/S 与 Cl/Br 始终分别保持同族。旧的 GFlowNet + FiLM 结果保留为早期消融，不再是当前 Top-K 最佳方案。

Stage 3 的族增益依任务而异。最终建议使用按任务头选择的 LightGBM：温度、时间和气氛使用无族头，反应方法使用族条件头。GPU 多任务宽网络完成了 3 组超参数搜索，但整体没有超过 LightGBM，因此保留为研究模型，不因计算量大而强制上线。

生产默认配置尚未替换，`configs/family_routed_v1.yaml` 中 `enabled: false`。

### 1.1 固定无泄漏 Top-K 口径的当前状态

模型选择使用 formula/material-disjoint validation；前驱体方案达到目标并冻结后才执行 test 审计。前驱体指标要求“整套前驱体集合精确相等”，工艺指标要求温度、时间和粗粒度气氛元组同时命中，均不以单标签或宽松命中替代。

| 任务 | 当前 validation Top-1 | Top-10 | 目标 | 状态 |
|---|---:|---:|---:|---|
| 前驱体整套精确命中 | **39.77%** | **80.97%** | 80% | validation 已达到；test 稳健性继续优化 |
| 合成条件缺失感知元组 | — | **81.87%** | 70% | 已达到 |
| 合成条件含 method 元组 | — | **75.31%** | 70% | 补充严格视图也达到 |

当前三源候选并集的 validation oracle recall 为 89.95%。五折 OOF 候选级 LambdaRank、RRF 与族路由把更多真实整套方案从第 11–100 位提升到前 10。当前主要消融如下：

| 前驱体方案 | Validation Top-10 |
|---|---:|
| 旧 GFlowNet/RRF + 元素笛卡尔积 | 68.62% |
| 同族深候选池 + 旧模型 + listwise 融合 | **69.63%** |
| OOF listwise CE 单模型 | 66.36% |
| OOF Top-K boundary 单模型 | 65.85% |
| OOF LambdaRank | 50.83% |
| 同族增强 GFlowNet beam | 57.14% |
| 候选多样性重排 | 未超过 69.63% |
| 零样本化学排序器（单模型） | 70.32% |
| 12 专家五折 OOF 候选堆叠 | 78.99% |
| 两个 OOF 堆叠器 + 原路由基线 RRF | 79.63% |
| 冻结族＋阴离子族局部路由 | 80.78% |
| 增加长尾族专家并限制 Top-1 不退化 | 80.88% |
| 加入两个组均衡长尾专家 | **80.97%** |

2026-07-15 新增了可处理训练未见前驱体的化学零样本排序器、候选内/候选—查询交叉注意力、候选级五折 OOF LambdaRank 堆叠，以及同时替换阳离子和阴离子的全元素族映射。12 专家 OOF 堆叠达到 78.99%，与另一 OOF 堆叠器及原路由基线融合后达到 79.63%；只对支持数不少于 10、且至少增加 1 条 Top-10 命中的族＋阴离子族切片启用冻结路由后达到 80.78%。随后加入长尾族专家和“每个路由不得损失 Top-1 命中”的约束，达到 80.88%；再加入两个按公式组频率均衡训练的长尾专家后，当前 validation Pareto 方案达到 **80.97%（1757/2170）**，Top-1 同时升至 39.77%。公式组宏平均 Top-10 为 74.80%。

冻结后执行的独立 test 暴露了显著分布偏移：test 中 LN 占 14.5%，而 validation 仅 2.8%；含训练未见前驱体的样本由 9.5% 上升至 16.0%。原可独立复现的精简 17 专家堆叠 test Top-10 为 56.04%。修复候选堆叠器中“同一公式组跨 OOF 折”的乐观偏差后，采用公式组完全隔离的 19 专家 LambdaRank 堆叠，冻结 test Top-10 首先提升至 56.78%。在同一 7911 种子、同一五折和同一 19 专家候选池下加入训练集拟合的 MatSciBERT 化学式表示后，validation OOF Top-10 从 68.48% 升到 71.75%（新增 71 条），冻结 test 升到 57.74%。再加入只由 validation 选择的 G01+G05、G02、G07、G12 四族多标签补救后，当前独立 test 最佳为 **57.91%（1332/2300）**，比原 17 专家增加 43 条、比旧 19 专家增加 26 条。它仍不能把 validation 的 80.97% 当作 test 成绩，生产开关继续关闭。

训练＋validation 最终拟合已按冻结轮数执行：20,297 条开发记录重新拟合缩放器和候选先验，test 仍保持公式组隔离。元素笛卡尔候选本身的 test Top-10 从约 45.13% 提升到 46.35%，但最终 chemistry-only 和 hybrid 专家的 Top-10 分别为 54.57% 和 54.91%；把最终拟合 chemistry-only 替换进固定 19 专家堆叠器后为 55.83%，均未超过 56.78%。该分支改善了若干 Top-1、Top-20 和 Top-50 指标，但没有改善目标 Top-10，故不替换当前独立最佳。

非参数最终拟合中，全元素族替换专家自身的 test Top-10 从 27.74% 提升至 35.74%，说明增加开发数据对同族类比检索有效；但替换固定堆叠器对应槽位后为 56.74%，比当前最佳少 1 条 Top-10 命中，仅改善 Top-20，故仍不替换主方案。

### 1.2 MatSciBERT 外部材料知识消融

本轮使用 [MatSciBERT 官方论文](https://www.nature.com/articles/s41524-022-00784-w) 和 [官方代码仓库](https://github.com/M3RG-IITD/MatSciBERT) 对应模型，远程下载后完全离线推理。缓存包含公共化学式提示和 target/precursor 角色提示两个 768 维视图；PCA 与 Ridge 只在 train 的可观察化学式和标签上拟合，validation/test 只做变换。受控容量消融中，16、32、64 维的 validation Top-10 分别为 70.41%、71.75%、71.20%，因此选择 32 维。

进一步用全部 18,127 条 train 记录微调 MatSciBERT 最后 4 层与 1,968 个前驱体标签表示，共 2,986 万个可训练参数。第 7 轮 validation 单标签 Recall@10 为 77.19%，但直接加入整套组合排序后 Top-10 为 70.88%，低于冻结表示的 71.75%；因此监督分数不替代主特征，只作为四个 validation 支持族的补救。最终四族路由 validation Top-10 为 72.12%，冻结 test 为 57.91%。

多标签组合生成器对每个目标枚举前 12 个标签的 1–4 元组合，共 793 个候选，validation 全候选 oracle 为 69.86%。作为第 20 专家后，19 专家并集 oracle 从 90.28% 升到 90.65%，但 Top-10 仍为 71.75%。在 validation 上继续给 G14/LN 路由虽升到 72.35%，冻结 test 却降到 57.26%，原因是 validation G14 主要为 SnS2，而 test G14 主要为单质碳；该路由已拒绝，不能依据已打开 test 反向修正规则。

### 1.3 独立 test 误差定位

作为候选覆盖审计基线，19 专家组隔离堆叠器在 2,300 条 test 中命中 1,306 条；另有 111 条真实方案位于第 11–20 位、321 条位于第 21–50 位、78 条位于第 51–100 位、119 条位于第 100 位以后，365 条未被候选并集生成。当前 MatSciBERT 四族路由把 Top-10 命中提高到 1,332 条，但不会改变底层候选 oracle 的定义。因此主要问题仍是“深候选向 Top-10 的组外排序”，G14 则仍是明显的覆盖例外。

LN 有 334 条，候选 oracle 为 95.51%，Top-10 仅 17.66%，属于排序问题；G14 有 209 条，候选 oracle 仅 13.40%，Top-10 为 1.91%，属于覆盖问题。进一步审计发现 validation G14 的 89 条中 84 条是 SnS2，而 test G14 的 209 条中 204 条是单质碳配方；test LN 的 334 条中 323 条是 CeO2。由于规范化学式必须整组隔离，C、CeO2、SnS2 这类巨型公式组不能同时出现在训练和评估，单一 split 的族内任务分布会发生大幅改变。

针对 LN/G14 已完成三类无泄漏消融：固定常见盐化学先验、公式组隔离的候选 LambdaRank、族专属 1,024 宽 listwise 网络。三者均未在 validation 路由上新增 Top-10 命中；族专属网络的 validation Top-10 为 56.38%、公式组宏平均为 64.64%，低于当前主模型在 LN 的 86.67% 和 G14 的 77.53%，因此没有把 test 用作正式模型选择。后续必须引入外部化学知识或整组交叉验证，不能依据已打开的 C/CeO2 标签手工调规则。

## 2. 全库划分

| 阶段 | 全库 | Train | Validation | Test | 特征族数/测试族数 |
|---|---:|---:|---:|---:|---:|
| Stage 2 前驱体 | 22,597 | 18,127 | 2,170 | 2,300 | 全库 256 / test 121 |
| Stage 3 工艺 | 24,631 | 19,788 | 2,422 | 2,421 | 全库 419 / test 200 |

Stage 2 原先看到的 2,978 条是 `gold_only/train` 高置信度子集，不是全库。新版本使用全部 22,597 条唯一记录；其中 solution-synthesis 15,565 条、solid-state 7,032 条。

主划分以规范化学式为 group，Stage 2 的规范化学式和 `material_id` 在 train/validation/test 间交集均为 0，Stage 3 的规范化学式交集也为 0。DOI 与化学式同时构造连通分量会产生约 17,000 条巨型分组，无法形成可用的 80/10/10 划分；因此本版优先阻断材料/化学式泄漏，Stage 2 DOI 重叠被明确审计为 train-val 276、train-test 220、val-test 79。

## 3. Stage 2 前驱体结果

公平消融使用相同的宽网络和训练参数，唯一差异是是否提供 24 维主阳离子族特征及 FiLM 调制。

| Split | 模型 | 严格整套命中率 | Samples-F1 | Jaccard |
|---|---|---:|---:|---:|
| Validation | 无族 768 | 21.52% | 34.12% | 30.45% |
| Validation | 族 FiLM 768 | **27.70%** | **37.40%** | **34.55%** |
| Test | 无族 768 | 14.17% | 26.26% | 22.55% |
| Test | 族 FiLM 768 | **20.35%** | **32.41%** | **28.98%** |

最佳 checkpoint 在 validation samples-F1 上选择，族模型为第 31 轮。相对严格同架构消融，test 变化为：

- 严格整套命中率：+6.17 个百分点；
- samples-F1：+6.15 个百分点；
- Jaccard：+6.43 个百分点。

冻结最佳策略后训练的大型候选 reranker 没有通过 validation top-1 验收：validation samples-F1 从 37.40% 降至 35.25%。因此 top-1 保留 greedy 策略。候选池仍有价值，排序后 exact-hit@5 为 validation 38.71%、test 29.39%，相对单一 greedy 候选分别增加约 11.01 和 9.04 个百分点，可作为 top-K 扩展而非 top-1 替换。

远程模型：

```text
/root/autodl-tmp/synthmind_family_routed_v1/runs/stage2_film_wide768_v1/best_model.pt
/root/autodl-tmp/synthmind_family_routed_v1/runs/stage2_film_wide768_frozen_rerank_v1/best_reranker.pt
```

## 4. Stage 3 工艺结果

Stage 3 使用化学检查后的线上一致输入 `predicted_precursor_set_chem_checked`。旧 schema 的 1,971 个前驱体 token 会漏掉 2,426 次有效输入；V1 从完整输入库扩展到 2,996 个 token，OOV 为 0。

### 4.1 LightGBM 严格消融

| Split | 模型 | 温度 MAE (°C) | 时间 MAE (h) | 气氛 Accuracy | 方法 Accuracy |
|---|---|---:|---:|---:|---:|
| Validation | 无族 | **204.41** | **30.02** | **91.65%** | 83.03% |
| Validation | 族条件 | 207.36 | 30.09 | 91.43% | **83.48%** |
| Test | 无族 | 228.54 | 30.52 | **90.76%** | 78.89% |
| Test | 族条件 | **227.44** | **30.28** | 90.40% | **80.55%** |

模型选择不查看 test：按 validation 主指标，温度、时间、气氛选无族头，反应方法选族条件头。这个组合在 test 上的结果为温度 MAE 228.54°C、时间 MAE 30.52h、气氛准确率 90.76%、方法准确率 80.55%。

### 4.2 GPU 多任务宽网络

GPU 搜索比较 512、768、1,024 三种宽度和 3–4 个残差块，同时学习温度/时间分位数、气氛和方法。validation 选出的族模型是 768 宽第 18 轮；无族模型是 1,024 宽第 6 轮。

| Test 模型 | 温度 MAE (°C) | 时间 MAE (h) | 气氛 Accuracy | 方法 Accuracy |
|---|---:|---:|---:|---:|
| GPU 无族 | 229.02 | **32.24** | **88.15%** | **80.50%** |
| GPU 族条件 | **228.20** | 38.94 | 85.90% | 80.30% |

族 GPU 模型只改善温度，其他任务退化；无族 GPU 的 validation 综合分数也略优。因此 GPU 模型不替换 LightGBM，但 checkpoint 和分位区间结果保留用于后续集成研究。

远程模型：

```text
/root/autodl-tmp/synthmind_family_routed_v1/runs/stage3_family_lgbm_v1/stage3_family_lgbm.joblib
/root/autodl-tmp/synthmind_family_routed_v1/runs/stage3_no_family_ablation_v1/stage3_family_lgbm.joblib
/root/autodl-tmp/synthmind_family_routed_v1/runs/stage3_gpu_multitask_family_v1/best_model.pt
/root/autodl-tmp/synthmind_family_routed_v1/runs/stage3_gpu_multitask_no_family_v1/best_model.pt
```

## 5. 代码与产物

核心代码：

- `synthmind/chemistry/families.py`：正式化学式解析和主阳离子族分配；
- `training/family/build_full_database_split.py`：Stage 2 全库划分、族特征和审计；
- `training/family/build_stage3_full_family_dataset.py`：Stage 3 全库数据与输入词表；
- `training/precursor/train_gflownet.py`：FiLM 族条件、最佳权重深拷贝修复、冻结策略 reranker；
- `training/family/train_stage3_family_lgbm.py`：Stage 3 LightGBM 严格消融；
- `training/family/train_stage3_gpu_multitask.py`：GPU 多任务宽网络搜索；
- `training/family/build_stage2_matscibert_embeddings.py`：双视角材料语言模型嵌入缓存；
- `training/family/train_stage2_matscibert_multilabel.py`：train-only MatSciBERT 多标签 GPU 微调；
- `training/family/evaluate_stage2_matscibert_set_generator.py`：多标签组合候选生成；
- `training/family/train_stage2_oof_candidate_stacker.py`：公式组隔离候选堆叠及 MatSciBERT 特征；
- `training/family/plot_family_accuracy.py`、`plot_stage3_family_accuracy.py`：总体和分族图表；
- `training/family/build_family_registry.py`：模型、数据和代码 SHA-256 注册表。

本地结果目录：`output/family_routed_v1/`。其中包含：

- `registry.json`：19 个关键产物及代码哈希；
- `data/stage2`、`data/stage3`：split manifest、schema 和统计；
- `metrics/`：所有最终模型与消融指标；
- `figures/stage2_film_wide768_v1/`：Stage 2 总体、分族、来源、支持度和学习曲线；
- `figures/stage3_lgbm_family_v1/`：最终 Stage 3 LightGBM 图；
- `figures/stage3_gpu_multitask_family_v1/`：GPU 模型消融图。

完整远程注册表：

```text
/root/autodl-tmp/synthmind_family_routed_v1/registry.json
```

## 6. 验证与限制

- 远程化学与候选模型回归测试共 23 项全部通过；
- `LiCl`、`NaCl`、`NaBr`、`Li2O`、`Na2S` 的等族断言已固化为测试；
- Stage 2/3 公式 group 跨 split 交集为 0；
- checkpoint 的最佳轮权重引用错误已修复为 `copy.deepcopy(state_dict)`；
- 当前结果为单一 split、单一主要随机种子，尚不能替代 3-seed 均值和置信区间；
- substitution-holdout、整族 OOD holdout 和完整端到端 Stage 3.5 路线排序尚未执行；
- Stage 3 使用的 `predicted_precursor_set_chem_checked` 需要在论文级发布前再次核对其逐行 OOF 生成链路；当前结果不能替代该 provenance 审计；
- DOI 不是本主 split 的硬隔离键，论文级报告必须同时披露 DOI 重叠或另建 DOI-disjoint 视图；
- V1 仍处于隔离版本，未改生产默认模型。
