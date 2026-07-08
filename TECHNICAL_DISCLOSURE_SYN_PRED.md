# SynPred 材料合成路线预测技术交底书

## 一、数据库规模介绍

本技术面向无机材料合成路线预测任务，输入为目标材料的晶体结构、化学组成及可选反应方式信息，输出包括推荐前驱体集合以及对应的合成工艺条件。系统使用结构化材料合成数据库作为训练基础，数据库中每条样本至少包含目标材料组成、材料结构标识、前驱体集合、反应方式，以及温度、时间、气氛、溶剂等合成条件字段中的一项或多项。

当前用于公平评估的主数据库采用按反应方式分层的 train / validation / test 拆分方式，总样本数为 36,530 条，其中训练集 29,208 条，验证集 3,658 条，测试集 3,664 条。该拆分方式避免只在单一固相数据上评估模型，能够更真实地反映模型在不同合成路线上的泛化能力。

| 数据集划分 | 样本数 | 说明 |
|---|---:|---|
| train | 29,208 | 用于 Stage2 前驱体预测和 Stage3 工艺预测训练 |
| validation | 3,658 | 用于模型早停、参数选择和候选排序校准 |
| test | 3,664 | 用于最终 method-stratified 公平测试 |
| 合计 | 36,530 | 覆盖多类反应方式的结构化合成数据 |

数据库覆盖 11 类主要反应方式，包括固相反应、溶液法、熔融合金/电弧法、水热/溶剂热、沉淀法、熔盐法、机械化学、溶胶凝胶、燃烧法、热分解以及其他类别。主测试集中各反应方式分布如下：

| 反应方式 | 测试集样本数 |
|---|---:|
| solid_state | 1,142 |
| solution | 1,102 |
| other | 737 |
| melt_arc | 224 |
| hydro_solvothermal | 191 |
| precipitation | 128 |
| flux_molten_salt | 68 |
| thermal_decomposition | 32 |
| mechanochemical | 28 |
| sol_gel | 7 |
| combustion | 5 |

在前驱体标签方面，原始数据库中存在多种同义写法、错误水合物标记、相态后缀和溶液后缀。经规范化处理后，主版本中前驱体标签由 4,520 个原始标签合并为 4,440 个 canonical 标签；进一步修正后得到 4,414 个 canonical 标签。系统还构建了前驱体 ontology，对 4,440 个 canonical 前驱体解析元素组成、阴离子类型、前驱体族以及是否为水合物、有机物、卤化物、碳酸盐、硝酸盐等化学属性。

| 前驱体数据库项目 | 数量 |
|---|---:|
| 原始前驱体标签 | 4,520 |
| canonical v2 前驱体标签 | 4,440 |
| canonical v4 前驱体标签 | 4,414 |
| ontology 前驱体标签 | 4,440 |
| ontology 解析失败标签 | 261 |
| family 类别数 | 13 |
| 元素类别数 | 107 |

核心反应方式子集进一步聚焦于固相反应、溶液法和熔融合金/电弧法，样本量为 24,631 条，其中训练集 19,722 条，验证集 2,441 条，测试集 2,468 条。该子集用于评估在样本规模较大、机理更清晰的主流路线上的预测能力。

| 核心子集划分 | 样本数 | 反应方式分布 |
|---|---:|---|
| train | 19,722 | solution 8,940；solid_state 8,899；melt_arc 1,883 |
| validation | 2,441 | solution 1,124；solid_state 1,061；melt_arc 256 |
| test | 2,468 | solid_state 1,142；solution 1,102；melt_arc 224 |

Stage3 工艺预测数据与 Stage2 主数据对齐，采用相同的 method-stratified 拆分。Stage3 连续变量包括合成温度和合成时间，离散变量包括气氛、溶剂和合成类型。为避免单位混乱，温度统一为摄氏度，时间统一为小时，原始文本中的天、小时、分钟等不同单位被转换到统一尺度。

## 二、前驱体合成算法介绍

前驱体预测任务定义为：给定目标材料结构、目标组成及反应方式，预测一组能够合成该目标材料的前驱体集合。该任务不同于普通单标签分类，因为真实输出是一个无序集合，而且同一目标材料可能存在多条可行合成路线。为此，本技术采用“标签规范化 + 候选生成 + 化学约束 + 集合排序”的组合式算法。

首先，系统对原始前驱体标签进行 canonical normalization。规范化规则包括统一水合物符号、修正 H20/H2O 错写、去除 aq/s/l/g/powder/solution/anhydrous 等无意义后缀，并将常见别名映射到统一化学式，例如将 aluminum nitrate nonahydrate 归并为 Al(NO3)3·9H2O，将 lithium carbonate 归并为 Li2CO3。系统使用 pymatgen Composition 和自定义解析逻辑共同校验化学式，生成 alias report，保留原始标签到 canonical 标签的映射关系。

其次，系统构建 precursor ontology。每个前驱体被解析为元素集合、可供给目标元素、前驱体族、阴离子族以及化学属性。前驱体族包括 oxide、carbonate、nitrate、hydroxide、acetate、sulfate、phosphate、halide、elemental、organic、other_salt、unknown 等。该 ontology 将具体化学式预测拆解为“元素来源”和“前驱体族”两个更稳定的子问题，从而缓解闭集标签分类无法处理未见前驱体的问题。

然后，系统训练 element-source / precursor-family predictor。对于每条训练样本，真实前驱体集合被转换为元素级监督标签，例如目标材料 LiMnO2 且真实前驱体为 Li2CO3 与 MnO2，则系统学习 Li 对应 carbonate family，Mn 对应 oxide family，O 可由前驱体或气氛共同提供。该 family predictor 在测试集上达到 per-element family top1 exact 73.44%、top1 recall 76.47%、top3 recall 94.03%，为后续候选生成提供化学先验。

在候选生成阶段，系统融合三类来源：

1. 闭集模型候选：由 MLP 或多标签分类模型输出训练集中出现过的前驱体概率。
2. 检索候选：从训练集中检索组成、结构描述符、反应方式相似的样本，复用其历史前驱体集合。
3. 开放式化学候选：根据目标元素和 family predictor 输出，按常见化学模板生成氧化物、碳酸盐、硝酸盐、氢氧化物、乙酸盐、硫酸盐、磷酸盐、卤化物和单质等候选。

候选集合通过 beam search 组合。组合时要求候选前驱体尽量覆盖目标元素，同时惩罚缺失目标元素、引入无关元素、前驱体数量过多或 family 与预测不一致等情况。候选集合得分综合 MLP 概率、元素覆盖率、family 概率、反应方式模板先验、前驱体共现先验、set-size predictor 以及检索相似度。

最后，系统使用验证集进行 score calibration 和 set-level ranking。排序器不是逐个前驱体打分，而是对整套候选前驱体集合打分。特征包括候选集合大小、预测集合大小误差、元素覆盖率、缺失元素数量、额外元素数量、family 匹配分数、前驱体共现分数、反应方式先验、候选来源、是否为 open-vocab 候选等。该设计直接优化“整套前驱体集合是否正确”，比单纯的逐标签 BCE 分类更贴合实际合成路线预测任务。

## 三、给定条件推荐方法

工艺条件推荐任务定义为：在给定目标材料结构、目标组成、反应方式以及候选前驱体集合的情况下，预测对应的合成条件，包括温度、时间、气氛、溶剂和合成类型等。由于不同反应方式的温度区间、时间尺度和气氛偏好差异较大，本技术采用按反应方式建模的专家模型框架。

首先，系统对工艺字段进行单位规范化。温度统一转换为摄氏度，时间统一转换为小时；原始数据中的分钟、小时、天等表达被统一映射到同一数值尺度。对于气氛、溶剂、合成类型等离散变量，系统进行标准化清洗和类别编码，避免同一含义由于大小写、缩写或文本噪声被视为不同类别。

其次，系统构建 Stage3 条件预测特征。输入特征包括目标材料组成描述符、结构描述符、反应方式、前驱体集合编码以及前驱体 family 信息。连续变量温度和时间使用回归模型预测，离散变量气氛和溶剂使用分类模型预测。当前主模型使用 LightGBM method experts，即按反应方式分别训练或加权训练专家模型，使固相、水热、溶液、熔盐、熔融等路线能够学习各自的条件分布。

为提高候选覆盖率，系统还构建 topK 条件候选推荐机制。除模型预测点估计外，系统引入反应方式模板、相似样本检索和候选条件重排。每个前驱体集合可对应多个候选工艺条件，例如不同温度、时间和气氛组合。系统对条件候选计算温度误差、时间误差和气氛匹配，并将其与 Stage2 前驱体候选一起组成完整合成路线。

完整路线评价采用 strict 与 relaxed 两级标准：

| 标准 | 定义 |
|---|---|
| strict condition | 温度误差 ≤100 °C，时间误差 ≤24 h，气氛正确 |
| relaxed condition | 温度误差 ≤200 °C，时间误差 ≤48 h，气氛正确 |
| strict route | 前驱体集合 exact，同时满足 strict condition |
| relaxed route | 前驱体 Jaccard ≥0.5，同时满足 relaxed condition |

系统还包含 Stage35 联合排序层。该层将 Stage2 生成的前驱体候选和 Stage3 生成的条件候选拼接为完整路线候选，并使用路线级特征进行 reranking。路线级特征包括前驱体集合分数、条件分数、温度/时间预测置信度、气氛匹配概率、反应方式先验、前驱体与工艺的共现先验等。Stage35 的目标是从多个“前驱体-工艺条件”组合中选择最可能的一条完整合成路线。

## 四、结果总结

在 method-stratified 公平测试集上，Stage2 前驱体预测从 v2 baseline 到 v5 final 有稳定提升。v5 final 的全方法测试结果为 top1 exact 39.47%、top10 exact 63.35%、top200 exact 77.02%、top500 exact 80.24%。其中 top500 exact 表示真实前驱体集合是否出现在前 500 个候选集合内，反映候选池覆盖上限；top1 exact 表示排序第一的候选集合是否完全正确，反映最终推荐精度。

| Stage2 版本 | 测试集 | top1 exact | top10 exact | top200 exact | top500 exact |
|---|---|---:|---:|---:|---:|
| v2 baseline | method-stratified | 33.95% | 60.81% | 70.41% | 73.06% |
| v3 ontology/open-vocab | method-stratified | 35.94% | 60.04% | 73.42% | 76.12% |
| v4 OOV calibration | method-stratified | 34.99% | 58.57% | 73.36% | 76.83% |
| v5 template/calibration final | method-stratified | 39.47% | 63.35% | 77.02% | 80.24% |
| core-method final | solid_state / solution / melt_arc | 46.15% | 69.17% | 81.69% | 85.33% |

在核心反应方式子集上，模型表现更稳定，说明固相反应、溶液法和熔融合金/电弧法具有较强的可学习规律。核心子集中 solid_state 的 top1 exact 为 50.44%，melt_arc 的 top1 exact 达到 76.79%，表明在机理清晰且候选空间较规范的路线中，前驱体推荐已具有较高实用性。

| 核心反应方式 | n | top1 exact | top10 exact | top200 exact | top500 exact |
|---|---:|---:|---:|---:|---:|
| solid_state | 1,142 | 50.44% | 67.16% | 80.39% | 85.11% |
| solution | 1,102 | 35.48% | 67.97% | 80.76% | 83.94% |
| melt_arc | 224 | 76.79% | 85.27% | 92.86% | 93.30% |

Stage3 工艺预测方面，当前 method-stratified 主模型采用 LightGBM method experts。测试集温度 MAE 为 200.5 °C，时间 MAE 为 34.2 h，气氛 accuracy 为 54.2%。在 mixed split 上，method experts 的温度 MAE 为 186.6 °C，时间 MAE 为 21.1 h，气氛 accuracy 为 72.6%，说明当训练和测试分布更接近时，工艺预测可取得更低误差。

| Stage3 版本 | 测试设定 | 温度 MAE | 时间 MAE | 气氛 accuracy |
|---|---|---:|---:|---:|
| LGBM direct mixed | mixed split | 203.6 °C | 21.5 h | 72.9% |
| LGBM method experts mixed | mixed split | 186.6 °C | 21.1 h | 72.6% |
| LGBM direct method-stratified | method-stratified | 205.6 °C | 34.5 h | 52.9% |
| LGBM method experts method-stratified | method-stratified | 200.5 °C | 34.2 h | 54.2% |

若按 strict / relaxed 条件命中率统计，当前 Stage3 top1 条件严格成功率约为 9.09%，宽松成功率约为 14.74%。在 topK condition MoE + retrieval 候选池中，oracle strict condition 可达 30.05%，oracle relaxed condition 可达 36.98%，说明候选池中已经包含较多合理工艺条件，但排序器仍需进一步优化以将正确条件排到第一。

| Stage3 条件推荐版本 | top1 strict condition | top1 relaxed condition | oracle strict condition | oracle relaxed condition |
|---|---:|---:|---:|---:|
| top10 method-stratified | 9.01% | 14.63% | 11.46% | 17.36% |
| top100 canonical | 9.03% | 14.66% | 12.55% | 18.31% |
| topK condition MoE + retrieval | 5.84% | 10.21% | 30.05% | 36.98% |
| Stage35 top10 rerank 后 | 9.09% | 14.74% | 11.46% | 17.36% |

完整路线成功率同时受到前驱体集合和工艺条件两部分影响，因此数值显著低于单独的前驱体预测或条件预测。当前 Stage35 rerank 后，在 method-stratified 测试集上 top1 strict route 为 3.49%，top1 relaxed route 为 6.85%。在 old/core 测试集上，top1 strict route 为 8.18%，top1 relaxed route 为 16.73%。

| 完整路线版本 | top1 strict route | top1 relaxed route | oracle strict route | oracle relaxed route |
|---|---:|---:|---:|---:|
| top10 method-stratified | 3.11% | 6.39% | 5.90% | 11.35% |
| top10 canonical enhanced | 3.30% | 6.50% | 6.25% | 12.01% |
| top100 canonical | 3.38% | 6.44% | 6.36% | 14.68% |
| Stage35 top10 rerank 后 | 3.49% | 6.85% | 5.90% | 11.35% |
| old/core test Stage35 | 8.18% | 16.73% | 10.97% | 21.75% |

综合来看，本技术已经在前驱体集合预测阶段形成较完整的算法闭环：通过标签规范化、前驱体 ontology、元素来源 family 预测、开放式化学候选生成、反应方式模板和集合级排序，使全方法 top500 exact 从 73.06% 提升到 80.24%，核心反应方式 top500 exact 达到 85.33%。Stage3 工艺推荐当前的主要瓶颈在于工艺条件本身的噪声、温度和时间的多模态分布、气氛标签不稳定以及长尾反应方式样本不足。下一步提升重点应放在 Stage3 条件候选排序、多模态条件生成、工艺文本信息引入以及高频反应方式模板清洗上。
