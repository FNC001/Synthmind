# 同族元素路由训练方案 V1

状态：全库划分和合成条件 Top-K 已完成；前驱体严格 Top-10 当前 72.17%、目标 80%，高成本优化进行中；生产默认仍关闭  
分支：`codex/family-routed-training-v1`  
适用范围：Stage 2 前驱体预测、Stage 3 工艺条件预测、Stage 3.5 路线排序

## 1. 目标

给定一个新的晶体结构或目标化学式，先识别它所属的“主阳离子族”，再利用该族及其相邻族的历史数据增强前驱体和工艺预测。

例如：

- `LiCl`、`NaCl`、`NaBr`、`Li2O`、`Na2S` 全部属于 `G01` 碱金属主族；
- Cl、Br、O、S 的身份以及 1:1、2:1 化学计量不参与主分库；
- 阴离子与计量仍作为模型输入特征保留，用来学习族内工艺差异。

V1 不采用“每次预测时临时训练一个新模型”。训练在离线完成；在线推理只执行族识别、软路由、候选合并和置信度计算。

## 2. 核心建议

### 2.1 不做单一硬分库，采用分层软路由

远程现有 Stage 2 模型就绪全库共有 22,597 条不重复记录，而不是 2,978 条。旧 `gold_only` 的 2,978 条只是高置信度子集；新版本合并旧 train/validation/test 后重新划分全库，所有记录都参与训练或评估。

采用四层信息：

1. 全局层：所有训练数据，始终保留为回退模型；
2. 主阳离子族：只按金属/主要骨架阳离子的周期表族集合分库，不看阴离子和计量；
3. 族内化学层：阴离子、元素身份、化学计量、氧化态和前驱体试剂族只作为特征，不再拆库；
4. 近邻层：结构嵌入和组成特征相似的 Top-K 历史样本。

推理结果由全局模型、合格的主阳离子族专家、族内化学特征和近邻检索共同产生。任何局部模型都不能成为唯一预测来源。

### 2.2 同族是迁移先验，不是化学等价

同族元素可发生类似取代，但并不保证合成工艺相同。例如 O/S/Se 同属 16 族，但氧化物、硫化物和硒化物对气氛、温度与前驱体的要求可能明显不同。

因此必须：

- 保留元素原子序数、电负性、半径、氧化态候选等原始特征；
- 保留 CHGNet/结构描述符，避免只看化学式；
- 将族信息作为条件特征、先验和路由权重，不覆盖原始元素身份；
- 在 Stage 3 单独学习“元素身份残差”，不直接把同族样本视为完全相同。

### 2.3 区分两种“族”

代码和数据字段中不得混用：

- `target_cation_family`：目标材料的主阳离子周期族分库，本方案新增；
- `precursor_reagent_family`：oxide、carbonate、halide、nitrate 等前驱体化学类别，项目中已有类似字段。

## 3. 主阳离子族定义

### 3.1 元素映射

- 金属和主要骨架元素使用 IUPAC 1–18 族；
- 镧系单独记为 `LN`，锕系单独记为 `AN`，避免不同库对第 3 族定义不一致；
- H、O、S、Se、F、Cl、Br、I 等常见阴离子不参与有金属目标的主分库；
- 没有金属时使用类金属作为骨架族；仍没有类金属时才使用非金属回退族；
- 未知、占位或解析失败的元素记为 `UNK`，并触发全局回退；
- 阴离子不能从模型特征或元素覆盖检查中删除。这里只是不让它们决定主数据库分区。

元素映射必须由版本化配置控制，不得在多个脚本中重复手写。

### 3.2 化学式归一化

目标结构优先使用 POSCAR/CIF 中的组成；表格数据使用 `pymatgen.core.Composition` 解析。不得使用简单正则表达式承担正式分族，因为括号、非整数计量、掺杂和水合物会被错误解析。

处理顺序：

1. 解析元素及占比；
2. 合并重复元素；
3. 选择全部金属元素；没有金属时依次回退到类金属、全部元素；
4. 将所选元素映射为周期族，去重并排序；
5. 原始阴离子、计量、占位和氧化态另存为族内特征；
6. 按规范顺序生成稳定主族签名和哈希 ID。

建议输出字段：

```text
target_formula_canonical
target_elements
target_cation_elements
target_anion_elements
family_signature_primary
family_id_primary
family_routing_level
family_internal_stoichiometry
family_parse_status
family_schema_version
```

### 3.3 主阳离子签名

主阳离子签名是排序、去重后的金属周期族集合，不包含阴离子和计量。

```text
LiCl   -> G01
NaCl   -> G01
NaBr   -> G01
Li2O   -> G01
Na2S   -> G01
SrTiO3 -> G02+G04
LiFePO4 -> G01+G08
```

同一个主阳离子族中的材料进入同一数据分区。阴离子、化学计量和具体元素身份不得再次生成子数据库；它们只用于族内模型特征、检索距离和置信度。

### 3.4 结构子族

主阳离子族内部保留结构相似度，但不再建立互斥子数据库：

- 优先使用已有 prototype/space group；
- 缺失时对 CHGNet embedding 聚类；
- 聚类只能在 train split 上拟合，validation/test 只做分配，避免泄漏；
- 结构聚类只用于检索和权重计算，不作为必须命中的硬门槛。

## 4. 数据工程与防泄漏

### 4.1 先划分，再统计和采样

先合并旧 Stage 2 train/validation/test 的 22,597 条互不重复记录，再以规范化学式为分组键重新做约 80/10/10 分组划分，并按 `target_cation_family × source_dataset × synthesis_method` 尽量分层。实际得到 18,127/2,170/2,300 条，规范化学式和 `material_id` 的跨集合交集均为 0。

所有记录必须被分配到 train、validation 或 test；不能因为不是 gold 数据而被排除。`gold/relaxed` 只作为样本质量等级和 loss 权重，不再决定是否进入训练库。

所有标准化器、族频率、聚类器、标签先验和重采样权重只能使用新 train split 拟合。不得先在全库建族统计后再切分。

数据审计表明，若同时对 DOI 和化学式构造连通分量，会形成约 17,000 条的巨型连通块，无法获得可用的 80/10/10 比例。因此 V1 主评估优先阻断目标化学式/材料泄漏；DOI 跨集合交集作为明确限制保留（train-val 276、train-test 220、val-test 79）。若后续任务更强调论文级反应抽取泛化，应另建 DOI-disjoint 评估视图，不能声称当前主划分同时 DOI-disjoint。

### 4.2 三套评估视图

1. `full_database_group_disjoint`：新 80/10/10 主划分，用于训练、选择和最终报告；
2. `substitution_holdout`：在已见主阳离子族中留出具体元素和阴离子组合，直接测试 Li→Na、Cl→Br、Cl→O 这类族内迁移；
3. `family_disjoint`：整族留出，测试未知族回退能力，只作为 OOD 报告，不用于调参；
4. `historical_fixed_split`：只用于与旧报告对照，不再作为新模型的主训练划分。

测试集不得用于选择族阈值、融合权重、Top-K 或专家启用条件。

### 4.3 分族审计报告

训练前必须输出：

- 解析成功率、`UNK` 比例和失败样例；
- 每个 split 的族数量、样本量、目标覆盖率；
- 每族前驱体标签数、长尾标签数和 Stage 3 有效目标数；
- train/validation/test 重复 ID、重复化学式和重复 DOI；
- 头部族、尾部族、未见族以及元素替换对的分布；
- 硬切分后可训练专家的数量，作为最终阈值依据。

## 5. 模型方案

### 5.1 族路由器

路由器是确定性化学规则与相似度检索的组合，不先引入额外神经分类器。

输入：规范化组成、结构描述符、CHGNet embedding。  
输出：主阳离子族、族内化学描述、Top-K 相邻样本、支持样本数、OOD 分数和路由权重。

建议权重形式：

```text
w_global + w_primary_family + w_neighbor = 1
```

权重只在 validation 上校准。主阳离子族样本越少、结构距离越大或解析越不可靠，`w_global` 越高。

### 5.2 Stage 2：前驱体预测

保留当前生产 GFlowNet 作为历史基线和安全回退，同时使用新全库 split 重训新的全局 GFlowNet。新模型在完整验收前不得覆盖生产 checkpoint。

V1 优先实现以下低风险增强，而不是为每个族从零训练一个 GFlowNet：

1. 将族 one-hot/embedding 和族支持度加入输入特征；
2. 从新 train split 计算 `P(precursor | target_cation_family)`，形成平滑后的族标签先验；
3. 将全局 GFlowNet 候选、同主阳离子族检索候选、结构近邻候选和现有 fallback/baseline 候选取并集；
4. 在候选 reranker 中加入族匹配、结构距离、族内标签支持度和 OOD 特征；
5. 对达到门槛的头部主阳离子族训练共享骨干上的 adapter/专家头，并从全局权重初始化；
6. 全程保持全局 precursor vocabulary，不能为每族创建不兼容的标签索引。

按新定义，全库 Stage 2 只有 256 个主阳离子族；其中总样本不少于 200 的有 25 个，覆盖 77.8% 全库。首轮结果表明，直接增加 24 维二元族条件特征已能提高测试指标，因此 V1 先保留共享全局模型，不急于为长尾族复制独立模型；尾部族仍由全局族条件模型和近邻检索处理。

训练数据使用方式：

1. 全部 22,597 条记录训练全局模型；
2. `gold/relaxed` 作为可校准的样本权重，不删除 relaxed 数据；
3. 从全局 checkpoint 初始化头部族专家；
4. 各族保持统一 precursor vocabulary；
5. 族专家、全局模型和检索候选在固定候选预算下融合。

候选预算必须和基线相同，防止因为生成更多候选而获得不公平提升。

### 5.3 Stage 3：工艺预测

保留全局 LightGBM quantile ensemble 和 method experts，并增加族条件特征及可选族专家：

- 温度和时间：全局分位数模型 + 合格族专家的分位数融合；
- 气氛、时间桶、合成方法：全局分类概率 + 族内校准；
- 族样本不足或专家/全局分歧过大时，自动提高全局权重并降低置信度；
- 分位数输出需要单调修正，并报告区间覆盖率和 pinball loss。

Stage 3 训练必须同时看到：

- gold precursor set；
- 通过 out-of-fold Stage 2 生成的 predicted precursor set。

不能只使用 gold precursor set 训练后直接接受线上预测前驱体，否则会产生明显的训练—推理暴露偏差。

V1 实际使用 `predicted_precursor_set_chem_checked` 作为线上一致的条件输入。旧 schema 只覆盖 1,971 个 token，造成全库 2,426 次有效输入被静默丢弃；构建器现从完整输入数据库扩充到 2,996 个 token，OOV 次数降为 0。该词表只描述模型输入，不使用温度、时间、气氛或方法标签，因此不会产生目标泄漏。

### 5.4 Stage 3.5：最终路线排序

新增但不限于以下特征：

```text
family_support_train
family_support_condition_target
family_route_weight_global
family_route_weight_primary
family_neighbor_distance
family_ood_score
family_precursor_prior_score
family_stage2_global_expert_disagreement
family_stage3_global_expert_disagreement
family_parse_ok
```

最终排序器仍需接受元素覆盖、前驱体 QC、方法一致性和条件可信度特征，族分数不能覆盖已有安全门槛。

## 6. 训练执行顺序

### Phase 0：冻结基线

- 使用现有固定 split 重现 `docs/FULL_RETRAIN_ACCURACY_REPORT_20260710.md` 中可复现指标；
- 固定环境、随机种子、数据指纹、候选预算和生产 checkpoint；
- 新模型写入新的 `runs/family_routed_v1/`，不得覆盖历史模型。

### Phase 1：实现并测试族规范

- 新建单一的 chemistry family 模块；
- 添加 Li/Na、Cl/Br、多元素、同族混合占位、分数计量、无效式等单元测试；
- 建立版本化 family schema 和稳定签名快照测试。

### Phase 2：构建族索引和审计

- 合并全库并为 Stage 2、Stage 3、Stage 3.5 元数据添加相同 family 字段；
- 生成 `family_assignments_<split>.csv`、`family_stats.json` 和审计报告；
- 根据真实样本分布确定专家门槛，禁止直接沿用候选值。

### Phase 3：生成无泄漏评估划分

- 将旧 split 合并后重新生成全库约 80/10/10 的 canonical-formula/material-disjoint 主 split；
- DOI-disjoint 另建为补充视图；主 split 的 DOI 重叠必须在报告中披露；
- 另行生成 substitution-holdout 和 family-disjoint 诊断 manifest；
- 所有 manifest 保存输入文件 SHA-256、schema 版本和 seed。

### Phase 4：训练全局族条件基线

- 先只增加 family features，不启用局部专家；
- 分别训练 Stage 2、Stage 3；
- 验证族信息本身是否带来增益，若无增益则暂停专家扩展并检查数据定义。

### Phase 5：训练族先验、检索与头部专家

- Stage 2 训练平滑族先验、族检索和头部族 adapter；
- Stage 3 训练满足样本门槛的族专家；
- 在 validation 上校准 global/primary-family/neighbor 融合权重；
- 对每个专家保留独立指标和是否启用的决策记录。

### Phase 6：端到端接入

- 在结构特征之后新增 `assign_target_cation_family` 步骤；
- Stage 2 候选生成、检索、rerank 读取同一个 family assignment；
- Stage 3 conditioned table 透传 family 特征；
- manifest 记录路由结果、模型版本、回退原因和各分量权重。

### Phase 7：消融和验收

至少比较：

1. 当前生产基线；
2. 基线 + family features；
3. + family prior/retrieval；
4. + Stage 2 头部专家；
5. + Stage 3 族专家；
6. 完整软路由；
7. 硬分库对照实验，仅用于证明软路由是否必要。

### Phase 8：影子运行与发布

- 先对历史测试集和一批未参与训练的新结构影子运行；
- 不自动替换生产模型；
- 只有验收门槛全部通过后，更新默认配置和模型注册表；
- 保留一键关闭 family routing 的配置开关。

## 7. 指标和验收门槛

### 7.1 数据门槛

- 正常目标化学式解析成功率不低于 99.5%；
- train/validation/test 的重复 material/公式/DOI 审计必须为零或有明确豁免清单；
- 每个输出都能追溯到 family schema、split manifest 和数据指纹；
- 未知族、稀疏族必须成功回退，不能导致空候选或流水线失败。

### 7.2 Stage 2

报告 exact@1/3/5/10、samples-F1、Jaccard、candidate recall、候选唯一数和元素覆盖率。

发布要求：

- 全体 test 的主要指标不低于生产基线的统计容差；
- substitution-holdout 的 exact@K 或 samples-F1 有稳定提升；
- 头部族提升不能以尾部族或 OOD 大幅退化为代价；
- 候选预算一致，至少 3 个 seed 报告均值和标准差。

### 7.3 Stage 3

报告温度/时间 MAE、median AE、R²、阈值命中率、pinball loss、区间覆盖率，以及气氛/时间桶的 accuracy、macro-F1、校准误差。

至少按主阳离子族、head/tail/unseen、阴离子类别、合成方法和前驱体来源分层报告。

### 7.4 端到端

继续使用现有 strict/relaxed route top1/3/5/10 指标，同时增加：

- family-route 覆盖率；
- 全局回退率；
- OOD 性能；
- 路由失败率；
- 每样本耗时和模型内存。

默认生产选择器只有在完整端到端指标通过后才能修改。

## 8. 建议的代码改动边界

建议新增：

```text
synthmind/chemistry/families.py
synthmind/chemistry/family_schema_v1.yaml
synthmind/research/family_splits.py
training/family/build_family_index.py
training/family/train_stage2_family_router.py
training/family/train_stage3_family_experts.py
training/family/calibrate_family_blend.py
pipeline/core/04_assign_target_cation_family.py
tests/chemistry/test_families.py
tests/research/test_family_splits.py
configs/family_routed_v1.yaml
```

建议修改：

```text
pipeline/run_pipeline.py
pipeline/core/runner.py
pipeline/core/steps_stage2.py
pipeline/core/steps_stage3.py
pipeline/core/05c_build_stage3_conditioned_feature_table_infer.py
training/precursor/train_gflownet.py
training/conditions/train_lgbm_quantile_ensemble.py
training/conditions/train_lgbm_method_experts.py
training/ranking/train_route_reranker.py
```

不要在第一轮删除或重写当前生产路径。所有行为通过 `family_routing.enabled` 开关接入，默认先设为 `false`。

## 9. 模型与数据产物约定

```text
data/interim/family_routed_v1/
  family_schema_snapshot.yaml
  split_manifests/
  stage2/family_assignments_{train,val,test}.csv
  stage3/family_assignments_{train,val,test}.csv
  audit/family_distribution.json
  audit/family_distribution.md

runs/family_routed_v1/
  registry.json
  stage2/global/
  stage2/family_priors/
  stage2/experts/<family_id>/
  stage3/global/
  stage3/experts/<family_id>/
  calibration/
  metrics/
```

`registry.json` 必须记录模型路径、训练数据指纹、代码 commit、family schema、split、seed、指标、启用状态和回退模型。

## 10. 远程训练环境与隔离策略

已确认的远程资源：

```text
GPU: NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB
CPU: 25 cores
RAM: 120 GB
Python: 3.12.3
PyTorch: 2.8.0+cu128
高速数据盘可用空间: 约 879 GB
文件存储可用空间: 约 134 GB
```

现有运行副本位于：

```text
/root/autodl-tmp/synthmind_autorun_20260613/synthmind
```

该目录包含约 7.3 GB 数据、4.1 GB 模型和 156 GB 输出，但没有 `.git`，不能作为代码版本源。采用以下隔离规则：

1. 本地 `codex/family-routed-training-v1` 是唯一代码源；
2. 远程新建独立目录 `/root/autodl-tmp/synthmind_family_routed_v1/`，不得原地修改现有运行副本；
3. 只读复用现有大数据；新中间数据、模型和日志全部写入新目录；
4. 阶段验收后的模型、registry、审计报告和 split manifest 再复制到文件存储长期保存；
5. 远程执行记录本地 commit SHA、同步时间和远程代码 SHA-256；
6. 长训练使用 `screen`/`tmux` 或等价后台任务，日志按 Phase 分文件；
7. SSH 凭据不得进入仓库、shell 脚本、配置、日志、命令历史或模型 registry。

远程已有的 `stage2_family_dataset` 和 `family_predictor` 指 oxide、carbonate、halide 等 `precursor_reagent_family`，不是本方案的 `target_cation_family`。两者应复用为互补特征，但不得共用字段名或模型目录。

### 10.1 首轮只读数据审计结果

以下结果由远程现有全库、pymatgen 公式解析和新的主阳离子族规则得到，未写入或修改远程文件：

| 数据集 | 合并后总行数 | 主阳离子族数 | 主要来源 |
|---|---:|---:|---|
| Stage 2 全库 | 22,597 | 256 | solution 15,565；solid-state 7,032 |
| Stage 3 core methods v5 全库 | 24,631 | 419 | solution 11,166；solid-state 11,102；melt-arc 2,363 |

主阳离子族覆盖（在重新划分前按全库总量统计）：

| 数据集 | 每族最少总样本 | 合格族数 | 全库覆盖率 |
|---|---:|---:|---:|
| Stage 2 | 1,000 | 6 | 41.2% |
| Stage 2 | 500 | 12 | 58.9% |
| Stage 2 | 200 | 25 | 77.8% |
| Stage 2 | 100 | 39 | 86.3% |
| Stage 3 | 1,000 | 4 | 23.5% |
| Stage 3 | 500 | 10 | 42.2% |
| Stage 3 | 200 | 30 | 70.1% |
| Stage 3 | 100 | 51 | 82.2% |

Stage 2 的 `G01` 主阳离子族现有 317 条记录，已经确认其中包含 `LiCl`、`Li2S`、磷酸盐、硅酸盐等不同阴离子和计量体系。它们进入同一个主分库，不再因阴离子或计量拆分。

旧训练方式的数字差异已经查明：

- `gold_only/train`：2,978 条；
- `relaxed_only/train`：21,333 条，其中 solution-synthesis 15,539 条；
- 旧 validation/test：635/629 条；
- 旧 relaxed train、validation、test 三者 ID 无重叠，合并后为 22,597 条；
- `gold_only` 的 2,978 条全部包含在旧 relaxed train 中，不能再作为额外样本重复加入。

旧 validation/test 几乎都是 solid-state，仅分别含 11/15 条 solution，无法代表 15,565 条 solution 全库。因此新版本必须合并后按来源、方法和主阳离子族重新分组划分，不能直接沿用旧固定 split。

远程还已有 `stage3_condition_dataset_predprec_oof_v4_20260612`，Phase 5 应优先验证并复用该 OOF predicted-precursor 数据，而不是重新生成一套不兼容格式。

## 11. 可直接交给 Codex 的主任务说明

> 在 `codex/family-routed-training-v1` 分支维护 `docs/FAMILY_ROUTED_TRAINING_PLAN_V1.md` 和结果报告。Stage 2 使用 22,597 条完整唯一记录，按 canonical formula/material ID 阻断泄漏，并按主阳离子族、source_dataset、synthesis_method 做约 80/10/10 分层分组划分；DOI 重叠单独审计。主分族只使用金属/骨架阳离子的周期表族集合，因此 LiCl、NaCl、NaBr、Li2O、Na2S 必须全部得到 `G01`。当前固定 validation 口径的 Stage 2 严格整套 Top-10 为 72.17%，由同族深候选池、化学式排序器和显式目标—候选化学关系排序器融合得到，80% 目标尚未完成；Stage 3 按任务头在族条件和无族 LightGBM 间选择，条件元组 Top-10 为 81.87%。任何后续优化必须在相同 split 的严格消融上比较，并只用 validation 选择；生产 feature flag 保持关闭，直到前驱体 Top-10 达标并完成 3-seed、substitution-holdout、family-OOD 和端到端路线排序验收。

## 12. 当前已知限制

- 公开仓库和本机不包含训练数据及 checkpoint；所需 Stage 2/3/3.5 artifacts 已在远程运行副本确认存在；
- 当前正式数据构建脚本并不完整地保留在公开树中，实施时需要先确认私有数据生产端能输出 formula/material_id/DOI/split 等连接字段；
- Stage 2 全库规模足以支持约 25 个样本数不少于 200 的候选主族专家，但实际门槛要在新 split 后重算；
- 同族替换的科学有效性必须由 substitution-holdout 结果证明，不能只看普通随机或固定测试集提升。
