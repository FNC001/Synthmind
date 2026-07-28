# Synthmind GNoME 冻结模型批量推理说明

## 1. 用途

`synthmind_gnome_frozen_adapter.py` 用冻结发布版对 CIF、VASP 或 POSCAR
结构执行批量推理，输出：

1. 前驱体候选集合；
2. 合成方法；
3. 温度分布；
4. 时间分布；
5. 气氛分布；
6. 可合成性排序代理分数；
7. 完整模型、输入、回退和质量标志。

程序不会修改冻结发布包。全部新结果写到独立输出目录，并按块保存，可使用
`--resume` 断点续跑。

## 2. 实际使用的模型

### Stage2 前驱体

- `stage2_factorized_h2048_b6_top20_g1_s9140`
- `stage2_factorized_h1536_b4_top20_g2_s9151`
- `stage2_factorized_h2048_b6_top20_g1_s9152`

三个冻结专家通过 chemistry-aware reciprocal-rank fusion 合并。候选必须优先
覆盖目标结构中除 H/O 外的必需元素，并惩罚额外的阳离子类元素。

冻结版 s9161 是验证集 miss gate；它依赖未作为新结构部署接口发布的历史基础
候选栈和 MatSciBERT PCA 特征，因此本适配器不会把它错误标成在线模型。

当三个专家在 GNoME 新化学空间中不能给出元素完整路线时，程序使用冻结训练库
中真实出现过的前驱体及其频率构造 composition training prior fallback。若冻结
词表完全没有某个目标元素的前驱体，则以该元素单质补全，并标记
`external_elemental_completion` 和 Stage3 OOV；不会隐藏该降级。

### Stage3 合成条件

- Normalizing Flow：64 个样本；
- CVAE：64 个样本；
- Conditional Diffusion：64 个样本。

三模型等权合并。温度和时间的交付点值取 192 个样本的中位数；P25–P75 是主要
稳健区间，P10–P90 保留在全量表中用于观察更宽分布。方法和气氛取样本众数，并
保存完整类别频率。

## 3. 排名分数的含义

`synthesizability_rank_score` 只用于当前 92,310 个结构内部排序。它由以下信息
组成：

- Stage2 三专家一致性与 RRF 排名；
- 目标阳离子覆盖；
- 目标必需元素覆盖；
- Stage3 三模型一致性与分布紧致度；
- 到训练特征域的接近程度；
- Stage2/Stage3 前驱体词表映射；
- 结构规模的弱复杂度项；
- OOV、额外元素、CHGNet 降级和不完整覆盖惩罚。

该值不是经过实验标定的合成成功概率，也不能解释成“0.8 就有 80% 成功率”。
最终实验前仍需做反应配平、热力学/动力学计算、危化品审查和人工路线复核。

## 4. 主要输出

- `all_predictions.csv.gz`：每个输入结构一行的完整结果；
- `top100_most_synthesizable.csv`：排序代理分数最高的 100 条；
- `top100_most_synthesizable.md`：便于阅读的前 100 列表；
- `model_manifest.json`：模型路径、权重 SHA-256、策略和限制；
- `summary.json`：结构数、成功/失败数、用时和输出路径；
- `chunks/chunk_*.csv.gz`：断点续跑分块。

Top100 默认只从严格质量合格池中选取：必需元素覆盖率为 1、Stage3 前驱体
映射率为 1、CHGNet 成功，且不存在外部元素补全、Stage3 OOV 或元素覆盖不完整
标志。只有严格合格池不足 100 条时才使用全部有效记录，并在
`top100_selection_policy` 中明确标注。

## 5. 重点字段

| 字段 | 含义 |
|---|---|
| `sample_id` | 与本地 CIF 文件同名的 GNoME ID |
| `input_sha256` | 远程 VASP 输入内容哈希 |
| `formula` | 从结构重新解析的约化化学式 |
| `predicted_precursors` | 最终推荐前驱体集合 |
| `top10_precursor_sets` | 前 10 候选及来源、覆盖和专家排名 |
| `precursor_candidate_source` | 模型、训练库先验或元素补全来源 |
| `precursor_target_required_element_coverage` | 除 H/O 外目标元素覆盖率 |
| `stage3_precursor_mapping_fraction` | 前驱体映射到 Stage3 词表的比例 |
| `pred_reaction_method` | Stage3 众数合成方法 |
| `pred_temperature_c_median` | 192 个样本温度中位数 |
| `pred_temperature_c_p25/p75` | 温度稳健四分位区间 |
| `pred_time_h_median` | 192 个样本时间中位数 |
| `pred_time_h_p25/p75` | 时间稳健四分位区间 |
| `pred_atmosphere` | Stage3 众数气氛 |
| `stage3_ensemble_consensus` | 三模型分布一致性代理值 |
| `synthesizability_rank_score` | 未标定、仅用于排序的代理分数 |
| `quality_flags` | OOV、回退、补全或异常标记 |

## 6. 复现命令

```bash
python scripts/infer/synthmind_gnome_frozen_adapter.py \
  --input-dir /path/to/GNoME_structures \
  --output-dir /path/to/new_output \
  --data-root /path/to/authorized_synthmind_artifacts \
  --source-root /path/to/Synthmind \
  --batch-size 64 \
  --chgnet-batch-size 32 \
  --stage3-row-batch-size 16 \
  --stage3-samples-per-model 64 \
  --stage2-candidates-per-expert 100 \
  --top-precursor-sets 10 \
  --seed 20260724 \
  --resume
```

## 7. 已执行的上线前验证

- 8 条端到端冒烟测试；
- 256 条跨完整 ID 范围的均匀抽样测试；
- 32 条完整 64+64+64 采样双运行稳定性测试；
- 双运行解压结果 SHA-256 完全一致；
- 双运行逐单元格完全一致；
- 双运行数值最大绝对差为 0；
- 本地 CIF 与远程 VASP 的 92,310 个去扩展名 ID 清单 SHA-256 一致；
- 双方重复 ID 均为 0。
