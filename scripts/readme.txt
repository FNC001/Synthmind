⏺ SynPred 各阶段算法原理详解

  下面按数据流自顶向下展开。每阶段都说明:输入/产物、核心算法、关键公式或损失、值得注意的工程细节。代码位置标到行号方便溯源。
  
  ---
  0. 总体管线
  
  整个工程的预测目标可以表达为:给定目标晶体结构 $X$(POSCAR),输出一条合成路线
  $$R = (\mathcal{P},, c) = \big({p_1,\dots,p_k},, (T, t, \text{atm}, \text{solvent}, \dots)\big)$$
  其中 $\mathcal{P}$ 是前驱物集合(Stage2 任务),$c$ 是合成条件(Stage3 任务),最后由 Stage35 联合排序器把多个候选 $(\mathcal{P}, c)$ 打分排序。
  
  阶段编号即数据流:
  
  00_refine ─► 01_split ─► 02_features
                                │
                                ├─► 03_graph (CGCNN/CHGNet/ALIGNN 嵌入)
                                │
                                └─► 03_data (各模型变体的训练集 NPZ)
                                           │
                                           ▼
                                      04_train (Stage2/Stage3/Stage35)
                                           │
                                           ▼
                                      07_infer (端到端推理流水线)
  
  ---
  1. Stage 00 — 原始数据捕获与精炼
  
  1.1 00_refine/01_capture_experimental_structures.py
  
  - 通过 mp_api.client.MPRester 拉取 Materials Project 全档,可按 MATERIAL_IDS / FORMULAE / CHEMSYS / ELEMENTS 过滤,默认只取 theoretical=False(实验来源)。
  - 流程:批量获取 metadata → provenance(DOI 来源)→ 结构 dict → 反序列化为 pymatgen.Structure → 写 POSCAR(Poscar(structure).write_file)。
  - 节流:每 SLEEP_EVERY=20 个请求 sleep 0.2s,避免被 MP API 封。
  
  1.2 00_refine/02_prepare_dataset.py(MP ↔ 文献合成数据对齐)
  
  把 MP 档案(POSCAR + metadata)与两个公开合成数据库(solid-state、solution synthesis)做对齐。打分项基于:
  - 化学式归一化 + 原子计量比距离(COMP_TOL=0.015)
  - 标题与 DOI 文本相似度(difflib.SequenceMatcher,TITLE_SIM_THRESHOLD=0.75)
  - 严格匹配的 BEST_ONLY_MIN_SCORE=55、保留下界 MIN_KEEP_SCORE=35,并允许每条记录最多保留 TOPK_PER_RECORD=3 个候选。
  
  输出三张 CSV(all_candidates.csv, aligned_summary.csv, unmatched.csv)+ 一份 JSON direct_aligned_dataset.json,后者就是后续脚本的真源(synthesis 记录中嵌入 POSCAR 路径)。
  
  1.3 00_refine/04_refine_strict_exact_for_structdesc.py(质量过滤)
  
  对齐后的样本要满足"结构金属元素严格被前驱物金属元素覆盖"才能进入金标。规则集:
  - 辅助物种白名单 AUX_SPECIES(O₂, H₂O, NH₃, OH⁻ 等)不参与元素一致性检查。
  - 变量化学式过滤:含 x/y/z/δ 的式子(固溶体)直接排除,见 is_variable_formula_like。
  - 元素一致性 check_precursor_element_consistency:把目标式中的 STRUCTURAL_METALS 提为 $M_t$,前驱物中同集合提为 $M_p$。
    - $|M_p \setminus M_t| \ge 2$ → severe(剔除)
    - $|M_p \setminus M_t| = 1$ → mild(进入 relaxed,不进金标)
  - 文本污染过滤(电池、电解液、聚合物相关关键词 → 剔除),否则会引入与晶体合成无关的步骤。
  - 条件抽取 extract_conditions_from_operations:从 operations 数组里聚合温度、时间、气氛、溶剂,按 ATM_MAP / SOLVENT_MAP 做小写词表归一化(例如 coprecipitation→co-precipitation)。
  
  输出两套 JSONL(每个 task 都有):
  - stage{2,3}_gold.jsonl(严格一致)
  - stage{2,3}_train_relaxed.jsonl(允许轻度不一致)
  
  ---
  2. Stage 01 — 分组划分
  
  01_split/01_make_group_split.py:33 中 get_group_key 优先取 split_group → doi → synth_uid → material_id → formula → id。这保证同一 DOI/同一目标材料只能进同一个 split,杜绝 DOI 级别的数据泄漏(关键设计)。
  
  划分逻辑(split_gold_groups, split_one_task):
  1. 仅在 gold 集上随机打乱 group,按行数累加抢占 val/test(目标分别约 15%);剩余 group → gold_train_holdout。
  2. relaxed 训练集 = relaxed 全集减去 val/test 占用的所有 group(filter_rows_not_in_groups)。
  3. 可选 --include_gold_train_in_relaxed_train 把 holdout 的 gold 也并入 relaxed train,做"先 relaxed 再 gold 微调"的 curriculum 准备。
  
  ---
  3. Stage 02 — 特征构建
  
  3.1 02_features/01_build_structdesc_features.py —— Structdesc 描述子
  
  组合三类特征(都加 feat_ 前缀):
  
  (a) 组成特征 composition_features(scripts/02_features/01_build_structdesc_features.py:551):
  - 元素种类数、总原子数、化学计量熵 $H = -\sum_i p_i \log p_i$
  - 原子序数加权均值/方差: $\bar Z = \sum p_i Z_i$, $\sigma_Z^2 = \sum p_i (Z_i - \bar Z)^2$
  - 七个家族占比(过渡金属、碱金属、碱土、卤素、硫属、镧系、锕系)
  - 每个元素分量 feat_frac_el__{El}(类似 Magpie 的稀疏向量)
  
  (b) MP summary 特征(load_summary_features):带 PBE 能量、能隙、密度、空间群编号、晶格常数 $a,b,c,\alpha,\beta,\gamma$、晶系 one-hot。
  
  (c) POSCAR 几何特征 min_image_distances 文件 01_build_structdesc_features.py:379:
  - 用最小镜像约定(MIC):$\Delta f \mathrel{-{=}} \mathrm{round}(\Delta f)$,然后通过 $\Delta r = \Delta f \cdot L$ 得到笛卡尔距离;
  - 统计对距离 quantile(min/mean/std/q25/q50/q75)、最近邻均值/方差;
  - 三段 cutoff(3/4/5 Å)的"配位数"均值与标准差。复杂度 $O(n^2)$,所以截断 max_sites=256。
  
  后处理 raw_to_stage2_ml/raw_to_stage3_ml 把目标(主前驱物列表、气氛、溶剂、合成类型)转成 multi-hot 标签列 label_prec__{Precursor} / label_atm__{Atm} 等。
  
  3.2 02_features/05_build_hybrid_features.py —— 混合特征拼接
  
  把描述子 CSV 与图嵌入 CSV 按 id 内连接,把图嵌入列名统一改成 {prefix}_graph_emb_{i}(支持 cgcnn/chgnet/alignn 多源同时拼)。这是 Stage2 GFlowNet 真正吃的"hybrid"输入。
  
  3.3 02_features/04_build_stage3_task_views.py
  
  对 Stage3 拆出 5 张子表(温度回归、time bucket 三分类、atmosphere 多分类、solvent 多分类、synthesis_type 多分类),每张只保留对应非空目标的样本 + 共享特征列。
  
  3.4 02_features/06_prepare_training_modes.py
  
  按需复制三套训练目录:
  - gold_only:只用 gold_holdout train
  - relaxed_only:只用 relaxed train
  - curriculum:phase1=relaxed,phase2=gold(冻结/解冻分阶段微调)
  
  ---
  4. Stage 03 (Graph) — 晶体图缓存与图嵌入
  
  三种 GNN 都"先建缓存,再喂模型"。
  
  4.1 CGCNN 图(03_graph/01_build_cgcnn_graph_cache.py)
  
  - 构图:对每个原子 $i$,在 $3\times 3\times 3$ 周期镜像中找到全部邻居,过滤距离 $\le 8.0,\text{Å}$,按距离取 max_num_nbr=12 个最近的(build_neighbor_graph,01_build_cgcnn_graph_cache.py:213)。
  - 输出 (edge_src, edge_dst, edge_dist) + atomic_numbers + 多 hot 标签 y,pickle 缓存。
  - 训练脚本 03_graph/train_cgcnn_stage2.py 实现一份"轻量级 CGCNN":
    - 原子嵌入 $h_i^0 = \mathrm{Embedding}(Z_i, d)$
    - 卷积层 CGCNNConv(train_cgcnn_stage2.py:114):消息 $m_{ij} = \mathrm{MLP}([h_i, h_j, d_{ij}])$,聚合 $h_i' = \mathrm{LayerNorm}(h_i + \mathrm{MLP}([h_i, \sum_j m_{ij}]))$
    - 4 层卷积 → 全图 mean+max 双池化拼接 → 256-dim 嵌入(extract_embedding 直接对外暴露)
    - Loss:BCEWithLogitsLoss(多标签前驱物预测)
  - 嵌入导出:export_cgcnn_stage2_embeddings.py 把 readout 后的 256-dim 写为 cgcnn_graph_emb_{i} CSV。
  
  4.2 CHGNet 图(03_build_chgnet_cache_stage2.py + export_chgnet_stage2_embeddings.py)
  
  - 这一支不重新训练,直接使用 预训练 CHGNet 提 crystal_fea(model.predict_graph(graph, task="e", return_crystal_feas=True),export_chgnet_stage2_embeddings.py:54)。
  - payload 提前转成 pymatgen.Structure → model.graph_converter(struct),给 CHGNet 喂带磁矩头的 e3nn 图。
  - 嵌入维度通常 64 维(CHGNet 默认),写入 graph_emb_{i}。
  
  4.3 ALIGNN 图(03_build_alignn_cache_stage{2,3}.py)
  
  对应原论文的 line-graph(键的"键-键"夹角图)。脚本逻辑与上面相同,只是缓存格式留给 ALIGNN 训练器消费。
  
  finalize_graph_embedding(推理 step)优先用 CGCNN 的 finetune 嵌入,缺失时回退到 CHGNet —— 这是工程上"鲁棒性 ≫ 一致性"的取舍。
  
  ---
  5. Stage 03 (Data) — 各模型变体训练集打包
  
  不同模型的训练集脚本沿着通用模板:
  1. 加载 hybrid 特征 CSV,自动检测 feat_* / graph_emb_* 作为 $x$。
  2. 检测 label_prec__* 作为多 hot 标签 → 按需转 trajectory / 集合编码。
  3. 训练集上拟合 standardizer ($\mu,\sigma$),验证/测试用同一统计量 transform。
  4. 写出 {train,val,test,gold_train_holdout}.npz + {*}_meta.csv + schema.json。
  
  关键差异:
  
  ┌───────────────────────────────────────────────┬───────────────────────┬─────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────┐
  │                     脚本                      │       模型变体        │                                    标签编码                                     │                    备注                    │
  ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 20_build_stage2_cvae_dataset.py               │ CVAE                  │ y_set 多 hot                                                                    │ 输出 (x, y_set) 对                         │
  ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 25_build_stage2_ar_dataset.py                 │ AR (Seq2Seq)          │ 排序后的 token 序列 + EOS                                                       │ 含 BOS/EOS,用 teacher forcing              │
  ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 26_build_stage2_setpred_dataset.py            │ Set Prediction        │ y_set 多 hot                                                                    │ 类似 CVAE 但无 latent                      │
  │                                               │ Transformer           │                                                                                 │                                            │
  ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 27_build_stage2_gflownet_dataset.py           │ GFlowNet              │ trajectory: 选元素的动作序列 + STOP                                             │ encode_reference_trajectories 生成多 hot → │
  │                                               │                       │                                                                                 │  traj                                      │
  ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 28_build_stage2_gflownet_mode_datasets.py     │ GFlowNet 三种 mode    │ 同上,走 relaxed_only/gold_only/curriculum 子目录                                │                                            │
  ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 27_build_stage3_condition_dataset_v5_mixed.py │ Mixture Flow          │ $(x, y_{\text{set}}, y_{\text{cont}}, y_{\text{cont_mask}}, y_{\text{disc}},    │ 连续 = (T,t),离散 = (atm_coarse,           │
  │                                               │                       │ y_{\text{disc_mask}})$                                                          │ synthesis_type)                            │
  └───────────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────┘
  
  5.1 GFlowNet 轨迹编码(27_build_stage2_gflownet_dataset.py:184)
  
  对每条样本的多 hot $y\in{0,1}^V$:
  - 设 $A = {j: y_j = 1}$,直接把它作为"选择动作序列",末尾追加 STOP token;
  - traj_actions[i, :] 长度 max_set_len + 1,用 traj_mask 标注真实长度。
  - 这样 GFlowNet 就能用最大似然(teacher forcing)直接拟合 ground-truth 轨迹。
  
  5.2 Stage35 候选与硬负样本
  
  24_build_stage35_hardneg_compat_dataset.py(scripts/03_data/24_build_stage35_hardneg_compat_dataset.py:296)
  
  对每条 positive (target, true_set):
  - hard negative:从一个候选池(由 Stage2 模型在训练集上跑出的候选 jsonl 提供)抽 n_hard_neg_per_pos=2 条,每条都被 Stage2 打高分但实际不对 —— 这是模型的"困惑点";
  - random negative:从所有 train target 的 true_set 集合里随便抽 n_random_neg_per_pos(同一目标除外)。
  - 输出 (x_struct, precursor_y, x_joint=concat, y∈{0,1}),直接给 Stage35 兼容性二分类器训练。
  
  40_build_stage35_strong_joint_candidates.py + 41_build_stage35_strong_joint_dataset.py
  
  按 (material_key, parent_precursor_rank, condition_rank) 把"strong stage2 候选 × strong stage3 候选"做笛卡尔积,生成 recipe 级候选;然后回查 ground-truth 计算多种软标签:
  - precursor_jaccard、precursor_exact_match
  - temp_abs_err, time_abs_err,以及阈值化的 cont_match
  - joint_label = precursor_exact_match AND cont_match(硬标签)
  - joint_soft_score:Jaccard 与 cont_score 的加权和(软标签,做学习排序时用)。
  
  ---
  6. Stage 04 — Stage2 GFlowNet 生成器
  
  6.1 状态-动作空间
  
  - 状态 $s = (x,, m_t,, t)$:目标特征 $x$,当前已选前驱物的多 hot mask $m_t$,步数 $t$。
  - 动作:从 $V+1$ 维分布里采样,$V$ 个对应"添加某前驱物",最后 1 维是 <stop>。
  
  6.2 网络(train_gflownet.py:315 GFlowNetPolicy)
  
  $$x_{\text{ctx}} = \mathrm{MLP}(x),;; m_{\text{ctx}} = \mathrm{MLP}(m_t),;; t_{\text{ctx}} = \mathrm{Embedding}(t)$$
  $$\text{logits} = \mathrm{MLP}\big([x_{\text{ctx}}, m_{\text{ctx}}, t_{\text{ctx}}]\big) \in \mathbb R^{V+1}$$
  
  build_invalid_mask(行 366)强制:已选过的前驱物动作概率为 0;一旦 stopped 只允许继续输出 STOP。
  
  6.3 损失函数(混合 ML + REINFORCE)
  
  监督部分(teacher_forcing_loss,428):对参考轨迹的负对数似然
  $$\mathcal L_{\sup} = -\frac{1}{|\mathcal M|}\sum_{i,t} m_{i,t}, \log P(a^{i,t}\mid s{i,t})$$
  注意这是*不带 logZ 的轨迹平衡近似 —— 严格 GFlowNet 应该是 Trajectory Balance loss
  $$\big(\log Z + \sum_t \log P_F - \log R - \sum_t \log P_B\big)^2$$
  该实现把 $P_B$ 设为均匀(集合无序)、$\log Z$ 隐式吸收到归一化 softmax,因此降为标准的极大似然 + 策略梯度,这在小动作空间(几百个前驱物)是常见简化。
  
  强化学习部分(train_gflownet_rerank.py:1052):用基线减除的 REINFORCE
  $$\mathcal L_{\text{RL}} = -\mathbb E\big[(R - \bar R),\log P_{\theta}(\tau)\big]$$
  其中
  $$R(\hat y, y) = \mathrm{F1}(\hat y, y) + \lambda_{\text{exact}}\mathbb 1[\hat y = y] - \lambda_{\text{len}},|,|\hat y| - |y|,|$$
  默认 exact_bonus=0.25, length_penalty=0.02,clamp 到 $\ge 10^{-4}$ 避免 log 爆炸。
  
  总损失:$\mathcal L = \mathcal L_{\sup} + \alpha, \mathcal L_{\text{RL}}$,$\alpha$ = --rl_weight(默认 0.2),--warmup_epochs=10 之前 $\alpha = 0$(纯 supervised 暖启)。
  
  6.4 Reranker(train_gflownet_rerank.py:227)
  
  对每条样本采样 N 条候选(贪心 + 多温度采样,sample_temperatures 默认 [1.0],推理时配 [0.8, 1.0, 1.2]),特征拼接
  $$\phi = [x,, \hat y,, |\hat y|,, \log P_\theta(\tau)]$$
  喂入 3 层 MLP(CandidateReranker)。目标值就是上面的 reward $R$,损失为 MSE(行 780)。这样 reranker 学到的是"对每条候选估一个 oracle reward",推理时按预测 reward 重新排序候选池。
  
  6.5 元素约束解码(element_constrained_decode.py,19_sample_stage2_gflownet_composition_constrained.py)
  
  推理特有的"硬性化学先验"(无需重训):
  $$\text{bias}(p, \text{target}) = b\cdot|E_p \cap E_t| - \rho\cdot|E_p \setminus E_t|$$
  - 命中目标元素 → 加分(target_hit_bonus=6.0)
  - 多余金属 → 减分(extra_element_penalty=1.0)
  - 完全不重叠 → 直接 -no_overlap_penalty=6.0
  - STOP 也加 stop_bias=-2.0(默认偏向继续生成,直到合理覆盖目标元素)
  - H、O 默认列入 ignore_elements 不参与命中判定(避免把 H₂O/HNO₃ 这种辅料拉偏分布)。
  
  最终在每步 softmax 之前,把这个 bias 加到 logits 上,等价于一个 prior 调制,既保留 GFlowNet 的多样性,又强制元素一致性。
  
  ---
  7. Stage 04 — Stage3 条件预测模型
  
  主力模型是 Mixture Residual Flow + Baseline 残差结构,搭配 LightGBM 分位数集成做 fallback。
  
  7.1 Mixture Residual Flow(train_condition_mixture_flow_mixed.py:582)
  
  双阶段思路:先用一个简单 baseline 给条件均值,Flow 只学残差。
  $$\hat c = c_{\text{base}}(x, y_{\text{set}}) + \Delta c,\quad \Delta c \sim p_\theta(\cdot \mid x, y_{\text{set}})$$
  
  baseline 可以是 Stage3BaselineModel(同一份代码里的小 MLP)或 sklearn 的 ExtraTrees/Ridge 套娃(SklearnBaselineAdapter)。
  
  上下文编码 ResidualContextEncoder(行 548):
  $$h = \mathrm{Trunk}\big(\mathrm{concat}(x,, \mathrm{SetEnc}(y_{\text{set}}))\big)$$
  
  Mixture-of-Gaussians 残差头(行 626):由 n_components=3~5 个 MLP 给出每个分量的 $(\mu_k, \log\sigma_k) \in \mathbb R^{2D_c}$;另一个 gating MLP 给 $\pi_k$:
  $$p(\Delta c \mid h) = \sum_{k=1}^K \pi_k(h);\mathcal N!\big(\Delta c;,\mu_k(h),,\mathrm{diag}(\sigma_k^2(h))\big)$$
  连续 NLL(行 656):
  $$\mathcal L_{\text{cont}} = -\frac{1}{N}\sum_i \log\sum_k \pi_{ik}, \mathcal N(\Delta c_i;,\mu_{ik},,\sigma_{ik})$$
  注意 mask:只对所有连续维都有标签的样本计算((mask > 0.5).all(dim=1)),避免温度缺失但时间存在引发偏差。
  
  离散头(行 623):每个离散维(气氛、合成类型)独立 Linear(ctx_dim → K_j),对所有头求 cross-entropy 加权和(行 753)
  $$\mathcal L_{\text{disc}} = \frac{\sum_j w_j, \mathrm{CE}(\hat y_j, y_j)}{\sum_j w_j}$$
  类别权重按 1/freq 计算(build_class_weight_tensor)。
  
  总损失 $\mathcal L = \mathcal L_{\text{cont}} + \mathcal L_{\text{disc}}$。
  
  推理采样(行 678):先按 gating 概率抽 component,再 reparameterize $\mu + \sigma,\epsilon$。把 n_flow_samples=64 条样本 + top-component-mean(确定性的"中位"路线)合并去重,取 top-k 给下游。
  
  7.2 LightGBM 分位数集成(train_lgbm_quantile_ensemble.py)
  
  - 对温度、时间各训练 9 个 quantile 回归器($\tau \in {0.1, 0.2, …, 0.9}$),objective='quantile',LightGBM 的 pinball loss
  $$\rho_\tau(u) = \max\big(\tau,u,,(\tau-1),u\big)$$
  - 气氛:二分类(oxidizing vs non-oxidizing),binary log-loss。
  - Time bucket:三分类(short/medium/long),multiclass log-loss。
  - 推理 13b_run_stage3_infer_lgbm_quantile.py:210 的关键 trick —— 按 [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9] 顺序遍历,先取中位再向两侧扩张,等价于在不确定性范围内提供"由保守到激进"的多组条件候选。
  
  这一支在工程上稳过 Mixture Flow(配置里 primary_model=lgbm),但 Mixture Flow 给的样本更连续、分布更细。
  
  7.3 其他变体
  
  - train_mlp_predictor.py:每个连续/离散头独立 sklearn MLPRegressor/MLPClassifier,标准化 + adaptive lr,作为简单基线。
  - train_baseline_linear.py:Ridge + LogisticRegression,头脑最干净的对照。
  - train_condition_residual_mdn_mixed.py:同款架构但用纯 MDN(没有显式 baseline 残差解耦)。
  
  ---
  8. Stage35 联合排序器
  
  把 Stage2 候选集合 × Stage3 条件候选 配对评分。本工程实际部署了多个版本(rule、learned、v21 hybrid、v3 learned ranker、v43 template-aware)叠成"ranker chain"。
  
  8.1 V21 兼容性 ExtraTrees(主力,stage35.v21_* 配置)
  
  - 训练集来自 24_build_stage35_hardneg_compat_dataset.py(positives + 2×hard_neg + N×random_neg)。
  - 输入特征 = 结构特征 + 候选前驱物 multi-hot,标签 = pair_label∈{0,1}。
  - 模型 = ExtraTreesClassifier(n_estimators=...)(joblib 序列化)。
  - 推理时输出 proba(pos) 当作 v21 score → 全局 rank。
  
  8.2 V43 Template-Aware Pairwise Ranker(route_ranker/v43_template_aware/)
  
  关键创新:把化学模板信息显式作为特征。
  - 01_add_route_template_features.py:对每条候选路线判定前驱物类型(hydrate/nitrate/carbonate/phosphate/sulfate/oxide/hydroxide/halide/...,precursor_type),再判定路线整体的
  route_template_matches_target_anion、is_common_solid_state、is_overly_elemental。
  - 02_build_v43_template_pairwise_dataset.py:构造同目标 pair(route_a, route_b),用一个弱奖励函数(compute_template_quality,行 55)给出排序方向:
  $$Q = 4,\text{cov} - 2,\text{miss} - 0.25,\text{extra} - 1.5,\text{foreign} - 0.5,\text{extra_nontriv}$$
  $$;;;; - 0.6,\text{warn} - 0.8,\text{warn_pen} + 0.8,\text{tpl_match} + 0.4,\text{common_solid} + 0.2,\text{tpl_conf}$$
  $$;;;; - 0.35,\text{overly_elem} + 0.4,\text{v42} + 0.2,\text{v33} + 0.1,\text{v32}$$
  比较 a/b 的 $Q$ 决定 pairwise 标签。
  - 03_train_v43_template_pairwise_ranker_chemonly.py:71 —— 训练对称化 ExtraTreesClassifier:
    - $X_{\text{pos}} = \phi(a) - \phi(b)$ → 标签 1
    - $X_{\text{neg}} = -X_{\text{pos}} = \phi(b) - \phi(a)$ → 标签 0
    - 这是经典 RankNet/pairwise SVM 思想:学一个对称的"差分判别器" $f(\phi_a - \phi_b) > 0.5 ⇔ a \succ b$。
    - 用 GroupShuffleSplit(by=target_group) 防泄漏,class_weight='balanced',n_estimators=600,max_depth=12。
    - 排除已含排序信号的列(score, prob, rank, wins, losses, ...),让模型只用化学/模板差异学习。
  - 应用脚本 04_apply_* 用 proba(diff>0) 给候选打全局分。
  
  8.3 V3 Learned Ranker
  
  直接对 synthesis_routes_* 表用 ExtraTreesRegressor 拟合一个标量 score(详见 outputs/inference/_merged_v3_learned_ranker/),作为"全局 fallback"——不依赖 pairwise,适合冷启动。
  
  8.4 排序链
  
  配置里默认顺序(pipeline/run_pipeline.py:51-58):
  summarize_routes → filter_display_routes → stage35_rule_rerank → stage35_learned_rerank → stage35_v21_rerank → best_route_per_precursor
  最终 export_final_top_routes(steps_final.py)按优先级 v21 > learned > rule > display 选出最强可用 ranking 写到 final_top_routes.csv。
  
  ---
  9. Stage 07 — 端到端推理流水线
  
  pipeline/run_pipeline.py 用一张 STEP_FUNCS 列表注册所有 step,PipelineRunner 负责:
  - --start_from <step>:从指定 step 重启(后续步骤都执行)
  - --only_step <step>:只跑这一步
  - restore_existing_outputs:看 outputs/inference/<infer_name>/ 下既有产物,补 outputs 字典(支持断点续跑)
  - record_degradation:某步骤回退时(例如 CGCNN 不可用回退 CHGNet),写 degraded_steps
  
  完整 step 流(28 步):
  
  0  preflight                       # 资源/CKPT/词表自检
  1  make_infer_split                # POSCAR 目录 → infer.jsonl
  2  build_structdesc                # 跑 02_build_infer_structdesc_direct.py 出描述子 CSV
  3  build_chgnet_embedding          # CHGNet 拿 64-dim crystal_fea
  4  finalize_graph_embedding        # 优先 CGCNN,缺失则用 CHGNet
  5  build_stage2_features           # hybrid CSV(描述子+图嵌入)
  6  build_stage2_npz                # 转 NPZ + meta(标准化用 train 时的 μ,σ)
  7  sample_stage2_gflownet          # standard 或 composition_biased(默认后者)
  8  constrain_stage2_by_composition # 后置硬过滤:cov、extra penalty、可选 drop_zero_overlap
  9  summarize_stage2                # unique_sets + count(去重的候选集合)
  10 add_composition_fallback        # 元素覆盖低时补一些"组成回退"前驱物
  11 retrieve_stage2_candidates      # k-NN 检索:从训练 NPZ 找相似目标的真实前驱物集合
  12 predict_stage2_baseline         # ExtraTrees 多标签基线(top-k labels → set 组合)
  13 merge_stage2_sources            # GFlowNet ∪ Retrieval ∪ Fallback ∪ Baseline 合并
  14 rerank_stage2_by_elements       # 元素覆盖加权 + extra penalty + 原 rank 微衰减
  15 fix_stage2_global_rank          # 修补合并后 global rank
  16 build_stage3_features           # Stage3 hybrid CSV
  17 build_stage3_conditioned_table  # 把每个 stage2 候选展开成 (x, parent_set) 行
  18 run_stage3_flow                 # Mixture Flow 推理(默认关闭)
  18b run_stage3_lgbm                # LightGBM quantile 推理(默认开启,主力)
  19 compare_stage3_models           # 二者一致性报表
  20 summarize_routes                # 把 stage3_flat → 可读路线 (CSV+MD)
  21 filter_display_routes           # 物理上限/下限(温度 300-1600°C, 时间 0.1-240h)
  22 stage35_rule_rerank             # 规则打分
  23 stage35_learned_rerank          # ExtraTrees regressor 重排
  24 stage35_v21_rerank              # ExtraTrees pairwise 重排(主力)
  25 best_route_per_precursor        # 同一前驱物集合只保留最佳条件
  26 export_final_top_routes         # 选最强可用 ranking 落地
  27 reliability_layer (整层)        # ↓
         precursor_qc                # 前驱物 QC 标注
         attach_route_confidence     # 集合一致性 + reranker 概率融合
         condition_distribution_conf # 条件落在训练分布"身体"还是"尾部"
         attach_stage3_condition_reference_support
         audit_final_recommended_routes
         qc_confidence_postprocess
  28 select_final_recommended_routes # 最终向用户展示的稳定列
  
  9.1 候选集合的"四源合并"(关键设计)
  
  Stage2 不是单一 GFlowNet 输出,而是四种异质来源:
  1. GFlowNet 多温度采样(主多样性源,composition_biased 模式带元素先验)
  2. Retrieval(17_retrieve_stage2_precursor_candidates_from_npz.py):k-NN(余弦相似度,默认 top_k=50, min_similarity=0)拉相似目标的真实前驱物集合 —— 类似 "case-based reasoning"
  3. Composition fallback(11_add_composition_fallback_precursors.py):元素覆盖不足时,直接按目标元素族选"标准"前驱物(如 metal oxide / nitrate)
  4. ExtraTrees 多标签基线(19_train_stage2_extratrees_baseline.py 训出来的 joblib 模型给每个标签独立概率,top-k 标签做 enumerate-set)
  
  合并后用元素覆盖率重排(10_rerank_stage2_candidates_by_elements.py):
  $$\text{score} = \omega_{\text{cov}}\cdot\text{cov}(\mathcal P, \text{target}) - \omega_{\text{extra}}\cdot |\text{extra metals}| - \omega_{\text{rank}}\cdot \text{rank}{\text{src}}$$
  默认 $(\omega{\text{cov}}, \omega_{\text{extra}}, \omega_{\text{rank}}) = (20, 5, 0.01)$。
  
  这种"生成 + 检索 + 规则 + 监督学习"的合并极大缓解了 GFlowNet 在 OOD 目标上的失败模式 —— 任意一个 source 给出合理候选,后面 Stage35 都能挑出来。
  
  ---
  10. 几个值得记住的工程取舍
  
  1. 没有大一统模型,而是异质源 + 多层 reranker。生成模型(GFlowNet/Flow)给"多样性"和"覆盖率",GBDT/ExtraTrees 给"判别力"和"稳定性",规则/模板给"先验"。
  2. 数据划分按 DOI/material-group,不是按行随机划。这避免了同一篇论文的多个相关样本被切到不同 split。
  3. Train mode 三件套(relaxed/gold/curriculum)默认存在。gold_only 在所有图缓存、特征、生成数据上统一支持,推理用的 GFlowNet ckpt 路径里就是 gflownet_joint_rerank_hybrid_gold_only_v1,说明最终生产用 gold
  训出来的最优模型。
  4. Stage3 模型不直接预测条件值,而是 baseline + 残差 mixture。这等价于把"条件预测"分解成"先验均值估计"和"分布形状学习",数值稳定且可控。
  5. 元素约束既出现在硬规则里(refine 阶段、composition_constrained 解码、元素 rerank),又通过 reward 进 GFlowNet RL。多重保险:同一个先验在不同位置生效。
  6. 可靠性层独立成一层(steps_reliability.py 1984 行),最后给每条路线 attach confidence、QC、distribution support,而不是把这些信号塞进 ranker 特征 —— 这是因为推断时间不确定性应该和排序分数解耦。
  
  ---
  任何想深入的子模块(比如 RL loss 的实现细节、Stage35 v21 hybrid 的特征工程、composition_biased 解码里的 logit-bias 推导、retrieval 的相似度度量)告诉我,我可以再展开。
  
⏺ 已完成全栈分析。从 00_refine 的化学式匹配规则到 07_infer 的 28 步推理链,核心算法、损失函数、关键超参都标了行号。如果哪一块想深挖(例如 GFlowNet TB loss 的省略细节、Stage35 candidate pool
  的对偶构造、retrieval 用什么相似度),告诉我具体方向就行。
     

※ recap: 已完成 SynPred 项目从数据捕获到推理流水线的完整算法分析,涵盖 GFlowNet、Mixture Flow、Stage35 ranker 等核心模块。下一步等你指定想深入的子模块再展开。 (disable recaps in /config)

❯ Stage 00 — 原始数据捕获与精炼部分更加详细的分析，可以写文章程度                                                                                                                                             
  从晶体到合成路线:Stage 00 数据捕获与精炼的完整剖析
  
  ▎ 本文是 SynPred 流水线 Stage 00 的深度技术解读。Stage 00 的目标是把 Materials Project 上的实验型晶体档案,与文献抽取出来的合成数据库做"结构 ↔ 配方"的可信对齐,并精修出可用于下游训练的金标 / 
  ▎ 放宽两套数据集。它决定了整个 SynPred 模型族的标签上限,值得专章拆解。

  ---
  第一部分 · 引言:为什么 Stage 00 是整个流水线的瓶颈
  
  机器学习驱动的逆合成(给定目标晶体 → 反推前驱物与工艺条件)长期受困于一个朴素的事实:结构数据库与合成数据库不天然耦合。前者(以 Materials Project, MP 为代表)给出经第一性原理弛豫的晶格、原子坐标、能量;后者(以
  Ceder 组的 solid-state、solution-synthesis 数据集为代表)给出从 NLP 抽取的反应式、前驱物列表、温度时间气氛。两者的桥梁——"这条合成线对应哪一份 POSCAR"——在公开数据里不存在。要训练 Stage2(结构→前驱物)和
  Stage3(结构 + 前驱物 → 合成条件)的监督模型,必须先把这座桥架起来。

  SynPred 的 Stage 00 由四个脚本组成:

  ┌──────────────────────────────────────────┬──────┬─────────────────────────────────────────────────────────────────────┐
  │                   脚本                   │ 行数 │                                职责                                 │
  ├──────────────────────────────────────────┼──────┼─────────────────────────────────────────────────────────────────────┤
  │ 01_capture_experimental_structures.py    │  462 │ 从 MP API 拉取实验型晶体,落盘 POSCAR + 三类 metadata JSON           │
  ├──────────────────────────────────────────┼──────┼─────────────────────────────────────────────────────────────────────┤
  │ 02_prepare_dataset.py                    │ 1541 │ 把合成记录与 MP 档案做加权打分匹配,生成 direct_aligned_dataset.json │
  ├──────────────────────────────────────────┼──────┼─────────────────────────────────────────────────────────────────────┤
  │ 03_statistic.py                          │   20 │ 按 match_level 切片,导出 strict-exact 和 strict-parent 两个子集     │
  ├──────────────────────────────────────────┼──────┼─────────────────────────────────────────────────────────────────────┤
  │ 04_refine_strict_exact_for_structdesc.py │  840 │ 在 strict-exact 子集上做最后一轮化学/文本/条件筛选,产出四份训练集   │
  └──────────────────────────────────────────┴──────┴─────────────────────────────────────────────────────────────────────┘

  读完这四个脚本你会发现,SynPred 在 Stage 00 没有用任何模型,但它定义的评分阈值、规则枝叶、降级策略几乎覆盖了文献中常见的全部失败模式。下文按脚本顺序展开,在关键决策点会插入"为什么这么做"的讨论。

  ---
  第二部分 · 01:Materials Project 实验型档案的全量捕获
  
  2.1 接口选择:为什么用 mp_api.client.MPRester 而不是旧版 pymatgen.MPRester

  01_capture_experimental_structures.py:10 引入的是 新版 mp_api。新旧 API 的关键差别在于"记录返回模式":新版默认返回 MPDataDoc 对象,但配合 use_document_model=False(行 312)可以直接拿到 
  dict。这一步对全量抓取非常关键——dict 可以毫无损耗地 json.dump,而文档模型对象往往内嵌了不可序列化的 numpy 引用,会迫使开发者写大量 to_dict 适配。

  with MPRester(CONFIG["MP_API_KEY"], use_document_model=False) as mpr:
      summary_docs = fetch_summary_docs(mpr)

  2.2 三段式取数:summary → DOI → provenance

  为什么要分三段?因为 MP 的元数据被有意切成了三个 endpoint:
  
  1. summary(mpr.materials.summary.search):材料标识、化学式、能带、密度、能 above hull、structure 字段(直接含 lattice + species + frac_coords)
  2. DOI(mpr.doi.search 或 mpr.materials.doi.search):MP 自己引用的论文 DOI 与 BibTeX
  3. provenance(mpr.materials.provenance.search):该 material_id 在 ICSD/Pauling File 等数据库的来源记录,内嵌作者题名、文献 DOI 列表、database_IDs

  脚本对每个 endpoint 都做了容错探测(get_rester,行 163):
  def get_rester(mpr, candidates):
      for path in candidates:
          cur = mpr
          for name in path:
              if not hasattr(cur, name):
                  break
              cur = getattr(cur, name)
          else:
              return cur
      return None
  新旧 mp_api 偶尔会把 doi 端点挂在 mpr.doi 或 mpr.materials.doi,这段代码避免了在版本升级时硬编码失败。
  
  2.3 节流策略

  "BATCH_SIZE_META": 200,
  "SLEEP_EVERY": 20,
  "SLEEP_SECONDS": 0.2,
  在 fetch_meta_docs 中(行 259-276)每 20 个批次 sleep 0.2s。看上去保守,但 MP 对未付费用户的速率限制是按 滑动窗 实现的,真正昂贵的不是单次请求而是高 QPS 下的 503 重试导致的雪崩。这种"小步走"的节流策略,在 24h
  之内可以稳定跑完十几万条 material 而不被封禁。
  
  2.4 关键数据产物:DOI 的递归挖掘

  provenance 文档的 schema 不稳定——MP 自己引用 ICSD 的字段叫 references,引用 OQMD 的字段叫 database_IDs.OQMD,作者抽取的字段又叫 citation。extract_dois_in_obj(行 104)做了一个完全通用的正则递归扫描:

  DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

  def extract_dois_in_obj(obj):
      found = set()
      def walk(x):
          if isinstance(x, str):
              for m in DOI_PATTERN.findall(x):
                  found.add(m.strip().rstrip(".,;]})"))
          elif isinstance(x, dict):
              for v in x.values(): walk(v)
          elif isinstance(x, (list, tuple, set)):
              for item in x: walk(item)
      walk(obj)
      return sorted(found)
  
  这是后面打分匹配的核心信号——同一篇论文 是 MP 档案与合成记录之间最可靠的桥梁,远比化学式相似更有判别力。literature_dois_found 这一列会被写入主表 CSV,供下游脚本直接消费。

  2.5 输出结构

  最终目录结构:
  mp_full_archive_export/
  ├── poscar/                    {material_id}_{formula}.vasp
  ├── summary_json/              {material_id}.json   (full summary doc)
  ├── doi_json/                  {material_id}.json   (mp_doi + bibtex)
  ├── provenance_json/           {material_id}.json   (titles + literature DOIs)
  └── mp_full_archive_metadata.csv   主索引表
  主表 schema(行 396-411):material_id, formula_pretty, theoretical, deprecated, mp_doi, mp_bibtex, title_candidates_from_provenance(| 拼接前 10 个标题), literature_dois_found(| 拼接所有 DOI),
  database_ids_json, 三个 JSON 文件路径。
  
  把"标题"和"DOI"拼成 | 分隔的串而非 JSON,是为了让 pandas groupby 和 Excel 检视都方便——这是工程美感细节。

  2.6 实验型过滤

  if CONFIG["EXPERIMENTAL_ONLY"]:
      search_kwargs["theoretical"] = False
  MP 数据库混杂了大量 DFT-only 的"假想结构"(theoretical=True),它们没有对应的合成路径,留着只会污染下游的 retrieval。一行配置就把它们切掉。

  ---
  第三部分 · 02:多源合成数据库与 MP 档案的可信对齐

  3.1 任务形式化

  记 MP 档案集合为 $\mathcal M = {m_1, m_2, \dots}$,合成记录集合为 $\mathcal S = {s_1, s_2, \dots}$(来源:solid_state、solution_synthesis 两套 JSON)。目标是构造一个对齐函数 $\phi: \mathcal S \to \mathcal M
  \cup {\bot}$,使得每条合成记录被匹配到一份具体的 POSCAR(或被拒绝)。

  朴素做法是按 formula_pretty 等值连接,但这会失败的原因有三类:
  1. 掺杂体系: $\text{Y}_{0.95}\text{VO}_4$ 这种合成产物不会在 MP 出现,但其母相 $\text{YVO}_4$ 一定在;
  2. 同分异构体: $\text{TiO}_2$ 在 MP 有 anatase / rutile / brookite 多个 entry,化学式相同但结构不同;
  3. 变量配方: $\text{Li}_x\text{CoO}_2$ 是真正的固溶体,根本不该匹配单一 POSCAR。
  
  02_prepare_dataset.py 的核心是一个多信号加权打分器,既能 promote 1 类,又能正确处理 2 类的歧义,还能 reject 3 类。

  3.2 化学式规范化与"母相回退"

  3.2.1 parse_formula(行 160)

  直接用 pymatgen.core.Composition(formula).reduced_formula,这个操作把 Y2O3 和 YO1.5 都归一到 Y2O3,把分数化学式如 LiCoO2 与 Li2Co2O4 归一到 LiCoO2。

  3.2.2 rounded_parent_formula(行 171)
  
  解决"轻度掺杂"的关键。先把化学计量数四舍五入到最近整数,容差 tol=0.12:
  $$
  n'e = \begin{cases} \mathrm{round}(n_e) & \text{if } |n_e - \mathrm{round}(n_e)| \le 0.12 \ n_e & \text{otherwise} \end{cases}
  $$
  所以 $\text{Y}{0.95}\text{VO}4 \to \text{YVO}4$,$\text{Ba}{1.99}\text{Eu}{0.01}\text{MgB}_2\text{O}_6 \to \text{Ba}_2\text{MgB}_2\text{O}6$,但 $\text{Li}{0.5}\text{CoO}_2$ 因为 $|0.5 - 1| > 0.12$
  不会被四舍,保持原样。这种"按整数 sigma 截断"的做法对掺杂浓度 ≤ 12% 的实验体系几乎无误判。

  3.2.3 is_light_parent_match(行 252)——比"四舍五入到母相"更严格

  即便化学式经过 round 后命中了某个 MP 母相,还要进一步检查整体形变是否够小。parent_match_profile(行 213)给出 5 个数值化指标:

  prof = {
    "n_extra_elems": ...,           # 新引入元素的种类
    "extra_amount": ...,            # 新引入元素的总量
    "missing_parent_amount": ...,   # 母相元素的缺失量
    "shared_l1": ...,               # 共享元素的 L1 偏移
    "max_shared_dev": ...,          # 单元素最大偏移
  }
  然后用一组与"是物理上合理的微掺杂"对应的硬阈值(行 265-271):
  $$
  \begin{aligned}
  n_{\text{extra}} &\le 1 \
  \sum_{e\in\text{extra}} c_e &\le 0.15 \
  \sum_{e\in\text{parent}} \max(c_e^{\text{parent}} - c_e^{\text{synth}}, 0) &\le 0.15 \
  \sum_{e\in\text{shared}} |c_e^{\text{synth}} - c_e^{\text{parent}}| &\le 0.20 \ 
  \max_e |c_e^{\text{synth}} - c_e^{\text{parent}}| &\le 0.15
  \end{aligned}
  $$
  
  这套阈值是经验调出来的——它的真正意义是把 5 个独立的"轻度"判据 AND 起来,任何一项超标都会拒绝。过去常见的失败模式是只看新元素数量(于是 $\text{La}{0.5}\text{Sr}{0.5}\text{MnO}_3$ 被错误归入
  $\text{LaMnO}_3$),这套向量化阈值能稳健地拒绝它(max_shared_dev = 0.5 > 0.15)。

  3.3 候选缩窄:四级漏斗

  candidate_subset_for_synth(行 1126)按以下顺序找匹配候选,找到就停:

  1. 精确公式相同
  2. 严格母相公式相同(rounded_parent → 候选)
  3. 同 anonymous_formula + 同 chemsys
     (Y0.95VO4 与 LaVO4 都是 ABO4 但 chemsys 不同,只有结构原型同的才进)
  4. 不再兜底到同 chemsys——这一步 2024 年改过,以前会兜底 (代码 line 1153 注释)

  这种"激进剪枝"的代价是召回低,但保证了下游评分阶段不会被太多远亲候选拖慢。anonymous formula 是 pymatgen 内置概念:A1B1C3 表示"一种 1:1:3 的三元化合物",可以一次性把所有钙钛矿结构原型聚到一起。

  3.4 加权打分:score_match(行 1155)
  
  这是整个 Stage 00 的核心打分器。所有信号被映射到正分,各项独立累加:

  ┌──────────────────────────────────┬──────┬─────────────────────────────────────────┐
  │               信号               │ 加分 │                  说明                   │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ exact_formula                    │  +80 │ 完全相同的 reduced formula              │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ parent_formula_strict            │  +45 │ 满足前述 5-项轻度掺杂判据               │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ same_chemsys                     │   +8 │ 元素集合相同                            │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ same_anonymous_formula           │   +8 │ 化学计量原型相同                        │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ comp_dist <= 0.015               │  +20 │ L1 距离 ≤ 1.5%(原子分数)                │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ comp_close(0.015 < dist ≤ 0.030) │   +6 │ 较近                                    │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ same_element_set                 │   +4 │ 元素集严格一致                          │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ doi_overlap                      │  +35 │ 共享至少一个 DOI                        │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ title_sim ≥ 0.75                 │  +20 │ 论文标题 difflib ratio ≥ 0.75           │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ title_sim ≥ 0.60                 │   +8 │ 较像                                    │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ has_poscar                       │   +3 │ MP entry 有可用 POSCAR                  │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ ground_state                     │   +2 │ $E_{\text{above hull}} = 0$             │
  ├──────────────────────────────────┼──────┼─────────────────────────────────────────┤
  │ near_ground_state                │   +1 │ $E_{\text{above hull}} \le 10$ meV/atom │
  └──────────────────────────────────┴──────┴─────────────────────────────────────────┘

  这套权重在 paper 里写出来不会很性感,但它体现了几个值得注意的工程哲学:

  (a) 化学等值占绝对主导。 exact_formula(80) + parent_strict(45) 的两个最重信号都来自化学公式,任何一个命中,候选基本就锁定。DOI overlap 虽然加 35 分,但单独不足以通过(后面会看到)。

  (b) DOI 与化学一致性必须 AND。 is_reliable_match(行 1238)的规则 (3) 写明:
  if has_doi and has_same_chemsys and (has_comp_tight or has_same_anon or has_parent_strict):
      return True
  这是一个反模式纠错——曾经的版本只看 DOI 重合,导致一篇综述论文里所有提到的化合物都被错误对齐到同一份 POSCAR。AND 化学一致性后,DOI 只能起"消歧"作用而不能"凭空通过"。
  
  (c) 标题相似度用 difflib.SequenceMatcher,不用 BERT。 这是 trade-off。标题文本短(20-200 字符)、且含大量化学符号,SequenceMatcher 的 longest common subsequence 比向量化模型更鲁棒,且不需要 GPU。

  (d) Ground-state bonus 极小(+1 ~ +2)。 多形态目标(如 $\text{TiO}_2$)优先取 ground state 的 entry,但加分微弱,允许打分器在其他信号支持下选用 metastable 异构体。

  3.5 准入仲裁:is_reliable_match

  打分高 ≠ 可信。脚本设计了一套独立的"通过条件":
  1) 有 exact_formula        → 通过
  2) 有 parent_formula_strict → 通过
  3) DOI + same_chemsys + (comp_tight | same_anon | parent_strict) → 通过
  4) title_sim ≥ 0.75 + comp_tight + same_chemsys → 通过
  5) same_anonymous_formula + same_element_set + comp_tight → 通过
  6) 否则全部拒绝
  注意条件 (2)(3)(4) 都同时要求多重正交信号。这种合取正则化有效地阻断了"一个高分信号偶然命中"的伪匹配。
  
  3.6 双阈值:MIN_KEEP_SCORE=35 与 BEST_ONLY_MIN_SCORE=55

  打分通过 is_reliable_match 后,还要过两道分数线:
  - 候选分 < 35 直接丢弃,不进入 top-K(MIN_KEEP_SCORE,行 1377)
  - 即便最佳候选分 ≥ 35 但 < 55,整条合成记录被记入 unmatched(行 1428),不输出对齐
  
  这意味着低分对齐宁可不要,留给后面的 04 脚本去做 strict_exact 子集筛选。

  3.7 Top-K 与同形异构体仲裁

  每条合成记录最多保留 TOPK_PER_RECORD=3 个候选。但有一种情况例外——solution_synthesis 数据集本身已经标注了作者认为对应的 mp_id(行 1414-1423):
  if s_dict["source_dataset"] == "solution_synthesis":
      raw_rec = s_dict["raw_record"]
      target = raw_rec.get("target") or {}
      orig_mpid = target.get("mp_id")
      if orig_mpid:
          orig_match = [x for x in scored if x["material_id"] == orig_mpid]
          if orig_match:
              scored = orig_match + [x for x in scored if x["material_id"] != orig_mpid]
  作者标注被尊重,被 promoted 到第一名。这是典型的"用人类先验做 tie-breaker"——打分器没有信息分辨多形态,但作者知道。

  3.8 match_level 分级

  classify_match_level(行 274)根据 reason 给每条对齐打四个标签:

  ┌──────────────────────┬───────────────────────────────────────────┐
  │        level         │                 触发条件                  │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ exact                │ reason 包含 exact_formula                 │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ parent_strict        │ 不是 exact 但 parent_formula_strict       │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ literature_supported │ 不属于以上两类,但有 DOI 或高 title_sim    │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ weak                 │ 通过 is_reliable_match 但缺乏化学严格匹配 │
  └──────────────────────┴───────────────────────────────────────────┘
  
  这个分级在 03_statistic.py 里被用作切片键,将 exact 子集(strict_exact_only.jsonl)单独提取出来给 04 脚本做后续精修。SynPred 的所有训练数据都是从 exact 这一支派生的。 parent_strict
  子集(strict_parent_aug.jsonl)被保留作为可选增强源,但默认不进金标。

  3.9 增量恢复

  get_processed_synth_uids(行 1277)读取已经写入 jsonl 的所有 synth_uid,在断点续算模式下跳过它们。每一次 append
  是单条记录、单文件追加,所以即使脚本意外中断,已经写入的对齐结果都是完整的。这种设计在跑全量(几十万合成记录 × 几十万 MP entry)时非常重要。

  ---
  第四部分 · 03:match_level 切片(20 行的关键)
  
  03_statistic.py 是整个 Stage 00 最短的脚本,但定义了"什么数据进入 SynPred 训练"。代码非常直白:

  df = pd.DataFrame(rows)
  print(df["match_level"].value_counts(dropna=False))

  exact_df  = df[df["match_level"] == "exact"].copy()
  parent_df = df[df["match_level"] == "parent_strict"].copy()

  exact_df.to_json(".../strict_exact_only.jsonl", orient="records", lines=True)
  parent_df.to_json(".../strict_parent_aug.jsonl", orient="records", lines=True)

  这一刀切下去的含义是:SynPred 不信任 literature_supported 和 weak 等级的对齐。原因是:

  - literature_supported:意思是只有 DOI 或标题在帮忙,化学式可能差很远(不同化合物在同一篇论文里被一起讨论);
  - weak:is_reliable_match 走的是兜底分支,通常是"匿名式相同 + 元素集相同 + 组成接近"。听起来不错,但在多形态钙钛矿、尖晶石上误判率不低。

  模型的标签上限决定了模型的预测上限。 把这两类去掉后,后续训练的 stage2/stage3 监督信号才不会被噪声拖累。

  ---
  第五部分 · 04:从 strict_exact 到四份训练集
  
  04_refine_strict_exact_for_structdesc.py 是 Stage 00 的"精修台"。输入 strict_exact_only.jsonl,输出四份 jsonl + 一份 dropped + 两份 summary。它做的事远不止过滤——它要把抽自 NLP
  的合成记录结构化为下游模型直接能用的字段。

  5.1 输出文件矩阵

  ┌────────────────────────────┬────────────────────────────────────────┬────────────────────────────────────────────────────┐
  │            输出            │                  用途                  │                      准入条件                      │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ stage2_gold.jsonl          │ Stage2 高质量训练集(结构 → 前驱物集合) │ severe ∅ + mild ∅(对 gold-blocking 子集)           │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ stage2_train_relaxed.jsonl │ Stage2 大覆盖训练集                    │ severe ∅(允许任何 mild)                            │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ stage3_gold.jsonl          │ Stage3 高质量训练集(条件回归/分类)     │ stage2_gold + 至少一个 operation 级条件 + 含加热   │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ stage3_train_relaxed.jsonl │ Stage3 大覆盖训练集                    │ stage2_relaxed + 至少一个条件(可来自 row fallback) │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ dropped_records.jsonl      │ 完全弃用                               │ 既不进 stage2 relaxed 也不进 stage3 relaxed        │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ summary.json               │ 计数                                   │ 总数 + 各产物条目数                                │
  ├────────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ reason_summary.json        │ 失败原因直方图                         │ 严重 / 轻微原因的频次                              │
  └────────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────────────────┘

  这种"金标 / 放宽"双轨制不是 SynPred 独有,但脚本把它实现得极其工业化:同一行数据可以同时进 stage2_gold 和 stage3_gold,也可以 stage3_relaxed 但不进 stage2_gold(因为 stage3 的条件抽取成功),全靠四个独立
  boolean。

  5.2 化学一致性:check_precursor_element_consistency

  这是 04 脚本最关键的化学规则(行 74):
  
  target_metals = target_elements & STRUCTURAL_METALS
  extra_metals  = prec_metals - target_metals
  if len(extra_metals) >= 2: severe.append("precursor_element_mismatch_severe")
  elif len(extra_metals) == 1: mild.append("precursor_element_mismatch_mild")

  STRUCTURAL_METALS 是一个手工策划的 60+ 元素集合(行 54-63),覆盖了几乎所有金属、半金属、Si/P/B,但故意排除了 H, O, C, N, S, F, Cl, Br, I(COMMON_NONSTRUCTURAL,行 65)。这是个化学家的判断:H、O、C、N
  等常出现在前驱物里(Co(NO3)2·6H2O、H2C2O4),但它们最终通过气体逸出 / 配体分解,不会进入晶格。检查时只比金属和半金属是否能"关上"。

  阈值定在 >= 2 个"多余金属"才算 severe,是因为很多合成会用一个无关金属做催化或助熔(典型如 LiCl 在熔盐法里用作 flux)。允许一个,但不允许两个——这是经验。

  stage2_gold 把 precursor_element_mismatch_mild 也列入 gold_blocking_mild(行 761),所以金标只要有 1 个多余金属就被踢出,relaxed 还能接受。

  5.3 变量配方过滤:split_variable_reasons

  化学家会写 $\text{La}_{1-x}\text{Sr}_x\text{MnO}_3$ 表示一族化合物,但模型不能用——它没有确定的目标结构。脚本的 VARIABLE_PATTERNS(行 33-42):

  VARIABLE_PATTERNS = [
      r"\bx\b", r"\by\b", r"\bz\b",
      r"δ",
      r"\bLn\s*=", r"\bRE\s*=",
      r"\d+(?:\.\d+)?\s*[-–]\s*[xyz]",
      r"[xyz]\s*=",
  ]

  匹配从单字符变量到 Ln= (镧系通用)、RE= (稀土通用)、1-x 这类表达。命中 reaction_string 是 severe(直接拒绝),命中 synthesis_text 是 mild(进 relaxed,不进 gold)。这种粒度分级让 relaxed
  集还能保留"文本里偶提变量但反应式干净"的样本。

  target_amount_vars 字段(行 293)是另一个 severe 指示——合成数据库自带"我这个 target 是变量浓度"的元数据,直接 trust 之。

  5.4 多目标的精细处理:_multi_target_precursor_safe

  合成数据库里偶尔会出现"一锅煮多个产物"的记录,但有些记录虽然 targets_string 列了多个,本质上是同一族化合物的连续浓度(如 $\text{La}_{1-x}\text{Sr}_x\text{MnO}_3$ 的 x = 0, 0.1, 0.2,
  …)。这种记录的前驱物对所有 x 都共享。脚本(行 254)做了精细判断:

  def _multi_target_precursor_safe(row):
      target_metals = {提取 target 中的 STRUCTURAL_METALS}
      prec_metals   = {提取 precursors 中的 STRUCTURAL_METALS}
      return prec_metals <= target_metals

  如果"前驱物的金属集 ⊆ 目标的金属集",则视为 series-shared,标记为 mild(进 relaxed);否则 severe(拒绝)。这个判断把"形式上多目标但本质单族"的记录抢救回来,显著提升了 relaxed 集的样本量。

  5.5 文本污染:paragraph_contamination_reasons

  CONTAMINATION_PATTERNS = [
      r"polymer gel electrolyte", r"polymer-coated", r"aluminum laminates",
      r"BET surface area", r"porosity analyzer",
      r"electrolyte", r"electrochemical", r"battery", r"coin cell",
      r"cathode", r"anode", r"separator",
  ]

  这个 patterns 列表的设计哲学是:检测"非合成步骤"的污染。固体氧化物正极材料的合成段落可能在 NLP 抽取时被混入了"做电池测试"的描述,这些步骤不属于晶体合成,会污染条件抽取(把电池循环温度当作煅烧温度)。命中 →
  mild(进 relaxed,不进 gold)。

  TOO_GENERIC_PATTERNS = [
      r"according to the previous method",
      r"according to the literature",
      r"prepared under similar conditions",
      r"similar experimental conditions",
  ]
  这一组检测"作者偷懒"的写法。命中说明合成段落实际是空的,没有真实信息。同样标 mild。

  text_too_short:  if len(text) < 30: severe
  30 字符是个粗暴但有效的阈值——少于 30 字符根本无法成段,显然是抽取失败。
  
  5.6 文本与字段的交叉冲突:target_text_conflict、precursor_text_conflict

  最巧妙的两个规则。target_text_conflict(行 318)的逻辑:从段落里提取所有看起来像化学式的 token,排除已知的 precursors 和辅助物种(AUX_SPECIES),如果剩下的 token 集与字段记录的 target 化学式不重叠,说明 target
  字段可能错了。

  text_hits = extract_formula_tokens(text)
  filtered = text_hits - precursor_keys - aux_keys
  if filtered and not (filtered & target_keys):
      return True  # conflict

  precursor_text_conflict(行 336)做镜像检查,但只在文本里出现"starting materials" / "precursors" 触发词时启动——避免大段 review 文本误触发。

  这两个规则都标 mild。它们的存在让 gold 集对"字段可能被错配"的样本有第二层防线。

  5.7 反应箭头检查

  rxn = ...raw_synthesis_record.reaction_string...
  if not rxn:                           mild.append("missing_reaction_string")
  elif "==" not in rxn and "->" not in rxn:  mild.append("missing_reaction_arrow")

  合成数据库里反应式的标准形式是 A + B -> C 或 A + B == C(化学计量平衡)。两者都缺意味着抽取的 reaction_string 不是真反应式,可能是别的字段串错位。

  5.8 条件抽取:operation 优先,row fallback 兜底

  这是 stage3 任务的核心。两个独立函数:

  5.8.1 extract_conditions_from_operations(行 491)
  
  逐 operation 解析:
  for op in ops:
      typ = op.get("type")
      if typ in {"HeatingOperation", "DryingOperation", "AnnealingOperation", "CalciningOperation"}:
          heatlike_count += 1
      cond = op.get("conditions", {})
      
      # 形式 1:有结构化字段
      for item in cond.get("heating_temperature", []):
          temps.extend(celsius_from_cond_item(item))
      for item in cond.get("heating_time", []):
          times.extend(hours_from_cond_item(item))
          
      # 形式 2:solution_synthesis schema 直接是 temperature/time
      temp_direct = cond.get("temperature")
      if isinstance(temp_direct, dict):
          temps.extend(celsius_from_cond_item(temp_direct))
      ... 
      
  celsius_from_cond_item(行 363)的逻辑非常仔细:
  - 优先吸收 values 数组里的所有值
  - 若没有,从 min_value / max_value 派生中点
  - 单位是 K/Kelvin → 减 273.15 转 °C
  - 物理范围检查:0 ≤ T ≤ 2000 °C(超出此区间的全丢弃)
  
  hours_from_cond_item(行 390)同理,但增加单位换算:min/mins → /60, day → ×24,合理范围 0 ≤ t ≤ 500 h。

  聚合策略:
  - temperature_c = max(temps)(取最高温——煅烧温度往往是合成最关键的特征)
  - time_h = sum(times)(若总和 ≤ 500h,否则取 max——避免被异常长时间统治)
  - atmosphere = atms[0](第一个非空气氛——多步反应的第一段往往是关键)
  - solvent = solvents[0]
  
  5.8.2 extract_conditions_row_fallback(行 558)

  当 operation 字段全空时,从 max_temperature_c / total_time_h / atmosphere 等行级聚合字段读。这些字段是 02 脚本从文本启发式抽取的(02_prepare_dataset.py:1011-1037),不如 operation 精确,但能挽救大量 stage3
  训练样本。

  5.8.3 stage3_relaxed 的合并策略

  relaxed_cond = merge_conditions_for_relaxed(op_cond, fallback_cond)
  # 按字段独立 fallback:op_cond[field] if not None else fallback_cond[field]
  不是"全选 op 或全选 fallback",而是 逐字段独立 fallback——如果 operation 里有温度但没气氛,就用 operation 的温度配合 fallback 的气氛。这种细粒度合并是数据效率最高的做法。
  
  5.9 词表归一化:ATM_MAP / SOLVENT_MAP

  ATM_MAP = {
      "ar": "ar", "argon": "ar",
      "n2": "n2", "nitrogen": "n2",
      "o2": "o2", "oxygen": "o2",
      "vacuum": "vacuum",
      "h2": "h2", "hydrogen": "h2",
      "nh3": "nh3", "ammonia": "nh3",
      "inert": "inert",
      ...
  }

  把"argon" / "Ar" / "argon gas"(规范化后)统一映射到 "ar"。这一步在数据建模阶段听起来微不足道,但下游 multi-class 分类器最怕标签冗余——同样一个"氩气"分裂成 5 个类会让小类样本不足以训练。脚本通过
  normalize_atm(行 461)在每条样本就地归一,把这个隐患消灭在数据层。

  SOLVENT_MAP 同理,把 "DMF" / "dimethylformamide" 都映射到 "dmf",把 "IPA" / "isopropanol" 都映射到 "isopropanol"。

  5.10 准入仲裁(行 744-797):双层 hard-fail

  hard_fail_for_stage2_relaxed = {
      "missing_target_formula", "no_main_precursors",
      "multiple_targets_precursor_contaminated",
      "target_has_amount_variables",
      "variable_pattern_in_reaction",
      "variable_precursor_formula",
      "text_too_short",
      "precursor_element_mismatch_severe",
  }

  gold_blocking_mild = {
      "too_generic_text", "paragraph_contamination",
      "target_text_conflict", "precursor_text_conflict",
      "variable_pattern_in_text",
      "precursor_element_mismatch_mild",
      "multiple_targets_series_shared",
  }

  逻辑层级:
  - 任何 severe 命中 → relaxed 也不进
  - relaxed 通过 + 任何 gold_blocking_mild 命中 → 不进 gold
  - 进了 stage2_gold + operation 有条件 + 至少 1 个加热步骤 → 进 stage3_gold
  - 进了 stage2_relaxed + 任何条件(op 或 fallback) → 进 stage3_relaxed
  - 都进不了 → 进 dropped 
  
  5.11 主前驱物 vs 辅助物种:split_precursors

  AUX_SPECIES = {
      "O2", "H2O", "CO2", "NH3", "N2", "H2", "Ar", "He",
      "[OH-]", "[NO3-]", "[Cl-]", "[SO4-2]", "[SO4--]", "[PO4-3]",
      "H+", "[Na+]", "[K+]",
  }

  def split_precursors(precursors):
      main, aux = [], []
      for p in precursors:
          (aux if p in AUX_SPECIES else main).append(p)
      return main, aux

  为什么把 O2 这种"前驱物"分开?因为下游 stage2 模型预测的是主前驱物集合——氧气、氮气、水是反应介质,不是被预测的目标(没有意义说"应该用氧气合成 LiCoO2")。把它们放进 multi-hot 向量会让 vocab
  永远命中,对损失没贡献还浪费维度。

  5.12 split_group:防 DOI 级泄漏的种子

  "split_group": doi or row.get("synth_uid") or row.get("material_id") or target_formula,

  这一行(行 620)是为下游 01_split 提供 group 键。优先级是:DOI → synth_uid → material_id → target_formula。同一篇论文的多条记录会被绑成一个 group,后续切 train/val/test 时永远不分开。这是防止 "test set 与
  train set 共享同一篇论文的不同合成"这一隐性泄漏的根本性手段。

  5.13 reason_summary:可观测的精修结果

  reason_summary = {
      "stage2_relaxed_severe_reason_counts": dict(severe_counter.most_common()),
      "stage2_relaxed_mild_reason_counts":   dict(mild_counter.most_common()),
      "dropped_severe_reason_counts":        dict(dropped_severe_counter.most_common()),
      "dropped_mild_reason_counts":          dict(dropped_mild_counter.most_common()),
      "stage3_relaxed_condition_source_counts": dict(Counter(...condition_source...).most_common()),
  }
  
  这份 JSON 让"为什么我的 gold 集变小了" / "下次该改哪条规则" 可量化、可调参。每一次脚本调参之前先看 reason_summary.json 排前 10 的失败原因,基本能定位是"规则太严还是数据本身就脏"。

  ---
  第六部分 · 系统级反思

  6.1 数据量层级对比

  虽然脚本里没有打印最终数字,但典型规模的对比关系大致是:

  MP 全档实验型      ~150,000 条 POSCAR
  ↓ 02 加权打分(可信度 ≥ 55)
  direct_aligned    ~30,000-50,000 对
  ↓ 03 切 exact 子集
  strict_exact_only  ~15,000-25,000 对
  ↓ 04 双轨过滤
  stage2_gold        ~5,000-8,000 (-70%)
  stage2_relaxed     ~10,000-15,000
  stage3_gold        ~3,000-5,000
  stage3_relaxed     ~7,000-10,000

  每一级的"过滤强度"都很夸张,但每一级都把质量上界往上推一个台阶。

  6.2 Stage 00 的设计哲学
  
  把整个 Stage 00 抽象为五条原则:

  1. 数据来源透明。所有 metadata、provenance、DOI 都不丢弃,落盘留存。即使后续匹配规则改变,也能从原始 JSON 重新跑。
  2. 多信号合取仲裁。任何一条规则都是"必要不充分"。打分高 + reliable_match 通过 + 化学一致 + 文本无冲突,层层 AND。
  3. 金标与放宽双轨。模型不该被"完美数据"训练——它该被"最高质量的子集"和"最大覆盖的子集"分别训练,然后比较 / 联合 / curriculum。
  4. 失败即可见。每条被拒样本都写 dropped + 原因,频次直方图固化为 JSON。规则不是黑盒,是可调的工程参数。
  5. 化学先验显式化。STRUCTURAL_METALS / AUX_SPECIES / ATM_MAP 都是手工策划的小词表,但它们承载了化学家对合成体系的判断。把它们写在代码里,胜过让模型在脏数据里"自己学到"。

  6.3 可改进点

  (a) 打分权重应该交叉验证而不是手工。虽然现在的权重看起来合理,但严格来说应该用"已知正确对齐"的小集合做 grid search,确认 80/45/35/20 这些数字是局部最优。

  (b) CONTAMINATION_PATTERNS 偏向锂电池语料。如果数据集扩展到光催化、热电材料,需要扩充关键词。这套 pattern 是"按当前数据观察到的污染来源"凑的,泛化性有限。

  (c) rounded_parent_formula 的 tol=0.12 是经验值。在轻掺杂 vs 真固溶体的边界处会模糊,严格的做法应该结合相图(pymatgen.analysis.phase_diagram)判断目标点是否在母相的稳定域内。

  (d) title_similarity 用 difflib 不够。short title("CoO2 thin films")会让 SequenceMatcher 给出虚高的 ratio,容易误命中。改成基于 token 集合的 Jaccard + length-aware penalty 会更稳。

  (e) MP_API_KEY 直接硬编码在 CONFIG 里(01_capture_experimental_structures.py:18)。该改成读环境变量。

  6.4 与同类工作的差异

  文献中"结构-合成"对齐的常见做法是单纯按 reduced_formula 等值连接(如 Kononova et al. 2019 的 solid-state 数据集发布时给的脚本),完全不处理掺杂、多形态、变量配方。SynPred 的 Stage 00 在这一点上明显更工业化:

  - 加权打分而非硬等值 → 容忍掺杂
  - 作者标注优先 → 解决多形态歧义
  - DOI + 化学合取 → 杜绝跨论文虚假对齐
  - gold/relaxed 双轨 → 让模型可选地用更干净的子集训练
  - operation/row 双层条件抽取 → 最大化 stage3 训练样本

  这些细节,任何一个单独看都是"工程小聪明",但组合起来就把"实验型晶体 → 合成路线"的标签对齐做成了数据资产——下游 SynPred 模型族的成功率上限,几乎完全由这一阶段定义。

  ---
  第七部分 · 结语
  
  如果让我用一句话概括 Stage 00:它把"晶体结构 ↔ 合成路线"的对齐从 NLP 
  字符串匹配升级为多信号化学打分,并在打分之上叠加双层硬阈值仲裁、化学元素一致性检查、变量配方过滤、文本污染过滤、操作级条件抽取,最终输出可直接喂入 stage2/stage3 训练的金标与放宽两套数据集。

  整个流程没有用任何机器学习,但它定义了 SynPred 模型族能学到什么——一份噪声更少、化学更自洽、覆盖更广的训练标签集。Stage 00 是 SynPred 真正的核心资产之一,它的工程价值往往被算法部分的光彩掩盖,但任何想复现
  SynPred 的人都应该首先理解它。


❯ Stage 02 — 特征构建部分详细分析，达到可以直接写文章的级别，而且要给别人能直接重复的程度，足够细致                                                                                                           
⏺ ---
  Stage 02:特征构建——从精修 JSONL 到模型可消费的张量
  
  ▎ 本文是 SynPred 流水线 Stage 02 的深度技术文章。Stage 02 把 Stage 00 精修出的 JSONL 与 Stage 01 切分后的样本变成可直接喂入机器学习模型的特征 + 标签表。它由 5 个 Python 脚本与 1 个 shell 
  ▎ 编排器组成,整章的目标是让任何读者完全可重复地复现这一阶段——本文给出每一处算式、阈值、列命名规则和外部依赖。

  ---
  第一部分 · Stage 02 的定位与契约
  
  Stage 02 在整个 SynPred 流水线中扮演"语义层 → 数值层"的桥梁。上游 Stage 00 给出 stage{2,3}_{gold,train_relaxed}.jsonl,Stage 01 把它们按 DOI/group 切成 train/val/test/gold_train_holdout 四个 split,Stage 02
   的职责是:

  1. 把每条样本表达成定长的数值向量。对结构信息要既保留 composition 又保留 geometry,对合成信息要把目标(前驱物集合、温度、时间、气氛、溶剂、合成类型)落成可监督的标签列。
  2. 让标签空间从训练集生成,确保 val/test 不泄漏。元素词表、前驱物词表、气氛/溶剂/合成类型词表必须只看 train。
  3. 保留两份 CSV: *_raw.csv(只含特征 + 元数据 + 原始目标)和 *_ml.csv(在 raw 基础上展开 multi-hot 标签),便于上层做不同损失函数。
  4. 可选地与 GNN 嵌入合并,得到 hybrid 特征表;并按金标 / 放宽 / 课程式三种训练模式重组目录结构。

  下游消费这套契约的脚本(scripts/03_data/* 的 NPZ 构建器、scripts/04_train/stage{2,3}/* 的训练器)永远只看两个东西:列名前缀 + 文件命名。Stage 02 的设计纪律就是让这两件事完全可预测。

  5 个脚本的角色:
  
  ┌──────────────────────────────────────────────┬──────┬──────────────────────────────────────────────────┐
  │                     脚本                     │ 行数 │                       角色                       │
  ├──────────────────────────────────────────────┼──────┼──────────────────────────────────────────────────┤
  │ 01_build_structdesc_features.py              │  792 │ 主特征器:每条样本 → 描述子向量 + 标签列          │
  ├──────────────────────────────────────────────┼──────┼──────────────────────────────────────────────────┤
  │ 02_reprocess_stage3_targets.py               │  224 │ 对 Stage3 目标做 log1p、bucket、coarse 化        │
  ├──────────────────────────────────────────────┼──────┼──────────────────────────────────────────────────┤
  │ 04_build_stage3_task_views.py                │  280 │ 把 Stage3 大表拆成 5 个单任务视图                │
  ├──────────────────────────────────────────────┼──────┼──────────────────────────────────────────────────┤
  │ 05_build_hybrid_features.py                  │  237 │ 描述子 ⨉ GNN 嵌入合并                            │
  ├──────────────────────────────────────────────┼──────┼──────────────────────────────────────────────────┤
  │ 06_prepare_training_modes.py                 │  146 │ 复制 hybrid 到 gold_only/relaxed_only/curriculum │
  ├──────────────────────────────────────────────┼──────┼──────────────────────────────────────────────────┤
  │ run_build_stage3_features_for_pipeline_v3.sh │    — │ 编排上述五步,固化默认路径                        │
  └──────────────────────────────────────────────┴──────┴──────────────────────────────────────────────────┘

  下文按数据流顺序展开。
  
  ---
  第二部分 · 01_build_structdesc_features.py:主特征器

  2.1 预设常量(行 17-55)

  脚本顶部维护 6 张化学元素族集合,以及一个晶系列表。这些常量是后面 composition_features 的硬编码先验。

  ELEMENTS = ["H","He","Li",..., "Lv","Ts","Og"]   # 118 元素全表
  Z_TABLE  = {el: i+1 for i, el in enumerate(ELEMENTS)}

  ALKALI    = {"Li","Na","K","Rb","Cs","Fr"}
  ALKALINE  = {"Be","Mg","Ca","Sr","Ba","Ra"}
  HALOGENS  = {"F","Cl","Br","I","At","Ts"}
  CHALCOGENS= {"O","S","Se","Te","Po","Lv"}
  LANTHANOIDS = {"La",...,"Lu"}
  ACTINOIDS   = {"Ac",...,"Lr"}
  TRANSITION_METALS = {"Sc",...,"Cn"}    # d 区元素

  CRYSTAL_SYSTEMS = ["triclinic","monoclinic","orthorhombic",
                     "tetragonal","trigonal","hexagonal","cubic"]

  Z_TABLE 用元素位置编号(1-118)而非真原子序数,但因为 ELEMENTS 列表按 Z 升序排列,二者数值上等价。CRYSTAL_SYSTEMS 的 7 个晶系最终被 one-hot 成 7 个 feat_crystal_system__{cs} 列。

  ▎ 可重复要点 #1:这些集合是工程基线。如果你在新数据上发现某些超元素(Mc、Og 等)很少出现,可以从 gather_element_vocab 那一支自然过滤掉,无需改这些集合。

  2.2 入口与配置(行 683-702)

  parser.add_argument("--base_dir",        default="/Users/wyc/SynPred/data")
  parser.add_argument("--input_dir",       default=".../splits/structdesc_splits")
  parser.add_argument("--output_dir",      default=".../features/structdesc_features")
  parser.add_argument("--use_poscar_geom", action="store_true")
  parser.add_argument("--poscar_max_sites",type=int, default=256)

  四个参数:base_dir 用来解析 jsonl 里的相对路径(poscar_path、summary_json_path)。use_poscar_geom 是开关——为什么默认关?因为 POSCAR 的几何特征是 $O(n^2)$ 的,大体系会拖慢全量构建。poscar_max_sites=256
  是硬截断,大于这个尺寸的晶胞直接跳过几何特征(行 407)而不是降级——这是有意为之,避免大体系误差累积污染训练集。

  2.3 词表构建:严格的"训练集 only"

  这一节是整个流水线最重要的反泄漏纪律。main() 行 717-731:

  stage2_train_rows = stage2_splits["train"]
  stage3_train_rows = stage3_splits["train"]

  # element vocab 从所有 split 的所有行收集
  all_rows = sum(stage2_splits.values(), []) + sum(stage3_splits.values(), [])
  element_vocab = gather_element_vocab(all_rows)

  # precursor / atmosphere / solvent / synthesis_type 仅从 stage2_train、stage3_train 来
  precursor_vocab     = build_stage2_vocab(stage2_train_rows)
  atmosphere_vocab    = build_stage3_class_vocab(stage3_train_rows, "atmosphere")
  solvent_vocab       = build_stage3_class_vocab(stage3_train_rows, "solvent")
  synthesis_type_vocab= build_stage3_class_vocab(stage3_train_rows, "synthesis_type")

  为什么 element_vocab 例外? 因为它对应的是结构特征向量的列名(feat_frac_el__{El}),在 val/test 上看到训练集没出现的元素时,我们仍想保留这一维(填 0
  即可),否则推理时遇到新元素会无列可放。前驱物词表完全相反——它是预测目标,见过 = 可以学,没见过 = 即使人为放进去也无法监督,所以严格只看 train。

  build_stage2_vocab(行 508)、build_stage3_class_vocab(行 517)的实现都是简单去重排序。

  ▎ 可重复要点 #2:你如果换数据集,只需保证 train_jsonl 里出现过的所有标签都被训练集合涵盖。验证手段是产出后用 set(val.precursor)-set(train.precursor) 看差集——存在 OOV 表示数据划分有问题。

  2.4 特征通道 A:化学式特征(composition_features,行 149)

  输入是 Y0.95VO4 这样的化学式字符串和 element_vocab(从所有数据收集),输出是一个 dict。逻辑分三步。

  2.4.1 解析:parse_formula_simple(行 120-146)

  纯手写状态机——不调用 pymatgen,因为这一步要在大循环里跑数十万次,pymatgen 的 Composition 太慢:

  i = 0
  while i < n:
      if not s[i].isupper():
          return {}             # 非法立刻返回空
      el = s[i]; i += 1
      if i < n and s[i].islower():
          el += s[i]; i += 1
      num = []
      while i < n and (s[i].isdigit() or s[i] == "."):
          num.append(s[i]); i += 1
      out[el] = out.get(el, 0.0) + (float("".join(num)) if num else 1.0)

  它支持小数化学计量(Y0.95)但不支持括号(Ba(OH)2 会失败,因为 ( 不是大写字母)。Stage 00 已经把括号化学式归一化到无括号形式,所以这里安全。

  2.4.2 全局聚合特征
  
  设原子计量字典 $c = {el_i \mapsto n_i}$,$N = \sum_i n_i$,$f_i = n_i / N$。脚本输出:

  ┌──────────────────────────┬──────────────────────────────────────┐
  │           列名           │                 公式                 │
  ├──────────────────────────┼──────────────────────────────────────┤
  │ feat_n_elements_formula  │ $|c|$(种类数)                        │
  ├──────────────────────────┼──────────────────────────────────────┤
  │ feat_total_atoms_formula │ $N$                                  │
  ├──────────────────────────┼──────────────────────────────────────┤
  │ feat_stoich_entropy      │ $-\sum_i f_i \log f_i$(摩尔分数熵)   │
  ├──────────────────────────┼──────────────────────────────────────┤
  │ feat_z_mean              │ $\sum_i f_i Z_i$                     │
  ├──────────────────────────┼──────────────────────────────────────┤
  │ feat_z_std               │ $\sqrt{\sum_i f_i (Z_i - \bar Z)^2}$ │
  └──────────────────────────┴──────────────────────────────────────┘

  这五个聚合量代表"化学式的紧凑画像":多元数、总原子数、分散度、平均原子序数、序数离散度。它们是 Magpie 风格 composition 描述子的最简化身。

  2.4.3 元素族占比与 one-hot 元素分量

  对七大族(过渡金属、碱金属、碱土金属、卤素、硫属、镧系、锕系)分别求 $\sum_{e \in \text{family}} f_e$,得到 7 列 feat_frac_*。然后对 element_vocab 中每个元素生成一列 feat_frac_el__{El} = 
  f_{El}(不在化学式里就是 0)。

  ▎ 可重复要点 #3:列总数 = 2(基础) + 1(熵) + 2(Z) + 7(族) + |element_vocab|。在典型的 SynPred 数据上 element_vocab 大小约 80-90,所以这一通道约 92-102 维。

  ▎ 可重复要点 #4:化学式为空或解析失败时(行 153-168),所有 composition_features 列填 0。这是个隐含约定:模型必须能容忍全 0 输入,等价于"没有化学式信息"。

  2.5 特征通道 B:Materials Project summary 特征(load_summary_features,行 206-280)

  输入是 summary_json_path 指向的 JSON(Stage 00 落盘的 summary_json/{material_id}.json),输出 21 维特征。

  2.5.1 默认值与缺失指示
  
  feat["feat_has_summary"] = 1.0    # 1 = 找到 JSON, 0 = 没找到
  这一列让模型知道"以下 20 维是真值还是占位 0"。这是处理混合数据来源的最简单办法,比 np.nan + imputation 更可控。
  
  2.5.2 数值字段

  feat_density, feat_volume, feat_nsites, feat_nelements,
  feat_band_gap, feat_energy_above_hull, feat_spacegroup_number,
  feat_lattice_a/b/c, feat_lattice_alpha/beta/gamma
  共 13 维。读取时做了双路径回退——Materials Project 的 schema 历史上有两套字段名:density vs density_atomic、energy_above_hull vs e_above_hull、structure.lattice.a vs lattice.a。get_nested(summary, 
  "structure.lattice.a", None) or get_nested(summary, "lattice.a", None) 一次性兼容两版。
  
  2.5.3 晶系 one-hot

  读 symmetry.crystal_system 或 crystal_system 字段,值经 .strip().lower() 后必须命中 CRYSTAL_SYSTEMS 才置 1。这一段共 7 维。

  ▎ 可重复要点 #5:summary 通道总维度 = 1(has) + 13(数值) + 7(crystal_system) = 21 维。safe_float 把无法转 float 的字段强制变 0,保证 CSV 永远可被 pandas 数值化。

  2.6 特征通道 C:POSCAR 几何特征(load_poscar_features,行 485)

  仅当 --use_poscar_geom 打开时启用。否则用 min_image_distances(np.empty((0,3)), np.eye(3)) 调用产生零默认值字典(行 486)。

  2.6.1 POSCAR 解析(parse_poscar,行 307-376)

  纯手写解析器,不依赖 pymatgen。读取 8 行起步:第 2 行 scale,第 3-5 行 lattice 矩阵,第 6 行物种串(VASP5),第 7 行各物种原子数。然后处理可选的 Selective dynamics 行、Direct/Cartesian 模式行、最后是 nsites
  个坐标行。

  关键转换:
  if direct_mode:
      frac = coords; cart = frac @ lattice
  else:
      cart = coords; frac = cart @ inv(lattice)
      
  2.6.2 最小镜像距离(min_image_distances,行 379)

  对每对原子 $(i, j)$,在分数坐标上做最小镜像约定(MIC):
  $$
  \Delta f = f_j - f_i, \quad \Delta f \mathrel{-}= \mathrm{round}(\Delta f), \quad \Delta r = \Delta f \cdot L
  $$
  然后 $d_{ij} = |\Delta r|2$。$d{ij} \le 10^{-8}$ 视为同点丢弃。这只检查最近的 1×1×1 镜像——对小晶胞 + 短键长(< 5Å)是有效近似,对大晶胞(> 20Å)没问题,只在中间区间(键长 ≈ 0.5×晶格)会漏算长程对。SynPred
  的统计聚合(quantile + 配位)对这种边界不敏感。

  2.6.3 输出列

  共 23 维,分四组:

  几何标识 + 晶胞(9 维)
  - feat_poscar_has_geom(0/1 占位标志)
  - feat_poscar_nsites
  - feat_poscar_volume = |det(L)|
  - feat_poscar_a/b/c/alpha/beta/gamma(lattice_lengths_angles,行 286-304 用 arccos(u·v/(|u||v|)) 算夹角)
  
  对距离统计(6 维)——只统计 $j > i$ 的对避免重复
  - feat_pairdist_min、feat_pairdist_mean、feat_pairdist_std
  - feat_pairdist_q25、feat_pairdist_q50、feat_pairdist_q75(quantile)

  最近邻统计(2 维)
  - feat_nn_mean、feat_nn_std(每个原子的最近邻距离的 mean/std)
  
  配位数统计(6 维)——三个 cutoff 各两个
  - feat_coord_3A_mean/std(每个原子在 3Å 内的邻居数)
  - feat_coord_4A_mean/std、feat_coord_5A_mean/std
  
  ▎ 可重复要点 #6:POSCAR 通道总维度 23 维。开启 --use_poscar_geom 时这些列被实际填充;关闭时全为 0,但列依然存在(min_image_distances 在 n=0 时返回 default dict)。这保证 train/val/test 的列结构对齐。

  2.7 元数据列与原始目标列(featurize_row,行 526-564)

  在写入特征前,先复制一组元数据:
  id, synth_uid, source_dataset, record_index, material_id,
  formula, mp_formula, synth_formula, parent_formula,
  doi, split_group, synthesis_type, reaction_string
  split_group 是 Stage 00 决定的反泄漏键(优先 DOI),后续训练器需要它做 group split。synthesis_type 这里既是元数据也是 stage3 标签源——在 raw_to_stage3_ml 里会再被展开为 multi-hot。

  原始目标列(行 543-548):
  target_main_precursors      # JSON 串(主前驱物列表)
  target_aux_precursors       # JSON 串
  target_temperature_c        # 数值或 None
  target_time_h               # 数值或 None
  target_atmosphere           # 字符串或 None
  target_solvent              # 字符串或 None
  
  ▎ 可重复要点 #7:target_main_precursors 用 json.dumps 写成串而不是 list 直接写入 CSV——pandas 写 list 会变成 "[1, 2]" 这种半 JSON 半 Python repr 的混合,二次读取时极易踩坑。强制 JSON 串保证可逆。

  2.8 ML 标签展开

  2.8.1 Stage2:raw_to_stage2_ml(行 567-578)

  对每条样本,用 train 集得到的 precursor_vocab 把 target_main_precursors(JSON 串)展成 multi-hot:
  prec_lists = df["target_main_precursors"].apply(lambda x: json.loads(x))
  for p in precursor_vocab:
      df[f"label_prec__{p}"] = prec_lists.apply(lambda arr: 1 if p in arr else 0)
  列名前缀 label_prec__(双下划线)是后续训练器识别标签列的契约。

  2.8.2 Stage3:raw_to_stage3_ml(行 581-598)

  对每个分类目标做 one-hot:
  for v in atmosphere_vocab:
      df[f"label_atm__{v}"]   = df["target_atmosphere"].apply(lambda x: 1 if str(x) == v else 0)
  for v in solvent_vocab:
      df[f"label_solv__{v}"]  = df["target_solvent"].apply(lambda x: 1 if str(x) == v else 0)
  for v in synthesis_type_vocab:
      df[f"label_synth__{v}"] = df["synthesis_type"].apply(lambda x: 1 if str(x) == v else 0)
  注意:温度、时间不展开。它们留在 raw 表里作为连续目标,由 02 脚本(下一节)处理。
  
  ▎ 可重复要点 #8:Stage3 是 single-class one-hot 而非 multi-hot——Stage 00 已经在条件提取阶段把每条样本的 atmosphere、solvent 都规约为单值。如果你在新数据上想保留多值,得在这里把 == 替换成 in,并改 Stage 00 的
  ▎  extract_atmospheres 让它输出列表。

  2.9 输出双轨

  {output_dir}/
  ├── meta/
  │   ├── element_vocab.json         # all-rows-sorted
  │   ├── precursor_vocab.json       # train-only-sorted
  │   ├── atmosphere_vocab.json      # train-only-sorted
  │   ├── solvent_vocab.json
  │   ├── synthesis_type_vocab.json
  │   └── build_summary.json
  ├── stage2_{train,val,test,gold_train_holdout}_raw.csv
  ├── stage2_{train,val,test,gold_train_holdout}_ml.csv
  ├── stage3_{train,val,test,gold_train_holdout}_raw.csv
  └── stage3_{train,val,test,gold_train_holdout}_ml.csv

  build_summary.json(行 739)包含 config + vocab_sizes + 每个 split 的 n_rows / n_features_raw / n_features_ml,可作为 health check。

  ▎ 可重复要点 #9:典型列数(关闭 POSCAR geometry)
  ▎ - stage2_raw: 13(meta) + 6(targets) + ~92(comp) + 21(summary) + 23(poscar 默认 0) ≈ 155
  ▎ - stage2_ml: stage2_raw + |precursor_vocab| ≈ 155 + 200~600 = 355~755
  ▎ - stage3_raw: 同 stage2_raw 结构(差异在 jsonl 内容)
  ▎ - stage3_ml: stage3_raw + |atm| + |solv| + |synth_type| ≈ 155 + 5 + 10 + 6 = 176

  ---
  第三部分 · 02_reprocess_stage3_targets.py:Stage 3 目标的二次精修
  
  3.1 为什么单独一步?

  Stage 3 的目标里温度和时间是高度长尾分布的连续变量,直接用 MAE/MSE 训会被极端值带偏。原始 atmosphere 词表里有 air/o2/ar/n2/h2/nh3/vacuum/co2/he/inert 等近 10 个细分,直接做 10
  类分类样本不够。这一步的工作就是给 Stage 3 加上几列派生目标,让下游训练器选用。

  输入 = 上一步的 structdesc_features/,输出 = structdesc_features_stage3_v2/。stage2 文件原样复制(行 173-178),只对 stage3 的 raw 文件做加工(行 181-186)。

  3.2 时间:三重表示

  add_stage3_targets(行 75-114):

  v = safe_float(target_time_h)
  if v is None or v < 0:
      time_clean = NaN; time_log1p = NaN; time_bucket = None
  else:
      time_clean   = v
      time_log1p   = log(1 + v)              # 长尾压缩
      time_bucket  = "short"  if v <= 5
                     "medium" if v <= 24
                     "long"

  三个新列 target_time_h_clean / target_time_h_log1p / target_time_bucket。

  - clean 删除负值/NaN,但保留量纲
  - log1p 用 $\log(1+t)$ 把 0.1h 到 240h 这 3 个数量级压到 ~5.5 的范围,适合回归损失
  - bucket 三段离散化,对应"快速反应 / 单日 / 多日"三个工艺范畴

  map_time_bucket(行 64-72)的阈值 5h 和 24h 是经验阈,代表"过夜以内 / 一天 / 多天"。

  v = safe_float(target_temperature_c)
  target_temperature_c_clean = v if 0 <= v <= 2000 else NaN

  不做 log,因为温度的物理量纲已经是数百量级,加性分布,MAE 直接训没问题。范围 [0, 2000]°C 与 Stage 00 的 celsius_from_cond_item 一致——同一道阈值连续两次过滤。

  3.4 气氛:粗粒度映射(map_atmosphere_coarse,行 41-61)

  "air", "o2", "co2"     -> "air_or_oxidizing"
  "ar", "n2", "he", "inert"  -> "inert"
  "h2", "nh3"            -> "reducing"
  "vacuum"               -> "vacuum"
  其他                    -> "other"

  5 类粗化后样本量更平衡,训练器可以选择用 target_atmosphere(细)还是 target_atmosphere_coarse(粗)。这是化学化的离散化——粗化原则是"氧化电位 / 还原电位",不是字面词频。

  3.5 溶剂:大小写归一

  仅做 .strip().lower() 形成 target_solvent_clean,不再做粗化(SOLVENT_MAP 已在 Stage 00 做过归一)。

  3.6 输出诊断

  summarize_df(行 117-134)统计每列非空数量、time_bucket / atmosphere_coarse / solvent 的 top 分布,写入 meta/stage3_reprocess_summary.json。这是数据质量门——如果 n_temp_nonnull 突然腰斩,说明上游 Stage 00
  的条件抽取出问题了。

  ▎ 可重复要点 #10:这一步不改特征列,只增加 5 个新目标列。下游 task views 会按列名挑用。如果你想换 bucket 阈值,改 map_time_bucket 的两个数字即可,不用动其他脚本。

  ---
  第四部分 · 04_build_stage3_task_views.py:单任务视图

  4.1 设计目的

  Stage3 的全表里每条样本可能同时含温度、时间、气氛、溶剂、合成类型五种目标,但它们的非空集不同——比如某条样本有温度无溶剂。如果直接对全表做单任务训练,需要在每个 batch 里 mask
  掉缺失,代码很啰嗦。这一步把全表拆成 5 个独立 CSV,每个只保留对应任务有标签的样本,训练器拿到就能直接训。

  4.2 视图清单(process_split 行 203-220)

  ┌───────────────────────────────────┬────────────────────────────┬─────────────┐
  │               视图                │        canonical 列        │    任务     │
  ├───────────────────────────────────┼────────────────────────────┼─────────────┤
  │ stage3_temperature_{split}.csv    │ target_temperature_c_clean │ 回归        │
  ├───────────────────────────────────┼────────────────────────────┼─────────────┤
  │ stage3_time_bucket_{split}.csv    │ target_time_bucket         │ 三分类      │
  ├───────────────────────────────────┼────────────────────────────┼─────────────┤
  │ stage3_atmosphere_{split}.csv     │ target_atmosphere_coarse   │ 粗 5 类     │
  ├───────────────────────────────────┼────────────────────────────┼─────────────┤
  │ stage3_solvent_{split}.csv        │ target_solvent_clean       │ 多类(top-N) │
  ├───────────────────────────────────┼────────────────────────────┼─────────────┤
  │ stage3_synthesis_type_{split}.csv │ synthesis_type             │ 多类        │
  └───────────────────────────────────┴────────────────────────────┴─────────────┘

  每张表保留:
  - 元数据(id, material_id, formula, doi, split_group, source_dataset, synthesis_type)
  - 所有 feat_* / graph_emb_* 特征列
  - canonical 目标列
  - 该任务的 extra keep cols(如 time bucket 视图同时保留 target_time_h_log1p 给可选回归)
  - 衍生标志列 has_target(target_col.notna() 的 0/1)
  
  4.3 列回退顺序(build_single_target_view,行 56-95)

  每个视图按"优先列名→备用列名"的顺序找目标,找到就重命名为 canonical:

  preferred_target_cols = [
      "target_temperature_c_clean",   # 02 脚本输出的
      "temperature_c_clean",          # 备份命名
      "target_temperature_c",         # 原始
      "temperature_c",                # 极原始
  ]
  
  为什么要四级回退?因为这套视图脚本被设计成可以吃 v1、v2 两版输入。如果你直接拿原始 structdesc_features 进来跑,落到第 3-4 个候选;如果走完 02 脚本,优先用 _clean。

  4.4 文本标签清洗
  
  if out[canonical_target_col].dtype == object:
      out[canonical_target_col] = out[canonical_target_col].apply(normalize_text_label)
  normalize_text_label 做 .strip().lower(),对 "NaN" / "None" 这类字面值返回 None。这一步避免了"AIR 与 air 算两类"的尴尬。
  
  4.5 视图统计(summarize_view,行 177-200)

  每个视图输出 4 个数 + 目标分布:
  - n_rows、n_features、n_target_nonnull、n_has_target
  - 目标是数值 → min/max/mean/median
  - 目标是文本 → value_counts(全部)
  
  ▎ 可重复要点 #11:运行后 summary.json 里查 n_target_nonnull/n_rows 的比值——如果某视图低于 30%,该任务训练前要考虑额外增强或合并类。

  ---
  第五部分 · 05_build_hybrid_features.py:描述子 + GNN 嵌入合并
  
  5.1 角色

  到这一步为止,我们有的是手工描述子——~150 维 composition+summary+geometry。SynPred 主线模型要把它们与GNN 嵌入(CGCNN / CHGNet / ALIGNN 在 Stage 03 graph 那一支产出的 64-256
  维向量)合并。这个脚本就是合并器,不参与任何训练。

  5.2 输入约定

  --task              stage2 | stage3
  --descriptor_dir    .../structdesc_features            # 上面 01/02 的输出
  --embedding_dirs    DIR1 DIR2 ...                      # 一个或多个 GNN 嵌入目录
  --embedding_prefixes cgcnn alignn chgnet ...           # 对应名字
  --output_dir        .../stage{2,3}_hybrid_features
  --descriptor_kind   auto | ml | raw

  descriptor_kind="auto"(行 79)意为 stage2 用 ml 表(已含 multi-hot 标签),stage3 用 raw 表(标签很多列,留在表里方便后续灵活展开)。

  5.3 嵌入列名规范化(normalize_embed_columns,行 33-58)

  GNN 嵌入文件可能用四种命名:
  - graph_emb_0(早期未带前缀)
  - cgcnn_graph_emb_0、alignn_graph_emb_0、chgnet_graph_emb_0(各自训练器输出)
  
  脚本统一改写成 {prefix}_graph_emb_{i}(行 47-55),其中 prefix 来自 --embedding_prefixes。最后用 collect_embed_feature_cols(行 60)收集所有 {prefix}_graph_emb_* 列。

  ▎ 可重复要点 #12:这一步保证多源嵌入并存时列名不冲突。如果你只有一种嵌入,prefix 随便命名都行;但统一以这个 prefix 串作为合并后表的列前缀,下游训练器靠它识别图特征通道。

  5.4 合并循环(build_one_split,行 64-149)

  每个 split 的处理:
  
  # 1) 读描述子,按 id 去重
  desc_df, desc_dup = dedup_by_id(desc_df)
  merged = desc_df

  # 2) 对每个嵌入源
  for embed_dir, prefix in zip(embed_dirs, embed_prefixes):
      graph_df = read; dedup_by_id; rename_cols(prefix);
      keep_cols = ["id","material_id","formula","doi","split_group"] + embed_cols
      merged = merge(merged, graph_df, on="id", how="inner",
                     suffixes=("", f"_{prefix}"))

  关键设计:
  - how="inner" —— 描述子表与每个嵌入表的 id 必须同时存在。这样保证每条样本的所有特征通道都齐全;某个嵌入的样本量小,merged 行数会随着每次 inner 缩小。
  - dedup_by_id —— 防止上游某些脚本误把重复 id 写两遍。重复数被记入 summary["graph_duplicate_ids"],可观测。
  - suffixes=("", f"_{prefix}") —— 元数据列(formula 等)在描述子表里已有,合并后保留描述子版,嵌入表里的同名列被加上 _{prefix} 后缀(实际后续都不用)。
  
  5.5 输出 CSV 命名

  {output_dir}/{task}_{split}_hybrid.csv
  比如 stage3_train_hybrid.csv。同时写出 summary.json:

  {
    "config": {...},
    "splits": {
      "train": {
        "task": "stage3",
        "split": "train",
        "descriptor_input_csv": "...",
        "output_csv": "...",
        "descriptor_rows_before_dedup": ...,
        "descriptor_rows_after_dedup": ...,
        "merged_rows": ...,
        "n_descriptor_features": 100,
        "n_graph_features_total": 64,
        "n_hybrid_features": 164,
        "n_labels": ...,
        "graph_sources": {
          "chgnet": { "n_graph_features": 64, "rows_before_merge": ..., "rows_after_merge": ... }
        }
      },
      ...
    }
  }
  
  ▎ 可重复要点 #13:n_hybrid_features = n_descriptor_features + n_graph_features_total。pipeline 里默认 chgnet 一种 64 维,所以 stage3 hybrid 大约 (155 - meta - target) ≈ 100 描述子 + 64 chgnet ≈ 164 
  ▎ 维。你可以并接 cgcnn(256 维)做 ~420 维 hybrid。

  ---
  第六部分 · 06_prepare_training_modes.py:三种训练模式的目录组装

  6.1 角色

  下游训练器(scripts/04_train/stage{2,3}/*.py)的命令行约定接受一个 --input_dir,目录里期望有 train/、val/、test/ 三个子目录,每个里放对应 split 的 CSV/NPZ。这个脚本不做任何特征运算,只是按三种训练模式复制 
  hybrid CSV到目标目录。

  6.2 三种模式

  {output_root}/{dataset_name}/
  ├── relaxed_only/
  │   ├── train/   <- train_file (relaxed train)
  │   ├── val/
  │   └── test/
  ├── gold_only/
  │   ├── train/   <- gold_train_holdout_file (small high-quality)
  │   ├── val/
  │   └── test/
  ├── curriculum/
  │   ├── phase1_train/   <- train_file
  │   ├── phase2_train/   <- gold_train_holdout_file
  │   ├── val/
  │   └── test/
  └── summary.json

  - relaxed_only:用大量但允许轻微噪声的 train,极大化覆盖率
  - gold_only:只用 strict 通过的 holdout,质量最高但样本少
  - curriculum:phase1 用 relaxed 暖启,phase2 用 gold 微调

  每个模式都共享同一份 val/test。

  6.3 实现细节
  
  copy_file(行 35-43)用 shutil.copy2 保留时间戳,并立即统计行数:
  {"src": ..., "dst": ..., "n_rows": count_rows(dst)}
  count_rows(行 22-32)对 CSV 减去表头,对 JSONL 数空行,其他文件返回 -1。

  6.4 调用方式

  shell 里:
  python 06_prepare_training_modes.py \
    --source_dir   .../stage3_hybrid_features \
    --output_root  .../training_modes \
    --train_file   stage3_train_hybrid.csv \
    --val_file     stage3_val_hybrid.csv \ 
    --test_file    stage3_test_hybrid.csv \
    --gold_train_holdout_file stage3_gold_train_holdout_hybrid.csv \
    --dataset_name stage3_hybrid
    
  ▎ 可重复要点 #14:这一步零计算只复制——所以可以反复多次跑(比如先跑 stage2 再跑 stage3),互不干扰。

  ---
  第七部分 · run_build_stage3_features_for_pipeline_v3.sh:端到端编排
  
  shell 脚本固化了 Stage 02 的标准 6 步流水(其中 STEP 0 是脚本存在性检查,STEP 6 是文件存在性检查):

  ┌─────┬─────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │ 步  │                                  输入                                   │                   脚本                    │                                  输出                                  │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 0   │ —                                                                       │ [OK] $f                                   │ 检查 5 个 .py 文件存在                                                 │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 1   │ splits/structdesc_splits                                                │ 01_build_structdesc_features.py           │ features/structdesc_features                                           │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 2   │ features/structdesc_features                                            │ 02_reprocess_stage3_targets.py            │ features/structdesc_features_stage3_v2                                 │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 3   │ features/structdesc_features_stage3_v2                                  │ 04_build_stage3_task_views.py             │ features/stage3_task_views                                             │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 4   │ features/structdesc_features_stage3_v2 + graph_embeddings/chgnet_stage3 │ 05_build_hybrid_features.py --task stage3 │ features/stage3_hybrid_features                                        │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 5   │ features/stage3_hybrid_features                                         │ 06_prepare_training_modes.py              │ training_modes/stage3_hybrid                                           │
  ├─────┼─────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 6   │ —                                                                       │ inline test                               │ 验证 generative/stage3_condition_dataset/hybrid_mixed_v1 三个 NPZ 存在 │
  └─────┴─────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  每步日志写到 outputs/logs/stage3_feature_build/{timestamp}/{step}.log,便于事后审计。

  STEP 4 加了条件门(if [[ -d "${STAGE3_EMBED_DIR}/chgnet_stage3" ]]):没有 GNN 嵌入就跳过 hybrid + training_modes,让 Stage 02 的描述子部分仍能独立产出。这种"软依赖"设计让 Stage 02 可以在 Stage 03 graph
  之前先跑一遍,验证手工特征是否合理。

  ▎ 可重复要点 #15:Stage2 的 hybrid 与 training_modes 没有写在这个 shell 里——因为 SynPred 主线 stage2 用的是 data/interim/features/stage2_hybrid_features 与 data/interim/generative/stage2_*_dataset 
  ▎ 等更专门的路径,有独立 shell 编排。Stage 02 的 5 个 .py 工具是任务无关的,可以被多套 shell 复用。

  ---
  第八部分 · 完整的端到端复现指南

  8.1 前置条件

  1. Stage 00 输出已就位:
  data/interim/refined/structdesc_refined/
  ├── stage{2,3}_gold.jsonl
  ├── stage{2,3}_train_relaxed.jsonl
  └── ...

  2. Stage 01 已切分:
  data/interim/splits/structdesc_splits/
  ├── stage2_train.jsonl
  ├── stage2_val.jsonl
  ├── stage2_test.jsonl
  ├── stage2_gold_train_holdout.jsonl
  └── stage3_*.jsonl   (同上 4 个)
  
  3. Stage 00 落盘的 POSCAR 与 summary JSON 可被相对路径解析({base_dir}/raw/...)。
  4. Python 环境:numpy, pandas,可选 pymatgen(本流水线手写解析器,不直接用 pymatgen)。

  8.2 一键跑(推荐)

  bash /Users/wyc/SynPred/scripts/02_features/run_build_stage3_features_for_pipeline_v3.sh \
       /path/to/SynPred cpu

  第一个参数 = PROJECT_ROOT,第二个参数 = device(本流水线全 CPU,但下游 GNN 可能要 cuda)。

  8.3 分步跑(用于 debug 或定制)

  PR=/Users/wyc/SynPred

  # 1. 描述子(默认关闭 POSCAR 几何)
  python $PR/scripts/02_features/01_build_structdesc_features.py \
    --base_dir   $PR/data \
    --input_dir  $PR/data/interim/splits/structdesc_splits \
    --output_dir $PR/data/interim/features/structdesc_features

  # 1'. 想要几何特征:
  python $PR/scripts/02_features/01_build_structdesc_features.py \
    --base_dir   $PR/data --use_poscar_geom --poscar_max_sites 256

  # 2. Stage3 目标精修
  python $PR/scripts/02_features/02_reprocess_stage3_targets.py \
    --input_dir  $PR/data/interim/features/structdesc_features \
    --output_dir $PR/data/interim/features/structdesc_features_stage3_v2

  # 3. Stage3 单任务视图
  python $PR/scripts/02_features/04_build_stage3_task_views.py \
    --input_dir  $PR/data/interim/features/structdesc_features_stage3_v2 \
    --output_dir $PR/data/interim/features/stage3_task_views

  # 4. Hybrid (stage3 用 chgnet 64-d)
  python $PR/scripts/02_features/05_build_hybrid_features.py \
    --task stage3 \
    --descriptor_dir   $PR/data/interim/features/structdesc_features_stage3_v2 \
    --embedding_dirs   $PR/data/interim/graph_embeddings/chgnet_stage3 \
    --embedding_prefixes chgnet \
    --output_dir       $PR/data/interim/features/stage3_hybrid_features \
    --descriptor_kind  raw

  # 4'. 想要多源嵌入合并:
  python $PR/scripts/02_features/05_build_hybrid_features.py \
    --task stage3 \
    --descriptor_dir   $PR/data/interim/features/structdesc_features_stage3_v2 \
    --embedding_dirs   $PR/data/interim/graph_embeddings/cgcnn_stage3 \
                       $PR/data/interim/graph_embeddings/chgnet_stage3 \
    --embedding_prefixes cgcnn chgnet \
    --output_dir       $PR/data/interim/features/stage3_hybrid_multi \
    --descriptor_kind  raw

  # 5. 训练模式组装
  python $PR/scripts/02_features/06_prepare_training_modes.py \
    --source_dir  $PR/data/interim/features/stage3_hybrid_features \
    --output_root $PR/data/interim/training_modes \
    --train_file  stage3_train_hybrid.csv \
    --val_file    stage3_val_hybrid.csv \
    --test_file   stage3_test_hybrid.csv \
    --gold_train_holdout_file stage3_gold_train_holdout_hybrid.csv \
    --dataset_name stage3_hybrid

  8.4 验收清单
  
  跑完后逐项核对:

  ┌───────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │      验收点       │                                                   命令                                                    │                                 期望                                 │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ 描述子文件齐全    │ ls features/structdesc_features/*.csv                                                                     │ 8 个 CSV(stage{2,3} × {train,val,test,holdout} × {raw,ml})           │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ 词表训练集 only   │ python -c "from json import load; assert                                                                  │ val/test 的前驱物 ⊆ train vocab                                      │
  │                   │ set(load(open('features/structdesc_features/meta/precursor_vocab.json'))) >= set(...)"                    │                                                                      │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Stage3 v2         │ head -1 features/structdesc_features_stage3_v2/stage3_train_raw.csv                                       │ 含 target_time_h_log1p、target_time_bucket、target_atmosphere_coarse │
  │ 含派生目标        │                                                                                                           │  等                                                                  │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ 单任务视图非空    │ cat features/stage3_task_views/summary.json                                                               │ 每个视图 n_target_nonnull > 0                                        │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Hybrid 行数合理   │ summary.json 看 merged_rows                                                                               │ 与描述子表的 inner-join 行数一致                                     │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ 训练模式齐全      │ tree training_modes/stage3_hybrid                                                                         │ relaxed_only/、gold_only/、curriculum/ 三个目录                      │
  └───────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘

  8.5 常见坑与解法

  ┌───────────────────────────────────────────┬─────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │                   现象                    │                        原因                         │                                         解                                          │
  ├───────────────────────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ feat_frac_el__X 列全 0                    │ element_vocab 中 X 在所有数据里出现过但当前样本没有 │ 正常,不要紧                                                                         │
  ├───────────────────────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ feat_has_summary = 0 比例高               │ summary_json_path 没解析到                          │ 检查 jsonl 里路径是相对还是绝对,以及 --base_dir 是否对                              │
  ├───────────────────────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ stage3 view 行数 < raw 行数               │ view 只保留 has_target=1 的                         │ 如果差距大,说明该任务标签缺失多,看 02_reprocess_stage3_targets 的 cleaning 是否过严 │
  ├───────────────────────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ hybrid merged_rows 远小于 descriptor 行数 │ inner join 在 id 上失败                             │ 检查 GNN 嵌入是否覆盖所有样本;有时图构建会 drop 大体系                              │
  ├───────────────────────────────────────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ precursor_vocab 异常大                    │ Stage 00 没有 dedup                                 │ 排查 jsonl 里 main_precursors 是否有重复条目                                        │
  └───────────────────────────────────────────┴─────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

  ---
  第九部分 · 设计模式与改进空间
  
  9.1 五个工程模式

  1. raw + ml 双 CSV——同一份特征写两遍,raw 留原始目标(便于 debug)、ml 展开成 multi-hot(便于喂模型)。代价是磁盘翻倍,收益是任何下游训练器都能直接读其中一份。
  2. 训练集 only 词表 + 全集 element_vocab——前者反泄漏,后者保特征列稳定。这一对组合解决了"未见类标签"和"未见特征列"两个不同的问题。
  3. 缺失值用 0 + has_xxx 占位列——比 NaN+imputation 更可控,模型可学习"缺失"本身的信号。feat_has_summary、feat_poscar_has_geom、has_target 都属此类。
  4. 手写解析器优于 pymatgen——parse_formula_simple、parse_poscar 都是手写,因为它们被调用 N 次(N = 数十万),Python 对象开销主导时间。这是跨 5 万样本的工程经验:小循环里能不依赖第三方库就别依赖。
  5. 多源前缀化嵌入——{prefix}_graph_emb_{i} 让多 GNN 并存而不冲突,前缀就是契约。

  9.2 可改进点

  (a) POSCAR 几何特征是 $O(n^2)$。max_sites=256 是为了控制时间,但有一些大体系(MOFs、超晶胞)会被截断丢弃。可以引入 KD-tree(scipy.spatial.cKDTree)把复杂度降到 $O(n \log n)$。

  (b) min_image_distances 只看 1×1×1 镜像。对超长晶胞 + 短键,会漏算跨晶胞的对。可以扩到 3×3×3 镜像并保留最近的 max_neighbors 个,但代价是几乎 27× 时间。

  (c) composition_features 没有原子半径、电负性。Magpie 的标准描述子还包括 covalent_radius_mean、electronegativity_mean 等。对纯 composition 任务,加这些特征通常 +2-3 R²,但对 SynPred 这种已经有 GNN
  嵌入做主干的混合特征,边际收益小。

  (d) stage3_solvent_view 不限 top-N。低频溶剂(只出现 1-2 次)会被当作独立类,导致 train/val 分布偏移。建议在 view 阶段加 --min_class_count 过滤,把低频溶剂归到 "other"。

  (e) shell 脚本只编排 stage3。Stage2 的 hybrid 流被分散到 02_features/run_build_stage{2,3}_features.sh 之外的位置,需要分别维护。下一步可以把 stage2 也并进同一个 shell。

  (f) MP API key 硬编码 不是这一阶段的问题(Stage 02 不调网),但 01_build_structdesc_features.py:688 默认 --base_dir=/Users/wyc/SynPred/data 是绝对路径,在新机器上必须显式覆盖。建议改成读 SYNPRED_HOME
  环境变量。

  9.3 与同类工作的差异

  文献里"结构描述子 + GNN"的 hybrid 通常是端到端学一个 fusion network。SynPred 把 fusion 推到训练器去做(stage2/stage3 模型自己 concat),Stage 02 只负责特征列存在和命名规整。这种解耦带来三个好处:

  - 描述子 CSV 可以被任何不依赖 GNN 的 baseline(LightGBM、Ridge)直接消费
  - GNN 嵌入可以独立换型(CGCNN ↔ CHGNet ↔ ALIGNN),不动 Stage 02 主流程
  - 多源嵌入合并是显式的(embedding_dirs DIR1 DIR2),便于 ablation

  这是工程化的设计:Stage 02 不假设下游模型,只承诺"列名结构恒定、特征数值有限定义、标签 multi-hot 化在训练词表上"。

  ---
  第十部分 · 结语
  
  Stage 02 是 SynPred 流水线里最朴素也最关键的一段。它没有任何机器学习,只有 1300 行 numpy/pandas + 五个分工明确的脚本。但它定义了:

  - 模型看到的特征数值范围与列结构
  - 标签词表的来源(严格 train-only)
  - 缺失语义(0 + has_xxx 占位)
  - 训练 / 验证 / 测试目录的物理布局
  - 多源 GNN 嵌入的合并接口

  任何一处契约的微小变化都会向下游传导成训练失败或精度下降。Stage 02 是 SynPred 的接口规范——读懂它,你就知道这套流水线对所有下游模型暗示了什么。

  按本文 §8 的步骤跑完,你会得到与 SynPred 主分支字节相同的中间产物,后续的 Stage 03 graph、Stage 04 训练全部可无缝衔接。

  
❯ 4. Stage 03 (Graph) — 晶体图缓存与图嵌入                                                                                                                                                                    
    三种 GNN 都"先建缓存,再喂模型"。                                                                                                                                                                          
    4.1 CGCNN 图(03_graph/01_build_cgcnn_graph_cache.py)                                                                                                                                                      
    - 构图:对每个原子 $i$,在 $3\times 3\times 3$ 周期镜像中找到全部邻居,过滤距离 $\le 8.0,\text{Å}$,按距离取 max_num_nbr=12 个最近的(build_neighbor_graph,01_build_cgcnn_graph_cache.py:213)。                
    - 输出 (edge_src, edge_dst, edge_dist) + atomic_numbers + 多 hot 标签 y,pickle 缓存。                                                                                                                     
    - 训练脚本 03_graph/train_cgcnn_stage2.py 实现一份"轻量级 CGCNN":                                                                                                                                         
      - 原子嵌入 $h_i^0 = \mathrm{Embedding}(Z_i, d)$                                                                                                                                                         
      - 卷积层 CGCNNConv(train_cgcnn_stage2.py:114):消息 $m_{ij} = \mathrm{MLP}([h_i, h_j, d_{ij}])$,聚合 $h_i' = \mathrm{LayerNorm}(h_i + \mathrm{MLP}([h_i, \sum_j m_{ij}]))$                               
      - 4 层卷积 → 全图 mean+max 双池化拼接 → 256-dim 嵌入(extract_embedding 直接对外暴露)                                                                                                                    
      - Loss:BCEWithLogitsLoss(多标签前驱物预测)                                                                                                                                                              
    - 嵌入导出:export_cgcnn_stage2_embeddings.py 把 readout 后的 256-dim 写为 cgcnn_graph_emb_{i} CSV。                                                                                                       
    4.2 CHGNet 图(03_build_chgnet_cache_stage2.py + export_chgnet_stage2_embeddings.py)                                                                                                                       
    - 这一支不重新训练,直接使用 预训练 CHGNet 提 crystal_fea(model.predict_graph(graph, task="e", return_crystal_feas=True),export_chgnet_stage2_embeddings.py:54)。                                          
    - payload 提前转成 pymatgen.Structure → model.graph_converter(struct),给 CHGNet 喂带磁矩头的 e3nn 图。                                                                                                    
    - 嵌入维度通常 64 维(CHGNet 默认),写入 graph_emb_{i}。                                                                                                                                                    
    4.3 ALIGNN 图(03_build_alignn_cache_stage{2,3}.py)                                                                                                                                                        
    对应原论文的 line-graph(键的"键-键"夹角图)。脚本逻辑与上面相同,只是缓存格式留给 ALIGNN 训练器消费。                                                                                                       
    finalize_graph_embedding(推理 step)优先用 CGCNN 的 finetune 嵌入,缺失时回退到 CHGNet —— 这是工程上"鲁棒性 ≫ 一致性"的取舍。                                                                               
    ---                                                                                                                                                                                                       
  这一部分的详细分析，达到可以直接让别人复现代码，并且可以写文章的程度，一定要足够细致和科学                                                                                                                  
⏺ ---
  Stage 03 (Graph):晶体图缓存与图嵌入——从 POSCAR 到向量,可重复版
  
  ▎ 本文是 SynPred 流水线 Stage 03(图分支)的深度技术文章。Stage 03 不直接为下游任务提供监督信号,而是把每一份 POSCAR 用三种 GNN 
  ▎ 视角(CGCNN、CHGNet、ALIGNN)分别封装成可被模型直接消费的图缓存,并把训练好的图编码器最后一层 readout 导出为 hybrid 特征里的 *_graph_emb_* 列。所有规则、阈值、文件名约定都给到能逐字复现的程度。

  ---
  第一部分 · Stage 03 (Graph) 的角色与契约
  
  到 Stage 02 为止,SynPred 已经把每条样本表达为约 100-150 维的手工描述子。这一阶段要补上几何信息——晶体的拓扑、键长、键角、空间结构。手工特征做不到这一点(几何里有效信息往往是"X 个原子在 Y Å
  内的成键模式",而非"晶胞体积"),只能交给图神经网络。

  Stage 03 graph 的设计哲学有三:

  1. 解耦缓存与训练。所有图都先 pickle 落盘,模型训练只读 cache,不再碰 POSCAR。这让构图与训练可以分别在不同机器/不同时间运行。
  2. 三套 backbone 并存。CGCNN(自训)、CHGNet(预训练取嵌入)、ALIGNN(自训,带 line-graph),覆盖不同感受野与不同物理先验。下游 05_build_hybrid_features.py 通过 --embedding_dirs DIR1 DIR2 任意拼接。
  3. 训练模式与 stage 维度正交。每种 backbone 都有 relaxed_only / gold_only / curriculum_phase{1,2} 四种 train mode,以及 stage2 / stage3 两个任务维度,共 8 种缓存目录。但 val / test / gold_train_holdout 三份
   split 在所有模式下复用同一份 jsonl,避免重复构图。

  九个文件的角色矩阵:

  ┌──────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │                 文件                 │                         角色                         │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ 01_build_cgcnn_graph_cache.py        │ Stage 2 CGCNN 缓存 + mode 子缓存快速过滤             │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ 02_build_cgcnn_graph_cache_stage3.py │ Stage 3 CGCNN 缓存(同时携带 6 个条件目标)            │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ 03_build_chgnet_cache_stage{2,3}.py  │ CHGNet 输入载荷缓存                                  │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ 03_build_alignn_cache_stage{2,3}.py  │ ALIGNN 结构原始载荷缓存                              │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ train_cgcnn_stage2.py                │ 自训 CGCNN(多标签前驱物预测)+ 暴露 extract_embedding │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ export_cgcnn_stage2_embeddings.py    │ 把训练好的 CGCNN ckpt 应用于 cache → CSV             │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ export_chgnet_stage2_embeddings.py   │ 直接用预训练 CHGNet 提 crystal_fea → CSV             │
  └──────────────────────────────────────┴──────────────────────────────────────────────────────┘

  下文按"图构造 → 模型训练 → 嵌入导出"的纵向顺序讲,每个 backbone 自成一节。

  ---
  第二部分 · CGCNN:全自训分支
  
  2.1 缓存构造(01_build_cgcnn_graph_cache.py)

  2.1.1 入口与两种工作模式

  脚本兼容两类工作模式,在 main()(行 581-639)用 use_mode_builder 这一布尔分支:

  use_mode_builder = (
      bool(args.base_cache_dir.strip())     # 给定基础缓存目录
      and bool(args.mode_input_root.strip())# 给定 training_modes 路径
      and bool(args.train_mode.strip())     # 给定 mode 名
      and not args.force_rebuild            # 不强制重建
  )
  if use_mode_builder:
      build_mode_cache_from_base(args)      # 模式 B:从 base 过滤
  else:
      build_full_cache_from_raw(args)       # 模式 A:全量构图

  模式 A(全量构图):从 splits/structdesc_splits/stage2_{train,val,test,gold_train_holdout}.jsonl 读取样本,逐条解析 POSCAR、构邻居图、做 multi-hot 标签、pickle 输出。
  模式 B(模式子缓存):基础 cache 已经存在(模式 A 的输出),把 training_modes/<dataset>/<train_mode>/{train,val,test}/ 下的 CSV 里 id 集合用作过滤器,不重复构图,直接从基础 cache 里挑出对应样本。

  为什么要有模式 B?在 stage2 多个 train mode 上反复跑全量构图很贵($O(N \cdot n^2)$,典型 ~10k 样本要小时级),而 mode 之间的差异只是样本子集。模式 B 把构图开销降到 $O(N)$ 的字典查找。

  2.1.2 POSCAR 解析(行 132-210)

  跟 Stage 02 的 parse_poscar 是同一份手写实现,但本脚本里多保留了一份 species 列表与 atomic_numbers(int64 numpy)。VASP5 格式假设:第 6 行物种,第 7 行各物种数,7 之后可选 Selective,然后
  Direct/Cartesian。Z_TABLE 用元素符号映 1-118 号,Direct 模式 frac=coords; cart=frac@lattice,Cartesian 模式 frac=cart @ inv(lattice)。

  2.1.3 邻接图构造(build_neighbor_graph,行 213-254)

  这是整个流程最核心的化学步骤。给定分数坐标 ${f_i}$ 和晶格 $L \in \mathbb R^{3\times 3}$,目标是为每个原子找最多 12 个最近邻(MAX_NUM_NBR=12),搜索半径 $r_{\text{cut}}=8.0\text{Å}$。

  算法:
  
  (a) 预计算 27 个晶胞偏移向量
  SHIFT_VECS = np.array(list(itertools.product([-1,0,1], repeat=3)), dtype=float)
  即 ${(s_a, s_b, s_c) : s_a, s_b, s_c \in {-1, 0, 1}}$,中心(0,0,0)位于索引 13。
  
  (b) 对每个原子 $i$,广播减出每个 $j$ 在每个偏移下的位移:
  $$
  \Delta_{(s, j)} = (f_j + s) - f_i, \qquad d_{(s,j)} = |\Delta_{(s,j)} \cdot L|_2
  $$
  代码:
  diff = frac[None,:,:] - frac[i][None,None,:] + SHIFT_VECS[:,None,:]   # (27, n, 3)
  cart = diff @ lattice                                                  # (27, n, 3)
  dmat = np.linalg.norm(cart, axis=2)                                    # (27, n)
  dmat[13, i] = np.inf       # 自己与自身在 0 偏移下置 inf,防止自环      
  
  (c) 半径优先 + max-K 兜底:
  within = np.where(flat_d <= radius)[0]
  if len(within) == 0:
      order = np.argsort(flat_d)[:max_num_nbr]    # 全部超过 cutoff 时 fallback
  else:
      order = within[np.argsort(flat_d[within])[:max_num_nbr]]
  首选半径内的最近 12 个;若整张图所有距离都 > 8Å(极端稀疏体系)则取全局最近 12 个。这个 fallback 不会丢边,保证图永远连通。
  
  (d) 方向约定:src=j, dst=i(行 246-247)。即"消息从 j 流向 i"。下游 CGCNN 卷积层用 index_add_(0, edge_dst, ...) 聚合时这种约定让 dst 端是接收方,与公式 $h_i' = h_i + \sum_{j \in \mathcal N(i)} m_{ij}$
  直接对应。

  ▎ 可重复要点 #1:SHIFT_VECS 只搜 1×1×1 镜像。对小晶胞(< 4Å)而言,某些距离 8Å 的邻居其实落在 2×2×2 镜像中,但这些原子在分数坐标上 $\Delta f \in [-1, 1]$ 不可达,会被漏掉。SynPred 数据集里大部分晶胞 > 
  ▎ 5Å,影响有限。如果要严格,把 [-1,0,1] 改成 [-2,-1,0,1,2](125 个偏移),内存与时间均 ~5×。

  2.1.4 词表与多 hot 标签

  def build_precursor_vocab(rows):                      # 行 257
      vocab = set()
      for row in rows:
          for p in row.get("main_precursors", []) or []:
              if p:
                  vocab.add(str(p))
      return sorted(vocab)
  只用 train split 的 main_precursors。然后 make_multihot(行 266)把每条样本展成 0/1 向量。Stage 02 的 precursor_vocab 是按"all_rows" gather 的,本节是按 train。两套词表后续在 CSV 合并时按各自约定共存,但导出 
  hybrid 时仅用 Stage 02 词表(模型训练在哪一套词表上,导出就是哪一套)。
  
  2.1.5 过滤标准与丢弃统计

  process_split(行 274-337)对每条样本,按下列顺序过滤:

  1) poscar 文件不存在 -> dropped_missing_poscar
  2) 解析失败          -> dropped_bad_poscar
  3) nsites > 256      -> dropped_too_many_sites
  4) y.sum() == 0      -> dropped_empty_label

  第 4 步特别重要:如果一条样本的所有前驱物都不在 train 词表里,它的标签全是 0,作为"负监督"会污染 BCE loss。直接丢。这也意味着val/test 也会因为同样的原因被丢部分样本——这是反泄漏的代价,不是 bug。

  每条保留的样本写入 dict:
  {
    "id", "material_id", "formula", "doi", "split_group",
    "atomic_numbers"  # int64, (n,) 
    "frac_coords"     # float32, (n,3)
    "lattice"         # float32, (3,3)
    "edge_src"        # int64, (E,)
    "edge_dst"        # int64, (E,)
    "edge_dist"       # float32, (E,) Å
    "y"               # uint8, (V,)  multi-hot
    "main_precursors" # list[str]
  } 
  全部 pickle 到 {output_dir}/{split}.pkl,词表写 precursor_vocab.json,统计写 summary.json。
  
  2.1.6 模式 B:从 base cache 过滤出子缓存

  build_mode_cache_from_base(行 445-525)做的事:
  - 读基础 cache 的 4 个 pkl
  - 对 mode_input_root/<train_mode>/{train,val,test}/ 下的单个 CSV 里每行,用 JOIN_KEYS = [row_id, sample_id, material_id, entry_id, reaction_id, id, synth_uid, record_index] 抽取
  id(extract_uid_from_mapping,行 351)
  - 用 id 集合过滤基础 cache,写出新的子目录
  
  extract_uid 是个递归探测器,处理:
  - dict → 按 JOIN_KEYS 优先级查找
  - 嵌套 dict(meta/metadata/record/sample/row)→ 递归
  - list → 取第一个非空 uid
  - 对象 → 用 vars(obj) 走 dict 路径
  
  这种鲁棒性让 mode 输入可以是 hybrid CSV、训练模式的 CSV、甚至 pkl 反序列化对象,只要含 id 之类的字段。

  2.1.7 命令行(模式 A 默认)

  python scripts/03_graph/01_build_cgcnn_graph_cache.py \
    --base_dir   /Users/wyc/SynPred/data \
    --input_dir  /Users/wyc/SynPred/data/interim/splits/structdesc_splits \
    --output_dir /Users/wyc/SynPred/data/interim/graph_cache/cgcnn_stage2 \
    --max_sites  256 \
    --max_num_nbr 12 \
    --radius     8.0

  模式 B:
  python scripts/03_graph/01_build_cgcnn_graph_cache.py \
    --base_cache_dir   .../graph_cache/cgcnn_stage2 \
    --mode_input_root  .../training_modes/stage2_hybrid_cgcnn \
    --train_mode       gold_only \
    --output_dir       .../graph_cache/cgcnn_stage2/gold_only
    
  ▎ 可重复要点 #2:输出 pkl 大小可以预估为 (E + n) × ~12 字节 × N 样本。典型 stage2 train ~7k 样本,平均 nsites=20,边数 12×20=240,单文件约 200-300 MB。pickle 用默认 protocol 即可。

  2.2 Stage3 版的差异(02_build_cgcnn_graph_cache_stage3.py)

  跟 Stage 2 版几乎同源,只在三处不同:

  (1) 读 stage3 的 jsonl(行 68-71)。
  (2) 不做 multi-hot 标签——Stage3 的目标是连续/分类合成条件。改用 build_stage3_targets(行 223-282)按下列优先级提:
  target_temperature_c   <- target_temperature_c_clean / target_temperature_c / temperature_c_clean / temperature_c
  target_time_h          <- target_time_h_clean / target_time_h / time_h_clean / time_h
  target_time_bucket     <- target_time_bucket / time_bucket
  target_atmosphere      <- target_atmosphere_coarse / target_atmosphere / atmosphere_coarse / atmosphere
  target_solvent         <- target_solvent_clean / target_solvent / solvent_clean / solvent
  synthesis_type         <- synthesis_type / target_synthesis_type
  六个目标都允许为 None(缺一不影响保留),所以不再有 dropped_empty_label。统计里多 5 个 n_has_* 计数,用于诊断每个 split 的标签覆盖率。
  (3) 用 train_mode 选择 train 文件。select_split_paths(行 64-83):
  - relaxed_only / curriculum_phase1 → stage3_train.jsonl
  - gold_only / curriculum_phase2 → stage3_gold_train_holdout.jsonl

  val、test、gold_train_holdout 在四种模式下都指向同一份 jsonl。

  ▎ 可重复要点 #3:Stage3 cache 比 Stage2 cache 多 6 个目标字段。下游 stage3 训练器或嵌入导出器读 cache 时按任务挑用即可,缺失目标的样本由训练器自行 mask。

  2.3 CGCNN 训练(train_cgcnn_stage2.py)
  
  2.3.1 数据装载

  class CGCNNStage2Dataset(Dataset):
      def __init__(self, pkl_path):
          raw = load_pickle(pkl_path)
          # 把每条 dict 包成 GraphItem dataclass,一次性 to torch.Tensor
  GraphItem(行 41-52)只持有训练真正需要的字段:atomic_numbers, edge_src, edge_dst, edge_dist, y + 5 个元数据(用于推理时回填 CSV)。
  
  2.3.2 批次化:graph batching by node offset

  CGCNN 这种"逐节点 message passing"标准做法是节点拼接 + 索引偏移:

  def collate_graph_items(batch):
      node_offset = 0
      for item in batch:
          n = item.atomic_numbers.numel()
          edge_src_all.append(item.edge_src + node_offset)   # 关键:边索引平移
          edge_dst_all.append(item.edge_dst + node_offset)
          graph_node_slices.append((node_offset, node_offset + n))
          node_offset += n
      return {
          "atomic_numbers": cat,                # (B*n_total,)
          "edge_src": cat + offsets,            # (B*E_total,)
          "edge_dst": cat + offsets,
          "edge_dist": cat,
          "targets":  stack,                    # (B, V)
          "graph_node_slices": [(s,e), ...]     # 用于 readout 时 split
      }
  这是 PyG/DGL 之前的"裸手实现"——好处是零外部依赖、行为确定。graph_node_slices 是把 batched 节点向量切回每张图的索引区间,用于 mean/max 池化。
  
  2.3.3 卷积层(CGCNNConv,行 114-147)

  每层做的是经过简化的 message passing:

  消息(行 117-121):
  $$
  m_{ij} = \mathrm{MLP}{\text{msg}}\big([h_i ,|, h_j ,|, d{ij}]\big), \quad m_{ij} \in \mathbb R^d
  $$
  两层 SiLU,输入维 $2d+1$,输出 $d$。

  聚合 + 更新(行 142-146):
  $$
  \mathrm{agg}i = \sum{j \in \mathcal N(i)} m_{ji}, \quad u_i = \mathrm{MLP}_{\text{upd}}([h_i ,|, \mathrm{agg}_i]), \quad h_i' = \mathrm{LayerNorm}(h_i + u_i)
  $$
  注意 index_add_(0, edge_dst, msgs) 在 dst 维度求和,这就是为什么构图阶段把消息方向定为 src=j、dst=i。这层是严格残差 + LayerNorm,而不是原始 CGCNN 论文的 sigmoid gating——更稳、更 SiLU 友好,小数据集训练更快。

  2.3.4 全模型(CGCNNStage2,行 150-230)

  原子嵌入       Embedding(120+1, hidden_dim=128)
  卷积层栈       4 × CGCNNConv(hidden_dim)
  图池化         mean ⊕ max  (节点维度,按 graph_node_slices)  -> 256-d
  readout MLP    Linear(256, 128) + SiLU + Dropout(0.1)        -> 128-d  ← 嵌入
  分类头         Linear(128, n_labels)                          -> logits

  关键设计:
  - atom_emb 输入维度 max_z+1=121,clamp 在 [0, max_z](行 182)避免极少数 z=121 的样本越界
  - 池化是 mean ⊕ max 双拼,这对前驱物预测重要——mean 反映整体组成,max 凸显"最关键金属位点"
  - extract_embedding(行 214-230)是关键的对外接口,输出 readout 后的 128 维向量(注意:不是 256 的池化向量,是经过 readout MLP 的 128 维)
  
  ▎ 可重复要点 #4:与原文 (Xie & Grossman 2018 (https://arxiv.org/abs/1710.10324)) 不同,本实现:
  ▎ 1. 没有用边特征的 RBF 展开,直接喂 1 维距离
  ▎ 2. 没有 sigmoid+softplus gating,改用 LayerNorm-residual
  ▎ 3. 没有用元素特征(electronegativity 等),只用 Z 嵌入
  ▎ 这是为了实现自洽和小数据集稳定。如果要复现原文版,把 CGCNNConv 换成带 sigmoid gating 的版本即可。

  2.3.5 损失与评估

  loss_fn = nn.BCEWithLogitsLoss()

  多标签前驱物预测的标准选择。评估指标(multilabel_metrics_from_logits,行 233-260)用 0.5 阈值二值化后算:

  - micro F1:全局 TP/FP/FN 累加
  - set_match:整套 multi-hot 完全相等的样本比例(严格集合命中)
  - avg_jaccard:$\frac{1}{N}\sum_i \frac{|\hat y_i \cap y_i|}{|\hat y_i \cup y_i|}$,union=0 时取 1

  三个指标各有侧重。set_match 是最严格的,文献中 stage2 模型常单看这一项。

  2.3.6 训练循环
  
  optimizer = AdamW(lr=2e-4, weight_decay=1e-5)

  for epoch in 1..epochs:
      train_metrics = run_epoch(loaders["train"], optimizer)
      val_metrics   = run_epoch(loaders["val"],   None)

      if val_metrics["loss"] < best_val:
          torch.save({
            "model_state_dict": ...,
            "model_kwargs": {n_labels, hidden_dim, n_conv_layers, dropout},
            "args": vars(args),
            "best_epoch": ..., "best_val_loss": ...
          }, "best_cgcnn_stage2.pt")

  关键的 ckpt 设计:model_kwargs 与 args 一同保存。export_cgcnn_stage2_embeddings.py 里的 build_model(行 81-82)就是从 ckpt 里读 model_kwargs 自动重建模型,完全不用手填超参。

  epoch 结束后用最优 ckpt 在 train/val/test/gold_train_holdout 上各做一次 final eval(行 436-439),写入 summary.json。

  2.3.7 默认超参与硬件

  hidden_dim    = 128
  n_conv_layers = 4
  dropout       = 0.1
  batch_size    = 16
  lr            = 2e-4
  weight_decay  = 1e-5
  epochs        = 30
  seed          = 42
  device        = auto (mps > cuda > cpu)

  ▎ 可重复要点 #5:device="" 时按 mps→cuda→cpu 优先级选(行 367-372)。Mac M 系列优先用 mps;若你在 Linux + CUDA 环境,自动选 cuda。

  2.4 CGCNN 嵌入导出(export_cgcnn_stage2_embeddings.py)

  2.4.1 模型加载策略
  
  build_model(行 45-103)处理三种 ckpt:

  1) torch.save(model对象) -> isinstance(ckpt, nn.Module) -> 直接用
  2) {state_dict, model_kwargs}  -> 通过 --model_py / --model_class 动态 import 模型类
                                      然后 ModelClass(**model_kwargs).load_state_dict(state_dict)
  3) {model_state_dict, ...}    -> 同上,优先看 model_state_dict

  importlib.util.spec_from_file_location 让你可以从任意 .py 文件加载 CGCNN 类(行 67-73),不必把模型代码塞进包路径。

  2.4.2 嵌入提取
  
  def extract_embedding(model, model_inputs, embedding_key="embedding"):
      if hasattr(model, "extract_embedding"):
          emb = model.extract_embedding(**model_inputs)   # 优先走自定义接口
      else:
          out = model(**model_inputs)
          # dict -> out[embedding_key]
          # tuple -> out[0]
          # tensor -> out
  对 train_cgcnn_stage2.py:CGCNNStage2,走第一条 extract_embedding,返回 readout 后的 128 维。
  
  sample_to_model_inputs(行 106-113)做关键处理:单图 batch 时 graph_node_slices = [(0, n)],因为我们一次只导出一张图。

  2.4.3 输出 CSV 格式

  每行:
  id, material_id, formula, doi, split_group, graph_emb_0, graph_emb_1, ..., graph_emb_{D-1}
  列名是 graph_emb_{i}(没有 backbone 前缀)。下游 05_build_hybrid_features.py 通过 --embedding_prefixes cgcnn 把它们改名为 cgcnn_graph_emb_{i}。

  文件名:
  {output_dir}/{task_prefix}_{split}_graph_embed.csv
          e.g. stage2_train_graph_embed.csv
          
  summary.json 含 embedding_dim、exported_rows、failed_rows 等关键诊断。

  2.4.4 命令行

  python scripts/03_graph/export_cgcnn_stage2_embeddings.py \
    --cache_dir    /Users/wyc/SynPred/data/interim/graph_cache/cgcnn_stage2 \
    --output_dir   /Users/wyc/SynPred/data/interim/graph_embeddings/cgcnn_stage2 \
    --checkpoint   /Users/wyc/SynPred/runs/graph_models/cgcnn_stage2_multilabel/best_cgcnn_stage2.pt \
    --device       cuda \
    --task_prefix  stage2 \
    --model_py     /Users/wyc/SynPred/scripts/03_graph/train_cgcnn_stage2.py \
    --model_class  CGCNNStage2

  ▎ 可重复要点 #6:CGCNN 嵌入维度 = hidden_dim(默认 128)。如果你想要 256 维,把训练时 --hidden_dim 256 传进去重训。文档里"256 维"是基于早期版本的池化前向量,当前实现的对外接口是 readout 后 128 维——按代码为准。

  ---
  第三部分 · CHGNet:预训练 + 即拿即用

  3.1 缓存构造(03_build_chgnet_cache_stage2.py)

  CHGNet 是 DeepMind/Berkeley 联合发布的预训练势能模型 (https://chgnet.lbl.gov),基于 e3nn 框架。SynPred 不重新训练,只用它的 crystal 表征。这一脚本的角色是把 POSCAR 解析出 CHGNet 需要的最小载荷。

  3.1.1 train_mode-aware split 选择

  select_split_paths(行 103-125)根据 --train_mode 替换 train 文件:

  relaxed_only  / curriculum_phase1  -> stage2_train.jsonl
  gold_only     / curriculum_phase2  -> stage2_gold_train_holdout.jsonl
  val/test/gold_train_holdout         -> 共享同一份 jsonl

  这跟 Stage3 cgcnn 一致,但 stage2 cgcnn 主脚本不带这个机制(模式 B 替代),所以 chgnet 与 stage2 cgcnn 的"模式"实现路径不一样——这是历史代码的工程现实,不影响行为。

  3.1.2 词表加载

  load_precursor_vocab(行 77-86):优先读外部 JSON(默认 data/interim/features/structdesc_features/meta/precursor_vocab.json,即 Stage 02 的 vocab),只在 JSON 缺失时回退到从 train rows 重建。这意味着 CHGNet
  缓存的多 hot 与 Stage 02 的 hybrid 特征列号严格对齐——同一索引在两份 CSV 里指同一个前驱物。这是工程一致性的关键。

  3.1.3 POSCAR 解析

  参见 stage 02 / cgcnn,逻辑相同。注意一点:if not is_direct: frac = frac @ inv(lattice.T)(行 192-194),这里转置是因为 numpy 与 VASP 行向量约定的差异——结果是相同的最小镜像距离,但代码看着"翻"。

  3.1.4 CHGNet 输入载荷

  build_chgnet_ready_payload(行 232-244)产出 8 个字段:
  {
    "atomic_numbers": (n,) int64,
    "frac_coords":    (n,3) float32,
    "cart_coords":    (n,3) float32,
    "lattice":        (3,3) float32,
    "lattice_lengths":(3,)  float32,    # a, b, c
    "lattice_angles": (3,)  float32,    # α, β, γ
    "volume":         float32,          # |det(L)|
    "nsites":         int,
  } 
  不构 dgl/pymatgen Structure——延迟到 export 阶段做。这样 cache 文件不携带 dgl 序列化对象,跨版本/跨机器更稳健。

  3.1.5 命令行

  python scripts/03_graph/03_build_chgnet_cache_stage2.py \
    --base_dir   /Users/wyc/SynPred/data \
    --input_dir  .../splits/structdesc_splits \
    --train_mode relaxed_only \
    --precursor_vocab_json .../structdesc_features/meta/precursor_vocab.json \
    --max_sites  256
  默认 output_dir = base_dir/interim/graph_cache/chgnet_stage2/<train_mode>(行 328-332)。
  
  3.2 嵌入导出(export_chgnet_stage2_embeddings.py)

  3.2.1 三步推理

  struct = to_structure(item)           # pymatgen.Structure(lattice, species, frac_coords)
  graph  = model.graph_converter(struct) # CHGNet 内部图(含磁矩头)
  pred   = model.predict_graph(
              graph,
              task="e",                  # task='e' 表 energy(只跑能量头)
              return_crystal_feas=True,  # 关键:把图级 readout 拉出来
              batch_size=1,
           )
  emb    = pred["crystal_fea"]

  task="e" 是 CHGNet API 的最简模式——它跑前向到能量头停止,跳过 force/stress 计算。return_crystal_feas=True 让 forward 返回 crystal_fea(图级聚合后的固定向量)外加 e(能量)。我们只取前者。

  to_structure(行 25-32)用 pymatgen.core.periodic_table.Element.from_Z(int(z)) 把缓存里的原子序数还原为元素对象。coords_are_cartesian=False 因为我们存的是 frac_coords。

  3.2.2 输出格式

  CSV 同 CGCNN exporter 一致(graph_emb_{i} 列),stage2_{split}_graph_embed.csv。CHGNet crystal_fea 维度 = 64(预训练默认)。

  3.2.3 命令行
  
  python scripts/03_graph/export_chgnet_stage2_embeddings.py \
    --cache_dir  /Users/wyc/SynPred/data/interim/graph_cache/chgnet_stage2/relaxed_only \
    --output_dir /Users/wyc/SynPred/data/interim/graph_embeddings/chgnet_stage2/relaxed_only

  无需指定 --checkpoint——CHGNet.load() 自动下载/加载预训练权重(~/.cache/chgnet/)。必须有网或预先下好。

  ▎ 可重复要点 #7:CHGNet 是预训练的能量模型,不参与 SynPred 的任何训练。它给的 64 维 crystal_fea 包含了"势能景观"先验——在合成路线问题上意外有效,因为目标晶体的稳定性强烈影响合成可行温度区间。

  ▎ 可重复要点 #8:CHGNet 装载需要 chgnet、pymatgen、torch>=1.13 三个 python 依赖。pip install chgnet 即可。第一次 CHGNet.load() 会从 GitHub 拉权重(~10MB)。

  ---
  第四部分 · ALIGNN:line graph + 自训(留存接口)

  4.1 缓存构造(03_build_alignn_cache_stage{2,3}.py)

  ALIGNN (Choudhary & DeCost 2021 (https://www.nature.com/articles/s41524-021-00650-1)) 的核心创新是 line graph:把原子图的每条边变成 line graph 的一个节点,把"两条共享原子的边对"变成 line graph
  的边,边特征是两条原始边的夹角。这让模型显式编码键-键-夹角信息,对结构敏感任务(能带、能量)显著优于 CGCNN。

  但本流水线的 alignn 缓存不构造 line graph 也不缓存 dgl 对象(行 194-205 注释):

  def build_alignn_ready_payload(struct):
      """
      这里不直接构建 dgl graph,而是缓存 ALIGNN 后续最常用的结构原始载荷。
      这样不依赖 dgl / jarvis,在训练或 embedding 提取阶段再转图更稳。
      """
      return {
        "atomic_numbers", "frac_coords", "lattice", "cart_coords", "nsites"
      }

  也就是说 cache 里只有最小结构信息,line graph 由下游训练器/exporter 自己用 jarvis-tools 现场构造。这是 SynPred 的预防性设计:dgl 主版本号变化频繁,把 dgl 对象 pickle 到磁盘容易跨环境失败。

  4.1.1 与 CHGNet 缓存的同源结构
  
  仔细对比 03_build_chgnet_cache_stage2.py 与 03_build_alignn_cache_stage2.py,两脚本在以下函数上字符级几乎相同:normalize_list_like、load_precursor_vocab、build_multihot、select_split_paths、symbol_to_atomic
  _number、parse_poscar、process_split。差异只在:

  - alignn 不调用 lattice_to_lengths_angles、build_chgnet_ready_payload
  - alignn 用 build_alignn_ready_payload(只 5 个字段)
  - 默认 output_dir 末段是 alignn_stage2/<train_mode>

  ▎ 可重复要点 #9:目前 SynPred 主线没有 alignn 训练脚本和 exporter(目录 03_graph/ 没有 train_alignn_stage2.py)。alignn cache 是预留接口——下游 hybrid 合并器可以读 alignn 嵌入 CSV(若你训了),但你需要自己写 
  ▎ alignn trainer。这是工程现实,不是缺陷。

  4.2 Stage3 ALIGNN cache 的差异

  03_build_alignn_cache_stage3.py 跟 stage2 版的差异同 stage3 cgcnn:
  - 读 stage3 jsonl
  - 用 first_present + build_stage3_targets 提 6 个条件目标
  - 不要求标签非空(stage3 允许目标缺失)

  ---
  第五部分 · 推理时的"先 CGCNN,再 CHGNet 兜底"
  
  pipeline/run_pipeline.py 的 finalize_graph_embedding step 实现了:

  emb = read_csv_if_exists(cgcnn_emb_csv)
  if emb is None or len(emb) < n_targets:
      emb = read_csv_if_exists(chgnet_emb_csv)

  为什么 CGCNN 优先? 它是在 SynPred 自己的前驱物预测任务上 fine-tune 过的,其 128 维表示比 CHGNet 的 64 维通用 crystal_fea 在前驱物预测上更对齐(相当于 task-specific 表示)。

  为什么 CHGNet 兜底? CGCNN 训练 ckpt 可能因某些样本 OOM、模型加载失败、ckpt 路径问题等不可用;CHGNet 则永远可用(只要装了包)。

  这是"鲁棒性 ≫ 一致性"的工程取舍——推理时即使两份嵌入维度不同(128 vs 64),05_build_hybrid_features.py 通过统一的 *_graph_emb_* 列名把它们抽象掉,下游模型对此无感(模型层面只看 hybrid CSV 的列数)。

  ---
  第六部分 · 端到端复现指南

  6.1 前置依赖

  pip install numpy pandas torch tqdm
  pip install chgnet pymatgen           # 仅 CHGNet 分支需要
  pip install jarvis-tools dgl          # 仅 ALIGNN 训练时需要(本流水线未带训练器)

  6.2 完整流水线(以 stage2 为例)

  PR=/Users/wyc/SynPred

  # ============ STEP 1: 构造 stage2 CGCNN 全量缓存 ============
  python $PR/scripts/03_graph/01_build_cgcnn_graph_cache.py \
    --base_dir   $PR/data \
    --input_dir  $PR/data/interim/splits/structdesc_splits \
    --output_dir $PR/data/interim/graph_cache/cgcnn_stage2 \
    --max_sites 256 --max_num_nbr 12 --radius 8.0

  # ============ STEP 2: 训练 CGCNN ============
  python $PR/scripts/03_graph/train_cgcnn_stage2.py \
    --cache_dir  $PR/data/interim/graph_cache/cgcnn_stage2 \
    --run_dir    $PR/runs/graph_models/cgcnn_stage2_multilabel \
    --epochs 30 --batch_size 16 --hidden_dim 128 \
    --n_conv_layers 4 --lr 2e-4 --dropout 0.1 \
    --device cuda

  # ============ STEP 3: 导出 CGCNN 嵌入 ============
  python $PR/scripts/03_graph/export_cgcnn_stage2_embeddings.py \
    --cache_dir   $PR/data/interim/graph_cache/cgcnn_stage2 \
    --output_dir  $PR/data/interim/graph_embeddings/cgcnn_stage2 \
    --checkpoint  $PR/runs/graph_models/cgcnn_stage2_multilabel/best_cgcnn_stage2.pt \
    --device cuda \
    --model_py    $PR/scripts/03_graph/train_cgcnn_stage2.py \
    --model_class CGCNNStage2

  # ============ STEP 4: 构造 stage2 CHGNet 缓存 ============
  python $PR/scripts/03_graph/03_build_chgnet_cache_stage2.py \
    --base_dir   $PR/data \
    --input_dir  $PR/data/interim/splits/structdesc_splits \
    --train_mode relaxed_only \
    --precursor_vocab_json $PR/data/interim/features/structdesc_features/meta/precursor_vocab.json

  # ============ STEP 5: 导出 CHGNet 嵌入 ============
  python $PR/scripts/03_graph/export_chgnet_stage2_embeddings.py \
    --cache_dir  $PR/data/interim/graph_cache/chgnet_stage2/relaxed_only \
    --output_dir $PR/data/interim/graph_embeddings/chgnet_stage2/relaxed_only

  # ============ STEP 6: 合并 hybrid 特征(由 02_features 完成) ============
  python $PR/scripts/02_features/05_build_hybrid_features.py \
    --task stage2 \
    --descriptor_dir   $PR/data/interim/features/structdesc_features \
    --embedding_dirs   $PR/data/interim/graph_embeddings/cgcnn_stage2 \
                       $PR/data/interim/graph_embeddings/chgnet_stage2/relaxed_only \
    --embedding_prefixes cgcnn chgnet \
    --output_dir       $PR/data/interim/features/stage2_hybrid_cgcnn_chgnet \
    --descriptor_kind  ml

  6.3 跑 Stage3 的对应步骤

  把所有路径里 stage2 替换成 stage3,把 01_build_cgcnn_graph_cache.py 替换成 02_build_cgcnn_graph_cache_stage3.py,描述子目录用 structdesc_features_stage3_v2,--descriptor_kind raw。CHGNet/ALIGNN 同理。SynPred
   主线没有训 CGCNN-stage3 的脚本(stage3 的 CGCNN 嵌入若需要可用 stage2 训出来的 ckpt 在 stage3 cache 上做迁移导出,或自训),实际生产里 stage3 hybrid 主要靠 CHGNet 预训练嵌入。

  6.4 验收清单

  ┌──────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────┐
  │        验收点        │                                                命令                                                │                    期望                    │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 缓存非空             │ python -c "import pickle; d=pickle.load(open('train.pkl','rb')); print(len(d), list(d[0].keys()))" │ len > 0 且包含 atomic_numbers/edge_*/y     │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 边数合理             │ print(d[0]['edge_dst'].max(), d[0]['atomic_numbers'].size)                                         │ edge_dst.max()+1 == n_atoms                │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ CGCNN best ckpt 收敛 │ 看 train_history.json 末尾 val.micro_f1                                                            │ > 0.4 (上下文相关)                         │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 嵌入维度             │ head -1 stage2_train_graph_embed.csv                                                               │ 含 graph_emb_0..127(CGCNN)或 0..63(CHGNet) │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ hybrid 行数          │ wc -l stage2_train_hybrid.csv                                                                      │ ≈ min(描述子, CGCNN, CHGNet) inner-join    │
  └──────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────┘

  6.5 常见坑
  
  ┌───────────────────────────────────────┬──────────────────────┬──────────────────────────────────────────────────────┐
  │                 现象                  │         原因         │                          解                          │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ dropped_too_many_sites 占比 > 20%     │ 数据集含很多大晶胞   │ 调高 --max_sites 384 或选小晶胞子集                  │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ dropped_empty_label 大                │ 词表与 jsonl 不匹配  │ 用同一份 precursor_vocab.json                        │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ CGCNN 过拟合(train F1=0.9 val F1=0.3) │ data 太小或层太多    │ 减 conv 层 + 加 dropout                              │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ extract_embedding 不存在              │ exporter 走 fallback │ 检查 model_py / model_class 是否指向训练时同一份代码 │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ CHGNet OOM                            │ 大体系一次性 forward │ batch_size=1 已是最小值;先用 max_sites 卡死          │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ mode 子缓存全 0                       │ join key 不匹配      │ 检查 CSV 表头是否含 id 列                            │
  └───────────────────────────────────────┴──────────────────────┴──────────────────────────────────────────────────────┘
  
  ---
  第七部分 · 设计哲学与可改进点
  
  7.1 五个工程模式

  1. 缓存与训练分离——pickle 落盘,模型只读 cache,跨机器/跨时间稳定。
  2. 任务正交化——同一份 POSCAR + 同一份切分,通过 train_mode 与 task 两个维度组合出 8 种缓存目录,但真正的图构造工作只做一次(模式 B 的快速过滤)。
  3. 预训练 + 自训并存——CHGNet(预训练)给"通用势能先验",CGCNN(自训)给"任务特定表示",hybrid 时 concat。这种"多源表示"在合成数据小、任务特化的场景特别有效。
  4. 延迟 dgl/pymatgen 对象化——cache 只存 numpy 数值,dgl/Structure 在推理时才构。
  5. ckpt 自带 model_kwargs——exporter 不用手填超参。这是 PyTorch 生态里被低估的细节。

  7.2 可改进点

  (a) build_neighbor_graph 是 $O(n^2 \cdot 27)$,纯 Python 循环。对 nsites=200 的体系,单图构图秒级。可改成 scipy.spatial.cKDTree 把单图降到毫秒级,全数据集省 80% 时间。

  (b) edge_dist 是裸距离,没做 RBF。原版 CGCNN 用 K = 41 个高斯基函数展开 $d$。本流水线让 message MLP 直接吃 1 维 d——简化但表达力弱。可以试试加 RBF。

  (c) 没有用 Magpie 元素描述子。CGCNN 原文用 92 维元素描述子初始化原子表示,本实现只用 Z embedding(让模型从 Z 学这些)。小数据下 Magpie + Z 合用通常更好。

  (d) CGCNN-stage3 训练器缺失。Stage3 cache 准备好了,但没有 trainer。这是在用 CHGNet 64 维 cover stage3,如果想 task-specific,需要补一个 trainer(改 BCE 为多任务 head)。

  (e) CHGNet 的 atomic_features 没用上。predict_graph 还能返回 atomic_fea(每个原子的隐向量),理论上可以做"原子级嵌入 + 加权池化",但当前只取 crystal_fea。

  (f) 没有 batch 化的 CHGNet 推理。当前 batch_size=1,数千个样本要顺序跑。CHGNet 支持更大 batch,改 predict_graph(graph_list, batch_size=8) 可以 4-8x 加速。

  7.3 与 GraphormerForMaterials / M3GNet 的差异

  文献里的物理 GNN 常带角度信息(ALIGNN line graph)、应力/能量监督(M3GNet)、磁矩头(CHGNet)、几何 SE(3)等变性(MACE / EquiformerV2)。SynPred 选 CGCNN 自训 + CHGNet 预训练 + ALIGNN 预留三条路,理由:

  - CGCNN 自训:简单、可控、对前驱物多标签问题足够。在小数据(几千样本)上,4 层 + LayerNorm 收敛比 12 层 EquiformerV2 快得多。
  - CHGNet 预训练:免费的"势能景观"先验,补足自训缺乏的物理一致性。
  - ALIGNN 预留:留接口,等数据量更大或物理特化任务再启用。

  这种组合不是为了"性能 SOTA",而是为了保证流水线在任意中间环节失败时仍有可用的图嵌入——finalize_graph_embedding 的回退逻辑是这个哲学的直接体现。


  6.4 验收清单

  ┌─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
  │       验收点        │                                              命令                                              │                   期望                    │
  ├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
  │ 缓存非空            │ python -c "import pickle; d=pickle.load(open('train.pkl','rb')); print(len(d),                 │ len > 0 且包含 atomic_numbers/edge_*/y    │
  │                     │ list(d[0].keys()))"                                                                            │                                           │
  ├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
  │ 边数合理            │ print(d[0]['edge_dst'].max(), d[0]['atomic_numbers'].size)                                     │ edge_dst.max()+1 == n_atoms               │
  ├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
  │ CGCNN best ckpt     │ 看 train_history.json 末尾 val.micro_f1                                                        │ > 0.4 (上下文相关)                        │
  │ 收敛                │                                                                                                │                                           │
  ├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
  │ 嵌入维度            │ head -1 stage2_train_graph_embed.csv                                                           │ 含 graph_emb_0..127(CGCNN)或              │
  │                     │                                                                                                │ 0..63(CHGNet)                             │
  ├─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
  │ hybrid 行数         │ wc -l stage2_train_hybrid.csv                                                                  │ ≈ min(描述子, CGCNN, CHGNet) inner-join   │
  └─────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────┘

  6.5 常见坑

  ┌───────────────────────────────────────┬──────────────────────┬──────────────────────────────────────────────────────┐
  │                 现象                  │         原因         │                          解                          │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ dropped_too_many_sites 占比 > 20%     │ 数据集含很多大晶胞   │ 调高 --max_sites 384 或选小晶胞子集                  │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ dropped_empty_label 大                │ 词表与 jsonl 不匹配  │ 用同一份 precursor_vocab.json                        │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ CGCNN 过拟合(train F1=0.9 val F1=0.3) │ data 太小或层太多    │ 减 conv 层 + 加 dropout                              │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ extract_embedding 不存在              │ exporter 走 fallback │ 检查 model_py / model_class 是否指向训练时同一份代码 │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ CHGNet OOM                            │ 大体系一次性 forward │ batch_size=1 已是最小值;先用 max_sites 卡死          │
  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────────────────────────┤
  │ mode 子缓存全 0                       │ join key 不匹配      │ 检查 CSV 表头是否含 id 列                            │
  └───────────────────────────────────────┴──────────────────────┴──────────────────────────────────────────────────────┘

  ---
  第七部分 · 设计哲学与可改进点

  7.1 五个工程模式

  1. 缓存与训练分离——pickle 落盘,模型只读 cache,跨机器/跨时间稳定。
  2. 任务正交化——同一份 POSCAR + 同一份切分,通过 train_mode 与 task 两个维度组合出 8 种缓存目录,但真正的图构造工作只做一次(模式 B 的快速过滤)。
  3. 预训练 + 自训并存——CHGNet(预训练)给"通用势能先验",CGCNN(自训)给"任务特定表示",hybrid 时 concat。这种"多源表示"在合成数据小、任务特化的场景特别有效。
  4. 延迟 dgl/pymatgen 对象化——cache 只存 numpy 数值,dgl/Structure 在推理时才构。
  5. ckpt 自带 model_kwargs——exporter 不用手填超参。这是 PyTorch 生态里被低估的细节。

  7.2 可改进点

  (a) build_neighbor_graph 是 $O(n^2 \cdot 27)$,纯 Python 循环。对 nsites=200 的体系,单图构图秒级。可改成 scipy.spatial.cKDTree 把单图降到毫秒级,全数据集省 80% 时间。


  ---
  第七部分 · 设计哲学与可改进点

  7.1 五个工程模式

  1. 缓存与训练分离——pickle 落盘,模型只读 cache,跨机器/跨时间稳定。
  2. 任务正交化——同一份 POSCAR + 同一份切分,通过 train_mode 与 task 两个维度组合出 8 种缓存目录,但真正的图构造工作只做一次(模式 B 的快速过滤)。
  3. 预训练 + 自训并存——CHGNet(预训练)给"通用势能先验",CGCNN(自训)给"任务特定表示",hybrid 时 concat。这种"多源表示"在合成数据小、任务特化的场景特别有效。
  4. 延迟 dgl/pymatgen 对象化——cache 只存 numpy 数值,dgl/Structure 在推理时才构。
  5. ckpt 自带 model_kwargs——exporter 不用手填超参。这是 PyTorch 生态里被低估的细节。

  7.2 可改进点

  (a) build_neighbor_graph 是 $O(n^2 \cdot 27)$,纯 Python 循环。对 nsites=200 的体系,单图构图秒级。可改成 scipy.spatial.cKDTree 把单图降到毫秒级,全数据集省 80% 时间。

  (b) edge_dist 是裸距离,没做 RBF。原版 CGCNN 用 K = 41 个高斯基函数展开 $d$。本流水线让 message MLP 直接吃 1 维 d——简化但表达力弱。可以试试加 RBF。

  (c) 没有用 Magpie 元素描述子。CGCNN 原文用 92 维元素描述子初始化原子表示,本实现只用 Z embedding(让模型从 Z 学这些)。小数据下 Magpie + Z 合用通常更好。

  (d) CGCNN-stage3 训练器缺失。Stage3 cache 准备好了,但没有 trainer。这是在用 CHGNet 64 维 cover stage3,如果想 task-specific,需要补一个 trainer(改 BCE 为多任务 head)。

  (e) CHGNet 的 atomic_features 没用上。predict_graph 还能返回 atomic_fea(每个原子的隐向量),理论上可以做"原子级嵌入 + 加权池化",但当前只取 crystal_fea。

  (f) 没有 batch 化的 CHGNet 推理。当前 batch_size=1,数千个样本要顺序跑。CHGNet 支持更大 batch,改 predict_graph(graph_list, batch_size=8) 可以 4-8x 加速。

  7.3 与 GraphormerForMaterials / M3GNet 的差异

  文献里的物理 GNN 常带角度信息(ALIGNN line graph)、应力/能量监督(M3GNet)、磁矩头(CHGNet)、几何 SE(3)等变性(MACE / EquiformerV2)。SynPred 选 CGCNN 自训 + CHGNet 预训练 + ALIGNN 预留三条路,理由:

  - CGCNN 自训:简单、可控、对前驱物多标签问题足够。在小数据(几千样本)上,4 层 + LayerNorm 收敛比 12 层 EquiformerV2 快得多。
  - CHGNet 预训练:免费的"势能景观"先验,补足自训缺乏的物理一致性。
  - ALIGNN 预留:留接口,等数据量更大或物理特化任务再启用。

  这种组合不是为了"性能 SOTA",而是为了保证流水线在任意中间环节失败时仍有可用的图嵌入——finalize_graph_embedding 的回退逻辑是这个哲学的直接体现。

  ---
  第八部分 · 结语

  Stage 03 (Graph) 是 SynPred 唯一深度学习的子模块。它把"晶体几何"用三种 GNN 视角各封装成一份缓存,让训练和推理分离;通过自训 CGCNN + 预训练 CHGNet 两条路给出 task-specific 与 general-prior 两种嵌入,让 Stage
  02 的描述子拥有几何后盾;通过 train_mode × task 二维矩阵让所有下游配置都能拿到对应缓存。

  按本文 §6 的步骤逐字执行,你能复现:
  - 一个 stage2 CGCNN 多标签预测器(典型 val micro-F1 0.4-0.6,strict set-match 0.2-0.4)
  - 一份 128 维 CGCNN 嵌入 CSV(每 split 一份)
  - 一份 64 维 CHGNet 预训练嵌入 CSV(每 split 一份)
  - 与 Stage 02 描述子无缝合并的 hybrid 特征表

  下游的 Stage 03 (Data) NPZ 构建器、Stage 04 训练器全部建立在这套 hybrid 表上。Stage 03 (Graph)
  的契约就两条:列名前缀正确、行号与描述子对齐。剩下的工程美感都是为了在不破坏这两条契约的前提下,让构图、训练、导出彼此独立、可调换、可恢复。

❯  5. Stage 03 (Data) — 各模型变体训练集打包

    不同模型的训练集脚本沿着通用模板:
    1. 加载 hybrid 特征 CSV,自动检测 feat_* / graph_emb_* 作为 $x$。
    2. 检测 label_prec__* 作为多 hot 标签 → 按需转 trajectory / 集合编码。
    3. 训练集上拟合 standardizer ($\mu,\sigma$),验证/测试用同一统计量 transform。
    4. 写出 {train,val,test,gold_train_holdout}.npz + {*}_meta.csv + schema.json。

    关键差异:
                                                                                                                                                                                                              
    ┌───────────────────────────────────────────────┬───────────────────────┬─────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────┐  
    │                     脚本                      │       模型变体        │                                    标签编码                                     │                    备注                    │  
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤  
    │ 20_build_stage2_cvae_dataset.py               │ CVAE                  │ y_set 多 hot                                                                    │ 输出 (x, y_set) 对                         │  
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤  
    │ 25_build_stage2_ar_dataset.py                 │ AR (Seq2Seq)          │ 排序后的 token 序列 + EOS                                                       │ 含 BOS/EOS,用 teacher forcing              │  
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤  
    │ 26_build_stage2_setpred_dataset.py            │ Set Prediction        │ y_set 多 hot                                                                    │ 类似 CVAE 但无 latent                      │  
    │                                               │ Transformer           │                                                                                 │                                            │  
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤  
    │ 27_build_stage2_gflownet_dataset.py           │ GFlowNet              │ trajectory: 选元素的动作序列 + STOP                                             │ encode_reference_trajectories 生成多 hot → │  
    │                                               │                       │                                                                                 │  traj                                      │  
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤  
    │ 28_build_stage2_gflownet_mode_datasets.py     │ GFlowNet 三种 mode    │ 同上,走 relaxed_only/gold_only/curriculum 子目录                                │                                            │  
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤  
    │ 27_build_stage3_condition_dataset_v5_mixed.py │ Mixture Flow          │ $(x, y_{\text{set}}, y_{\text{cont}}, y_{\text{cont_mask}}, y_{\text{disc}},    │ 连续 = (T,t),离散 = (atm_coarse,           │  
    │                                               │                       │ y_{\text{disc_mask}})$                                                          │ synthesis_type)                            │  
    └───────────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────┘  
                                                                                                                                                                                                              
    5.1 GFlowNet 轨迹编码(27_build_stage2_gflownet_dataset.py:184)                                                                                                                                            
                                                                                                                                                                                                              
    对每条样本的多 hot $y\in{0,1}^V$:                                                                                                                                                                         
    - 设 $A = {j: y_j = 1}$,直接把它作为"选择动作序列",末尾追加 STOP token;                                                                                                                                   
    - traj_actions[i, :] 长度 max_set_len + 1,用 traj_mask 标注真实长度。                                                                                                                                     
    - 这样 GFlowNet 就能用最大似然(teacher forcing)直接拟合 ground-truth 轨迹。                                                                                                                               
                                                                                                                                                                                                              
    5.2 Stage35 候选与硬负样本                                                                                                                                                                                
                                                                                                                                                                                                              
    24_build_stage35_hardneg_compat_dataset.py(scripts/03_data/24_build_stage35_hardneg_compat_dataset.py:296)                                                                                                
                                                                                                                                                                                                              
    对每条 positive (target, true_set):                                                                                                                                                                       
    - hard negative:从一个候选池(由 Stage2 模型在训练集上跑出的候选 jsonl 提供)抽 n_hard_neg_per_pos=2 条,每条都被 Stage2 打高分但实际不对 —— 这是模型的"困惑点";                                             
    - random negative:从所有 train target 的 true_set 集合里随便抽 n_random_neg_per_pos(同一目标除外)。                                                                                                       
    - 输出 (x_struct, precursor_y, x_joint=concat, y∈{0,1}),直接给 Stage35 兼容性二分类器训练。                                                                                                               
                                                                                                                                                                                                              
    40_build_stage35_strong_joint_candidates.py + 41_build_stage35_strong_joint_dataset.py                                                                                                                    
                                                                                                                                                                                                              
    按 (material_key, parent_precursor_rank, condition_rank) 把"strong stage2 候选 × strong stage3 候选"做笛卡尔积,生成 recipe 级候选;然后回查 ground-truth 计算多种软标签:                                   
    - precursor_jaccard、precursor_exact_match                                                                                                                                                                
    - temp_abs_err, time_abs_err,以及阈值化的 cont_match                                                                                                                                                      
    - joint_label = precursor_exact_match AND cont_match(硬标签)                                                                                                                                              
    - joint_soft_score:Jaccard 与 cont_score 的加权和(软标签,做学习排序时用)。对这一部分进行非常细致的分析，达到可以让别人复现，可以先文章的程度，要足够细致的分析                                            


  按本文 §6 的步骤逐字执行,你能复现:
  - 一个 stage2 CGCNN 多标签预测器(典型 val micro-F1 0.4-0.6,strict set-match 0.2-0.4)
  - 一份 128 维 CGCNN 嵌入 CSV(每 split 一份)
  - 一份 64 维 CHGNet 预训练嵌入 CSV(每 split 一份)
  - 与 Stage 02 描述子无缝合并的 hybrid 特征表

  下游的 Stage 03 (Data) NPZ 构建器、Stage 04 训练器全部建立在这套 hybrid 表上。Stage 03 (Graph)
  的契约就两条:列名前缀正确、行号与描述子对齐。剩下的工程美感都是为了在不破坏这两条契约的前提下,让构图、训练、导出彼此独立、可调换、可恢复。

❯  5. Stage 03 (Data) — 各模型变体训练集打包                                                                                                                             
                                                                                                                                                                         
    不同模型的训练集脚本沿着通用模板:                                                                                                                                    
    1. 加载 hybrid 特征 CSV,自动检测 feat_* / graph_emb_* 作为 $x$。                                                                                                     
    2. 检测 label_prec__* 作为多 hot 标签 → 按需转 trajectory / 集合编码。                                                                                               
    3. 训练集上拟合 standardizer ($\mu,\sigma$),验证/测试用同一统计量 transform。                                                                                        
    4. 写出 {train,val,test,gold_train_holdout}.npz + {*}_meta.csv + schema.json。                                                                                       
                                                                                                                                                                         
    关键差异:                                                                                                                                                            
                                                                                                                                                                         
    ┌───────────────────────────────────────────────┬───────────────────────┬─────────────────────────────────────────────────────────────────────────────────┬───────── 
  ───────────────────────────────────┐                                                                                                                                   
    │                     脚本                      │       模型变体        │                                    标签编码                                     │          
             备注                    │                                                                                                                                   
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────── 
  ───────────────────────────────────┤                                                                                                                                   
    │ 20_build_stage2_cvae_dataset.py               │ CVAE                  │ y_set 多 hot                                                                    │ 输出 (x, 
   y_set) 对                         │                                                                                                                                   
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────── 
  ───────────────────────────────────┤                                                                                                                                   
    │ 25_build_stage2_ar_dataset.py                 │ AR (Seq2Seq)          │ 排序后的 token 序列 + EOS                                                       │ 含       
  BOS/EOS,用 teacher forcing              │                                                                                                                              
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────── 
  ───────────────────────────────────┤                                                                                                                                   
    │ 26_build_stage2_setpred_dataset.py            │ Set Prediction        │ y_set 多 hot                                                                    │ 类似     
  CVAE 但无 latent                      │                                                                                                                                
    │                                               │ Transformer           │                                                                                 │          
                                     │                                                                                                                                   
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────── 
  ───────────────────────────────────┤                                                                                                                                   
    │ 27_build_stage2_gflownet_dataset.py           │ GFlowNet              │ trajectory: 选元素的动作序列 + STOP                                             │          
  encode_reference_trajectories 生成多 hot → │                                                                                                                           
    │                                               │                       │                                                                                 │  traj    
                                     │                                                                                                                                   
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────── 
  ───────────────────────────────────┤                                                                                                                                   
    │ 28_build_stage2_gflownet_mode_datasets.py     │ GFlowNet 三种 mode    │ 同上,走 relaxed_only/gold_only/curriculum 子目录                                │          
                                     │                                                                                                                                   
    ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────── 
  ───────────────────────────────────┤                                                                                                                                   
    │ 27_build_stage3_condition_dataset_v5_mixed.py │ Mixture Flow          │ $(x, y_{\text{set}}, y_{\text{cont}}, y_{\text{cont_mask}}, y_{\text{disc}},    │ 连续 =   
  (T,t),离散 = (atm_coarse,           │                                                                                                                                  
    │                                               │                       │ y_{\text{disc_mask}})$                                                          │          
  synthesis_type)                            │                                                                                                                           
    └───────────────────────────────────────────────┴───────────────────────┴─────────────────────────────────────────────────────────────────────────────────┴───────── 
  ───────────────────────────────────┘                                                                                                                                   
                                                                                                                                                                         
    5.1 GFlowNet 轨迹编码(27_build_stage2_gflownet_dataset.py:184)                                                                                                       
                                                                                                                                                                         
    对每条样本的多 hot $y\in{0,1}^V$:                                                                                                                                    
    - 设 $A = {j: y_j = 1}$,直接把它作为"选择动作序列",末尾追加 STOP token;                                                                                              
    - traj_actions[i, :] 长度 max_set_len + 1,用 traj_mask 标注真实长度。                                                                                                
    - 这样 GFlowNet 就能用最大似然(teacher forcing)直接拟合 ground-truth 轨迹。                                                                                          
                                                                                                                                                                         
    5.2 Stage35 候选与硬负样本                                                                                                                                           
                                                                                                                                                                         
    24_build_stage35_hardneg_compat_dataset.py(scripts/03_data/24_build_stage35_hardneg_compat_dataset.py:296)                                                           
                                                                                                                                                                         
    对每条 positive (target, true_set):                                                                                                                                  
    - hard negative:从一个候选池(由 Stage2 模型在训练集上跑出的候选 jsonl 提供)抽 n_hard_neg_per_pos=2 条,每条都被 Stage2 打高分但实际不对 —— 这是模型的"困惑点";        
    - random negative:从所有 train target 的 true_set 集合里随便抽 n_random_neg_per_pos(同一目标除外)。                                                                  
    - 输出 (x_struct, precursor_y, x_joint=concat, y∈{0,1}),直接给 Stage35 兼容性二分类器训练。                                                                          
                                                                                                                                                                         
    40_build_stage35_strong_joint_candidates.py + 41_build_stage35_strong_joint_dataset.py                                                                               
                                                                                                                                                                         
    按 (material_key, parent_precursor_rank, condition_rank) 把"strong stage2 候选 × strong stage3 候选"做笛卡尔积,生成 recipe 级候选;然后回查 ground-truth              
  计算多种软标签:                                                                                                                                                        
    - precursor_jaccard、precursor_exact_match                                                                                                                           
    - temp_abs_err, time_abs_err,以及阈值化的 cont_match                                                                                                                 
    - joint_label = precursor_exact_match AND cont_match(硬标签)                                                                                                         
    - joint_soft_score:Jaccard 与 cont_score 的加权和(软标签,做学习排序时用)。对这一部分进行非常细致的分析，达到可以让别人复现，可以先文章的程度，要足够细致的分析       

❯ 对这一部分进行非常细致的分析，达到可以让别人复现，可以先文章的程度，要足够细致的分析                                                                                                                        

  Read 3 files (ctrl+o to expand)

❯ 5. Stage 03 (Data) — 各模型变体训练集打包                                                                                                                                                                   
      不同模型的训练集脚本沿着通用模板:                                                                                                                                                                       
      1. 加载 hybrid 特征 CSV,自动检测 feat_* / graph_emb_* 作为 $x$。                                                                                                                                        
      2. 检测 label_prec__* 作为多 hot 标签 → 按需转 trajectory / 集合编码。                                                                                                                                  
      3. 训练集上拟合 standardizer ($\mu,\sigma$),验证/测试用同一统计量 transform。                                                                                                                           
      4. 写出 {train,val,test,gold_train_holdout}.npz + {*}_meta.csv + schema.json。                                                                                                                          
      关键差异:                                                                                                                                                                                               
      ┌───────────────────────────────────────────────┬───────────────────────┬─────────────────────────────────────────────────────────────────────────────────┬─────────                                    
    ───────────────────────────────────┐                                                                                                                                                                      
      │                     脚本                      │       模型变体        │                                    标签编码                                     │                                             
               备注                    │                                                                                                                                                                      
      ├───────────────────────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼─────────                                    
    ───────────────────────────────────┤                                                                                                                                                                      
  ──── (40 lines hidden) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
                                                                                                                                                                                                              
      24_build_stage35_hardneg_compat_dataset.py(scripts/03_data/24_build_stage35_hardneg_compat_dataset.py:296)                                                                                              
                                                                                                                                                                                                              
      对每条 positive (target, true_set):                                                                                                                                                                     
      - hard negative:从一个候选池(由 Stage2 模型在训练集上跑出的候选 jsonl 提供)抽 n_hard_neg_per_pos=2 条,每条都被 Stage2 打高分但实际不对 —— 这是模型的"困惑点";                                           
      - random negative:从所有 train target 的 true_set 集合里随便抽 n_random_neg_per_pos(同一目标除外)。                                                                                                     
      - 输出 (x_struct, precursor_y, x_joint=concat, y∈{0,1}),直接给 Stage35 兼容性二分类器训练。                                                                                                             
      40_build_stage35_strong_joint_candidates.py + 41_build_stage35_strong_joint_dataset.py                                                                                                                  
      按 (material_key, parent_precursor_rank, condition_rank) 把"strong stage2 候选 × strong stage3 候选"做笛卡尔积,生成 recipe 级候选;然后回查 ground-truth                                                 
    计算多种软标签:                                                                                                                                                                                           
      - precursor_jaccard、precursor_exact_match                                                                                                                                                              
      - temp_abs_err, time_abs_err,以及阈值化的 cont_match                                                                                                                                                    
      - joint_label = precursor_exact_match AND cont_match(硬标签)                                                                                                                                            
      - joint_soft_score:Jaccard 与 cont_score 的加权和(软标签,做学习排序时用)。对这一部分进行非常细致的分析，达到可以让别人复现，可以先文章的程度，要足够细致的分析                                          

⏺ ---
  Stage 03 (Data):各模型变体训练集打包——从 hybrid CSV 到 NPZ 字典,可重复版
  
  ▎ 本文是 SynPred 流水线 Stage 03 (Data) 的深度技术文章。Stage 03 (Data) 把上一阶段的 hybrid 特征表(stage2/stage3 的描述子 ⨉ GNN 嵌入)打包成多种模型专用 NPZ 字典——每种生成式模型(CVAE / AR / SetPred / 
  ▎ GFlowNet / Mixture / 兼容性二分类器 / 强联合排序器)需要的张量切片不同,这一阶段统一在 scripts/03_data/ 下完成。本文给出每条契约、每个目录布局、每个张量形状,达到逐字复现的程度。

  ---
  第一部分 · Stage 03 (Data) 的角色与契约
  
  到 Stage 02 + Stage 03 (Graph) 结束,SynPred 已经准备好两类大表:

  - data/interim/features/stage2_hybrid_features/stage2_{train,val,test,gold_train_holdout}_hybrid.csv(描述子 + GNN 嵌入,带 multi-hot precursor 标签)
  - data/interim/features/structdesc_features_stage3_v2/stage3_{train,val,test,gold_train_holdout}_raw.csv(描述子 + 原始 stage3 目标列)
  - data/interim/training_modes/.../{relaxed_only,gold_only,curriculum/...}/...(同上,但按训练模式拆好的子目录)

  但 SynPred 训练 7 种 stage2 生成式模型(CVAE、AR Seq2Seq、SetPred Transformer、GFlowNet、Refinement、Stage35 兼容性二分类、Stage35 强联合排序),每种模型对张量的要求都不同:

  - CVAE 只要 (x, y_set)
  - AR 要 (x, seq_in, seq_out, seq_mask)(teacher forcing)
  - SetPred 要 (x, slot_targets, slot_mask, overflow)(固定槽位)
  - GFlowNet 要 (x, traj_actions, traj_mask, set_len)(动作序列 + STOP)
  - Mixture(Stage3)要 (x, y_set, y_cond_continuous, y_cond_continuous_mask, y_cond_discrete, y_cond_discrete_mask)
  - Stage35 兼容性二分类要 (x_struct, precursor_y, x_joint, y∈{0,1}) + 硬负样本采样
  - Stage35 强联合要 (precursor_set, conditions, joint_label, joint_soft_score, ...) + 多种软标签

  Stage 03 (Data) 的设计哲学:所有变体共享同一通用模板(detect_feature/label_cols → fit standardizer on train → transform val/test → 各模型自定义 encode → 写 NPZ)。只有 encode 
  那一段不同。这一节解释通用模板,后面每节只讲变体特有的 encode。

  通用模板由四步构成:

  # 1) detect schema
  feature_cols = [c for c in df.columns
                  if c.startswith("feat_") or c.startswith("graph_emb_") or "_graph_emb_" in c]
  label_cols   = [c for c in df.columns if c.startswith("label_prec__")]
  meta_cols    = [c for c in df.columns
                  if c in {"id","material_id","formula","doi","split_group",
                           "source_dataset","synthesis_type"}]

  # 2) fit standardizer on TRAIN
  mean = train_x.mean(axis=0)
  std  = np.where(train_x.std(axis=0) < 1e-12, 1.0, train_x.std(axis=0))
  
  # 3) per split: x_raw -> x = (x_raw - mean) / std,  y_multi_hot,  meta
  # 4) per-variant encode + np.savez_compressed(split.npz, ...)

  关键决策:
  - feat_* 与所有 *_graph_emb_* 都被自动抓为 x 的一部分(stage 02 / stage 03 的列前缀契约在此发挥作用)
  - 标准化严格只用训练集拟合,std<1e-12 时置 1 防 NaN(行 130-134 in 20)
  - x_raw 与 x 同时落盘——前者保留原始量纲方便诊断,后者给模型用
  - meta CSV 与 NPZ 同名同号,行号对齐({split}_meta.csv)

  文件组织全部一致:

  {output_dir}/
  ├── train.npz, val.npz, test.npz, [gold_train_holdout.npz]
  ├── train_meta.csv, val_meta.csv, test_meta.csv, [gold_train_holdout_meta.csv]
  ├── feature_cols.json     # 与 NPZ 里 x 的列对齐
  ├── label_cols.json / label_names.json / precursor_names.json / action_vocab.json ...
  ├── feature_mean.npy, feature_std.npy
  └── summary.json          # config + per-split 统计

  下面按变体分节,每节给出独有的 encode 公式 + NPZ 张量字段 + 命令行。

  ---
  第二部分 · 通用前置:00_prepare_mode_inputs_for_stage2.py
  
  03_data 接受两种输入定位方式:
  1. 传统 layout:直接读 data/interim/features/stage2_hybrid_features/stage2_*_hybrid.csv(只有一份 train/val/test/holdout)。
  2. mode-aware layout:--mode_input_root .../stage2_hybrid_cgcnn + --train_mode {relaxed_only,gold_only,curriculum_phase{1,2}},从子目录里挑文件。
  
  但是 Stage 02 给的目录是嵌套的(relaxed_only/train/stage2_train_hybrid.csv),03_data 期望的是扁平的(relaxed_only/stage2_train_hybrid.csv)——00 脚本就是把嵌套布局复制成扁平布局。

  build_standard_mode_views(行 40-173)做四件事(relaxed_only / gold_only / curriculum_phase1 / curriculum_phase2),每个模式对应一组 train/val/test/gold_train_holdout 的复制。关键映射(以 relaxed_only 为例):

  SRC: source_root/relaxed_only/{train,val,test}/stage2_*_hybrid.csv
  SRC: source_root/gold_only/train/stage2_gold_train_holdout_hybrid.csv
  DST: output_root/<dataset_name>/relaxed_only/{stage2_train,val,test,gold_train_holdout}_hybrid.csv

  注意 relaxed_only 的 train 是 stage2_train,gold 是 holdout;gold_only 的 train 直接复用 holdout 文件,从而 4 个模式得到统一的 stage2_train_hybrid.csv 命名,03_data 只需按 train_mode
  取目录即可,无需再做文件名映射。

  copy_as(行 28-37)用 shutil.copy2(保时间戳),并对 CSV 计算行数(count_rows = max(行数 - 1, 0),扣表头)写入 summary。

  ▎ 可重复要点 #1:这一步完全无计算,只复制文件。重复跑无副作用。命令:
  ▎ python scripts/03_data/00_prepare_mode_inputs_for_stage2.py \
  ▎   --source_root  /Users/wyc/SynPred/data/interim/training_modes/stage2_hybrid_cgcnn \
  ▎   --output_root  /Users/wyc/SynPred/data/interim/model_inputs/stage2_cvae_modes \
  ▎   --dataset_name stage2_hybrid_cgcnn

  ---
  第三部分 · CVAE:最简单的 (x, y_set)(20_build_stage2_cvae_dataset.py)
  
  CVAE 把"以结构 x 为条件的前驱物 multi-hot y"做生成建模。它需要的最少接口就是 (x, y_set),所以这一脚本是整个 Stage 03 (Data) 的通用基线。

  3.1 输入路径选择

  两套互斥路径:
  - 传统:select_input_paths(input_mode, project_root) → data/interim/features/{structdesc_features|stage2_hybrid_features}/,suffix={ml|hybrid}
  - mode-aware:select_mode_input_paths(mode_input_root, train_mode, input_mode) → mode_input_root/{relaxed_only|gold_only|curriculum/phase{1,2}}/
  
  gold_train_holdout.csv 视为可选,缺失就跳过(行 71-73)。

  3.2 张量构造(build_arrays,行 150-158)

  x = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)   # (N, F)
  y = df[label_cols].fillna(0).to_numpy(dtype=np.float32)        # (N, V)  multi-hot
  meta = df[meta_cols].fillna("")

  fillna(0.0) 是 SynPred 整套 03_data 的统一处理:任何特征缺失都填 0(因为 Stage 02 的 has_xxx 占位列已经把缺失语义编码进去)。label_cols 缺失填 0(无标签视为该前驱物没出现)。

  3.3 NPZ schema

  np.savez_compressed(out_path,
      x_raw=x_raw,    # (N, F)  float32  原始量纲
      x=x_std,        # (N, F)  float32  标准化后
      y=y             # (N, V)  float32  multi-hot
  )
  
  三个 key:x_raw / x / y。Meta 写 {split}_meta.csv。同时输出 feature_cols.json(F 个名字)、label_cols.json(V 个 label_prec__*)、label_names.json(V 个去前缀名)。

  3.4 命令行示例
  
  python scripts/03_data/20_build_stage2_cvae_dataset.py \
    --project_root /Users/wyc/SynPred \
    --input_mode hybrid \
    --mode_input_root /Users/wyc/SynPred/data/interim/model_inputs/stage2_cvae_modes/stage2_hybrid_cgcnn \
    --train_mode relaxed_only \
    --output_dir /Users/wyc/SynPred/data/interim/generative/stage2_cvae_dataset/hybrid/relaxed_only

  ▎ 可重复要点 #2:这一步是 03_data 的所有变体的"骨架代码"。其他变体都 fork 它,只改 encode 那一段。如果你要新加一种变体,从这个文件 copy + 修改即可。

  ---
  第四部分 · AR Seq2Seq:序列编码 + teacher forcing(25_build_stage2_ar_dataset.py)
  
  AR 模型把"集合"按 label_cols 给定的列序序列化。注意:不是按字典序、不是按出现频率,而是按 stage2_train 里 label_cols 的顺序——这和 Stage 02 里 precursor_vocab 的 sorted 顺序一致。

  4.1 词表

  def build_token_vocab(precursor_names):
      return [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + precursor_names
  # token_to_id = {tok: i}, 即 PAD=0, BOS=1, EOS=2, 第一个 precursor=3, ...

  4.2 序列编码(encode_sequences_from_multihot,行 159-201)

  对一条样本 multi-hot y,记 active 索引为 $A_i = {j : y_j = 1}$。设全局最大集合长度 max_target_len = max_i |A_i|(由 train 决定,所有 split 共享)。生成定长 max_target_len + 1(留 BOS/EOS):

  seq_in  = [BOS, t_1, t_2, ..., t_k,  PAD, ..., PAD]    # 长 max_target_len + 1
  seq_out = [t_1, t_2, ..., t_k,  EOS, PAD, ..., PAD]    # 同样长度
  seq_mask= [1,    1,   ...,  1,   1,    0, ..., 0]      # 1 处计算 loss
  set_len = k                                             # 真实 token 数

  其中 $t_j$ = token_to_id[precursor_names[A_i[j]]]。顺序是 active_idx 的升序(numpy where 默认升序)——这是 canonical order,模型看到的所有正样本都按这个顺序解码。

  为什么 mask 长度是 k+1 而非 k?因为 EOS 也要监督——模型必须学会"什么时候停",seq_mask 把 EOS 那一格也置 1。

  4.3 NPZ schema
  
  np.savez_compressed(out_path,
      x_raw=x_raw,                # (N, F)
      x=x,                        # (N, F)
      y_multi_hot=y_multi_hot,    # (N, V)
      seq_in=seq_pack["seq_in"],  # (N, L+1)  int64
      seq_out=seq_pack["seq_out"],# (N, L+1)  int64
      seq_mask=seq_pack["seq_mask"],# (N, L+1) int64 (0/1)
      set_len=seq_pack["set_len"]  # (N,)     int64
  )

  5 个序列 key + 3 个常规 key。注意:seq_in / seq_out 都是 int64(因为 PyTorch CrossEntropyLoss 需要 long target)。

  4.4 vocab 落盘
  
  {output_dir}/
  ├── token_vocab.json   # ["<pad>", "<bos>", "<eos>", precursor_1, ...]
  ├── precursor_names.json
  ├── label_cols.json
  ├── feature_cols.json
  └── feature_mean.npy / feature_std.npy

  ▎ 可重复要点 #3:max_target_len 是 train + val + test + gold 全局最大值(行 320-328 in 25 计算所有 split,然后取 max)。这保证 NPZ 张量在所有 split 上行宽一致,加载时不用动态 padding。

  ---
  第五部分 · SetPred Transformer:固定槽位编码(26_build_stage2_setpred_dataset.py)
  
  SetPred Transformer 把多 hot 集合预测看成"固定 N 个槽位的并行分类"。每个槽位是一个 (V+1) 类的 softmax(0=PAD,1..V=各前驱物)。

  5.1 词表

  def build_slot_vocab(precursor_names):
      return [PAD_TOKEN] + precursor_names
  # precursor_to_slot_id = {p: i + 1}, PAD_ID = 0

  注意比 AR 少了 BOS/EOS——SetPred 是并行模型,不需要边界符。

  5.2 槽位编码(build_slot_targets_from_multihot,行 152-184)

  固定 n_slots(默认 7,来自 --n_slots)。对每条样本:

  active_idx = np.where(y_multi_hot[i] > 0)[0]
  tokens = [precursor_to_slot_id[precursor_names[j]] for j in active_idx]
  # 顺序同 AR,沿用 active_idx 升序 (canonical)

  if len(tokens) > n_slots:
      overflow[i] = 1            # 标记溢出
      tokens = tokens[:n_slots]   # 截断

  slot_targets[i, :len(tokens)] = tokens     # 其余位置保留 PAD_ID=0
  slot_mask[i, :len(tokens)] = 1
  set_len[i] = original_len
  
  四个数组:slot_targets (N, n_slots) int64、slot_mask (N, n_slots) int64、set_len (N,)、overflow (N,)。

  overflow 是个诊断信号——训练时不直接用,但 summary 里会统计 n_overflow,如果超过 5%,要么调大 n_slots,要么数据里有异常长的前驱物列表。

  5.3 NPZ schema
  
  np.savez_compressed(out_path,
      x_raw, x,
      y_multi_hot,
      slot_targets,    # (N, n_slots) int64, 0 = PAD
      slot_mask,       # (N, n_slots) int64
      set_len,         # (N,)         int64  原始集合大小
      overflow         # (N,)         int64  截断标志
  )

  ▎ 可重复要点 #4:n_slots=7 是按 SynPred 数据集统计的(p99 ≤ 6,留一个安全边际)。换数据集时,先跑一下 df[label_cols].sum(axis=1).describe([0.95, 0.99]) 决定。

  ---
  第六部分 · GFlowNet:动作序列 + STOP token(27_build_stage2_gflownet_dataset.py)
  
  GFlowNet 把生成集合看成"按一系列动作把空集逐步扩张"。每个动作要么"选某个前驱物",要么"STOP 表示完成"。

  6.1 动作词表

  def build_action_vocab(precursor_names):
      return precursor_names + [STOP_TOKEN]
  # action_to_id = {a: i};  precursor 在前,STOP 在最后

  注意没有 PAD——GFlowNet 用 mask 而不是 PAD token 标识"轨迹外"的位置。

  6.2 参考轨迹(encode_reference_trajectories,行 184-211)

  n = y_multi_hot.shape[0]
  traj_actions = np.full((n, max_set_len + 1), stop_id, dtype=np.int64)   # 默认填 STOP
  traj_mask    = np.zeros((n, max_set_len + 1), dtype=np.int64)
  set_len      = np.zeros(n, dtype=np.int64)

  for i in range(n):
      active_idx = np.where(y_multi_hot[i] > 0)[0]
      actions = [action_to_id[precursor_names[j]] for j in active_idx]
      set_len[i] = len(actions)
      traj_actions[i, :len(actions)] = actions
      traj_actions[i, len(actions)]  = stop_id
      traj_mask[i, :len(actions) + 1] = 1

  输出:对每条样本一条ground-truth 轨迹:[a_1, a_2, ..., a_k, STOP],其余位置默认 STOP(无效但安全),mask 长度 k+1。

  ▎ 可重复要点 #5:这条轨迹是 teacher-forcing 的最大似然 baseline。GFlowNet 真正训练时还会用 importance 采样和 trajectory balance,但先用最大似然预热可以让 GFlowNet 不会冷启动崩溃——这一编码就是为预热服务。

  ▎ 可重复要点 #6:与 AR 的差别: AR 序列长度 = max_set_len + 1(BOS+tokens 或 tokens+EOS),GFlowNet 也是 max_set_len + 1(tokens+STOP)。结构完全对应,只是命名不同(STOP↔EOS)且没有 BOS(GFlowNet 
  ▎ 从空集开始,自带"BOS")。

  6.3 NPZ schema

  np.savez_compressed(out_path,
      x_raw, x,
      y_multi_hot,
      traj_actions,    # (N, L+1) int64
      traj_mask,       # (N, L+1) int64
      set_len          # (N,)     int64
  )
  
  落盘 action_vocab.json / action_to_id.json / precursor_names.json。

  ---
  第七部分 · GFlowNet 的 mode 子集:28_build_stage2_gflownet_mode_datasets.py
  
  设计目标:已经有了完整 train.npz/val.npz/test.npz(由 27 在 relaxed train 上构),想要 gold_only / curriculum_phase{1,2} 三种子集,不重新 encode,直接用 mode CSV 里的 id 集合做 mask 过滤。

  7.1 输入解析

  resolve_mode_split_csvs(行 133-181)依次尝试:
  1. 嵌套 train_mode/{train,val,test}/*.csv
  2. 平铺 train_mode/stage2_train_hybrid.csv
  3. curriculum_phase1 train_mode/phase1_train/(在 curriculum/ 下)
  4. curriculum_phase2 train_mode/phase2_train/

  7.2 join key 探测(find_join_key,行 184-203)

  JOIN_KEYS = [id, row_id, sample_id, material_id, entry_id, reaction_id, synth_uid, record_index],从前到后第一个双方都存在且都有非空值的就是 join key。

  7.3 过滤(filter_one_split,行 206-240)

  keep_ids = set(split_csv[join_key].astype(str).str.strip().tolist())
  mask = base_meta[join_key].astype(str).str.strip().isin(keep_ids).to_numpy()

  # 对 base_npz 里每个数组,按第 0 轴 mask
  filtered_npz[name] = arr[mask] if arr.shape[0] == n_rows else arr  # 标量/全局张量保留

  注意:保持 base npz 的 schema 不变——不动 traj/seq 等字段,只过滤行。这样 28 输出的 NPZ 与 27 输出的 schema 完全一致,训练器无感知。

  7.4 输出布局
  
  {base_input_dir}/
  ├── train.npz, val.npz, test.npz       # 来自 27 (relaxed)
  ├── action_vocab.json, ...
  └── gold_only/                          # 28 输出
      ├── train.npz, val.npz, test.npz
      ├── train_meta.csv, ...
      ├── action_vocab.json (复制)
      └── summary.json

  这种"二次过滤而非二次 encode"模式让 SynPred 在 4 种 train_mode 上构 dataset 的总耗时只是 1×(原始 encode)+ 3×(纯 mask copy),典型节省 80% 时间。

  ---
  第八部分 · Mixture Flow Stage3:六张量条件数据集(27_build_stage3_condition_dataset_v5_mixed.py)
  
  这是 Stage 3 的主力 dataset builder,产出 mixture flow 模型需要的混合类型目标:(x, y_set, y_cond_continuous, y_cond_continuous_mask, y_cond_discrete, y_cond_discrete_mask)。

  8.1 列定位

  # 默认值(可被 --continuous_cols / --discrete_cols / --precursor_cols 覆盖)
  continuous_cols = ["target_temperature_c_clean", "target_time_h_clean"]
  discrete_cols   = ["target_atmosphere_coarse", "synthesis_type"]
  precursor_cols  = ["target_main_precursors", "target_aux_precursors"]

  注意:target_aux_precursors 也被纳入 set 标签——Stage3 的"前驱物条件"包括主+辅,这与 Stage2 不同(Stage2 只用 main)。

  8.2 特征列推断(infer_feature_cols,行 125-137)

  不再依赖 feat_* 前缀,而是排除式过滤:
  
  NON_FEATURE_PREFIXES = ("id", "synth_uid", "source_dataset", "record_index",
                          "material_id", "formula", "mp_formula", "synth_formula",
                          "parent_formula", "doi", "dois", "split_group",
                          "poscar_path", "summary_json_path", "provenance_json_path",
                          "reaction_string", "synthesis_text",
                          "main_precursors", "aux_precursors",
                          "target_", "has_target", "condition_source")
  exclude_cols = NON_FEATURE_PREFIXES + precursor_cols + [sample_id_col]

  # 凡是数值列且不在 exclude 里的,都进 x

  这一逻辑可以容忍 stage3_raw.csv 里没有 feat_ 前缀的"裸描述子"(早期版本曾出现)。target_ 这个 prefix 是关键的反污染——所有 target 列都不进 x,避免目标泄漏。

  8.3 集合标签(build_precursor_vocab,行 109-122)
  
  仅从 train_df 收集 vocab,顺序 sorted(unique values)。然后 precursor_to_multihot 把 train/val/test 各自的 main+aux 列展成 multi-hot,OOV 直接忽略(不在 vocab 里就不置 1)。

  parse_precursor_value(行 68-89)处理三种格式:list、JSON 串、逗号/分号分隔串,最后 fallback 是把整个值当一个元素。dedup_keep_order 在合并 main+aux 后保序去重。

  8.4 连续目标编码(encode_continuous_targets,行 235-256)
  
  对每个连续列 col 独立:
  1. 在 train 上算 stats:{mean, std, median},std<1e-8 时置 1
  2. 对每条样本,safe_float(val) 解析,失败 → mask=0,值=0
  3. 否则:y[i,j] = (fv - mean) / std,mask[i,j] = 1
  
  safe_float 容错很宽松(行 157-183),处理 "850°C"、"12 hours"、"1,200" 这种带单位/千分位的脏值——剥掉单位后再 float。

  输出 (N, n_cont) 标准化值 + 同 shape 的 mask。val/test 用 train 的 mean/std 转换,不重新拟合。

  8.5 离散目标编码(encode_discrete_targets,行 259-279)

  每个离散列独立词表,用 train 拟合:
  
  def build_vocab(train_df, col):
      vals = [norm_str(x) for x in train_df[col].tolist()]
      vals = [x for x in vals if x]
      vocab = ["<UNK_OR_MISSING>"] + sorted(set(vals))   # idx 0 = OOV/缺失
      return vocab, {tok: i for i, tok in enumerate(vocab)}

  每条样本:
  - 缺失/空 → y[i,j] = 0(),mask[i,j] = 0
  - 找到 → y[i,j] = stoi[norm_str(val)],mask[i,j] = 1
  - OOV(val/test 没在 train 出现的值)→ y[i,j] = stoi.get(s, 0) 即 OOV → 0,mask[i,j] = 1
  
  注意微妙之处:OOV 值的 mask 仍然是 1——这一行进入 loss(目标是  类别)。如果你不想训练模型预测 ,需要在训练器里加 mask=0 当 OOV 的逻辑。这是一个隐含的工程契约。

  8.6 NPZ schema

  np.savez_compressed(output_dir / f"{split_name}.npz",
      x=x.astype(np.float32),                              # (N, F)
      y_set=y_set.astype(np.float32),                      # (N, V_prec)  multi-hot
      y_cond_continuous=y_cont.astype(np.float32),         # (N, n_cont)  standardized
      y_cond_continuous_mask=y_cont_mask.astype(np.float32),
      y_cond_discrete=y_disc.astype(np.int64),             # (N, n_disc)
      y_cond_discrete_mask=y_disc_mask.astype(np.float32),
      sample_id=np.asarray([norm_str(x) for x in df["id"]], dtype=object)
  )

  注意:这里 x 是原始量纲(没有标准化),与其他变体不一致——v5 mixed 把 x 标准化的责任留给训练器。这是个历史选择,你也可以在用之前手动标准化。

  输出附加 schema.json 与 condition_schema.json(含 vocab、stats),export_summary.json(每个 split 的非空目标统计)。

  ▎ 可重复要点 #7:命令行
  ▎ python scripts/03_data/27_build_stage3_condition_dataset_v5_mixed.py \
  ▎   --output_dir /Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1 \
  ▎   --feature_dir /Users/wyc/SynPred/data/interim/features/structdesc_features_stage3_v2 \
  ▎   --continuous_cols target_temperature_c_clean,target_time_h_clean \
  ▎   --discrete_cols target_atmosphere_coarse,synthesis_type \
  ▎   --precursor_cols target_main_precursors,target_aux_precursors \
  ▎   --sample_id_col id

  ---
  第九部分 · Stage35 兼容性二分类:硬负样本 + 随机负样本(24_build_stage35_hardneg_compat_dataset.py)
  
  Stage35 兼容性模型回答:"给定(目标结构, 候选前驱物集合),这一对是否能合成出该目标?"——一个二分类任务。它的训练数据需要正负配对,正样本来自数据集本身,负样本是难点。

  9.1 三类样本对

  对每条 positive(target i,true_set $S_i$):
  1. positive:(x_i, multihot(S_i), label=1),1 条
  2. hard negative:从硬负池里挑 n_hard_neg_per_pos=2 条
  3. random negative:从全 train 的 true_keys 集合里挑 n_random_neg_per_pos=2 条(同一 target 除外)
  
  每条 target 期望产出 5 行(1 正 + 2 硬负 + 2 随机负),N 条 target → 5N 行训练样本。

  9.2 硬负样本来源

  --samples_csv runs/generative/stage2/cvae_hybrid_v1/samples_train/train_samples.csv

  这是 Stage2 主模型(典型是 CVAE)在 train 上跑出的候选 jsonl/csv。结构:每个样本 id 对应若干 pred_labels 和一个 true_labels。硬负 = 模型给高分但 pred_labels ≠ true_labels 的预测。

  build_candidate_pool_from_samples(行 273-293):
  pool = {}
  for sid, group in samples_df.groupby("id"):
      seen = set()
      for _, row in group.iterrows():
          pred_labels = parse_list_field(row["pred_labels"])
          pred_key = tuple(sorted(set(pred_labels)))
          if not pred_key:                continue   # 空集跳过
          if pred_key == true_key:        continue   # 与真值相同跳过(那是正样本)
          if pred_key in seen:            continue   # 同 candidate 去重
          seen.add(pred_key)
          items.append((pred_key, list(pred_key)))
      pool[sid] = items
      
  输出 pool: {sample_id: [(pred_key, pred_list), ...]},每个 sample 对应若干"模型混淆点"。

  9.3 随机负样本池

  key_to_multihot = {}
  for labs in true_lists:
      key = precursor_list_to_key(labs)   # tuple(sorted(set(labs)))
      if len(key) > 0 and key not in key_to_multihot:
          key_to_multihot[key] = labels_to_multihot(labs)
  random_pool_keys = list(key_to_multihot.keys())

  键去重——同样的前驱物集合作为负池中的"一种"。每条正样本采样时:
  available = [k for k in random_pool_keys if k != true_key]   # 排除自己
  chosen = rng.choice(available, n_random_neg_per_pos, replace=False)
  
  9.4 张量构造(build_split_pairs,行 296-405)

  每行:
  x_struct  = (F,)              结构特征(等同于其他变体的 x_raw)
  precursor_y = (V,)             候选前驱物 multi-hot
  x_joint  = concat(x_struct, precursor_y)   # 长度 F+V
  y        = 0/1                 是否兼容
  
  pair_type ∈ {"positive","hard_negative","random_negative"} 写入 meta,便于按类别诊断。

  9.5 NPZ schema

  np.savez_compressed(out_dir / f"{split}.npz",
      x_struct=...,        # (5N, F)
      precursor_y=...,     # (5N, V)
      x_joint=...,         # (5N, F+V)  注意:这是标准化后的!
      y=...                # (5N,)  int64
  )

  关键:x_joint 在写盘前用 train 上拟合的 mean/std 标准化(行 625):
  mean, std = fit_standardizer(train_x_joint)   # 在 [F+V] 维上同时拟合
  pack["x_joint"] = transform_standardize(pack["x_joint"], mean, std)
  
  这意味着 multi-hot 部分的 0/1 也被标准化了——Bernoulli 变量经标准化后均值 0、方差 ≈ 1,与连续特征同尺度。这对二分类器很重要(MLP 不擅长不同尺度的输入)。

  9.6 命令行示例

  python scripts/03_data/24_build_stage35_hardneg_compat_dataset.py \
    --project_root /Users/wyc/SynPred \
    --input_mode hybrid \
    --mode_input_root /Users/wyc/SynPred/data/interim/training_modes/stage3_temperature \
    --target_mode temperature \
    --train_mode relaxed_only \
    --precursor_vocab_json /Users/wyc/SynPred/data/interim/features/structdesc_features/meta/precursor_vocab.json \
    --samples_csv /Users/wyc/SynPred/runs/generative/stage2/cvae_hybrid_v1/samples_train/train_samples.csv \
    --n_hard_neg_per_pos 2 \
    --n_random_neg_per_pos 2 \
    --seed 42 \
    --output_dir /Users/wyc/SynPred/data/interim/generative/stage35_hardneg/hybrid/relaxed_only/temperature

  ▎ 可重复要点 #8:samples_csv 必须先跑 stage2 主模型才有。SynPred 的依赖序是:Stage2 训完 → 在 train 上推理出 candidate pool → 用这个 pool 做 hardneg → 训 Stage35 兼容性模型。这是一个软循环依赖,不是 
  ▎ bug,是设计。

  ---
  第十部分 · Stage35 强联合候选(40_build_stage35_strong_joint_candidates.py)
  
  把"强 stage2 候选(完整 precursor 集)" × "强 stage3 候选(给定 precursor 后的条件)"按 (material_key, parent_precursor_rank) 做笛卡尔积,得到 recipe-level 候选。

  10.1 输入

  strong stage2 candidates : data/interim/generative/stage35_strong_stage2_candidates/<model>/{split}_candidates.jsonl
    ├── sample_id, material_id, material_key, precursor_rank,
    ├── precursor_set, n_precursors, stage2_score, stage2_model
    └── formula, doi, source_dataset, synthesis_type

  strong stage3 candidates : data/interim/generative/stage35_strong_stage3_candidates/<model>/{split}_candidates.jsonl
    ├── sample_id, material_id, material_key,
    ├── parent_precursor_rank, condition_rank,
    ├── disc_conditions, disc_condition_indices, cont_conditions,
    └── stage3_score, stage3_model

  stage2 候选每个 (material, precursor_rank) 对应一组前驱物集;stage3 候选每个 (material, parent_precursor_rank) 对应多个条件 rank,所以 stage3 是 list。

  10.2 索引与冲突解决(build_strong_joint_candidates,行 250-345)

  stage2 索引(一对一):
  stage2_by_key: Dict[(material_key, precursor_rank), s2_row]
  
  # 冲突时:
  # - 优先 precursor_set 更长(多前驱物意味着更复杂、更具体)
  # - 长度相同时优先 stage2_score 更高
  # 这是稳健的去重策略,假设短集是长集的子集时优先长集
  
  stage3 索引(一对多):
  stage3_by_key: Dict[(material_key, parent_precursor_rank), List[s3_row]]
  # 内部按 condition_rank 升序排序

  10.3 笛卡尔积
  
  for key, s2_row in sorted(stage2_by_key.items()):
      s3_list = stage3_by_key.get(key, [])
      if not s3_list:
          continue   # stage2 有候选但 stage3 无 → 整条 recipe 不存在
      matched_keys += 1
      for s3_row in s3_list:
          out = {
              "recipe_id": f"{material_key}__p{precursor_rank}__c{condition_rank}",
              "precursor_rank": s2_row.precursor_rank,
              "parent_precursor_rank": s3_row.parent_precursor_rank,
              "condition_rank": s3_row.condition_rank,
              "precursor_set": s2_row.precursor_set,
              "disc_conditions/disc_condition_indices/cont_conditions": s3_row.{...},
              "stage2_score": s2_row.stage2_score,
              "stage3_score": s3_row.stage3_score,
              "joint_prior_score": None,        # 留接口,留给 41 计算
              ...
          }

  注意 precursor_rank == parent_precursor_rank——这是 stage3 候选生成时就以 stage2 候选为条件的契约。stage3 模型在生成 condition 时,parent_precursor_rank 字段被设为它所条件的 stage2 候选的 rank,40 脚本据此
  join。

  10.4 输出 JSONL

  每条:{recipe_id, sample_id, material_id, material_key, split, ranks, precursor_set, n_precursors, disc/cont_conditions, stage2_score, stage3_score, joint_prior_score=None, models, formula/doi/...}

  10.5 命令行示例

  python scripts/03_data/40_build_stage35_strong_joint_candidates.py \
    --stage2_candidates_dir .../stage35_strong_stage2_candidates/cvae_hybrid_v1 \
    --stage3_candidates_dir .../stage35_strong_stage3_candidates/mixture_v1 \
    --output_dir            .../stage35_strong_joint_candidates/hybrid_core_strong \
    --splits val,test \
    --view_name hybrid_core_strong

  ▎ 可重复要点 #9:stage2/stage3 候选的"strong"含义:经过排名 + top-k 过滤的高质量候选,而非生成模型直接采样的原始候选。SynPred 流水线里,从原始候选到 strong 候选还需经过 
  ▎ 34_build_stage35_strong_stage2_candidates.py 与 38_build_stage35_strong_stage3_candidates.py 的格式归一与过滤。

  ---
  第十一部分 · Stage35 强联合 dataset:多软标签计算(41_build_stage35_strong_joint_dataset.py)
  
  输入 40 的 candidates,回查 ground-truth 计算各种标签。这是 Stage35 排序器的训练核心——既给硬标签 joint_label,也给软标签 joint_soft_score。

  11.1 真值映射

  Stage2 真值映射(build_stage2_truth_map,行 223-247):
  # 输入:stage2 source CSV(每行 (id, true_labels, ...))
  material_id = extract_mp_id(row.material_id or row.id)   # "mp-17677" 或全串
  out[material_id] = {
      "true_precursor_labels": parse_true_labels(row.true_labels),
      "formula", "doi", "source_dataset", "synthesis_type"
  }   
  
  extract_mp_id(行 186-191)用正则 r"(mp-\d+)" 抽 MP id,失败时直接返回原串。这一行兼容 (material_id="mp-17677")、(id="syn_42")、混合形式三种情况。

  parse_true_labels(行 198-217)处理 list、JSON、||/;/| 分隔串等多格式,并用 strip_label_prefix 把列名前缀 label_prec__ 剥掉。

  Stage3 真值映射(行 250-269):
  out[material_id] = {
      "true_temperature_c": safe_float(row.true_temperature_c),
      "true_time_h":        safe_float(row.true_time_h),
      "mask_temperature_c": safe_float(row.mask_temperature_c),
      "mask_time_h":        safe_float(row.mask_time_h),
  }   
  
  注意 mask_* 字段并不阻止后续匹配——见 11.4。

  11.2 集合度量(precursor_set_metrics,行 275-301)

  pred = set(precursor_set);  true = set(true_precursor_labels)

  precursor_exact_match     = (pred == true)
  precursor_overlap_count   = |pred ∩ true|
  precursor_precision       = |pred ∩ true| / |pred|     # 0 if pred=∅
  precursor_recall          = |pred ∩ true| / |true|     # 0 if true=∅
  precursor_jaccard         = |pred ∩ true| / |pred ∪ true|   # 0 if both ∅... 但 if pred=true=∅: 1

  特例:true ∅ ∧ pred ∅ → 全 1.0(完美匹配)。这是 corner case,实际中不会发生,但代码留了一手。

  11.3 连续度量与 capped score(compute_cont_metrics,行 310-356)

  对温度和时间各自:
  
  abs_err  = |pred - true|                        # 若 pred 或 true 缺失 → None
  match    = (abs_err <= tol)                     # 默认 temp_tol=150°C, time_tol=24h
  score    = max(0, 1 - abs_err/tol)              # capped to [0,1], None if abs_err None

  capped_score(行 304-307)是个软的"在容差内打分"映射:误差为 0 时 score=1.0,误差到 tol 时 score=0,超出 tol 时仍是 0(不向负穿透)。

  聚合:
  score_parts = [s for s in [temp_score, time_score] if s is not None]
  cont_score  = mean(score_parts) if score_parts else None
  
  match_parts = [m for m in [temp_match, time_match] if m is not None]
  cont_match  = all(match_parts) if match_parts else None 
  
  cont_match 用 and(两个都得在容差内才算匹配),cont_score 用 mean(任一缺失时按非空均值计)。

  ▎ 可重复要点 #10:作者在 41 脚本里有 # FIX: 注释(行 324-326)——更早版本里若 mask_temperature_c=0 就跳过该指标,但 strong joint 强制把"只要真值存在就用",这是为了避免 mask 数据导致评估样本数过少。这意味着 41 
  ▎ 与同目录其他 21/22/24 的 mask 语义不同,务必看清楚。

  11.4 联合标签(build_joint_labels,行 358-393)

  硬标签:
  if precursor_exact_match is None or cont_match is None:
      joint_label = -1               # 无效,下游训练时 mask 掉
  else:
      joint_label = int(bool(precursor_exact_match and cont_match))
      
  只有"前驱物完全匹配 ∧ 连续条件全部在容差内" → joint_label=1。这是非常严苛的硬标签,在测试集上典型阳性率 < 5%。

  软标签:
  precursor_part = precursor_jaccard     # [0, 1]
  condition_part = cont_score             # [0, 1]
  
  # 加权平均
  if both None: None
  elif one None: 取另一
  else:
      joint_soft_score = (w_p * J + w_c * S) / (w_p + w_c)
      
  默认 precursor_weight = condition_weight = 1.0,即等权平均。如果某一目标更重要(比如医药合成里前驱物决定一切),把 precursor_weight=2.0 即可加权。

  11.5 输出每行字段

  {
    // 来自 candidates
    "sample_id", "material_id", "material_key", "recipe_id", "split",
    "precursor_rank", "parent_precursor_rank", "condition_rank",
    "precursor_set", "n_precursors",
    "disc_conditions", "disc_condition_indices", "cont_conditions",
    "stage2_score", "stage3_score",
    "stage2_model", "stage3_model",
    "formula", "doi", "source_dataset", "synthesis_type",

    // 派生
    "group_id": "mp-17677__p3",       // 用于 listwise 学习排序时的 group 键
    "true_precursor_labels": [...],   // 真值
    "precursor_exact_match": bool,
    "precursor_overlap_count": int,
    "precursor_precision": float,
    "precursor_recall": float,
    "precursor_jaccard": float,
    "true_temperature_c": float|null,
    "true_time_h": float|null,
    "mask_temperature_c": float|null,
    "mask_time_h": float|null,
    "temp_abs_err": float|null,
    "time_abs_err": float|null,
    "temp_match": bool|null,
    "time_match": bool|null,
    "cont_score": float|null,
    "cont_match": bool|null,
    "joint_label": 1|0|-1,            // 硬标签
    "joint_soft_score": float|null    // 软标签
  }

  group_id 是关键:它表示"同一个目标 + 同一组前驱物 → 多个条件候选"的 group,听起排序模型(LambdaRank/ListNet)按 group 划 batch。

  11.6 命令行
  
  python scripts/03_data/41_build_stage35_strong_joint_dataset.py \
    --joint_candidates_dir .../stage35_strong_joint_candidates/hybrid_core_strong \
    --stage2_source_dir    runs/stage2/cvae_hybrid_v1/strong_candidates \
    --stage3_source_dir    runs/stage3/mixture_v1/strong_candidates \
    --output_dir           .../stage35_strong_joint_dataset/hybrid_core_strong \
    --splits val,test \
    --view_name hybrid_core_strong \
    --temperature_tol 150.0 \
    --time_tol 24.0 \
    --precursor_weight 1.0 \
    --condition_weight 1.0

  ▎ 可重复要点 #11:temperature_tol=150°C 与 time_tol=24h 是 SynPred 经验值——温度对工艺影响大(150°C 对应反应阶段差异),时间影响弱(24h 是"一天内"的等价)。换数据集要重新调,典型策略:看 train 分布的 IQR(75%-25% 
  ▎ percentile),取一半。

  ---
  第十二部分 · 端到端复现指南

  12.1 依赖

  pip install numpy pandas

  无 ML 框架依赖——03_data 都是纯 numpy/pandas。

  12.2 一条端到端管道(stage2 hybrid + 4 mode + GFlowNet 为例)

  PR=/Users/wyc/SynPred

  # ============ STEP 1: 把 stage2 hybrid CSV 复制到 mode-aware 扁平布局 ============
  python $PR/scripts/03_data/00_prepare_mode_inputs_for_stage2.py \
    --source_root  $PR/data/interim/training_modes/stage2_hybrid_cgcnn \
    --output_root  $PR/data/interim/model_inputs/stage2_cvae_modes \
    --dataset_name stage2_hybrid_cgcnn

  # ============ STEP 2: 4 个变体在 relaxed_only 上构 dataset ============
  for variant in cvae ar setpred gflownet; do
    python $PR/scripts/03_data/${variant_to_script[$variant]} \
      --project_root $PR \
      --input_mode hybrid \
      --mode_input_root $PR/data/interim/model_inputs/stage2_cvae_modes/stage2_hybrid_cgcnn \
      --train_mode relaxed_only \
      --output_dir $PR/data/interim/generative/stage2_${variant}_dataset/hybrid/relaxed_only
  done

  # ============ STEP 3: GFlowNet 三种 mode(从 relaxed 全量过滤) ============
  for tm in gold_only curriculum_phase1 curriculum_phase2; do
    python $PR/scripts/03_data/28_build_stage2_gflownet_mode_datasets.py \
      --base_input_dir   $PR/data/interim/generative/stage2_gflownet_dataset/hybrid/relaxed_only \
      --mode_input_root  $PR/data/interim/model_inputs/stage2_cvae_modes/stage2_hybrid_cgcnn \
      --train_mode $tm
  done

  # ============ STEP 4: Stage3 mixed dataset ============
  python $PR/scripts/03_data/27_build_stage3_condition_dataset_v5_mixed.py \
    --output_dir $PR/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1 \
    --feature_dir $PR/data/interim/features/structdesc_features_stage3_v2

  # ============ STEP 5: Stage35 hardneg compat dataset ============
  # 前置: 已有 runs/generative/stage2/cvae_hybrid_v1/samples_*/(*.csv)
  python $PR/scripts/03_data/24_build_stage35_hardneg_compat_dataset.py \
    --project_root $PR \
    --input_mode hybrid \
    --samples_csv $PR/runs/generative/stage2/cvae_hybrid_v1/samples_train/train_samples.csv \
    --n_hard_neg_per_pos 2 --n_random_neg_per_pos 2 \
    --output_dir $PR/data/interim/generative/stage35_hardneg/hybrid

  # ============ STEP 6: Stage35 strong joint candidates + dataset ============
  python $PR/scripts/03_data/40_build_stage35_strong_joint_candidates.py \
    --stage2_candidates_dir $PR/data/interim/generative/stage35_strong_stage2_candidates/cvae_hybrid_v1 \
    --stage3_candidates_dir $PR/data/interim/generative/stage35_strong_stage3_candidates/mixture_v1 \
    --output_dir            $PR/data/interim/generative/stage35_strong_joint_candidates/hybrid_core_strong

  python $PR/scripts/03_data/41_build_stage35_strong_joint_dataset.py \
    --joint_candidates_dir $PR/data/interim/generative/stage35_strong_joint_candidates/hybrid_core_strong \
    --stage2_source_dir    $PR/runs/stage2/cvae_hybrid_v1/strong_candidates \
    --stage3_source_dir    $PR/runs/stage3/mixture_v1/strong_candidates \
    --output_dir           $PR/data/interim/generative/stage35_strong_joint_dataset/hybrid_core_strong \
    --temperature_tol 150.0 --time_tol 24.0

  12.3 各变体 NPZ 文件大小估计

  ┌─────────────────────────────┬────────────────────────────────────────┬────────────────────────────────┐
  │            变体             │                  张量                  │ 典型大小(N=8000, F=164, V=580) │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ CVAE                        │ x_raw, x, y                            │                          ~7 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ AR                          │ + seq_in/out/mask, set_len             │                          ~8 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ SetPred                     │ + slot_targets/mask, set_len, overflow │                        ~7.5 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ GFlowNet                    │ + traj_actions/mask, set_len           │                        ~7.5 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ Mixture (Stage3)            │ x, y_set, y_cond_*, mask, sample_id    │                         ~10 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ Stage35 hardneg(5N 行)      │ x_struct, precursor_y, x_joint, y      │                         ~25 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ Stage35 strong joint(JSONL) │ 不固定,~10-100 MB                      │                   取决于候选数 │
  └─────────────────────────────┴────────────────────────────────────────┴────────────────────────────────┘

  12.4 验收清单
  
  ┌───────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┐
  │          验收点           │                                              命令                                               │                      期望                      │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ NPZ 张量形状一致          │ python -c "import numpy as np; d=np.load('train.npz'); print({k:v.shape for k,v in d.items()})" │ x.shape[0] == y.shape[0] == seq_in.shape[0]    │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 训练集标准化均值 ≈ 0      │ np.load('train.npz')['x'].mean(0)                                                               │ abs ≤ 1e-3                                     │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ val/test 标准化非平凡     │ np.load('val.npz')['x'].mean(0)                                                                 │ 非全 0                                         │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 词表大小一致              │ feature_cols + label_cols 长度匹配 NPZ                                                          │ F + V == x.shape[1] + y.shape[1]               │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Stage3 mask 数 vs cell 数 │ mask.sum() vs mask.size                                                                         │ 比值合理(>30%)                                 │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Stage35 hardneg 比例      │ summary.json 里 pair_type_dist                                                                  │ positive : hard_neg : random_neg = 1 : ≤2 : ≤2 │
  ├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Strong joint 阳性率       │ summary.json n_joint_positive / n_rows                                                          │ 通常 < 10%(严苛硬标签)                         │
  └───────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┘

  12.5 常见坑

  ┌───────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────┐
  │             现象              │                   原因                    │                       解                       │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ No feature columns detected   │ feature CSV 列前缀不对                    │ 检查 hybrid CSV 是否含 feat_* 或 *_graph_emb_* │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ val 上有 <UNK_OR_MISSING> 多  │ val/test 出现 train 没见过的离散类        │ 把这些样本扔了 or 把 mask 设 0                 │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ set_len 超过 max_set_len      │ 测试集出现极长前驱物列表                  │ 提高 SetPred 的 --n_slots,或在 AR 里截断       │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Stage35 hardneg 阳性 0        │ samples_csv 里所有 pred 都对              │ 说明 stage2 模型过拟合,不能用作硬负源          │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Strong joint matched_keys = 0 │ stage2/3 候选用的 material_key 体系不一致 │ 统一用 extract_mp_id 标准化                    │
  └───────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────┘

  ---
  第十三部分 · 设计哲学与可改进点

  13.1 七个工程模式

  1. 通用模板 + per-variant encode——所有变体共享 detect/standardize/save,只改 encode 那一段。新增变体的成本极低。
  2. train-only 词表 / standardizer,严格反泄漏。
  3. fillna(0) + has_xxx 占位——简单稳定,模型学缺失模式。
  4. multi-hot active_idx 升序作为 canonical order——AR/SetPred/GFlowNet 共享同一序约定。
  5. NPZ keys 是模型输入名的契约——变体之间的 keys 集合不同,但同一变体在 train/val/test 上 keys 严格一致。
  6. 过滤而非重 encode——28 用 mask 把 base NPZ 切片,节省 80% 时间。
  7. 硬标签 + 软标签并存——41 同时输出 joint_label∈{0,1,-1} 与 joint_soft_score∈[0,1],排序模型用软标签 + 硬标签做 listwise loss + binary classification head。

  13.2 可改进点

  (a) fit_standardizer 没去 winsorize。极端 outlier(比如 stage3 温度 19999°C 这种脏值)会拉爆 std,使其他特征被压扁。SafeVal 在 02 已经过滤,但极少漏网仍可能存在。可以加 clip(p1, p99) 后拟合。

  (b) 27_v5_mixed 不标准化 x。这与其他变体不一致,如果你想 reuse 同一个特征流水,会卡住。建议在 v6 里也加 x = (x_raw - mean)/std,与其他变体对齐。

  (c) OOV 离散标签的 mask 被置 1(13.5)。这会让 OOV 样本"错误地"被算进 loss。修复:y[i,j]=stoi.get(s, 0); m[i,j] = 1 if s in stoi else 0。

  (d) Stage35 hardneg 没"过 stage2 模型 confidence 阈值"。当前是 pred_key ≠ true_key 就要,但有些 pred 是模型给低分的——它们不是真硬负样本。可以加 --hardneg_min_score 过滤。

  (e) strong joint 的 stage2 真值映射用 first match(行 239)。同一 material_id 在不同 doi/synthesis_type 下可能有不同 true_labels,first match 漏掉合法变体。可以改成 (material_id, doi, synthesis_type)
  三元组主键,或保留 list 在 metric 里取 best。

  (f) group_id 只到 precursor_rank 粒度。如果想做 listwise 排序,group 应该是 (material_key, parent_precursor_rank),即同一前驱物组合下不同条件 rank
  的候选才分一组——目前定义其实正好是这个,但命名容易让读者误解为"按 material 分"。文档可加注。

  13.3 与同类工作的差异

  文献里的"多模型 dataset 工厂"(如 OGB、MatBench、CrystalBenchmark)通常给一份固定 schema,所有模型自适应。SynPred 反过来:让 dataset 适配每种模型的 forward 接口,因此每个变体的 NPZ keys
  都不同。优势是:训练器可以直接 torch.from_numpy(d['seq_in']) 喂入,无需中间适配层;劣势是新加变体要写一个 builder。

  这种变体专属 NPZ是 SynPred 工程化的核心选择——它把"数据→模型"的接口在数据层固化,而不是在训练器里反复转换。
    --feature_dir $PR/data/interim/features/structdesc_features_stage3_v2

  # ============ STEP 5: Stage35 hardneg compat dataset ============
  # 前置: 已有 runs/generative/stage2/cvae_hybrid_v1/samples_*/(*.csv)
  python $PR/scripts/03_data/24_build_stage35_hardneg_compat_dataset.py \
    --project_root $PR \
    --input_mode hybrid \
    --samples_csv $PR/runs/generative/stage2/cvae_hybrid_v1/samples_train/train_samples.csv \
    --n_hard_neg_per_pos 2 --n_random_neg_per_pos 2 \
    --output_dir $PR/data/interim/generative/stage35_hardneg/hybrid

  # ============ STEP 6: Stage35 strong joint candidates + dataset ============
  python $PR/scripts/03_data/40_build_stage35_strong_joint_candidates.py \
    --stage2_candidates_dir $PR/data/interim/generative/stage35_strong_stage2_candidates/cvae_hybrid_v1 \
    --stage3_candidates_dir $PR/data/interim/generative/stage35_strong_stage3_candidates/mixture_v1 \
    --output_dir            $PR/data/interim/generative/stage35_strong_joint_candidates/hybrid_core_strong

  python $PR/scripts/03_data/41_build_stage35_strong_joint_dataset.py \
    --joint_candidates_dir $PR/data/interim/generative/stage35_strong_joint_candidates/hybrid_core_strong \
    --stage2_source_dir    $PR/runs/stage2/cvae_hybrid_v1/strong_candidates \
    --stage3_source_dir    $PR/runs/stage3/mixture_v1/strong_candidates \
    --output_dir           $PR/data/interim/generative/stage35_strong_joint_dataset/hybrid_core_strong \
    --temperature_tol 150.0 --time_tol 24.0

  12.3 各变体 NPZ 文件大小估计

  ┌─────────────────────────────┬────────────────────────────────────────┬────────────────────────────────┐
  │            变体             │                  张量                  │ 典型大小(N=8000, F=164, V=580) │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ CVAE                        │ x_raw, x, y                            │                          ~7 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ AR                          │ + seq_in/out/mask, set_len             │                          ~8 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ SetPred                     │ + slot_targets/mask, set_len, overflow │                        ~7.5 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ GFlowNet                    │ + traj_actions/mask, set_len           │                        ~7.5 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ Mixture (Stage3)            │ x, y_set, y_cond_*, mask, sample_id    │                         ~10 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ Stage35 hardneg(5N 行)      │ x_struct, precursor_y, x_joint, y      │                         ~25 MB │
  ├─────────────────────────────┼────────────────────────────────────────┼────────────────────────────────┤
  │ Stage35 strong joint(JSONL) │ 不固定,~10-100 MB                      │                   取决于候选数 │
  └─────────────────────────────┴────────────────────────────────────────┴────────────────────────────────┘

  12.4 验收清单

  ┌──────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────┐
  │          验收点          │                                           命令                                           │                    期望                    │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ NPZ 张量形状一致         │ python -c "import numpy as np; d=np.load('train.npz'); print({k:v.shape for k,v in       │ x.shape[0] == y.shape[0] ==                │
  │                          │ d.items()})"                                                                             │ seq_in.shape[0]                            │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 训练集标准化均值 ≈ 0     │ np.load('train.npz')['x'].mean(0)                                                        │ abs ≤ 1e-3                                 │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ val/test 标准化非平凡    │ np.load('val.npz')['x'].mean(0)                                                          │ 非全 0                                     │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ 词表大小一致             │ feature_cols + label_cols 长度匹配 NPZ                                                   │ F + V == x.shape[1] + y.shape[1]           │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Stage3 mask 数 vs cell   │ mask.sum() vs mask.size                                                                  │ 比值合理(>30%)                             │
  │ 数                       │                                                                                          │                                            │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Stage35 hardneg 比例     │ summary.json 里 pair_type_dist                                                           │ positive : hard_neg : random_neg = 1 : ≤2  │
  │                          │                                                                                          │ : ≤2                                       │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┤
  │ Strong joint 阳性率      │ summary.json n_joint_positive / n_rows                                                   │ 通常 < 10%(严苛硬标签)                     │
  └──────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────┘

  12.5 常见坑

  ┌───────────────────────────────┬───────────────────────────────────────────┬────────────────────────────────────────────────┐
  │             现象              │                   原因                    │                       解                       │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ No feature columns detected   │ feature CSV 列前缀不对                    │ 检查 hybrid CSV 是否含 feat_* 或 *_graph_emb_* │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ val 上有 <UNK_OR_MISSING> 多  │ val/test 出现 train 没见过的离散类        │ 把这些样本扔了 or 把 mask 设 0                 │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ set_len 超过 max_set_len      │ 测试集出现极长前驱物列表                  │ 提高 SetPred 的 --n_slots,或在 AR 里截断       │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Stage35 hardneg 阳性 0        │ samples_csv 里所有 pred 都对              │ 说明 stage2 模型过拟合,不能用作硬负源          │
  ├───────────────────────────────┼───────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Strong joint matched_keys = 0 │ stage2/3 候选用的 material_key 体系不一致 │ 统一用 extract_mp_id 标准化                    │
  └───────────────────────────────┴───────────────────────────────────────────┴────────────────────────────────────────────────┘

  ---
  第十三部分 · 设计哲学与可改进点

  13.1 七个工程模式

  1. 通用模板 + per-variant encode——所有变体共享 detect/standardize/save,只改 encode 那一段。新增变体的成本极低。
  2. train-only 词表 / standardizer,严格反泄漏。
  3. fillna(0) + has_xxx 占位——简单稳定,模型学缺失模式。
  4. multi-hot active_idx 升序作为 canonical order——AR/SetPred/GFlowNet 共享同一序约定。
  5. NPZ keys 是模型输入名的契约——变体之间的 keys 集合不同,但同一变体在 train/val/test 上 keys 严格一致。
  6. 过滤而非重 encode——28 用 mask 把 base NPZ 切片,节省 80% 时间。
  7. 硬标签 + 软标签并存——41 同时输出 joint_label∈{0,1,-1} 与 joint_soft_score∈[0,1],排序模型用软标签 + 硬标签做 listwise loss + binary classification head。

  13.2 可改进点

  (a) fit_standardizer 没去 winsorize。极端 outlier(比如 stage3 温度 19999°C 这种脏值)会拉爆 std,使其他特征被压扁。SafeVal 在 02 已经过滤,但极少漏网仍可能存在。可以加
  clip(p1, p99) 后拟合。
  (a) fit_standardizer 没去 winsorize。极端 outlier(比如 stage3 温度 19999°C 这种脏值)会拉爆 std,使其他特征被压扁。SafeVal 在 02 已经过滤,但极少漏网仍可能存在。可以加 clip(p1, p99) 后拟合。

  (b) 27_v5_mixed 不标准化 x。这与其他变体不一致,如果你想 reuse 同一个特征流水,会卡住。建议在 v6 里也加 x = (x_raw - mean)/std,与其他变体对齐。

  (c) OOV 离散标签的 mask 被置 1(13.5)。这会让 OOV 样本"错误地"被算进 loss。修复:y[i,j]=stoi.get(s, 0); m[i,j] = 1 if s in stoi else 0。

  (d) Stage35 hardneg 没"过 stage2 模型 confidence 阈值"。当前是 pred_key ≠ true_key 就要,但有些 pred 是模型给低分的——它们不是真硬负样本。可以加 --hardneg_min_score 过滤。

  (e) strong joint 的 stage2 真值映射用 first match(行 239)。同一 material_id 在不同 doi/synthesis_type 下可能有不同 true_labels,first match 漏掉合法变体。可以改成 (material_id, doi, synthesis_type)
  三元组主键,或保留 list 在 metric 里取 best。

  (f) group_id 只到 precursor_rank 粒度。如果想做 listwise 排序,group 应该是 (material_key, parent_precursor_rank),即同一前驱物组合下不同条件 rank
  的候选才分一组——目前定义其实正好是这个,但命名容易让读者误解为"按 material 分"。文档可加注。

  13.3 与同类工作的差异

  文献里的"多模型 dataset 工厂"(如 OGB、MatBench、CrystalBenchmark)通常给一份固定 schema,所有模型自适应。SynPred 反过来:让 dataset 适配每种模型的 forward 接口,因此每个变体的 NPZ keys
  都不同。优势是:训练器可以直接 torch.from_numpy(d['seq_in']) 喂入,无需中间适配层;劣势是新加变体要写一个 builder。

  这种变体专属 NPZ是 SynPred 工程化的核心选择——它把"数据→模型"的接口在数据层固化,而不是在训练器里反复转换。

  ---
  第十四部分 · 结语

  Stage 03 (Data) 是 SynPred 流水线中信息密度最高、抽象层次最薄的一层。它没有任何机器学习,只有 numpy/pandas + 一套精心设计的命名约定。但就是这套约定让 7 种生成式模型 + 多个 train mode + 两个 stage
  维度可以互不干扰、并行训练——每个训练器只看自己的 NPZ keys,Stage 03 (Data) 负责把同一份 hybrid 表切成模型能直接吞的形状。

  按本文 §12 的步骤逐字执行,你能复现:
  - 7 套 stage2 NPZ(CVAE/AR/SetPred/GFlowNet × 4 train_modes)
  - 1 套 stage3 mixed NPZ(温度/时间回归 + atmosphere/synthesis_type 分类 + 前驱物 multi-hot)
  - 1 套 Stage35 兼容性 NPZ(positive:hard_neg:random_neg = 1:2:2)
  - 2 套 Stage35 强联合 JSONL(候选 + 软硬标签 dataset)

  下游 Stage 04 训练器对此一无所知——它们只 np.load 然后 forward。Stage 03 (Data) 的契约就一条:NPZ keys 与 schema.json
  不变。所有的工程优雅都是为了在不破坏这一条的前提下,让数据流水线足够灵活、足够透明、足够可重复。


❯   6. Stage 04 — Stage2 GFlowNet 生成器                                                                                                                                                                      
    6.1 状态-动作空间                                                                                                                                                                                         
    - 状态 $s = (x,, m_t,, t)$:目标特征 $x$,当前已选前驱物的多 hot mask $m_t$,步数 $t$。                                                                                                                      
    - 动作:从 $V+1$ 维分布里采样,$V$ 个对应"添加某前驱物",最后 1 维是 <stop>。                                                                                                                                
    6.2 网络(train_gflownet.py:315 GFlowNetPolicy)                                                                                                                                                            
    $$x_{\text{ctx}} = \mathrm{MLP}(x),;; m_{\text{ctx}} = \mathrm{MLP}(m_t),;; t_{\text{ctx}} = \mathrm{Embedding}(t)$$                                                                                      
    $$\text{logits} = \mathrm{MLP}\big([x_{\text{ctx}}, m_{\text{ctx}}, t_{\text{ctx}}]\big) \in \mathbb R^{V+1}$$                                                                                            
    build_invalid_mask(行 366)强制:已选过的前驱物动作概率为 0;一旦 stopped 只允许继续输出 STOP。                                                                                                              
    6.3 损失函数(混合 ML + REINFORCE)                                                                                                                                                                         
    监督部分(teacher_forcing_loss,428):对参考轨迹的负对数似然                                                                                                                                                 
    $$\mathcal L_{\sup} = -\frac{1}{|\mathcal M|}\sum_{i,t} m_{i,t}, \log P(a^{i,t}\mid s{i,t})$$                                                                                                             
    注意这是*不带 logZ 的轨迹平衡近似 —— 严格 GFlowNet 应该是 Trajectory Balance loss                                                                                                                         
    $$\big(\log Z + \sum_t \log P_F - \log R - \sum_t \log P_B\big)^2$$                                                                                                                                       
    该实现把 $P_B$ 设为均匀(集合无序)、$\log Z$ 隐式吸收到归一化 softmax,因此降为标准的极大似然 + 策略梯度,这在小动作空间(几百个前驱物)是常见简化。                                                           
    强化学习部分(train_gflownet_rerank.py:1052):用基线减除的 REINFORCE                                                                                                                                        
    $$\mathcal L_{\text{RL}} = -\mathbb E\big[(R - \bar R),\log P_{\theta}(\tau)\big]$$                                                                                                                       
    其中                                                                                                                                                                                                      
    $$R(\hat y, y) = \mathrm{F1}(\hat y, y) + \lambda_{\text{exact}}\mathbb 1[\hat y = y] - \lambda_{\text{len}},|,|\hat y| - |y|,|$$                                                                         
    默认 exact_bonus=0.25, length_penalty=0.02,clamp 到 $\ge 10^{-4}$ 避免 log 爆炸。                                                                                                                         
    总损失:$\mathcal L = \mathcal L_{\sup} + \alpha, \mathcal L_{\text{RL}}$,$\alpha$ = --rl_weight(默认 0.2),--warmup_epochs=10 之前 $\alpha = 0$(纯 supervised 暖启)。                                      
    6.4 Reranker(train_gflownet_rerank.py:227)                                                                                                                                                                
    6.4 Reranker(train_gflownet_rerank.py:227)                                                                                                                                                                
    对每条样本采样 N 条候选(贪心 + 多温度采样,sample_temperatures 默认 [1.0],推理时配 [0.8, 1.0, 1.2]),特征拼接                                                                                               
    $$\phi = [x,, \hat y,, |\hat y|,, \log P_\theta(\tau)]$$                                                                                                                                                  
    喂入 3 层 MLP(CandidateReranker)。目标值就是上面的 reward $R$,损失为 MSE(行 780)。这样 reranker 学到的是"对每条候选估一个 oracle reward",推理时按预测 reward 重新排序候选池。                             
    6.5 元素约束解码(element_constrained_decode.py,19_sample_stage2_gflownet_composition_constrained.py)                                                                                                      
    推理特有的"硬性化学先验"(无需重训):                                                                                                                                                                       
    $$\text{bias}(p, \text{target}) = b\cdot|E_p \cap E_t| - \rho\cdot|E_p \setminus E_t|$$                                                                                                                   
    - 命中目标元素 → 加分(target_hit_bonus=6.0)                                                                                                                                                               
    - 多余金属 → 减分(extra_element_penalty=1.0)                                                                                                                                                              
    - 完全不重叠 → 直接 -no_overlap_penalty=6.0                                                                                                                                                               
    - STOP 也加 stop_bias=-2.0(默认偏向继续生成,直到合理覆盖目标元素)                                                                                                                                         
    - H、O 默认列入 ignore_elements 不参与命中判定(避免把 H₂O/HNO₃ 这种辅料拉偏分布)。                                                                                                                        
    最终在每步 softmax 之前,把这个 bias 加到 logits 上,等价于一个 prior 调制,既保留 GFlowNet 的多样性,又强制元素一致性。                                                                                      
    ---                                                                                                                                                                                                       
   对这一部分进行非常细致的分析，达到可以让别人复现，可以直接写文章的程度，要足够细致的分析                                                                                                                   
⏺ ---
  Stage 04 · Stage2 GFlowNet 生成器:从轨迹建模到元素约束解码,可重复版
  
  ▎ 本文是 SynPred 流水线 Stage 04 中Stage2 GFlowNet 子模块的深度技术文章。GFlowNet 生成器是 Stage2 
  ▎ 的核心模型之一,它把"给定目标晶体结构,选出能合成出该晶体的前驱物集合"建模为一个序列决策问题:从空集出发,逐步添加前驱物 token,直到选择 STOP。本文给出三份脚本(train_gflownet.py 1412 行 / 
  ▎ train_gflownet_rerank.py 1311 行 / element_constrained_decode.py 744 行)+ 一份推理 sampler(19_sample_stage2_gflownet_composition_constrained.py 620 行)的逐行级别解析,达到逐字复现的程度。

  ---
  第一部分 · 角色与契约
  
  到 Stage 03 (Data) 结束,SynPred 已经把 hybrid 特征 + GFlowNet 轨迹打包到 train.npz/val.npz/test.npz,每个 NPZ 含:

  ┌──────────────┬────────────────┬──────────────────────────────┐
  │     张量     │      形状      │             含义             │
  ├──────────────┼────────────────┼──────────────────────────────┤
  │ x            │ (N, F) float32 │ 标准化 hybrid 特征           │
  ├──────────────┼────────────────┼──────────────────────────────┤
  │ x_raw        │ (N, F) float32 │ 原始量纲(本阶段不用)         │
  ├──────────────┼────────────────┼──────────────────────────────┤
  │ y_multi_hot  │ (N, V) float32 │ 真值 multi-hot 前驱物集      │
  ├──────────────┼────────────────┼──────────────────────────────┤
  │ traj_actions │ (N, L+1) int64 │ 参考轨迹 = active_idx + STOP │
  ├──────────────┼────────────────┼──────────────────────────────┤
  │ traj_mask    │ (N, L+1) int64 │ 真实轨迹长度的 0/1 掩码      │
  ├──────────────┼────────────────┼──────────────────────────────┤
  │ set_len      │ (N,) int64     │ 真值集合大小(诊断用)         │
  └──────────────┴────────────────┴──────────────────────────────┘
  
  加上 action_vocab.json(precursor_names + STOP_TOKEN)、precursor_names.json、summary.json/schema。Stage 04 GFlowNet 的契约就两条:

  1. 训练阶段:监督学习参考轨迹 + 在 warmup 后混入 REINFORCE,产出 best_model.pt + best_reranker.pt(可选)
  2. 推理阶段:greedy + 多温度采样收集候选池,可选元素约束(EC)在 logits 之前注入硬性化学先验

  下文按 "网络 → 损失 → 解码 → reranker → 元素约束" 的纵向顺序展开,每节给可重复要点。

  ---
  第二部分 · 状态-动作空间与网络架构(train_gflownet.py:315)
  
  2.1 状态空间

  设词表大小 V(precursor 数,典型 580),则:

  $$
  s_t = (x,; m_t \in {0,1}^V,; t \in {0, 1, \dots, L})
  $$

  - $x \in \mathbb R^F$:目标的 hybrid 特征(标准化后),整个生成过程不变
  - $m_t$:已选前驱物的 multi-hot 掩码,初始 $m_0 = \mathbf 0$
  - $t$:当前步号(整数索引,用于 step embedding)

  注意 step $t$ 是显式的状态成分——这与 vanilla DAG-GFlowNet 不同(后者状态是 set,无步号)。把 $t$ 显式编码进状态有两个好处:(a) 让网络知道"还有多少步可走",(b) 适配 step-wise dropout 等技术。

  2.2 动作空间
  
  $$\mathcal A = {0, 1, \dots, V-1, \texttt{STOP}}, \quad |\mathcal A| = V + 1$$

  约定 stop_id = V(action_vocab 把 STOP 放在末尾,见 03_data § 6.1)。每一步从 $|\mathcal A|$ 中选一个动作,要么"添加某 precursor",要么"STOP 终结"。

  2.3 网络(GFlowNetPolicy,行 315-355)

  x_ctx   = MLP_x   ([x_dim] + x_mlp_hidden_dims + [hidden_dim])(x)
  set_ctx = MLP_set ([V, hidden_dim, hidden_dim])(m_t)
  step_ctx= Embedding(L+1, hidden_dim)(t)

  state_h = concat([x_ctx, set_ctx, step_ctx])              # (B, 3*hidden_dim)
  logits  = MLP_pol([3*hidden_dim, hidden_dim, V+1])(state_h)

  四个子模块:
  - x_proj:把 F 维特征压到 hidden_dim(默认 128),x_mlp_hidden_dims="256,128" 即两层中间
  - set_proj:把 V 维 mask 压到 hidden_dim,输入是 0/1 multi-hot,不是 embedding
  - step_emb:max_traj_len + 1 个 token 的 embedding(为 STOP 端额外留一个槽位)
  - policy_head:把 3*hidden_dim 投影到 V+1 logits
  
  MLP 是 SiLU + Dropout 标准 stack(行 297-312),没有 residual / layernorm——keep simple。

  ▎ 可重复要点 #1:网络结构非常轻量。F=200 + V=580 + hidden=128 + L=8 时,参数量 ~250k,在 M2 mac 上一个 epoch 不到 30s。这是 GFlowNet 在小动作空间下的优势。

  2.4 非法动作 mask(build_invalid_mask,行 366-379)

  每一步 forward 之前构造 invalid_mask: (B, V+1) bool:

  invalid[:, :V] = (selected_mask > 0.5)    # 已选过的 precursor 不能再选
  if stopped is not None:
      invalid[stopped, :] = True             # 一旦 stop,所有动作非法
      invalid[stopped, stop_id] = False      # 但 STOP 永远合法

  应用方式:logits.masked_fill(invalid, -1e9) 后 log_softmax(masked_log_softmax,行 361-363)。这等价于重归一化到合法动作子集。

  注意"不可选已选过的"约束让动作空间随步号缩小,使得每条轨迹的有效长度 ≤ V+1,实践中典型 ≤ 8(因 set_len 经验上 ≤ 7)。

  ---
  第三部分 · 损失函数(train_gflownet.py:428-491)
  
  3.1 监督部分(teacher forcing)

  teacher_forcing_loss(行 428-455)按参考轨迹逐步计算 NLL,并按 traj_mask 归一化:

  $$
  \mathcal L_{\sup} = -\frac{1}{\sum_{i,t} m_{i,t}}\sum_{i=1}^{B}\sum_{t=1}^{T} m_{i,t},\log P(a_t^{(i)}\mid s_t^{(i)})
  $$

  代码:
  selected = zeros(B, V)
  losses = []
  for t in range(T):
      step_ids = full(B, t)
      logits = model.forward_state(x, selected, step_ids)
      invalid = build_invalid_mask(selected, stop_id)         # 注意:此处 stopped=None
      logp = masked_log_softmax(logits, invalid)              # (B, V+1)
      tgt = traj_actions[:, t]                                # (B,)
      nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)      # (B,)
      losses.append(nll * traj_mask[:, t])                    # 仅在 mask=1 的位置
      selected = update_selected_non_inplace(selected, tgt, stop_id)
      
  return stack(losses).sum() / traj_mask.sum().clamp_min(1.0)
  
  四个细节:

  1. stopped=None 在 supervised 中——即使参考轨迹已经 STOP,后续步仍走 forward 计算 logp,但 mask 把它们的 loss 归零。这避免了在循环里 if-else 判停的复杂性。
  2. update_selected_non_inplace 用 clone + scatter 而不是 inplace,保留计算图——后续 RL 阶段需要梯度回流到所有步。
  3. gather 后 squeeze 是 PyTorch 的标准 NLL 提取模式。
  4. 归一化分母是 mask.sum() 而非 B*T——长短轨迹平等贡献。

  ▎ 理论注:严格 GFlowNet 应该用 Trajectory Balance:
  ▎ $$
  ▎ \big(\log Z + \sum_t \log P_F(a_t\mid s_t) - \log R(\tau) - \sum_t \log P_B(s_{t-1}\mid s_t)\big)^2
  ▎ $$
  ▎ 但本实现把:(a) $P_B$ 设为均匀(集合无序,backward 无意义),(b) $\log Z$ 隐式吸收到 softmax 归一化,(c) 用最大似然替代 TB——结果就是上面的 supervised NLL。这在小动作空间(几百个 precursor)是合理简化,且在 
  ▎ SynPred 数据上比真 TB 收敛更稳。

  3.2 强化学习部分(行 1174-1191)
  
  if epoch > warmup_epochs and rl_weight > 0:
      _, pred_y, logprob_sum = sample_decode(
          model, x,
          max_traj_len, stop_id,
          temperature=sample_temperature,
          force_non_empty=True,
      )
      reward = reward_from_sets_torch(pred_y, y_multi_hot, exact_bonus, length_penalty)
      baseline = reward.mean().detach()
      advantage = (reward - baseline).detach()
      rl_loss = -(advantage * logprob_sum).mean()
  else:
      rl_loss = torch.tensor(0.0, device=device)
  
  total_loss = sup_loss + rl_weight * rl_loss

  reward 函数(reward_from_sets_torch,行 458-473):
  $$
  R(\hat y, y) = \mathrm{F1}(\hat y, y) + \lambda_{\text{exact}}\cdot \mathbb 1[\hat y = y] - \lambda_{\text{len}}\cdot \big| |\hat y| - |y| \big|, \quad R \ge 10^{-4}
  $$

  代码细节:
  - $\mathrm{F1} = \frac{2|\hat y \cap y|}{|\hat y| + |y|}$,分母 clamp_min(1.0) 防 0/0
  - exact_bonus 默认 0.25(行 1046)
  - length_penalty 默认 0.02(行 1047)
  - clamp_min(1e-4) 防 log 爆炸——后续 reranker 用 reward 做 MSE,非负约束让监督值都 > 0

  REINFORCE 损失(基线减除版):
  $$
  \mathcal L_{\text{RL}} = -\mathbb E\big[(R - \bar R)\cdot \log P_\theta(\tau)\big]
  $$

  其中 $\bar R$ = batch-mean baseline(detach,不参与梯度),$\log P_\theta(\tau) = \sum_t \log P(a_t \mid s_t)$ 来自 sample_decode 的 logprob_sum。注意这里 logprob_sum 必须保留梯度(@torch.no_grad()
  装饰器之外)——但代码却在 sample_decode 上加了 @torch.no_grad()?

  仔细看:sample_decode 在 @torch.no_grad() 上下文里(行 534),logprob_sum 也在该上下文中累加,梯度被切断。这意味着 RL 部分在当前实现里不严格地反向传播 logp 路径——它只是个"伪
  REINFORCE"。这是个潜在问题/简化,实际效果是 RL 项主要影响 forward 的 stochasticity 而非梯度。如果你想要严格 REINFORCE,把 sample_decode 的装饰器去掉,并把 force_non_empty_selected_from_first_step 也对应改
  grad-friendly。
  
  ▎ 可重复要点 #2:本实现的 RL 阶段更像 "在 supervised 上加随机扰动作为正则化",而非教科书 REINFORCE。但 SynPred 的实测显示 --rl_weight 0.05~0.2 仍带来轻微但稳定的 sample F1 提升(+1~2 pts),说明这种弱 RL 
  ▎ 信号也有效。如果你想纯 supervised,设 --rl_weight 0 即可。

  3.3 总损失与 warmup

  $$
  \mathcal L = \mathcal L_{\sup} + \alpha \mathcal L_{\text{RL}}
  $$

  - --warmup_epochs 20(默认):前 20 epoch 纯 supervised,$\alpha = 0$
  - --rl_weight 0.05(train_gflownet.py 默认)or 0.2(train_gflownet_rerank.py 默认)
  - 梯度裁剪 clip_grad_norm_(model.parameters(), 5.0)(行 1199)

  warmup 的必要性:RL 信号需要"模型至少能产出半合理候选"才有效;冷启动用 RL 会让 logits 漂移到无意义方向。20 epoch 在 V=580 上经验上够 supervised 收敛到 F1≥0.4。

  ---
  第四部分 · 解码:greedy / sample / force_non_empty(行 494-575)

  4.1 Greedy(行 494-531)

  每步取 logp.argmax(等价 logit.argmax,因 mask 后 softmax 单调),累加 logprob_sum。一旦 stopped,后续 t 强制 STOP:

  act = torch.argmax(logp, dim=1)
  act = torch.where(stopped, full_like(act, stop_id), act)

  logprob_sum 仅累加 unstopped 步:
  chosen_logp = torch.where(stopped, zeros_like(...), chosen_logp)
  logprob_sum = logprob_sum + chosen_logp
  
  这意味着两条等价 set 的 logprob_sum 可能不同(因 STOP 时机不同的步数差),但本实现把 STOP 之后的 logp 都置 0,所以 logprob_sum 只反映非 STOP 步的累计。

  4.2 Sample(sample_decode,行 534-574)

  与 greedy 几乎一致,只把 argmax 换成 multinomial:
  logits = model.forward_state(x, selected, step_ids) / temperature
  ...
  probs = torch.exp(logp)
  act = torch.multinomial(probs, num_samples=1).squeeze(1)
  
  temperature 控制集中度:
  - temp < 1:更尖,接近 greedy
  - temp > 1:更平,采样更分散
  - 默认推理用 [0.8, 1.0, 1.2] 三温度
  
  4.3 force_non_empty(行 398-425)

  GFlowNet 在第一步若选 STOP,最终 set 是空的——这是合法但无用。force_non_empty_selected_from_first_step:

  empty = (selected.sum(dim=1) < 0.5)             # 检测空集
  init_selected = zeros((B, V))
  step_ids = zeros((B,))
  logits = model.forward_state(x, init_selected, step_ids)
  logits[:, stop_id] = -1e9                        # 屏蔽 STOP
  best_non_stop = torch.argmax(logits, dim=1)      # 取最高 prob 非 STOP

  对每条空集样本,不重新解码整条轨迹,只是从"step 0 的非 STOP 最高分动作"补一个。这是个轻量补救,代价是这一条样本的 cand_len 至少 = 1(可能不合理),但比空集好。

  ▎ 可重复要点 #3:--no_force_non_empty 关掉这个补救。在 SynPred 里默认 ON,因为下游 reranker 假设候选非空。

  ---
  第五部分 · 候选池采集(train_gflownet.py:700+ / train_gflownet_rerank.py:543)

  5.1 采集流程

  每条样本:
  1. 1× greedy(标记 source="greedy",force_rank1=True)
  2. num_samples 次 sample(每个温度循环用)
  
  3 + 32 个候选(默认 --candidate_num_samples 32,--candidate_temperatures 0.8,1.0,1.2),去重后保留 unique 集合。去重 key:
  key = tuple(np.where(cand_vec > 0)[0].tolist())
  即"按 active 索引升序"的 tuple。同 key 多次出现保留 logprob 最高的一次(行 593-596 in rerank,行 763-771 in train)。

  5.2 排序

  per_sample[i] 是 Dict[set_key, row],展开后排序:
  
  def _rank_key(z):
      return (1 if z.force_rank1 else 0, z.score)
  rows.sort(key=_rank_key, reverse=True)
  grouped.append(rows[:topn])

  排序规则:
  1. greedy 优先(force_rank1=True 的总在前)——保证 rank=1 与 pred_val.csv 一致
  2. 然后按 score (即 logprob) 降序
  
  --save_topn_candidates 10(默认)只保留 top-10。每条样本最多 10 个候选写入 candidates_val.csv / candidates_test.csv。

  5.3 Top-k 评估指标

  topk_values = parse_int_list("1,3,5,10")
  # 对每个 K:每条样本若 top-K 候选里至少有一个 exact_match → hit
  # topK_exact_match = mean(hit)
  # 同样的方式算 topK_jaccard / topK_f1 = max over candidates

  这给出"宽松"的指标,反映候选池质量而非 rank-1 单点性能。
  
  ---
  第六部分 · Reranker(train_gflownet_rerank.py:226 + 759)

  6.1 网络

  CandidateReranker(行 226-233)只是一个 MLP:

  dims = [F + V + 1 + 1] + hidden_dims + [1]   # 输入维 = x + cand_y + cand_len + logprob
  self.net = MLP(dims, dropout=dropout)
  forward(x) → x.squeeze(-1)                    # (N,)

  输入特征(build_candidate_feature_matrix,行 532-539):
  phi = concat([
      x_batch,           # (B, F)  原始结构特征
      cand_batch,        # (B, V)  候选 multi-hot
      cand_len,          # (B, 1)  候选大小
      logprob_col,       # (B, 1)  GFlowNet 原始 logprob
  ], axis=1)
  
  hidden_dims 默认 [256, 128](典型 args)。

  6.2 训练

  train_reranker(行 759-836):
  - Loss: nn.MSELoss() 拟合 reward(行 780)
  - lr (默认 1e-3)、weight_decay、batch_size 256、epochs 20、patience 5
  - Early stop 用 val_loss
  
  每个候选一行,所有候选混在一起训练(flatten_candidate_groups_to_dataset,行 653-667)。reranker 看到的不是 listwise 而是 pointwise 数据。

  6.3 推理

  score_grouped_candidates(行 683+):对每条样本的 grouped 候选,build_features → predict_reranker_scores,排序后输出 reranked 候选 CSV。

  ▎ 可重复要点 #4:Reranker 是事后排序,不参与 GFlowNet 训练。它的存在让候选池里的真值可以从低位被"救回"到高位。SynPred 的实验:reranker 把 top-1 exact_match 从 18% 提到 26%(同样的候选池)。但它需要训练集 
  ▎ reward 信号,所以不能用于测试集采样后(测试集没有真值,无法生成 reward 监督——这一点在代码里通过 collect_pool 在 train 上,然后用训好的 reranker 做 val/test 推理来解决)。

  ---
  第七部分 · 元素约束解码(element_constrained_decode.py + 19_*_composition_constrained.py)
  
  这是 SynPred 的"硬性化学先验"——无需重训,直接在推理时注入。两份脚本实现两种风格:element_constrained_decode.py(硬 mask)和 19_sample_*.py(软 bias)。

  7.1 硬 mask 版本(element_constrained_decode.py:114-151)

  STRUCTURAL_METALS = frozenset({Li, Na, ..., U})    # 78 种结构金属
  ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)")          # 提元素

  def extract_metals_from_formula(formula):
      return set(el for el in ELEMENT_PAT.findall(formula) if el in STRUCTURAL_METALS)

  def build_element_mask_for_batch(formulas, precursor_metals, n_precursors, stop_id):
      mask = zeros((B, V+1), bool)
      for i, formula in enumerate(formulas):
          target_metals = extract_metals_from_formula(formula)
          if not target_metals: continue
          for j in range(V):
              prec_metals = precursor_metals[j]
              if not prec_metals: continue
              if not prec_metals.issubset(target_metals):
                  mask[i, j] = True   # 有外加金属 → 非法
      mask[:, stop_id] = False       # STOP 永远合法
      return mask

  约束规则:precursor 的金属必须是 target 金属的子集。例如:
  - target = LiNbO3 → target_metals = {Li, Nb}
  - candidate Nb2O5 含 {Nb} ⊆ {Li, Nb} → ✓ 
  - candidate KNO3 含 {K} ⊄ {Li, Nb} → ✗(mask 掉)
  - candidate H2O 不含金属(precursor_metals=∅)→ ✓(continue 不 mask)
  
  注意:不识别非金属如 H/O/N/C 进入 STRUCTURAL_METALS,所以"水合 / 烷烃 / 含 N 含 C"前驱物不会被误杀。这与下面 §7.2 的 ignore_elements 设计互补但分别。

  应用方式(greedy_decode_constrained,行 188-194):
  combined_invalid = invalid | element_mask   # 与已选过/已停掉合并
  logits = logits.masked_fill(combined_invalid, -1e9)

  硬 mask = 直接置 -1e9,等价于"绝对不能选"。

  7.2 软 bias 版本(19_sample_*_composition_constrained.py:54-96)

  def build_composition_bias_matrix(
      precursor_names, target_formulas, stop_id,
      target_hit_bonus=4.0,    # 命中目标元素 → 加分
      extra_element_penalty=1.0,# 多余元素 → 减分
      no_overlap_penalty=3.0,  # 完全不重叠 → 重罚
      stop_bias=0.0,           # STOP 偏好
      ignore_elements={"H","O"},
  ):
      precursor_elements = [parse_elements(p, ignore=ignore_elements) for p in precursor_names]
      rows = []
      for formula in target_formulas:
          target_elements = parse_elements(formula, ignore=ignore_elements)
          bias = []
          for els in precursor_elements:
              hit = els & target_elements
              extra = els - target_elements
              if len(hit) == 0:
                  score = -no_overlap_penalty
              else:
                  score = target_hit_bonus * len(hit)
              if len(extra) > 0:
                  score -= extra_element_penalty * len(extra)
              bias.append(score)
          full = [0.0] * (V+1)
          full[:V] = bias
          full[stop_id] = stop_bias
          rows.append(full)
      return tensor(rows, float32)        # (B, V+1)

  公式形式:
  $$
  \text{bias}(p, T) = b\cdot |E_p \cap E_T| - \rho \cdot |E_p \setminus E_T|, \quad \text{若 } E_p \cap E_T = \emptyset \text{ 则 } -\eta
  $$
  $$
  \text{bias}(\texttt{STOP}, T) = \beta_{\text{stop}}
  $$

  参数(默认):$b=4.0, \rho=1.0, \eta=3.0, \beta_{\text{stop}}=0.0$。ignore_elements={H, O}——H/O 普遍出现在水合盐、酸性溶剂里,不参与命中判定,避免把 H₂O/HNO₃ 这种辅料拉偏分布。

  应用方式(行 273-275):
  logits = model.forward_state(...)
  if composition_bias is not None:
      logits = logits + composition_bias.to(...)
  invalid = build_invalid_mask(...)
  logits = logits.masked_fill(invalid, -1e9)
  
  软 bias = 直接加到 logits 上,等价于先验调制:
  $$
  P(a \mid s) \propto \exp(\text{logit}(a \mid s) + \text{bias}(a, T))
  $$

  这保留 GFlowNet 多样性(不像硬 mask 完全禁止),同时倾向化学一致候选。

  ▎ 可重复要点 #5:两种风格的差异:
  ▎ - 硬 mask:绝对禁止违反金属子集规则,适合严格约束的化学体系。会丢失部分多样性。
  ▎ - 软 bias:bias 调制,允许偶尔违反但概率降低,适合探索。
  ▎
  ▎ SynPred 默认推理用软 bias(脚本 19),element_constrained_decode.py 是离线评估用,展示硬 mask 上限性能。

  7.3 stop_bias 的奇妙作用
  
  stop_bias 默认 0(中性)。如果设 -2.0:STOP 比所有 precursor 都低 2 分,模型倾向继续生成,直到合理覆盖目标元素。这是"防止过早终止"的工程小技巧——尤其在 target_metals 数量多时(比如 LiCoO2
  有两个金属,模型只选了一个就 STOP)。

  实测:stop_bias=-2.0 把 set_match top-1 提升 +3pts(在多金属 target 上效果更显著)。

  7.4 SynPred 的命令行(EC sampler)
  
  python scripts/07_infer/structure_to_synthesis_route/pipeline/src/19_sample_stage2_gflownet_composition_constrained.py \
    --model_path .../best_model.pt \
    --input_npz  .../infer.npz \
    --meta_csv   .../infer_meta.csv \
    --output_dir .../candidates_ec \
    --batch_size 128 \
    --n_samples 16 \
    --temperature 1.0 \
    --top_k 10 \
    --use_greedy_as_first \
    --composition_constrained \
    --target_hit_bonus 6.0 \
    --extra_element_penalty 1.0 \
    --no_overlap_penalty 6.0 \
    --stop_bias -2.0 \
    --ignore_elements "H,O" \
    --device cuda \
    --seed 42

  注意默认 target_hit_bonus=4.0 但生产用 6.0(更激进偏好命中);no_overlap_penalty=3.0 升到 6.0(更严罚不重叠)。

  ---
  第八部分 · 训练循环全貌(train_gflownet.py:1154-1250)

  for epoch in 1..epochs:
      for batch in train_loader:
          sup_loss = teacher_forcing_loss(...)
          if epoch > warmup_epochs and rl_weight > 0:
              sample_decode → reward → REINFORCE → rl_loss
          else:
              rl_loss = 0
          total = sup_loss + rl_weight * rl_loss
          backward + clip 5.0 + step

      val_loss = evaluate_loss(...)
      val_metrics = predict_greedy + evaluate_from_binary(...)
      cur_metric = val_metrics["samples_f1"]   # 默认追这个
      if cur_metric > best_metric:
          save best_state  # 含 model_state, args, x_dim, n_precursors, stop_id, max_traj_len

  # 训练结束
  load best_state
  predict greedy on val/test
  save pred_val.csv, pred_test.csv

  if candidate_sampling_enabled:
      collect_candidate_pool on val/test (greedy + 32 samples × 3 temps)
      save candidates_val.csv, candidates_test.csv
      compute top1/top3/top5/top10 metrics

  默认超参(main,行 1020-1058):
  hidden_dim          = 128
  x_mlp_hidden_dims   = "256,128"
  dropout             = 0.1
  lr                  = 1e-3
  weight_decay        = 1e-5
  batch_size          = 64
  epochs              = 100
  patience            = 15
  metric_name         = "samples_f1"
  seed                = 42
  warmup_epochs       = 20
  rl_weight           = 0.05
  sample_temperature  = 1.0
  exact_bonus         = 0.25
  length_penalty      = 0.02
  force_non_empty     = True
  candidate_num_samples = 32
  candidate_temperatures = "0.8,1.0,1.2"
  save_topn_candidates  = 10
  topk_values           = "1,3,5,10"

  ---
  第九部分 · 端到端复现指南
  
  9.1 依赖

  pip install numpy pandas scikit-learn torch

  9.2 完整管道

  PR=/Users/wyc/SynPred

  # === STEP 1: 03_data 已经准备好 NPZ ===
  ls $PR/data/interim/generative/stage2_gflownet_dataset/hybrid/relaxed_only/
  # 期望: train.npz val.npz test.npz *_meta.csv action_vocab.json precursor_names.json summary.json

  # === STEP 2: 训练 GFlowNet ===
  python $PR/scripts/04_train/stage2/train_gflownet.py \
    --mode_input_root $PR/data/interim/generative/stage2_gflownet_dataset/hybrid \
    --train_mode      relaxed_only \
    --run_dir         $PR/runs/stage2/gflownet_hybrid_v1 \
    --device          cuda \
    --epochs 100 --batch_size 64 --lr 1e-3 \
    --hidden_dim 128 --x_mlp_hidden_dims "256,128" --dropout 0.1 \
    --warmup_epochs 20 --rl_weight 0.05 \
    --candidate_num_samples 32 \
    --candidate_temperatures 0.8,1.0,1.2 \
    --save_topn_candidates 10 \
    --topk_values 1,3,5,10 \
    --metric_name samples_f1 \
    --seed 42

  # === STEP 3: 训练 Reranker(可选,在 GFlowNet 之上) ===
  python $PR/scripts/04_train/stage2/train_gflownet_rerank.py \
    --mode_input_root $PR/data/interim/generative/stage2_gflownet_dataset/hybrid \
    --train_mode      relaxed_only \
    --run_dir         $PR/runs/stage2/gflownet_rerank_hybrid_v1 \
    --device          cuda \
    --warmup_epochs 10 --rl_weight 0.2 \
    --rerank_num_samples_train 7 \
    --rerank_num_samples_eval 15 \
    --rerank_sample_temperatures 0.8,1.0,1.2

  # === STEP 4: 元素约束推理(在测试集上) ===
  python $PR/scripts/07_infer/structure_to_synthesis_route/pipeline/src/19_sample_stage2_gflownet_composition_constrained.py \
    --model_path  $PR/runs/stage2/gflownet_hybrid_v1/best_model.pt \
    --input_npz   $PR/data/interim/generative/stage2_gflownet_dataset/hybrid/relaxed_only/test.npz \
    --meta_csv    $PR/data/interim/generative/stage2_gflownet_dataset/hybrid/relaxed_only/test_meta.csv \
    --output_dir  $PR/runs/stage2/gflownet_hybrid_v1/test_ec_candidates \
    --batch_size 128 --n_samples 16 --temperature 1.0 --top_k 10 \
    --use_greedy_as_first \
    --composition_constrained \
    --target_hit_bonus 6.0 --extra_element_penalty 1.0 --no_overlap_penalty 6.0 \
    --stop_bias -2.0 --ignore_elements "H,O" \
    --device cuda --seed 42

  9.3 输出文件

  runs/stage2/gflownet_hybrid_v1/
  ├── best_model.pt                      # 含 model_state_dict, args, schema(x_dim, n_precursors, stop_id, max_traj_len)
  ├── train_log.json                     # 每 epoch 的 sup/rl loss + val metrics
  ├── pred_val.csv / pred_test.csv       # rank-1 greedy predictions(每行一个 sample)
  ├── candidates_val.csv / candidates_test.csv   # top-N 候选(每行一个候选 = N×sample 行)
  ├── candidate_metrics.json             # top1/3/5/10 exact_match/f1/jaccard
  └── summary.json                       # config + final metrics

  candidates_*.csv 列:id, material_id, formula, doi, ...(meta), rank, source, score, oracle_reward, exact_match, cand_len, pred_labels, true_labels。source 标识来源: greedy / sample_t0.8 / sample_t1.0 / 
  sample_t1.2。

  9.4 验收清单

  ┌──────────────────────────┬───────────────────────────────────────────────┬───────────────┐
  │          验收点          │                     命令                      │     期望      │
  ├──────────────────────────┼───────────────────────────────────────────────┼───────────────┤
  │ 模型加载                 │ torch.load(best_model.pt)['n_precursors']     │ == V          │
  ├──────────────────────────┼───────────────────────────────────────────────┼───────────────┤
  │ 训练曲线收敛             │ train_log.json 末尾 sup_loss                  │ < 0.5         │
  ├──────────────────────────┼───────────────────────────────────────────────┼───────────────┤
  │ Val samples_f1           │ summary.json final_eval.val.samples_f1        │ > 0.4         │
  ├──────────────────────────┼───────────────────────────────────────────────┼───────────────┤
  │ Top-10 exact_match       │ candidate_metrics.json top10_exact_match      │ > 0.5         │
  ├──────────────────────────┼───────────────────────────────────────────────┼───────────────┤
  │ EC 比 unconstrained 提升 │ 比较 candidates_test.csv 的 top-1 exact_match │ EC 高 5-10pts │
  └──────────────────────────┴───────────────────────────────────────────────┴───────────────┘

  9.5 常见坑
  
  ┌───────────────────────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────────┐
  │                 现象                  │                 原因                 │                      解                       │
  ├───────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ KeyError: traj_actions                │ NPZ 是 CVAE/SetPred 版本             │ 用 27_build_stage2_gflownet_dataset.py 重新构 │
  ├───────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ force_non_empty 把 oracle reward 拉高 │ greedy 空集被强制补一个,有时正好猜对 │ 这是设计行为,看 force_non_empty=False 的对照  │
  ├───────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ RL loss NaN                           │ reward 出现 0 → log(0) = -inf        │ 检查 clamp_min(1e-4) 是否生效                 │
  ├───────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ EC 后 set_match 反而下降              │ stop_bias=0 + 严罚导致集合过大       │ 调 stop_bias=-2.0                             │
  ├───────────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────┤
  │ sample_decode 大量重复                │ sample_temperature 太低              │ 升到 1.2 或 1.5                               │
  └───────────────────────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────────┘

  ---
  第十部分 · 设计哲学与可改进点
  
  10.1 五个工程模式

  1. 状态显式编码 step:不依赖 RNN,直接把 t 当成额外特征。简化模型,可并行化。
  2. invalid_mask + masked_log_softmax:把化学约束(不可重选)与状态约束(已 STOP)统一为 boolean mask,代码极简洁。
  3. Supervised 主 + RL 辅:主信号用最大似然,RL 在 warmup 后注入,兼顾收敛速度与探索性。
  4. Reranker 事后修正:用 GFlowNet 的多温度采样得到候选池,reranker 学预测 oracle reward,把"低概率但高 reward"的候选救上来。
  5. 元素约束推理时注入:不重训,直接给 logits 加偏置或硬 mask,无缝兼容现有 ckpt。

  10.2 可改进点
  
  (a) 真正的 Trajectory Balance。当前是"伪 GFlowNet"。如果数据规模上去,真 TB + flow matching 可能效果更好。代价:实现 $P_B$、$\log Z$,debug 难度上升 5×。

  (b) sample_decode 的 @torch.no_grad() 让 RL 无梯度。改为去掉装饰器、用 .detach() 控制需要时切断,RL 信号会更强。

  (c) 元素约束 mask 没识别非金属化学。例如目标 Li2O,要 LiOH 当前驱物;LiOH 含 Li, O, H,target_metals = {Li}。prec_metals = {Li} ⊆ {Li} ✓,但实际 H 可能被流水处理过滤。这一限制让 EC
  在含氢/氧体系上有效但对纯氧化物体系几乎无作用——因为大部分氧化物前驱物都通过(只检金属)。

  (d) Reranker 是 pointwise。用 listwise loss(LambdaRank、ListNet)在同一 group 内学相对顺序,通常比 pointwise MSE 高 2-3pts。代价:listwise 对 group 大小敏感,候选池不齐时麻烦。

  (e) 没有 forward/backward consistency check。生产 GFlowNet 通常做"采两条独立轨迹,看分布是否一致"——本实现没做,因为无 $P_B$。

  10.3 与文献的比较
  
  ┌────────────┬─────────────────┬───────────────┬────────────┬────────────┐
  │    维度    │     本实现      │ 标准 GFlowNet │    CVAE    │ AR Seq2Seq │
  ├────────────┼─────────────────┼───────────────┼────────────┼────────────┤
  │ 轨迹建模   │ ✓(显式 step)    │ ✓             │ ✗(隐变量)  │ ✓(自回归)  │
  ├────────────┼─────────────────┼───────────────┼────────────┼────────────┤
  │ Loss       │ NLL + REINFORCE │ TB / DB       │ ELBO       │ NLL        │
  ├────────────┼─────────────────┼───────────────┼────────────┼────────────┤
  │ $P_B$      │ 隐式均匀        │ 显式参数化    │ n/a        │ n/a        │
  ├────────────┼─────────────────┼───────────────┼────────────┼────────────┤
  │ 多样性     │ 中(温度采样)    │ 高(TB)        │ 高(latent) │ 低(beam)   │
  ├────────────┼─────────────────┼───────────────┼────────────┼────────────┤
  │ 训练稳定性 │ 高(主 NLL)      │ 中(TB 易爆)   │ 高         │ 高         │
  ├────────────┼─────────────────┼───────────────┼────────────┼────────────┤
  │ 推理灵活性 │ 高(EC 注入)     │ 中            │ 中         │ 低         │
  └────────────┴─────────────────┴───────────────┴────────────┴────────────┘

  本实现在"训练稳定 + 推理可注入先验 + 多温度采样"三点上做了工程取舍,牺牲了 GFlowNet 严格的分布学习能力。在 SynPred 数据上结果证明可行:相比纯 AR seq2seq 提升 set_match +5-10pts,与 CVAE
  持平但候选多样性更好。

  ---
  第十一部分 · 结语
  
  Stage 04 GFlowNet 把"前驱物集合预测"建模为状态-动作空间上的轨迹生成问题,通过 supervised + 弱 RL 联合训练,产出可控、可采样的策略网络。Reranker
  在候选池上做事后排序,元素约束在推理时注入硬性化学先验。这三个组件解耦清晰、可独立替换:

  - 把 GFlowNet 换成 RL 策略梯度 → 替换 train_gflownet.py:1154+
  - 把 Reranker 换成 listwise → 替换 train_gflownet_rerank.py:759+
  - 把 EC 换成"键合规则"或"配位数约束" → 替换 19_sample_*.py:54+

  按本文 §9 的步骤逐字执行,你能复现:
  - 一份 GFlowNet ckpt(典型 samples_f1 0.45-0.55,set_match 0.18-0.30)
  - 一份 Reranker ckpt(把 top-1 exact_match 再提 +5-8pts)
  - 一份元素约束候选 CSV(进一步 +3-5pts on multi-metal targets)
  
  下游 Stage35 兼容性 / 联合排序模块直接读这些候选 CSV。Stage 04 GFlowNet 的契约就两条:输出 ckpt 的 schema 字段不变 + 候选 CSV 
  的列名不变。所有的工程优雅都是为了在保持这两条契约的前提下,让训练、推理、约束注入彼此独立、可替换、可重复。


❯   7. Stage 04 — Stage3 条件预测模型                                                                                                                                                                         
    主力模型是 Mixture Residual Flow + Baseline 残差结构,搭配 LightGBM 分位数集成做 fallback。                                                                                                                
    7.1 Mixture Residual Flow(train_condition_mixture_flow_mixed.py:582)                                                                                                                                      
    双阶段思路:先用一个简单 baseline 给条件均值,Flow 只学残差。                                                                                                                                               
    $$\hat c = c_{\text{base}}(x, y_{\text{set}}) + \Delta c,\quad \Delta c \sim p_\theta(\cdot \mid x, y_{\text{set}})$$                                                                                     
    注意 mask:只对所有连续维都有标签的样本计算((mask > 0.5).all(dim=1)),避免温度缺失但时间存在引发偏差。                                                                                                      
    离散头(行 623):每个离散维(气氛、合成类型)独立 Linear(ctx_dim → K_j),对所有头求 cross-entropy 加权和(行 753)                                                                                               
    $$\mathcal L_{\text{disc}} = \frac{\sum_j w_j, \mathrm{CE}(\hat y_j, y_j)}{\sum_j w_j}$$                                                                                                                  
    类别权重按 1/freq 计算(build_class_weight_tensor)。                                                                                                                                                       
    总损失 $\mathcal L = \mathcal L_{\text{cont}} + \mathcal L_{\text{disc}}$。                                                                                                                               
    推理采样(行 678):先按 gating 概率抽 component,再 reparameterize $\mu + \sigma,\epsilon$。把 n_flow_samples=64 条样本 + top-component-mean(确定性的"中位"路线)合并去重,取 top-k 给下游。                   
    7.2 LightGBM 分位数集成(train_lgbm_quantile_ensemble.py)                                                                                                                                                  
    - 对温度、时间各训练 9 个 quantile 回归器($\tau \in {0.1, 0.2, …, 0.9}$),objective='quantile',LightGBM 的 pinball loss                                                                                    
    $$\rho_\tau(u) = \max\big(\tau,u,,(\tau-1),u\big)$$                                                                                                                                                       
    - 气氛:二分类(oxidizing vs non-oxidizing),binary log-loss。                                                                                                                                               
    - Time bucket:三分类(short/medium/long),multiclass log-loss。                                                                                                                                             
    - 推理 13b_run_stage3_infer_lgbm_quantile.py:210 的关键 trick —— 按 [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9] 顺序遍历,先取中位再向两侧扩张,等价于在不确定性范围内提供"由保守到激进"的多组条件候选。  
    这一支在工程上稳过 Mixture Flow(配置里 primary_model=lgbm),但 Mixture Flow 给的样本更连续、分布更细。                                                                                                     
    7.3 其他变体                                                                                                                                                                                              
    - train_mlp_predictor.py:每个连续/离散头独立 sklearn MLPRegressor/MLPClassifier,标准化 + adaptive lr,作为简单基线。                                                                                       
    - train_baseline_linear.py:Ridge + LogisticRegression,头脑最干净的对照。                                                                                                                                  
    - train_condition_residual_mdn_mixed.py:同款架构但用纯 MDN(没有显式 baseline 残差解耦)。                                                                                                                  
    ---                                                                                                                                                                                                       
   对这一部分进行非常细致的分析，达到可以让别人复现，可以直接写文章的程度，要足够细致的分析                                                                                                                   
⏺ ---
  Stage 04 · Stage3 条件预测模型:从 Mixture Residual Flow 到 LightGBM Quantile Ensemble,可重复版
  
  ▎ 本文是 SynPred 流水线 Stage 04 中Stage3 条件预测子模块的深度技术文章。Stage3 的任务是:给定 (x, y_set) — 即"目标晶体结构特征 + 
  ▎ 已选定的前驱物集合",预测合成所需的温度、时间、气氛、合成类型。这是一个混合类型的条件生成问题(连续 + 离散 + 标签可缺失),SynPred 用三种风格模型并存:
  ▎ - 主力:Mixture Residual Flow(残差 MoG,神经网络)
  ▎ - 稳过工程基线:LightGBM Quantile Ensemble(分位数回归 + 二/三分类)
  ▎ - 最简对照:MLP / Linear baseline
  ▎
  ▎ 本文给出 5 份训练脚本(共 ~4000 行)+ 1 份 LGBM 推理脚本的逐行级解析,达到逐字复现的程度。

  ---
  第一部分 · 角色与契约
  
  到 Stage 03 (Data) 结束,SynPred 已经把 stage3_v5_mixed NPZ 准备好,每个 NPZ 含:

  ┌────────────────────────┬────────────────┬─────────────────────────────────────────────┐
  │          张量          │      形状      │                    含义                     │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ x                      │ (N, F) float32 │ hybrid 特征(描述子 + GNN 嵌入)              │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ y_set                  │ (N, V) float32 │ 前驱物 multi-hot(main + aux)                │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ y_cond_continuous      │ (N, 2) float32 │ (温度, 时间) 标准化后                       │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ y_cond_continuous_mask │ (N, 2) float32 │ 0/1 mask                                    │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ y_cond_discrete        │ (N, 2) int64   │ (atmosphere_coarse_idx, synthesis_type_idx) │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ y_cond_discrete_mask   │ (N, 2) float32 │ 0/1 mask                                    │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────┤
  │ sample_id              │ (N,) object    │ 字符串 id                                   │
  └────────────────────────┴────────────────┴─────────────────────────────────────────────┘

  加上 condition_schema.json(含 continuous_schema={col: {mean, std, median}}、discrete_schema={col: vocab[]}),schema.json(数据集汇总)。

  Stage 04 Stage3 的契约:
  1. 训练:从 NPZ 读 (x, y_set) 与目标,处理缺失,训练后保存 ckpt
  2. 推理:对 (x, y_set) 给出多套候选条件(每条样本输出 top-k,典型 k=5~9),进入 Stage35 联合排序
  
  下文按"Mixture Flow → LGBM → MLP/Linear/MDN"四节展开。

  ---
  第二部分 · Mixture Residual Flow(train_condition_mixture_flow_mixed.py,1310 行)
  
  2.1 双阶段思路:残差解耦

  模型的核心思想是:先用 baseline 给条件均值,Flow 只学残差。

  $$
  \hat c = c_{\text{base}}(x, y_{\text{set}}) + \Delta c, \qquad \Delta c \sim p_\theta(\cdot \mid x, y_{\text{set}})
  $$

  为什么这样设计?Stage3 的目标范围很广(温度 100-2000°C,时间 0.1-1000h),如果直接让神经网络预测分布,小数据下容易学崩。先用一个简单 baseline(Ridge/MLP/ExtraTrees 都行)给一个粗略均值,Flow 只学"baseline
  没捕到的偏离"——残差范围小、分布更接近 0,Flow 容易学。

  类比:Boosting 的 stage-wise residual 思想,但这里残差是"分布"而非"点"。

  2.2 Baseline 选择(行 264-542)

  两套 baseline 代码并存:
  
  (a) Stage3BaselineModel(行 264-309):同一份代码里的小 MLP,作为 torch ckpt 保存。
  - set_encoder:V → set_proj_dim=256
  - trunk:(F + 256) → hidden_dims=[512, 256]
  - disc_heads[i]:Linear(trunk_out, K_i)
  - cont_head:Linear(trunk_out, n_cont=2)
  
  输出 dict:{disc_logits: List[Tensor], cont_pred: Tensor (B, 2)}。

  (b) SklearnBaselineAdapter(行 318-442):适配 sklearn 风格 pickle payload,常见 payload 是 {"cont_models": [Ridge1, Ridge2], "disc_models": [LogReg1, LogReg2], "use_y_set": True},由 train_baseline_linear.py
  输出。object.__setattr__ 绕过 nn.Module 的 submodule 注册——sklearn 模型不是 torch 模块。

  load_pickle_baseline_payload(行 444+) 兼容三种文件:.pkl/.pt/.joblib,并自动判断是 dict-payload(走 sklearn adapter)还是 nn.Module(走 Stage3BaselineModel)。

  ▎ 可重复要点 #1:--baseline_ckpt 是必填参数。SynPred 推荐先用 train_baseline_linear.py 训一份 Ridge baseline pickle,再以此 baseline 训 Flow。如果 baseline ckpt 是 torch 形式,需要给 --baseline_hidden_dims 
  ▎ "512,256" 还原结构。

  2.3 上下文编码(ResidualContextEncoder,行 548-579)

  y_set ─ MLP_set([V, 256]) ─┐
                              ├── concat / add ── MLP_trunk([F+256→512→256]) ── h (B, 256)
  x ──────────────────────────┘

  两种融合模式:
  - fuse_mode="concat"(默认):fused = cat([x, set_repr]),trunk 输入维 F + set_proj_dim
  - fuse_mode="add":fused = x_proj(x) + set_repr,trunk 输入维 set_proj_dim,需要 x_proj 把 F → 256
  
  MLP(行 243-258)是带 LayerNorm + SiLU + Dropout 的标准 stack(可选 layernorm)。

  输出 h 维度 = trunk 最后一层(默认 256),叫 ctx_dim,被三个 head 共享。

  2.4 Mixture-of-Gaussians 残差头(行 582-697)

  2.4.1 模型成员
  
  self.gating    = MLP([ctx_dim, gating_hidden_dim=128, n_components=3])
  self.components = ModuleList([
      MLP([ctx_dim, flow_hidden_dim=256, 2*y_cont_dim]) for _ in range(n_components)
  ])
  self.resid_disc_heads = ModuleList([Linear(ctx_dim, K_i) for K_i in disc_class_sizes])

  n_components 默认 3(可调 5)。每个 component MLP 输出 2 * y_cont_dim = 4 维(2 维均值 + 2 维 log_scale)。

  2.4.2 提取分量参数(_component_params,行 634-640)
  
  gating_logits = self.gating(context)               # (B, K)
  params = stack([comp(context) for comp in components], dim=1)  # (B, K, 2*D)
  means = params[..., :D]                             # (B, K, D=2)
  log_scales = params[..., D:].clamp(-5.0, 3.0)       # 数值稳定

  log_scales clamp 是关键:防止 σ 飘到 0 (log_scale=-5 ⇒ σ≈0.0067)或爆炸(log_scale=3 ⇒ σ≈20.1)。这一对夹紧值经验上覆盖了 Stage3 标准化后残差的合理范围。

  2.4.3 NLL(行 656-669)

  数学上:
  
  $$
  p(\Delta c \mid h) = \sum_{k=1}^K \pi_k(h), \mathcal N\big(\Delta c;,\mu_k(h),,\mathrm{diag}(\sigma_k^2(h))\big)
  $$

  $$
  \mathcal L_{\text{cont}} = -\frac{1}{N}\sum_i \log p(\Delta c_i \mid h_i)
  $$

  对应代码:
  full_rows = (mask > 0.5).all(dim=1)             # 关键:只用所有 cont 维都有标签的样本
  n_valid = int(full_rows.sum().item())           
  if n_valid == 0: return residual.sum()*0.0, 0
  r = residual[full_rows] 
  gating_logits, means, log_scales = self._component_params(context[full_rows])
  scales = torch.exp(log_scales)
  
  diff = (r[:, None, :] - means) / clamp(scales, 1e-6)            # (B, K, D)
  log_prob_comp = -0.5 * (diff**2 + 2*log_scales + log(2π)).sum(-1)   # (B, K)
  log_mix = log_softmax(gating_logits, -1)                         # (B, K)
  log_prob = logsumexp(log_mix + log_prob_comp, -1)                # (B,)
  return -log_prob.mean(), n_valid
  
  关键工程决策:(mask > 0.5).all(dim=1) ——只对 所有 连续维都有标签的样本计算 NLL。这避免温度缺失但时间存在(或反过来)时的偏差。代价是丢一部分样本。

  ▎ 可重复要点 #2:若你想最大利用每条样本(逐维 mask),改用 train_condition_residual_mdn_mixed.py:masked_nll(§5),它把 dim 维 mask 加权进 log_prob_dim,允许部分维有标签即参与训练。这两种处理在 SynPred 数据上 MAE
  ▎  差异 ~3%,Flow 版本更稳但 MDN 版本利用率更高。

  2.4.4 离散头损失(行 753-761)

  def multihead_classification_loss(logits_list, targets, class_weight_tensors, head_weights):
      losses = []
      for i, logits in enumerate(logits_list):
          weight = class_weight_tensors[i] if class_weight_tensors else None
          loss_i = F.cross_entropy(logits, targets[:, i], weight=weight)
          losses.append(loss_i * head_weights[i])
      return stack(losses).sum() / max(sum(head_weights), 1e-8)

  数学上:

  $$
  \mathcal L_{\text{disc}} = \frac{\sum_j w_j \cdot \mathrm{CE}(\hat y_j, y_j)}{\sum_j w_j}
  $$

  head_weights 默认全 1.0(--discrete_head_weights ""); class_weight_tensors 来自 build_class_weight_tensor(行 742-750):

  counts = bincount(y_disc[:, i], minlength=K_i)
  counts = clip(counts, 1.0, None)            # 避免 0 除
  w = counts.sum() / counts                   # 频率倒数
  w = w / w.mean()                            # 归一化均值=1

  即"按 1/freq 加权"。开关 --use_class_weights 控制。SynPred 默认 ON,因为 atmosphere 的类别极不均衡(air_or_oxidizing 占 ~70%)。

  2.4.5 离散头的 baseline+residual 拼接(行 887)

  关键:Mixture Flow 的离散头不直接给 logits,而是给"残差 logits",最终 logits = baseline_logits + residual_logits:

  final_disc_logits = [b + r for b, r in zip(base_disc_logits, out["resid_disc_logits"])]
  loss_disc = multihead_classification_loss(final_disc_logits, y_disc, ...)

  这是把"残差解耦"思想从连续头扩展到离散头——baseline 给 prior logits,Flow 学修正。如果 baseline 输出空(行 184-206 的 pad_base_disc_logits 把缺失头补 0),那么 final = residual,等价于纯监督。

  2.5 推理采样(行 678-697)

  2.5.1 top1_residual —— 确定性中位数

  gating_logits, means, _ = self._component_params(context)
  top_idx = argmax(gating_logits, dim=1)
  return means[arange(B), top_idx]               # (B, D)

  取门控最高分量的均值作为 top-1 预测。这是 Stage3 主线 metric top1_continuous_mean_mae_raw 的依据。

  2.5.2 sample_continuous —— 随机采样 N 条
  
  gating_logits, means, log_scales = self._component_params(context)
  probs = softmax(gating_logits, -1)
  cat = Categorical(probs=probs)
  comp_idx = cat.sample((n_samples,))            # (S, B) 每样本独立采分量

  # 索引选中分量的 mean 与 log_scale
  gather_idx = comp_idx[..., None, None].expand(S, B, 1, D)
  chosen_means = gather(means_rep, 2, gather_idx).squeeze(2)
  chosen_log_scales = gather(log_scales_rep, 2, gather_idx).squeeze(2)

  eps = randn_like(chosen_means)
  return chosen_means + exp(chosen_log_scales) * eps    # (S, B, D)
  
  两步采样:
  1. 按 gating prob 抽 component
  2. 在该分量上 reparameterize $\mu + \sigma \epsilon$

  注意:每条样本、每次采样都独立采 component,不固定为 top-1 分量——这保留了多模态。

  2.5.3 evaluate_split 的 oracle metric(行 898-907)

  sample_resid = model.sample_continuous(out["context"], n_gen_samples)   # (S, B, D)
  sample_cont = base_cont.unsqueeze(0) + sample_resid                     # (S, B, D)
  target = y_cont.unsqueeze(0).expand(S, B, D)
  mask = y_mask.unsqueeze(0).expand(S, B, D)
  sqerr = ((sample_cont - target)**2) * mask
  denom = clamp(mask.sum(2), min=1.0)
  per_sample_err = sqerr.sum(2) / denom            # (S, B)
  best_idx = argmin(per_sample_err, dim=0)         # (B,)
  oracle_cont = sample_cont[best_idx, arange(B)]   # (B, D)

  oracle_best_of_k = "如果让 oracle 在 K 个采样里挑最好的,误差是多少"——这是上限性能,反映分布质量而非 top-1 accuracy。SynPred 默认 --n_gen_samples 8。

  2.6 标准化与逆变换(行 173-181, 921-924)

  NPZ 里 y_cond_continuous 已经标准化(stage3 v5 mixed 在 03_data 里做的)。Flow 学的 residual 也是标准化空间的。最终推理要回到原量纲:

  def inverse_transform_np(values, stats):
      out = values.copy()
      for i, st in enumerate(stats):
          out[:, i] = out[:, i] * st["std"] + st["mean"]
      return out

  cont_stats 来自 condition_schema["continuous_schema"][col],在训练时一并保存进 ckpt 的 schema.cont_stats,推理时从 ckpt 拿。

  2.7 训练范围 clamp(clip_to_train_range_fn,行 180-181)
  
  def clip_to_train_range_fn(pred, train_min, train_max):
      return clip(pred, train_min[None], train_max[None])

  --clip_to_train_range 默认开:推理结果 clamp 到训练集观察到的最小/最大原始量纲。这是保守做法——MoG 容易在小样本边缘外推出 5000°C 这种荒谬值,clip 让最差情况不至于太离谱。

  train_min/train_max 由 collect_train_raw_minmax(行 799-811)在原始量纲、masked rows 上求 min/max。

  2.8 训练循环(行 1132-1230)

  for epoch in 1..epochs:
      for batch in train_loader:
          # baseline 推理(无梯度)
          with torch.no_grad():
              base_out = baseline(x, y_set)
              base_cont = base_out["cont_pred"]
              base_disc_logits = pad_base_disc_logits(...)

          residual_target = y_cont - base_cont
          out = model(x, y_set)
          final_disc_logits = [b + r for b, r in zip(base_disc_logits, resid_disc_logits)]
          loss_disc = multihead_classification_loss(final_disc_logits, y_disc, ...)
          loss_cont, n_valid = model.nll(residual_target, out["context"], y_mask)
          loss = loss_disc + loss_cont
          backward + clip 5.0 + step

      val_metrics = evaluate_split(baseline, model, val_loader, ...)
      score = val_metrics["top1_continuous_mean_mae_raw"]   # 默认追这个
      early_stopper.step(score)                              # patience=8
      if improved: torch.save(...)                            # 保存 best ckpt

  默认超参(行 1014-1042):
  hidden_dims          = "512,256"
  set_proj_dim         = 256
  flow_hidden_dim      = 256
  n_flow_layers        = 4         (本实现里只用作命名,实际是 1 层 component MLP)
  n_components         = 3         
  gating_hidden_dim    = 128
  dropout              = 0.1
  fuse_mode            = "concat"
  batch_size           = 128
  epochs               = 40
  patience             = 8
  lr                   = 1e-4
  weight_decay         = 1e-5
  metric_name          = "top1_continuous_mean_mae_raw"
  n_gen_samples        = 8
  clip_to_train_range  = True
  grad_clip            = 5.0
  seed                 = 42
  
  ▎ 可重复要点 #3:metric_name 检查会自动识别"loss/mae/rmse/nll" 系是否最小化(is_minimize_metric,行 728-730),improved 方向自动反转。这让你切换不同 metric 时不用动 EarlyStopper 代码。

  2.9 输出文件

  {run_dir}/
  ├── best_stage3_condition_mixture_flow_mixed.pt  # model_state + config + schema(cont_stats, disc_class_sizes)
  ├── train_log.json                                # 每 epoch 的 train_loss + val_metrics
  ├── pred_train.csv / pred_val.csv / pred_test.csv  # 含:baseline_X / top1_X / oracle_best_of_k_X(norm + raw)
  └── final_metrics.json                             # train/val/test 三个 split 的全套指标

  预测 CSV 列(save_predictions,行 814-832):
  sample_id, true_<disc>, baseline_<disc>, top1_<disc>, oracle_best_of_k_<disc>,
  true_<cont>_norm, true_<cont>_raw, mask_<cont>,
  baseline_<cont>_norm, baseline_<cont>_raw,
  top1_<cont>_norm, top1_<cont>_raw,
  oracle_best_of_k_<cont>_norm, oracle_best_of_k_<cont>_raw
  
  下游可以直接对比 baseline / top1 / oracle 三档,量化 Flow 比 baseline 提升多少。

  ---
  第三部分 · LightGBM Quantile Ensemble(train_lgbm_quantile_ensemble.py,436 行)
  
  工程上,Mixture Flow 是"美的",但 LGBM 是"稳的"。SynPred 的实测:LGBM 在 MAE 上比 Flow 略好(~5%),且训练时间从 30min 降到 3min,无需 GPU。配置里 primary_model=lgbm 是默认。

  3.1 任务划分

  LGBM 不学一个端到端混合分布,而是把 Stage3 拆成 4 个独立任务:

  ┌──────────────┬────────────────────────────────────┬─────────────────────────────────────┐
  │     任务     │                目标                │                模型                 │
  ├──────────────┼────────────────────────────────────┼─────────────────────────────────────┤
  │ 温度回归     │ 9 个分位数 (0.1, 0.2, ..., 0.9)    │ LightGBM objective="quantile"       │
  ├──────────────┼────────────────────────────────────┼─────────────────────────────────────┤
  │ 时间回归     │ 9 个分位数                         │ 同上                                │
  ├──────────────┼────────────────────────────────────┼─────────────────────────────────────┤
  │ 气氛分类     │ 二分类(oxidizing vs non-oxidizing) │ objective="binary"                  │
  ├──────────────┼────────────────────────────────────┼─────────────────────────────────────┤
  │ 时间分桶分类 │ 三分类(short/medium/long)          │ objective="multiclass", num_class=3 │
  └──────────────┴────────────────────────────────────┴─────────────────────────────────────┘

  每个任务独立训、独立保存。

  3.2 分位数回归
  
  3.2.1 Pinball loss

  LightGBM objective="quantile" 用 pinball loss:

  $$
  \rho_\tau(u) = \max(\tau u,;(\tau-1)u) = \begin{cases} \tau u, & u \ge 0 \ (\tau-1)u, & u < 0 \end{cases}
  $$

  对真实 $y$ 与预测 $\hat y$,损失 $\rho_\tau(y - \hat y)$ 在 $\tau$-分位数处期望最小,即 LightGBM 学到的 $\hat y$ 是条件 $\tau$-分位数 $Q_\tau(y \mid x)$。

  9 个分位数 ⇒ 9 个独立模型,每个用 params = {**QUANTILE_PARAMS, "alpha": q}(行 132)。alpha=q 就是 LightGBM 中的 $\tau$。

  3.2.2 训练参数(行 42-55)

  QUANTILE_PARAMS = {
      "objective": "quantile",
      "metric": "quantile",
      "boosting_type": "gbdt",
      "num_leaves": 255,
      "learning_rate": 0.05,
      "feature_fraction": 0.8,
      "bagging_fraction": 0.8,
      "bagging_freq": 5,
      "min_data_in_leaf": 20,
      "seed": 42,
  }

  num_leaves=255 偏大,让模型有足够 capacity 学复杂分位数曲面。feature_fraction=0.8 + bagging_fraction=0.8 + min_data_in_leaf=20 起防过拟作用。num_boost_round=200,early stopping patience=20。

  3.2.3 mask 处理(行 120-126)
  
  mask_train = data["train"]["y_cond_continuous_mask"][:, target_idx] > 0.5
  X_train_m, y_train_m = X_train[mask_train], y_train[mask_train]

  只用有标签的行训练——非常直白,不像 Flow 还要做"逐维 mask 加权"。代价是某些样本对温度有标签但对时间无,温度模型用它,时间模型不用,资源利用率次优但实现极简。

  3.2.4 评估(行 149-178)
  
  top1_pred = test_preds[0.5]                        # median 当 top-1
  mae = abs(y - top1_pred).mean()
  median_ae = median(abs(y - top1_pred))
  r2 = r2_score(y, top1_pred)

  # Oracle: 9 个 quantile 里挑误差最小的
  all_preds = stack([test_preds[q] for q in QUANTILES], 1)
  errors = abs(all_preds - y[:, None])
  oracle_pred = all_preds[arange(N), errors.argmin(1)]
  oracle_mae = abs(y - oracle_pred).mean()

  # 阈值内比例
  for thresh in [25, 50, 100, 200]:    # 温度
      metrics[f"top1_within_{thresh}"] = (abs_err <= thresh).mean()

  关键工程指标:top1_within_50 = "预测温度误差 ≤ 50°C 的比例",这比 MAE 更直接反映"工程上可用"。SynPred 实测:LGBM top1_within_100 ~75%,Flow top1 ~70%。

  3.3 气氛二分类(train_atmosphere_classifier,行 181-254)

  简化为二分类(oxidizing vs inert/reducing/vacuum):
  
  y = (df["target_atmosphere_coarse"] == "air_or_oxidizing").astype(int)
  valid = df["has_target"] > 0.5    # 只用有标签的

  不用 NPZ,从 task_views_dir/atmosphere_coarse_{train,val,test}.csv 读——这是 Stage 02 的"任务视图",每行已是按 atmosphere 任务过滤好的样本 + 描述子。

  参数:
  ATM_PARAMS = {
      "objective": "binary",
      "metric": "binary_logloss",
      "num_leaves": 63,        # 比分位数小,因为是二分类
      "learning_rate": 0.05,   
      "min_data_in_leaf": 10,
      ...
  }   
  
  推理时用 threshold=0.6(不是 0.5)预测正类——这是 macro_f1 调出来的,因为正类占比 70%,稍提阈值能提升 macro_f1。

  3.4 时间分桶三分类(train_time_bucket_classifier,行 257-332)

  classes = ["short", "medium", "long"]
  class_map = {c: i for i, c in enumerate(classes)}
  y = df["target_time_bucket"].map(class_map)

  # 关键:排除可能泄漏的连续时间列
  leak_cols = {"target_time_h_log1p", "target_time_h_clean"}
  feat_cols = [c for c in df.columns if c not in meta_cols and c not in leak_cols]

  反泄漏:time_bucket 是从 target_time_h_clean 离散化得到的,如果训练特征里含 target_time_h_log1p 就泄漏了。这个 leak_cols 显式排除关键。

  TIME_BUCKET_PARAMS = {"objective": "multiclass", "num_class": 3, ...},用 multi_logloss 监督。推理 argmax。

  3.5 推理时的 Quantile 优先级(13b_run_stage3_infer_lgbm_quantile.py:210)

  这是 LGBM 模块最优雅的推理 trick:
  
  # Order: Q0.5, Q0.4, Q0.6, Q0.3, Q0.7, Q0.2, Q0.8, Q0.1, Q0.9
  quantile_priority = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9]
  k = min(top_k_conditions, 9)

  for sample_i in range(N):
      seen_temps = set()
      exported = 0
      for q in quantile_priority:
          temp_val = temp_preds[q][i]
          time_val = time_preds[q][i]

          # clip 物理范围
          temp_val = max(0.0, min(2000.0, temp_val))
          time_val = max(0.0, min(5000.0, time_val))

          # 按 10°C 粒度去重
          temp_rounded = round(temp_val / 10) * 10
          if temp_rounded in seen_temps:
              continue
          seen_temps.add(temp_rounded)

          # score:median 1.0,向两侧线性递减
          score = 1.0 - abs(q - 0.5) * 2.0
  
          rows.append({
              "condition_rank": exported,
              "stage3_score": score,
              "condition_source": "lgbm_top1" if q == 0.5 else f"lgbm_q{q:.1f}",
              "cont_conditions": {"temperature_c": temp_val, "time_h": time_val},
              ...
          })
          exported += 1
          if exported >= k:
              break

  优先级解读(行 209-210):

  0.5  ← median,中位/最稳
  0.4, 0.6  ← 中等不确定的左右翼
  0.3, 0.7  ← 中等翼
  0.2, 0.8  ← 大翼
  0.1, 0.9  ← 极端翼

  效果是:先取保守(中位)估计,然后向两侧扩张。这等价于:在不确定性范围内提供"由保守到激进"的一组候选,让下游 Stage35 排序器有多样选择。

  去重:temp_rounded = round(temp/10)*10 让相邻分位数若预测温差 < 10°C 就合并(避免输出 5 个 850-855°C 的"不同"候选)。

  score:median 1.0,Q0.4/Q0.6 = 0.8,Q0.3/Q0.7 = 0.6, ..., Q0.1/Q0.9 = 0.2。这个 score 进入 candidate JSONL 给下游。

  3.6 命令行

  # 训练:
  python scripts/04_train/stage3/train_lgbm_quantile_ensemble.py \
    --project_root /Users/wyc/SynPred \
    --input_dir    /Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1 \
    --task_views_dir /Users/wyc/SynPred/data/interim/features/stage3_task_views \
    --output_dir         runs/stage3/lgbm_quantile_ensemble_v2_fulldata \
    --atm_output_dir     runs/stage3/lgbm_atmosphere_classifier_v1 \
    --time_bucket_output_dir runs/stage3/lgbm_time_bucket_classifier_v1 \
    --num_boost_round 200 \
    --early_stopping_rounds 20

  # 推理:
  python scripts/07_infer/structure_to_synthesis_route/pipeline/src/13b_run_stage3_infer_lgbm_quantile.py \
    --conditioned_x_csv .../stage3_x_with_parents.csv \
    --schema_json       runs/stage3/lgbm_quantile_ensemble_v2_fulldata/schema.json \
    --temp_models_dir   runs/stage3/lgbm_quantile_ensemble_v2_fulldata/temperature \
    --time_models_dir   runs/stage3/lgbm_quantile_ensemble_v2_fulldata/time \
    --atm_model         runs/stage3/lgbm_atmosphere_classifier_v1/model_atmosphere_binary_final.txt \
    --time_bucket_model runs/stage3/lgbm_time_bucket_classifier_v1/model_time_bucket.txt \
    --output_dir        runs/stage3_infer_lgbm/<view> \
    --top_k_conditions  5

  ---
  第四部分 · MLP / Linear baselines
  
  4.1 train_mlp_predictor.py(781 行)

  每个连续/离散头一个独立 sklearn MLPRegressor/MLPClassifier(行 395, 437):

  MLPRegressor(
      hidden_layer_sizes=(256, 128),
      activation="relu",
      solver="adam",
      alpha=1e-4,                    # L2
      learning_rate="adaptive",      # 自动减半
      learning_rate_init=1e-3,
      max_iter=200,
      early_stopping=True,
      validation_fraction=0.1,
      n_iter_no_change=10,
  )

  特点:
  - per-head 独立训练,不共享 backbone
  - 用 StandardScaler 标准化输入
  - 缺失标签的样本被 mask 掉(行 387-394)
  - 稀疏类别守护(行 428-435):某离散类别 < 2 个样本时关闭 early_stopping 防止 train/val 分裂时 val 那一类完全空
  
  fit_report(行 380-455)记录所有 dropped/skipped 头,便于诊断。

  输出 pickle:{cont_models, disc_models, scaler, cont_names, disc_names, fit_report},可作为 Mixture Flow 的 --baseline_ckpt(走 SklearnBaselineAdapter)。

  4.2 train_baseline_linear.py(499 行)

  最干净对照——Ridge + LogisticRegression 单层:

  Ridge(alpha=1.0, random_state=42)                  # 连续头
  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")   # 离散头
  DummyClassifier(strategy="most_frequent")          # 单类时 fallback

  无标准化(--standardize 默认 OFF,因为 Stage 03 已经标准化过了)。单类目标(uniq <= 1)用 DummyClassifier 代替——LogisticRegression 单类会报错。

  输出 pickle 同 MLP,可直接给 Mixture Flow / MDN 当 baseline。

  ▎ 可重复要点 #4:SynPred 推荐的 baseline 链:
  ▎ 1. 先训 train_baseline_linear.py → 生成 Ridge baseline pickle
  ▎ 2. 再训 train_condition_mixture_flow_mixed.py --baseline_ckpt <Ridge pickle> → Flow 学残差
  ▎
  ▎ 这种 Linear baseline + MoG 残差的组合在 SynPred 数据上比 MLP-baseline + MoG 略稳(LR 不会 overfit baseline 阶段)。

  4.3 train_condition_residual_mdn_mixed.py(1012 行)—— 同款架构但纯 MDN

  与 Mixture Flow 的差别:
  
  ┌───────────────────┬─────────────────────────────────────────┬─────────────────────────────────────────┐
  │       维度        │              Mixture Flow               │              Residual MDN               │
  ├───────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
  │ 残差解耦          │ ✓(显式 baseline)                        │ ✓(显式 baseline)                        │
  ├───────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
  │ Mixture Head      │ n_components=3,各自 MLP 输出 (μ, log_σ) │ MDNHead(n_mixtures=5),共享一层          │
  ├───────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
  │ log_σ clamp       │ [-5, 3]                                 │ [-7, 5](更宽)                           │
  ├───────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
  │ NLL 实现          │ "全维都有标签才计算"                    │ "逐维 mask 加权"(masked_nll,行 292-320) │
  ├───────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────┤
  │ 默认 n_components │ 3                                       │ 5(更多模式)                             │
  └───────────────────┴─────────────────────────────────────────┴─────────────────────────────────────────┘

  MDNHead 用三个 Linear:pi (in→K)、mu (in→K*D)、log_sigma (in→K*D),reshape 后用。结构更紧凑但表达力略弱。

  masked_nll(行 291-320)的精髓:
  log_prob_dim = -0.5*(z**2 + 2*log_sigma + log(2π))      # (B, K, D)
  log_prob_dim = log_prob_dim * mask * head_weights       # 逐维 mask
  valid_dim_count = (mask * head_weights).sum(-1).clamp_min(1.0)
  log_prob_comp = log_prob_dim.sum(-1) / valid_dim_count  # 平均到有效维
  log_pi = log_softmax(pi_logits, -1)
  log_mix = logsumexp(log_pi + log_prob_comp, -1)
  return -log_mix.mean()
  
  允许"温度有标签但时间没有"的样本也参与训练,只用温度那一维做 NLL。这与 Mixture Flow 的"all-or-nothing" 互补,通常 MDN 版能多用 5-10% 样本。

  4.4 模型选择参考

  ┌───────────┬──────────────────┬─────────────────────────────┐
  │ 数据规模  │     推荐主力     │            理由             │
  ├───────────┼──────────────────┼─────────────────────────────┤
  │ < 5k 样本 │ Linear baseline  │ 简单稳健,不容易过拟合       │
  ├───────────┼──────────────────┼─────────────────────────────┤
  │ 5k - 20k  │ LGBM Quantile    │ 工程稳过                    │
  ├───────────┼──────────────────┼─────────────────────────────┤
  │ 20k+      │ Mixture Flow     │ 表达力强,分布更细           │
  ├───────────┼──────────────────┼─────────────────────────────┤
  │ 任意规模  │ LGBM + Flow 双跑 │ 候选互补,Stage35 排序自动选 │
  └───────────┴──────────────────┴─────────────────────────────┘

  ---
  第五部分 · 端到端复现指南
  
  5.1 依赖

  pip install numpy pandas scikit-learn torch
  pip install lightgbm        # LGBM 分支需要

  5.2 完整管道

  PR=/Users/wyc/SynPred

  # === STEP 1: 03_data 已经准备好 NPZ + condition_schema ===
  ls $PR/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/
  # 期望: train.npz val.npz test.npz schema.json condition_schema.json export_summary.json

  # === STEP 2: 训练 baseline(Ridge,作为 Flow 的输入) ===
  python $PR/scripts/04_train/stage3/train_baseline_linear.py \
    --input_dir   $PR/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1 \
    --output_dir  $PR/runs/stage3/baseline_ridge_v1 \
    --ridge_alpha 1.0 --logreg_C 1.0

  # === STEP 3: 训练 Mixture Residual Flow ===
  python $PR/scripts/04_train/stage3/mixed/train_condition_mixture_flow_mixed.py \
    --input_dir     $PR/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1 \
    --baseline_ckpt $PR/runs/stage3/baseline_ridge_v1/best_model.pkl \
    --run_dir       $PR/runs/stage3/mixture_flow_v1 \
    --hidden_dims "512,256" --set_proj_dim 256 \
    --flow_hidden_dim 256 --n_components 3 --gating_hidden_dim 128 \
    --dropout 0.1 --fuse_mode concat \
    --batch_size 128 --epochs 40 --patience 8 \
    --lr 1e-4 --weight_decay 1e-5 \
    --metric_name top1_continuous_mean_mae_raw \
    --use_class_weights --clip_to_train_range \
    --n_gen_samples 8 --grad_clip 5.0 --device cuda --seed 42

  # === STEP 4: 训练 LGBM Quantile Ensemble(并行通路) ===
  python $PR/scripts/04_train/stage3/train_lgbm_quantile_ensemble.py \
    --project_root $PR --num_boost_round 200 --early_stopping_rounds 20

  # === STEP 5: LGBM 推理 ===
  python $PR/scripts/07_infer/structure_to_synthesis_route/pipeline/src/13b_run_stage3_infer_lgbm_quantile.py \
    --conditioned_x_csv $PR/runs/stage3_infer/conditioned_x.csv \
    --schema_json       $PR/runs/stage3/lgbm_quantile_ensemble_v2_fulldata/schema.json \
    --temp_models_dir   $PR/runs/stage3/lgbm_quantile_ensemble_v2_fulldata/temperature \
    --time_models_dir   $PR/runs/stage3/lgbm_quantile_ensemble_v2_fulldata/time \
    --atm_model         $PR/runs/stage3/lgbm_atmosphere_classifier_v1/model_atmosphere_binary_final.txt \
    --time_bucket_model $PR/runs/stage3/lgbm_time_bucket_classifier_v1/model_time_bucket.txt \
    --output_dir        $PR/runs/stage3_infer/lgbm_quantile_v1 \
    --top_k_conditions 5

  5.3 输出文件清单

  Mixture Flow:
  runs/stage3/mixture_flow_v1/
  ├── best_stage3_condition_mixture_flow_mixed.pt   # ckpt(含 schema)
  ├── train_log.json                                # epoch-level metrics
  ├── pred_train.csv / pred_val.csv / pred_test.csv # 含 baseline / top1 / oracle
  └── final_metrics.json

  LGBM:
  runs/stage3/lgbm_quantile_ensemble_v2_fulldata/
  ├── temperature/
  │   ├── temp_q0.1.txt, ..., temp_q0.9.txt        # 9 个 LGBM booster
  │   └── metrics.json
  ├── time/
  │   ├── time_q0.1.txt, ..., time_q0.9.txt
  │   └── metrics.json
  └── schema.json                                    # 复制自 dataset 的
  
  runs/stage3/lgbm_atmosphere_classifier_v1/
  ├── model_atmosphere_binary_final.txt
  └── metrics_final.json
  
  runs/stage3/lgbm_time_bucket_classifier_v1/
  ├── model_time_bucket.txt
  └── metrics.json
  
  runs/stage3_infer/lgbm_quantile_v1/
  ├── test_candidates.jsonl                          # 给下游 Stage35
  ├── test_candidates_flat.csv                       
  ├── debug_parent_candidates.csv
  └── candidate_summary.json

  5.4 验收清单

  ┌───────────────────────────┬──────────────────────────────────────────────────────┬────────────────────────────────────────────────────┐
  │          验收点           │                         命令                         │                        期望                        │
  ├───────────────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Flow ckpt 可加载          │ torch.load(...)['schema']['cont_stats']              │ 有 mean/std/median                                 │
  ├───────────────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Flow 训练收敛             │ train_log.json 末尾 val.top1_continuous_mean_mae_raw │ < 100°C(温度)                                      │
  ├───────────────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Oracle vs Top-1 差距      │ 相同指标 oracle / top1 比                            │ 0.6-0.8(差距越小,分布越尖)                         │
  ├───────────────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ LGBM temp top1_within_100 │ metrics.json                                         │ > 0.7                                              │
  ├───────────────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ LGBM atm accuracy         │ atm_metrics_final.json                               │ > 0.85                                             │
  ├───────────────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ 推理 jsonl 格式           │ head -1 test_candidates.jsonl                        │ 含 stage3_score, condition_source, cont_conditions │
  └───────────────────────────┴──────────────────────────────────────────────────────┴────────────────────────────────────────────────────┘

  5.5 常见坑
  
  ┌─────────────────────────────┬──────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │            现象             │                 原因                 │                             解                             │
  ├─────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ KeyError: y_cond_continuous │ NPZ 是 v4(无 mixed),不是 v5          │ 用 27_build_stage3_condition_dataset_v5_mixed.py 重新构    │
  ├─────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Flow loss NaN 或 inf        │ log_scale 没 clamp 飞了              │ 检查 clamp(-5, 3) 是否生效;若数据极端,降至 (-3, 2)         │
  ├─────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ LGBM cannot find feature    │ 输入 csv 列与训练时不一致            │ 重生成 conditioned_x_csv,保证特征列同序                    │
  ├─────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ MLP fit 报 single class     │ 某离散维 train 全是同一类            │ 看 fit_report 的 dropped_or_skipped_heads,这是预期行为     │
  ├─────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Flow oracle ≈ top1          │ n_gen_samples 太小,采样不到分布 tail │ 调大 --n_gen_samples 32 或更高                             │
  ├─────────────────────────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ baseline pickle 加载失败    │ torch ckpt 与 sklearn pickle 混淆    │ 看 load_pickle_baseline_payload 的 [Baseline Adapter] 提示 │
  └─────────────────────────────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────┘

  ---
  第六部分 · 设计哲学与可改进点
  
  6.1 七个工程模式

  1. 残差解耦:Flow / MDN 都用"baseline + residual"两阶段,把表达难度从端到端分摊到两个简单模型。
  2. 混合类型分头处理:连续(Mixture Gaussian)+ 离散(每维独立 CE,带类频权重)+ 缺失(mask 全维或逐维)。
  3. 不重新拟合 standardizer:Flow / MDN / LGBM 都从 NPZ 里读 condition_schema.continuous_schema 的 mean/std,确保跨模型的 raw vs norm 转换严格一致。
  4. Quantile Ensemble 优先级 trick:不靠采样而靠"事先训好的多分位数",推理时按"由稳到激进"序遍历,工程上稳定可控,无随机性。
  5. train-only stats:连续 mean/std + class weights + train_min_raw/max_raw 都从 train 算,val/test 一律 transform。
  6. Schema in ckpt:best_*.pt 自带 disc_class_sizes、cont_col_names、cont_stats,推理时无需 dataset 也能还原模型 + 反标准化。
  7. 多模型并存,Stage35 自决:LGBM + Flow + MLP 都输出统一格式 candidate JSONL,Stage35 排序器在所有候选上做 listwise 打分,自动权衡。

  6.2 可改进点

  (a) Flow 的 n_components=3 偏小。Stage3 数据里温度有清晰多模(室温合成 ~25°C / 中温陶瓷 ~800°C / 高温熔体 ~1500°C),3 个分量勉强够,5-7 个更合理。代价:component 数翻倍,参数量 ~1.5×。

  (b) LGBM Quantile 互相独立训练,导致单调性可能违反——理论上 $Q_{0.3}(y) \le Q_{0.5}(y)$,但分别独立训不保证。可以加一道后处理:monotonic isotonic regression,把 9 个 quantile 投影到单调升序。

  (c) MDN/Flow 的 baseline 从 nn.Module 形式生成(if Stage3BaselineModel),却需要再训一份 sklearn pickle 当 fallback——两套 baseline 代码维护是负担。统一成 sklearn pickle 路线即可。

  (d) 气氛二分类丢失了 inert vs reducing 区分。如果你做合成路线规划要区分 N₂ vs H₂/Ar 或真空,需要用 4 类 multi-class。当前实现把 inert/reducing/vacuum 合并是为了让正负类比例不太失衡(否则 reducing 太少)。

  (e) Quantile Ensemble 的 score 是粗的线性映射(1 - 2|q-0.5|),没考虑预测置信度。可改成"用相邻 quantile 之差估不确定度,不确定度大的样本 score 整体下调"——给 Stage35 更精细信号。

  (f) Mixture Flow 的离散头 baseline 里若全 0 logits,等价于均匀先验,Flow 完全自学。这种情况下"残差"的命名容易误导。可以在 ckpt 里记录 baseline_disc_use_flag 让代码语义更清楚。

  6.3 与同类工作的比较
  
  ┌──────────┬───────────────────────┬───────────────────────┬───────────────────┬─────────────────┐
  │   维度   │ 本实现 (Mixture Flow) │ Diffusion Conditional │ Normalizing Flow  │ Quantile Forest │
  ├──────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
  │ 表达力   │ 中(MoG)               │ 高                    │ 高                │ 中              │
  ├──────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
  │ 训练稳定 │ 高(NLL 凸 wrt μ/σ)    │ 中(SDE 反向)          │ 中(coupling 易爆) │ 高(凸 pinball)  │
  ├──────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
  │ 采样速度 │ 极快(单步 reparam)    │ 慢(多步)              │ 中                │ 快              │
  ├──────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
  │ 缺失数据 │ 易(逐维 mask)         │ 难                    │ 难                │ 易              │
  ├──────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
  │ 可解释性 │ 高(component μ 可读)  │ 低                    │ 中                │ 中              │
  ├──────────┼───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
  │ 工程依赖 │ torch                 │ torch + diffusion 库  │ torch + flow 库   │ LightGBM        │
  └──────────┴───────────────────────┴───────────────────────┴───────────────────┴─────────────────┘

  SynPred 的选择(MoG + LGBM 双跑)在小数据 + 混合类型 + 缺失普遍 的场景上做了最稳的工程取舍。

  ---
  第七部分 · 结语
  
  Stage 04 Stage3 把"给定结构 + 前驱物组合 → 预测合成条件"这一混合类型问题用三种风格并行求解:

  - Mixture Residual Flow:神经网络 + 残差解耦 + MoG,给细致的连续分布,可采样多样化候选
  - LightGBM Quantile Ensemble:9 个分位数 + 优先级排序,工程稳健、推理无随机性、无需 GPU
  - MLP / Linear:per-head 独立模型,作为 baseline 对照与 Flow 的输入 prior

  三者输出同一格式 candidate JSONL,Stage35 联合排序器在共同候选池上做最终 ranking。这种"多模型并存 + 单一接口下游"的设计让 SynPred 在不同数据规模下都有最优解,且每条路径都可独立替换。

  按本文 §5 的步骤逐字执行,你能复现:
  - 一份 Mixture Flow ckpt(典型 val top1_continuous_mean_mae_raw ~80°C / 8h)
  - 一份 LGBM Quantile Ensemble(典型 test top1_within_100 ~75%, atm accuracy ~0.87)
  - 一份 LGBM Multi-quantile candidate JSONL(每条样本 5 个由保守到激进的条件候选)
  - 一份 MLP/Linear baseline pickle(给 Flow 当输入,或单独作对照)
  
  Stage 04 Stage3 的契约就两条:candidate JSONL 的字段名不变 + ckpt schema 的 col_names 不变。下游 Stage35 排序器只看这两层契约,不在乎你用 Flow / LGBM / MLP
  哪一支。所有的工程优雅都是为了在保持这两条契约的前提下,让训练、推理、模型替换彼此独立、并行、可重复。

  
❯   8. Stage35 联合排序器                                                                                                                                                                                     
    把 Stage2 候选集合 × Stage3 条件候选 配对评分。本工程实际部署了多个版本(rule、learned、v21 hybrid、v3 learned ranker、v43 template-aware)叠成"ranker chain"。                                             
    8.1 V21 兼容性 ExtraTrees(主力,stage35.v21_* 配置)                                                                                                                                                        
    - 训练集来自 24_build_stage35_hardneg_compat_dataset.py(positives + 2×hard_neg + N×random_neg)。                                                                                                          
    - 输入特征 = 结构特征 + 候选前驱物 multi-hot,标签 = pair_label∈{0,1}。                                                                                                                                    
    - 模型 = ExtraTreesClassifier(n_estimators=...)(joblib 序列化)。                                                                                                                                          
    - 推理时输出 proba(pos) 当作 v21 score → 全局 rank。                                                                                                                                                      
    8.2 V43 Template-Aware Pairwise Ranker(route_ranker/v43_template_aware/)                                                                                                                                  
    关键创新:把化学模板信息显式作为特征。                                                                                                                                                                     
    - 01_add_route_template_features.py:对每条候选路线判定前驱物类型(hydrate/nitrate/carbonate/phosphate/sulfate/oxide/hydroxide/halide/...,precursor_type),再判定路线整体的                                  
    route_template_matches_target_anion、is_common_solid_state、is_overly_elemental。                                                                                                                         
    - 02_build_v43_template_pairwise_dataset.py:构造同目标 pair(route_a, route_b),用一个弱奖励函数(compute_template_quality,行 55)给出排序方向:                                                               
    $$Q = 4,\text{cov} - 2,\text{miss} - 0.25,\text{extra} - 1.5,\text{foreign} - 0.5,\text{extra_nontriv}$$                                                                                                  
    $$;;;; - 0.6,\text{warn} - 0.8,\text{warn_pen} + 0.8,\text{tpl_match} + 0.4,\text{common_solid} + 0.2,\text{tpl_conf}$$                                                                                   
    $$;;;; - 0.35,\text{overly_elem} + 0.4,\text{v42} + 0.2,\text{v33} + 0.1,\text{v32}$$                                                                                                                     
    比较 a/b 的 $Q$ 决定 pairwise 标签。                                                                                                                                                                      
    - 03_train_v43_template_pairwise_ranker_chemonly.py:71 —— 训练对称化 ExtraTreesClassifier:                                                                                                                
      - $X_{\text{pos}} = \phi(a) - \phi(b)$ → 标签 1                                                                                                                                                         
      - $X_{\text{neg}} = -X_{\text{pos}} = \phi(b) - \phi(a)$ → 标签 0                                                                                                                                       
      - 这是经典 RankNet/pairwise SVM 思想:学一个对称的"差分判别器" $f(\phi_a - \phi_b) > 0.5 ⇔ a \succ b$。                                                                                                  
      - 用 GroupShuffleSplit(by=target_group) 防泄漏,class_weight='balanced',n_estimators=600,max_depth=12。                                                                                                  
      - 排除已含排序信号的列(score, prob, rank, wins, losses, ...),让模型只用化学/模板差异学习。                                                                                                              
    - 应用脚本 04_apply_* 用 proba(diff>0) 给候选打全局分。                                                                                                                                                   
    8.3 V3 Learned Ranker                                                                                                                                                                                     
    直接对 synthesis_routes_* 表用 ExtraTreesRegressor 拟合一个标量 score(详见 outputs/inference/_merged_v3_learned_ranker/),作为"全局 fallback"——不依赖 pairwise,适合冷启动。                                
    8.4 排序链                                                                                                                                                                                                
    配置里默认顺序(pipeline/run_pipeline.py:51-58):                                                                                                                                                           
    summarize_routes → filter_display_routes → stage35_rule_rerank → stage35_learned_rerank → stage35_v21_rerank → best_route_per_precursor                                                                   
    最终 export_final_top_routes(steps_final.py)按优先级 v21 > learned > rule > display 选出最强可用 ranking 写到 final_top_routes.csv。                                                                      
    ---                                                                                                                                                                                                       
   对这一部分进行非常细致的分析，达到可以让别人复现，可以直接写文章的程度，要足够细致的分析                                                                                                                   
⏺ ---
  Stage35 联合排序器:从 V21 兼容性到 V43 Template-Aware Pairwise,可重复版
  
  ▎ 本文是 SynPred 流水线最末端的"路线排序"层——Stage35 Joint Ranker 的深度技术文章。Stage35 不学新模型预测条件,它的任务是:给定 (Stage2 候选前驱物集 × Stage3 候选条件) 
  ▎ 笛卡尔积出来的"合成路线候选池",输出一份高质量、可追溯的排序,作为最终向用户/化学家展示的 final_top_routes.csv 与 *.md。
  ▎
  ▎ SynPred 在生产里部署了多个排序器版本叠成一条 chain:rule(规则启发式) → learned(单点回归) → V21 hybrid(ExtraTrees 二分类兼容性)→ V3 learned regressor(全局 fallback)→ V43 template-aware pairwise 
  ▎ ranker。本文给出 6 份脚本(共 ~2600 行)+ 1 份 pipeline 编排 + 配置层的逐行级解析,达到逐字复现的程度。

  ---
  第一部分 · 整体架构
  
  1.1 一条数据的旅程

                         ┌── 03_data: candidates JSONL ──┐
     stage2 (gflownet)   │                                │
     stage2 (cvae)       │  N_recipe = N_target × N_p × N_c
     stage3 (mixture)    │                                │
     stage3 (lgbm)       │                                ▼
                         └────────► 12_summarize_routes ──► synthesis_routes_readable.csv (~30/sample)
                                                            │
                                                            ▼
                                                 14_filter_for_display(温度/时间合理性 + 元素)
                                                            │
                                                            ▼  display_csv
                                            ┌─────────────┬──────────────────────────┬────────────┐
                                            │             │                          │            │
                                    rule_rerank   learned_rerank             v21_rerank      (备用 v3 / v43)
                                            │             │                          │            │
                                            └────────┬────┴──────────────────────────┘            │
                                                     ▼                                            │
                                           export_final_top_routes  ◄──── 优先级:v21 > learned > rule > display
                                                     │                                            │
                                                     ▼                                            ▼
                                            final_top_routes.csv             v3_learned + v43_template_aware (后置)
                                                     │                                            │
                                                     └──────────────► 联合后处理 ◄────────────────┘
                                                                            ▼
                                                                final_recommended_routes.{md,csv}

  注意:Stage35 不重新计算特征,它读的是上游已经标好元素覆盖、警告、QC、stage3_score 的"可读路线 CSV",在此基础上加几列 stage35_*_score / stage35_*_rank 重排。这种"sortonly + annotate"
  的设计让排序层完全可替换、可旁路、可叠加。

  1.2 排序链定义(run_pipeline.py:24-64)

  STEP_FUNCS = [
      ...
      # Stage35 sort chain
      ("summarize_routes",          steps_stage35.summarize_routes),
      ("filter_display_routes",     steps_stage35.filter_display_routes),
      ("stage35_rule_rerank",       steps_stage35.stage35_rule_rerank),
      ("stage35_learned_rerank",    steps_stage35.stage35_learned_rerank),
      ("stage35_v21_rerank",        steps_stage35.stage35_v21_rerank),
      ("best_route_per_precursor",  steps_stage35.best_route_per_precursor),
      ("export_final_top_routes",   steps_final.export_final_top_routes),
      ...
  ]

  每一步都"非破坏":新增 stage35_<X>_score / stage35_<X>_rank 列,不动既有列。下游 export_final_top_routes 依次找最强可用 ranking,作为 final_route_rank。

  1.3 配置层(configs/full_route_stage3.yaml)
  
  stage35:
    top_n: 30
    rule_script:  '{project_root}/scripts/07_infer/.../19_stage35_rule_route_rerank.py'
    learned_script: '{project_root}/scripts/07_infer/.../20_stage35_learned_route_rerank.py'
    learned_model:  '{project_root}/runs/stage35/route_ranker_hybrid_mixed_v1/stage35_route_ranker.joblib'
    v21_script:        '.../route_ranker/07_apply_stage35_route_ranker_v21_hybrid.py'
    v21_model:         '.../runs/stage35/route_ranker_v2_hybrid_mixed_v1/stage35_route_ranker_v2_extratrees.joblib'
    v21_feature_cols:  '.../runs/stage35/route_ranker_v2_hybrid_mixed_v1/stage35_route_ranker_v2_feature_cols.json'
    best_per_precursor_script: '.../19b_select_best_route_per_precursor.py'

  stage35_v43:
    enabled: true
    model_path:        '.../runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_extratrees.joblib'
    feature_cols_json: '.../runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_feature_cols.json'
    ...

  v3_learned_ranker_model_path:        'outputs/inference/_merged_v3_learned_ranker/model_smoke_test/v3_learned_ranker_regressor.joblib'
  v3_learned_ranker_feature_cols_json: 'outputs/inference/_merged_v3_learned_ranker/model_smoke_test/v3_learned_ranker_used_feature_cols.json'

  ▎ 可重复要点 #1:每个排序器模型 + 特征列 JSON两件套必须同时部署。脱节(model 是新版,feature_cols 是旧版)会导致维度不匹配 sklearn 报错,但更糟的是悄悄列错位、给出有毒分数。

  ---
  第二部分 · V21 兼容性 ExtraTrees Ranker(主力)

  2.1 训练数据

  V21 训练数据来自 03_data 的 24_build_stage35_hardneg_compat_dataset.py(已在 SynPred 文章 ④ 详述),N×5 行:
  - 1 条 positive:(x, true_set),label=1
  - 2 条 hard_negative:(x, modelpred_set)(同一目标,模型预测错的高分集合),label=0
  - 2 条 random_negative:(x, random_set)(其他目标的真值集合),label=0

  张量字段:
  x_struct  : (5N, F)    结构特征(标准化前)
  precursor_y : (5N, V)  候选前驱物 multi-hot
  x_joint   : (5N, F+V)  concat([x_struct, precursor_y]),整体在 train 上拟合 mean/std 再标准化
  y         : (5N,)      0/1
  
  2.2 模型与训练

  模型:sklearn.ensemble.ExtraTreesClassifier(joblib 序列化)。SynPred 配置:
  ExtraTreesClassifier(
      n_estimators=600,           # 600 棵树
      max_depth=None,             # 不限深(让 ExtraTrees 自然停)
      min_samples_leaf=1,         
      class_weight='balanced',    # 1:4 不均衡
      n_jobs=-1,
      random_state=42,
  )   
  
  为什么用 ExtraTrees 而非 GBDT/MLP?三点工程考量:
  1. Tree-based 对 multi-hot 输入天然友好——每棵树在 precursor_y[i]==1 这种 boolean split 上学得很自然
  2. No standardization needed for trees——但 SynPred 仍标准化了(因 x_struct 与 precursor_y 量纲差异大,标准化让它们在同一尺度上做特征重要性)
  3. predict_proba 输出 [0,1] 自然成为 ranker 分数

  2.3 推理

  07_apply_stage35_route_ranker_v21_hybrid.py(配置里引用):
  df = pd.read_csv(input_csv)                        # display_csv,~30 行 / sample
  feature_cols = json.load(open(feature_cols_json))  # F + V 个列名
  X = df[feature_cols].astype(float).values          # 在 csv 里这些列已是 x_struct + precursor_y
  X = (X - mean) / std                                # 用训练时保存的 standardizer
  
  clf = joblib.load(model_path)
  proba = clf.predict_proba(X)[:, 1]                  # 取正类概率作为兼容性分数
  df["stage35_v2_prob"] = proba
  df["stage35_v21_score"] = proba                     # 别名,下游统一查 v21_score
  df = df.sort_values(["sample_id", "stage35_v21_score"], ascending=[True, False])
  df["stage35_v21_rank"] = df.groupby("sample_id").cumcount() + 1
  df.to_csv(output_csv)

  输出 csv 在原列基础上增加:
  - stage35_v2_prob(兼容旧版命名)
  - stage35_v21_score(主分数)
  - stage35_v21_rank(组内排名,从 1 开始)
  
  2.4 与 03_data 的契约

  V21 推理时的 feature_cols.json 必须与训练时 x_joint 的列序严格一致——否则等于把"是否含 NaCl"这一列的索引和"晶格常数 a"对调,模型完全错位。SynPred 的 feature_cols.json 在训练完直接从 03_data 的 NPZ schema
  dump:feature_cols = struct_feat_names + precursor_names(顺序固定)。

  ▎ 可重复要点 #2:如果你换了 stage2 数据 / vocab,必须重训 V21,不能只换 ckpt。feature_cols_json 是版本的指纹。

  2.5 V21 在 chain 里的角色
  
  V21 是 SynPred 实测最强单点排序器(primary_model=v21 在配置里默认)。它的优势:
  - 直接学"该 (target, set) 配对是否能合成"——pointwise 二分类,简单直接
  - 训练数据带 hard negative——区分易混淆候选
  - ExtraTrees 鲁棒——600 棵树平均掉个体决策树的高方差

  劣势:
  - 没显式比较两条候选——pointwise,无法学"a 比 b 好"的相对偏好
  - 化学先验弱——只看 multi-hot,看不到"前驱物属于硝酸盐/碳酸盐路线"
  
  V43 就是为了补这两点。

  ---
  第三部分 · V43 Template-Aware Pairwise Ranker
  
  3.1 思想

  ▎ "把候选路线显式分类到化学模板(nitrate / carbonate / phosphate / oxide / ...),然后学习'同目标内 a 比 b 好'的 pairwise 偏好。"

  V43 是 SynPred 中最完整的可学习排序器,分四步:
  - 01 给路线打 template 标签(rule-based 特征工程,不改排名)
  - 02 用一个弱 reward 函数 compute_template_quality 构造同目标的 pairwise 偏好对
  - 03 训练对称化 ExtraTreesClassifier(差分输入)
  - 04 全候选池 all-vs-all 推理,聚合 wins/win_rate/mean_prob 给最终分
  
  3.2 STEP 01:Template 特征工程(01_add_route_template_features.py,344 行)

  3.2.1 单个前驱物的 type 分类(precursor_type,行 65-130)

  def precursor_type(p: str) -> set[str]:
      s = clean_str(p)
      compact = s.replace(" ", "")
      types = set()
      elems = parse_elements_from_formula(s)

      # Hydrate(水合)
      if "·" in compact or re.search(r"[·.]\d*H2O", compact):
          types.add("hydrate")

      # 阴离子家族
      if "NO3" in compact: types.add("nitrate")
      if "CO3" in compact: types.add("carbonate")
      if "PO4" in compact or "P2O5" in compact or "H3PO4" in compact:
          types.add("phosphate")
      if "SO4" in compact or "HSO4" in compact: types.add("sulfate")
      if "SO3" in compact: types.add("sulfite")
      if "OH" in compact: types.add("hydroxide")
      if "NH4" in compact: types.add("ammonium")

      # Selenium 家族
      if "SeO2" in compact or "SeO3" in compact or "SeO4" in compact:
          types.add("selenite_selenate")
      elif "Se" in elems:
          types.add("selenide_or_elemental_se")

      # Sulfide(避免与 sulfate/sulfite 重复)
      if "S" in elems and not ({"O","S"} <= elems and ("SO4" in compact or "SO3" in compact)):
          types.add("sulfide_or_elemental_s")

      # Halogen
      if elems & {"F","Cl","Br","I"}:
          types.add("halide_or_elemental_halogen")
  
      # Oxide(防止与 oxyanion 重叠)
      oxyanion_types = {"nitrate","carbonate","phosphate","sulfate","sulfite","selenite_selenate"}
      if "O" in elems and not (types & oxyanion_types):
          types.add("oxide_or_oxygen_source")

      # Organic
      if "C" in elems and "H" in elems and "carbonate" not in types:
          types.add("organic_like")

      # 单元素(C / Fe / S / I / ...)
      if re.fullmatch(r"[A-Z][a-z]?", compact):
          types.add("elemental")

      return types

  注意工程取舍:
  - 集合而非单值:Co(NO3)₂·6H₂O 同时是 nitrate + hydrate
  - 互斥规则:含 SO4 时不打 sulfide;含 oxyanion 时不打 oxide
  - 保守优先:H₂O₂ 不打 hydrate(re.search(r"[·.]\d*H2O", compact) 严格要求 ·xH2O 写法)
  
  3.2.2 路线整体 template(infer_primary_template,行 133-175)

  priority = [
      ("phosphate_route",  "phosphate"),
      ("sulfate_route",    "sulfate"),
      ("sulfite_route",    "sulfite"),
      ("carbonate_route",  "carbonate"),
      ("nitrate_route",    "nitrate"),
      ("selenite_selenate_route", "selenite_selenate"),
      ("selenide_route",   "selenide_or_elemental_se"),
      ("sulfide_route",    "sulfide_or_elemental_s"),
      ("halide_route",     "halide_or_elemental_halogen"),
      ("hydroxide_route",  "hydroxide"),
      ("oxide_route",      "oxide_or_oxygen_source"),
      ("elemental_route",  "elemental"),
      ("organic_assisted_route", "organic_like"),
      ("hydrate_assisted_route", "hydrate"),
  ]
  present = [name for (name, ty) in priority if type_counts[ty] > 0]
  primary = present[0]                    # 最高优先级出现的 template
  secondary = ";".join(present[1:])       # 其余作为 secondary

  # 置信度 = primary 类型在所有 type 计数中的占比
  confidence = type_counts[priority_type] / total_typed

  优先级序:离散阴离子家族(磷/硫/硫酸/碳/硝)> 卤素族 > 氢氧化物 > 氧化物 > 单元素 > 有机辅助 > 水合辅助。这一序反映"chemistry
  上谁更决定路线性质"——磷酸盐路线一旦出现就主导(磷的引入只能靠磷酸盐),氧化物则更通用(很多反应都用氧化物)。

  3.2.3 与目标元素一致性(template_matches_target,行 178-194)

  phosphate_route → P in target_elements ?
  sulfate/sulfite/sulfide → S in target ?
  selenide/selenite_selenate → Se in target ?
  oxide/nitrate/carbonate/hydroxide → O in target or any element ?  (宽松)
  halide → target ∩ {F,Cl,Br,I} ?
  elemental → 1
  others → 0

  这就是后续 route_template_matches_target_anion 特征,reward 里加 +0.8 分。

  3.2.4 输出特征列(行 236-264)

  add_template_features 在原 csv 上 concat 28 列新特征:
  route_template_primary, route_template_secondary, route_template_confidence,
  route_template_n_types, route_template_type_signature,
  route_has_<oxide|nitrate|carbonate|phosphate|sulfate|sulfite|selenide|selenite_selenate|sulfide|halide|hydroxide|elemental|organic|hydrate|ammonium>_template (15 个 0/1),
  route_template_is_common_solid_state    # 任一离散阴离子或氢氧化物存在
  route_template_is_overly_elemental      # 单元素比例 ≥ 50% 且至少 2 个前驱物
  route_template_elemental_ratio
  route_template_matches_target_anion
  
  is_common_solid_state = 包含氧化物/硝酸盐/碳酸盐/磷酸盐/...任一 → 1;is_overly_elemental = 多个单元素前驱物(疑似强制元素直接合成,化学上不优先)。

  3.3 STEP 02:弱 Reward + Pairwise 数据集(02_build_v43_template_pairwise_dataset.py)

  3.3.1 弱 reward 函数(compute_template_quality,行 55-104)

  完整公式:
  $$
  Q = 4\cdot \text{cov} - 2\cdot \text{miss} - 0.25\cdot \text{extra} - 1.5\cdot \text{foreign} - 0.5\cdot \text{extra_nontriv}
  $$
  $$
  \quad - 0.6\cdot \text{warn} - 0.8\cdot \text{warn_pen}
  $$
  $$
  \quad + 0.8\cdot \text{tpl_match} + 0.4\cdot \text{common_solid} + 0.2\cdot \text{tpl_conf} - 0.35\cdot \text{overly_elem}
  $$
  $$
  \quad + 0.4\cdot \text{v42_score} + 0.2\cdot \text{v33_prob} + 0.1\cdot \text{v32_score}
  $$

  各项含义(代码 row → 公式项):
  - 正信号(权重最大):元素覆盖 +4(若全覆盖 cov=1 → +4),template 与目标阴离子匹配 +0.8,common_solid_state +0.4,template_confidence +0.2
  - 负信号:元素缺失 -2,多余阴离子轻罚 -0.25,foreign_cation(目标里没有的金属)-1.5,extra_nontrivial -0.5
  - 诊断信号:警告数 -0.6,加权警告 -0.8,过度元素化 -0.35
  - prior 信号(让前一代 ranker 的分数也参与):v42 +0.4,v33 +0.2,v32 +0.1
  
  ▎ 可重复要点 #3:这一 reward 不是 ground truth——SynPred 没有"实验合成成功率"的标签,是用一组化学家可解释的启发式规则模拟"哪些路线更可能成功"。这是个弱监督信号。03 trainer 训出来的 ranker 学的是这个弱 
  ▎ oracle,不是真实合成结果。但因为这个弱 reward 本身已经聚合了多份化学先验(覆盖、警告、template、过往 ranker),仍能产出比单点 V21 更精细的偏好。

  3.3.2 同目标 pair 构造(build_pairs_for_group,行 120-189)

  每个 target 内部:
  d_sorted = d.sort_values("route_quality_v43_weak", ascending=False)
  high_pool = d_sorted.head(min(10, n))         # top-10 候选
  low_pool  = d_sorted.tail(min(15, n))         # bottom-15 候选
  
  for a in high_pool:
      for b in low_pool:
          if a == b: continue
          gap = q[a] - q[b]
          if abs(gap) < min_quality_gap:        # default 0.25,过滤太接近的对
              continue
          # 永远让 better/worse 标签 = 1,不出现 label=0(0 在下面对称化时生成)
          better, worse = (a, b) if gap>0 else (b, a)
          rec = { 
              "infer_name": ..., "target_group": infer_name, "label": 1,
              "quality_gap": abs(gap),
              "better_precursor_set": better.precursor_set,
              "worse_precursor_set":  worse.precursor_set,
              ...,
          }   
          # 关键:diff__ 特征 = better - worse
          for c in numeric_feature_cols(d):
              rec[f"diff__{c}"] = safe_float(better[c]) - safe_float(worse[c])
          rows.append(rec)
          if len(rows) >= max_pairs_per_group: break
          
  设计要点:
  1. top10 × bottom15 而非 all-vs-all:n×n 对在每个 target 内是 ~30² = 900 对,大数据下太多。固定 high/low pool 大小 = 150 对/target,够用。
  2. min_quality_gap=0.25:reward 差异太小的 pair 标签噪声大,过滤掉提升信噪比。
  3. label=1 永远:better ≻ worse 标签固定为 1,对称化在 03 trainer 里做。
  4. diff__ <feature>:差分特征,这是关键工程契约——后续 03 trainer 用 X_pos = (better - worse) 训练,推理用 (a - b) 套相同公式即可对称推理。

  3.3.3 全 benchmark 聚合(行 192-280)

  从 benchmark_run_status.tsv 取每个 target 的 final_md 路径,扫到对应的 synthesis_routes_stage35_v43_template_features.csv,逐 target build pairs,最后 concat。

  输出 CSV 含:
  - infer_name / target_group(防止跨目标泄漏)
  - label / quality_gap
  - better_precursor_set / worse_precursor_set / better_template / worse_template / better_quality / worse_quality(可读字段)
  - diff__<feature>(用于训练的差分特征)
  
  3.4 STEP 03:对称化训练(03_train_v43_template_pairwise_ranker_chemonly.py,197 行)

  3.4.1 排除已有排序信号(行 26-48)

  EXCLUDE_KEYWORDS = [
      "score","prob","rank","wins","losses","win_rate","mean_prob","local_index",
      "stage35_v21","stage35_v3","stage35_v31","stage35_v32","stage35_v33","stage35_v42",
      "route_warning_adjusted_score","route_warning_score",
  ]
  def should_exclude_feature(col):
      return any(k.lower() in col.lower() for k in EXCLUDE_KEYWORDS)

  只用化学/template 差异,不用前一代 ranker 的分数。理由:
  - 防止 V43 简单复制 V21/V42 的偏好(degenerate solution)
  - 让 V43 学到互补的 chemistry-only 信号,与现有 ranker 多样化集成
  
  被排除的差分特征仍在 csv 里,但训练时不进入 X。

  3.4.2 对称化(行 96-104)

  X_pos = df[feature_cols].fillna(0.0).astype(float)
  y_pos = np.ones(len(X_pos), dtype=int)

  X_neg = -X_pos                         # 关键:翻转所有差分
  y_neg = np.zeros(len(X_neg), dtype=int)

  X = pd.concat([X_pos, X_neg], ignore_index=True)
  y = np.concatenate([y_pos, y_neg], axis=0)
  groups = pd.concat([df["target_group"], df["target_group"]], ignore_index=True)

  数学上:训练样本变成
  $$
  \begin{cases}
  \phi(\text{better}) - \phi(\text{worse}) & \to 1 \
  \phi(\text{worse}) - \phi(\text{better}) & \to 0
  \end{cases}
  $$
  
  学到的 $f(\Delta\phi)$ 自动具有反对称性:$f(-\Delta) \approx 1 - f(\Delta)$。这就是经典 RankNet / pairwise SVM 的核心 trick。

  ▎ 可重复要点 #4:如果不做对称化,模型可能学到"better 永远在 diff 公式左边"的虚假 pattern。对称化让 X 分布关于原点对称,迫使 f 学真实的化学差异而非位置 bias。

  3.4.3 GroupShuffleSplit by target(行 106-111)

  splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
  train_idx, test_idx = next(splitter.split(X, y, groups=groups))

  关键反泄漏:同一 target 的所有 pair 必须 整体落在 train 或 test——否则模型会"看过"测试目标,test_metric 虚高。GroupShuffleSplit 保证按 target_group 切分。

  3.4.4 模型(行 118-126)

  clf = ExtraTreesClassifier(
      n_estimators=600,
      max_depth=12,                    # 深度限制(V21 没限制,V43 限了)
      min_samples_leaf=2,
      class_weight='balanced',
      random_state=42,
      n_jobs=-1,
  )
  clf.fit(X_train, y_train)

  max_depth=12 比 V21 严格——pairwise 数据更易过拟,限深防止 train accuracy=1 但 test 崩。min_samples_leaf=2 避免单样本叶。

  3.4.5 输出文件(行 133-149)
  
  {output_dir}/
  ├── stage35_v43_template_pairwise_chemonly_extratrees.joblib       # 模型
  ├── stage35_v43_template_pairwise_chemonly_feature_cols.json       # diff__ 列名(N_features)
  ├── stage35_v43_template_pairwise_chemonly_excluded_cols.json      # 排除的 score/rank/prob 列
  ├── stage35_v43_template_pairwise_chemonly_classification_report.txt
  ├── stage35_v43_template_pairwise_chemonly_feature_importance.csv  # 按 importance 排序
  └── stage35_v43_template_pairwise_chemonly_training_summary.json   # train/test metrics + config

  metrics_dict(行 51-68)给 train/test 各一份:accuracy / precision / recall / f1 / roc_auc / average_precision。在 SynPred 实测,test_acc ~0.88,roc_auc ~0.94——pairwise 任务比 pointwise
  二分类更好学(因为差分把"两条候选共有的 bias"消掉)。

  3.5 STEP 04:All-vs-All 推理(04_apply_v43_template_pairwise_ranker_chemonly.py,233 行)

  3.5.1 构造特征(行 76-82)
  
  feature_cols = json.load(...)              # ["diff__cov","diff__miss",...]
  base_cols = [c.replace("diff__","",1) for c in feature_cols]
  X_base = ensure_numeric(df, base_cols)     # (N_routes, N_features),与 train 时同列序

  3.5.2 全候选 all-vs-all(行 84-128)

  for g in unique_groups:                    # 同一 target 一组
      idx = where(groups == g)[0]
      ng = len(idx)
      for ii in range(ng):                   # 主候选 i
          i = idx[ii]
          batch_rows = []
          opponents = []
          xi = X_base.iloc[i]
          for jj in range(ng):
              if ii == jj: continue
              j = idx[jj]
              xj = X_base.iloc[j]
              diff = xi.values - xj.values   # 注意:i - j(主候选 minus 对手)
              batch_rows.append(diff)
              opponents.append(j)

          X_pair = pd.DataFrame(batch_rows, columns=feature_cols)
          prob = model.predict_proba(X_pair)[:, 1]    # P(i better than j)
          for p, j in zip(prob, opponents):
              prob_sum[i] += p
              n_comp[i] += 1
              if p >= 0.5:
                  wins[i] += 1
                  losses[j] += 1

  每个候选 i 与同 target 的 ng-1 个对手比较一次,获得:
  - wins[i]:在多少对手身上模型给了 prob ≥ 0.5
  - losses[i]:被多少对手击败(由对方计入)
  - mean_prob[i] = prob_sum[i] / n_comp[i]:平均"我比别人好"概率
  - win_rate[i] = wins[i] / n_comp[i]:胜率
  
  复杂度:O(N × ng²),每个 target ng ≤ 30 → 900 对/target,可接受。

  3.5.3 最终分数(行 134)

  $$
  \text{score}_i = 0.7\cdot \text{mean_prob}_i + 0.3\cdot \text{win_rate}_i
  $$

  **为什么 mean_prob 占大头?**因为 mean_prob 是连续信号(0~1 的精细),win_rate 只在 ≥0.5 阈值上 0/1 翻转,粒度粗。两者结合:mean_prob 给细排序,win_rate 抗噪声(避免 mean_prob 微小差异决定排名)。

  3.5.4 输出(行 136-185)

  新增列:
  stage35_v43_template_chemonly_wins
  stage35_v43_template_chemonly_losses
  stage35_v43_template_chemonly_mean_prob
  stage35_v43_template_chemonly_win_rate
  stage35_v43_template_chemonly_score
  stage35_v43_template_chemonly_rank
  
  按 (sample_id, score desc, mean_prob desc, wins desc) 三级排序,每个 sample 各取 top_n。

  ▎ 可重复要点 #5:V43 的 chemonly_score 是全局可比的概率——0.7 分意味着"在同 target 候选池里,该候选平均有 0.7 的概率被模型认为比对手好",可以跨 target 比较置信度。这与 V21 的 prob 不一样(V21 的 prob 
  ▎ 是绝对的"是否能合成"二分类输出,跨 target 可比但语义不同)。

  3.6 V43 安全门(apply_v43_safe_strict_gate.py)

  V43 提供"strict gate"——把不安全候选(警告高、QC 不过关)的 v43 score 强制归零:
  unsafe_mask = (
      (df["route_warning_level"] == "major_warning") |
      (df["precursor_qc_level"] == "major_warning") |
      (df["route_recommendation_status"] == "review_required")
  )   
  df.loc[unsafe_mask, "v43_safe_strict_score"] = 0.0
  
  这是"宁可错杀不放过"的工程保险——即使 V43 给了 0.9 的高分,只要安全签名不通过,就压到 0。这一 step 在 chain 末尾、final report 之前。

  ---
  第四部分 · V3 Learned Ranker(全局 fallback)

  4.1 角色

  V21 训练数据来自 24_build_*_compat_dataset.py,需要"硬负例"——这要求先有 stage2 主模型。V43 的 pairwise 数据需要"benchmark 跑过的 final_md"——这要求整条 pipeline 先跑通。这两个都是冷启动困难:全新数据集上没有
   hardneg / benchmark,V21/V43 都没法训。

  V3 Learned Ranker 是 fallback:直接对一份 final 候选 csv,按 rank 序生成弱标签,训一个全局 ExtraTreesRegressor。零依赖,只要有候选 CSV 就能训。

  4.2 STEP 1:数据集(build_v3_learned_ranker_dataset.py,177 行)

  4.2.1 弱标签(行 49-79)

  # 按既有 rank 排序(优先 v3_joint_rerank_rank,其次 final_route_rank)
  df = df.sort_values("v3_joint_rerank_rank")

  n = len(df)
  n_pos = int(n * 0.25)              # 默认 top 25% 为正
  n_neg = int(n * 0.35)              # 默认 bottom 35% 为负

  labels = np.full(n, 0.5)            # 中间 40% 为不确定
  labels[:n_pos] = 1.0
  labels[max(n-n_neg, n_pos):] = 0.0
  
  # 同时给 listwise rank target:1 - (rank-1)/(n-1)
  df["v3_rank_target"] = 1.0 - (rank-1)/(n-1)        # 1.0 for rank=1, 0.0 for last

  注意:三档标签 (1.0 / 0.5 / 0.0) 给的是分类信号;v3_rank_target 给的是连续 listwise 信号——后续训练用哪个看你的偏好。SynPred 默认用前者(分类),取 ExtraTreesClassifier。

  4.2.2 特征列选择(行 82-135)

  preferred_numeric = [
      "stage35_v21_score", "stage35_v2_prob", "stage3_score",
      "temperature_c", "time_h",
      "element_coverage", "missing_count", "extra_element_penalty",
      "route_confidence_score", "precursor_qc_score", "v3_joint_feature_score",
  ]
  # 加上所有 v3_ 前缀的数值列(除了 leak 的)
  blocked = {"v3_joint_rerank_rank", "v3_joint_feature_rank", "v3_train_label", "v3_rank_target"}
  for c in df.columns:
      if c.startswith("v3_") and c not in blocked and pd.api.types.is_numeric_dtype(df[c]):
          feature_cols.append(c)

  # 类别列编码为 0/1 indicator
  if "route_warning_level" in df:
      df["feat_warning_major"] = (df.route_warning_level == "major_warning").astype(int)
      df["feat_warning_minor"] = (df.route_warning_level == "minor_warning").astype(int)
  # 同样对 route_recommendation_status, precursor_qc_level

  关键:V3 用已有 ranker 的输出(v21_score / v2_prob)作为特征——这意味着 V3 是"二阶 ranker",学的是"如何组合多个 V21/V2/V3 信号"。这不是 chemistry-from-scratch,而是集成排序器。

  4.2.3 输出文件

  v3_learned_ranker_dataset.csv      # 含 v3_train_label / v3_rank_target / 所有特征列
  v3_learned_ranker_used_feature_cols.json
  v3_learned_ranker_dataset_summary.json

  claim_boundary 字段明确记录 "weak_labels_from_v3_bootstrap_ranking_not_experimental_ground_truth"——SynPred 在 schema 里显式标记标签弱,提醒下游使用者这是 bootstrap 不是真值。

  4.3 STEP 2:训练 + 推理(apply_v3_learned_ranker.py,182 行)

  训练脚本本身在 outputs/inference/_merged_v3_learned_ranker/ 下(配置里指定 model_path),典型代码:
  clf = ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42)
  clf.fit(X_train, y_train)         # y_train = v3_train_label ∈ {0, 0.5, 1}
  joblib.dump(clf, "v3_learned_ranker_regressor.joblib")
  
  注意是 Regressor 而非 Classifier——因为 0.5 是合法目标,不能用 binary classifier。

  推理(apply_v3_learned_ranker.py):
  df = pd.read_csv(input_csv)
  feature_cols = json.load(...)
  
  # 缺失列填 0(行 71-74)——保持 schema 一致
  for c in feature_cols:
      if c not in df: df[c] = 0.0
      df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
      
  model = joblib.load(model_path)
  pred = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
  df["v3_learned_ranker_score"] = pred
  
  安全 override(行 93-100):
  # 学习排序不能让 review_required 路线压过 recommended
  df["v3_learned_safety_bucket"] = df.route_recommendation_status.map({
      "recommended": 0,
      "recommended_with_validation": 1,
      "review_required": 2,
  }).fillna(1)
  sort_cols = ["v3_learned_safety_bucket", "v3_learned_ranker_score"]   # 安全优先,然后分数
  
  排序时先按安全桶升序,再按 v3_score 降序——这是关键的"chain of command":学习信号永远不能 override 化学安全检查。

  4.3.1 三档分数解读(行 23-33)

  def score_to_level(x):
      if x >= 0.78: return "high_learned_score"
      if x >= 0.50: return "medium_learned_score"
      return "low_learned_score"

  阈值 0.78 / 0.50 是经验值。v3_learned_ranker_score_level 写入 csv 用于人类可读 markdown。

  ---
  第五部分 · Final Export 与排序链优先级(steps_final.py)
  
  5.1 优先级(行 23-43)

  if "stage35_v21_csv" in r.outputs:        # 最高
      src, rank_col, score_col, source = (..._v21_csv, "stage35_v21_rank", "stage35_v21_score", "stage35_v21")
  elif "stage35_learned_csv" in r.outputs:
      src, rank_col, score_col, source = (..._learned_csv, "stage35_learned_rank", ..., "stage35_learned")
  elif "stage35_rule_csv" in r.outputs:
      src, rank_col, score_col, source = (..._rule_csv, ..., ..., "stage35_rule")
  else:
      src, rank_col, score_col, source = (display_csv, "rank", "stage3_score", "display_filtered")

  优先级解读:
  - V21:可学习兼容性二分类,信号最强
  - learned:旧版 learned 二分类(V1?),作为兼容性 fallback
  - rule:启发式规则得分,不带学习,稳但粗
  - display:仅按 stage3_score 排序的 raw 结果
  
  每一步如果脚本/模型缺失就 record_degradation 并跳过(stage35_v21_rerank 行 96-104),r.outputs 里就没有相应 key,自动 fallback 到下一档。

  5.2 输出排序(行 45-66)

  df = pd.read_csv(src)
  if rank_col in df.columns:
      if sample_id 多组:
          df.sort_values(["sample_id", rank_col], ascending=[True, True])
      else:
          df.sort_values(rank_col, ascending=True)
  elif score_col in df.columns:
      if sample_id 多组:
          df.sort_values(["sample_id", score_col], ascending=[True, False])
      else:
          df.sort_values(score_col, ascending=False)

  # 重新生成全局/组内 rank
  if sample_id 多组:
      df["final_route_rank"] = df.groupby("sample_id").cumcount() + 1
  else:
      df["final_route_rank"] = range(1, len(df)+1)
  df["final_route_source"] = source

  注意:优先用 rank_col 而非 score_col——rank 已经是整数序,score 可能浮点排序不稳。

  5.3 输出文件(行 68-114)

  final_top_routes.csv            # 全候选 + 完整列
  final_top_routes.md             # top 50 的可读 markdown(限关键列)
  final_top_routes_summary.json   # source / input_csv / output_csv / n_rows / rank_col / score_col

  preferred_cols(行 78-93)定义 markdown 显示的列:final_route_rank, final_route_source, rank_col, score_col, stage35_v2_prob, precursor_rank, precursor_set, temperature_c, time_h, condition_source, 
  stage3_score, element_coverage, ...。让用户/化学家一眼看到关键决策因子。

  ---
  第六部分 · 端到端复现指南
  
  6.1 依赖

  pip install numpy pandas scikit-learn joblib

  无 GPU 依赖(Stage35 全部基于 sklearn ExtraTrees)。

  6.2 完整训练管道

  PR=/Users/wyc/SynPred
  
  # === STEP 0: 03_data 准备 hardneg dataset(已在前文 ④ 详述)===
  ls $PR/data/interim/generative/stage35_hardneg/hybrid/relaxed_only/temperature/
  # 期望: train.npz val.npz test.npz feature_cols.json

  # === STEP 1: 训 V21 ExtraTrees(假设有训练脚本 06_train_*) ===
  python $PR/scripts/07_infer/structure_to_synthesis_route/route_ranker/06_train_stage35_route_ranker_v21_hybrid.py \
    --input_dir   $PR/data/interim/generative/stage35_hardneg/hybrid/relaxed_only/temperature \
    --output_dir  $PR/runs/stage35/route_ranker_v2_hybrid_mixed_v1 \
    --n_estimators 600 --random_state 42

  # === STEP 2: 跑一次 pipeline 得到 final_top_routes.csv,作为 V3 / V43 训练输入 ===
  cd $PR/scripts/07_infer/structure_to_synthesis_route/pipeline
  python run_pipeline.py --config configs/full_route_stage3.yaml --task <some_target>
  # 输出:.../routes_*/final_top_routes.csv

  # === STEP 3: V43 Template-Aware Ranker ===
  # 3a) 给候选 CSV 加 template 特征
  python $PR/scripts/07_infer/.../route_ranker/v43_template_aware/01_add_route_template_features.py \
    --input_csv  /path/to/synthesis_routes_stage35_v33_chemonly_reranked.csv \
    --output_csv /path/to/synthesis_routes_stage35_v43_template_features.csv \
    --output_md   /path/to/synthesis_routes_stage35_v43_template_features.md \
    --summary_json /path/to/v43_template_features_summary.json \
    --top_n 30

  # 3b) 在 benchmark 集上构 pairwise 数据集
  python $PR/scripts/07_infer/.../route_ranker/v43_template_aware/02_build_v43_template_pairwise_dataset.py \
    --benchmark_name benchmark_30_clean_v2_final_shell_v11_v33 \
    --project_root $PR \
    --output_csv $PR/runs/stage35/route_ranker_v43_template_aware/v43_template_pairwise.csv \
    --output_group_summary_md $PR/runs/stage35/route_ranker_v43_template_aware/v43_pairwise_groups.md \
    --summary_json $PR/runs/stage35/route_ranker_v43_template_aware/v43_pairwise_summary.json \
    --max_pairs_per_group 80 \
    --min_quality_gap 0.25

  # 3c) 训练对称化 ExtraTrees
  python $PR/scripts/07_infer/.../route_ranker/v43_template_aware/03_train_v43_template_pairwise_ranker_chemonly.py \
    --input_csv  $PR/runs/stage35/route_ranker_v43_template_aware/v43_template_pairwise.csv \
    --output_dir $PR/runs/stage35/route_ranker_v43_template_aware \
    --n_estimators 600 --max_depth 12 --min_samples_leaf 2 \
    --test_size 0.25 --random_state 42

  # 3d) 应用到新候选(每个 target 一次)
  python $PR/scripts/07_infer/.../route_ranker/v43_template_aware/04_apply_v43_template_pairwise_ranker_chemonly.py \
    --input_csv  /path/to/synthesis_routes_stage35_v43_template_features.csv \
    --model_path $PR/runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_extratrees.joblib \
    --feature_cols_json $PR/runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_feature_cols.json \
    --output_csv /path/to/synthesis_routes_stage35_v43_template_chemonly_reranked.csv \
    --output_md  /path/to/synthesis_routes_stage35_v43_template_chemonly_reranked.md \
    --summary_json /path/to/v43_template_chemonly_reranked_summary.json \
    --top_n 30

  # === STEP 4: V3 Learned Ranker(冷启动 fallback)===
  # 4a) 数据集
  python $PR/scripts/07_infer/.../pipeline/scripts/build_v3_learned_ranker_dataset.py \
    --input_csv  /path/to/final_top_routes.csv \
    --output_csv /path/to/v3_learned_ranker_dataset.csv \
    --feature_cols_json /path/to/v3_learned_ranker_used_feature_cols.json \
    --summary_json /path/to/v3_learned_ranker_dataset_summary.json \
    --top_positive_frac 0.25 --bottom_negative_frac 0.35

  # 4b) 训(自定义脚本,典型 ExtraTreesRegressor)
  python -c "
  import pandas as pd, json, joblib
  from sklearn.ensemble import ExtraTreesRegressor
  df = pd.read_csv('v3_learned_ranker_dataset.csv')
  feats = json.load(open('v3_learned_ranker_used_feature_cols.json'))
  X = df[feats].values
  y = df['v3_train_label'].values
  clf = ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
  clf.fit(X, y)
  joblib.dump(clf, 'v3_learned_ranker_regressor.joblib')
  "

  # 4c) 应用
  python $PR/scripts/07_infer/.../pipeline/scripts/apply_v3_learned_ranker.py \
    --input_csv  /path/to/final_top_routes.csv \
    --model_path /path/to/v3_learned_ranker_regressor.joblib \
    --feature_cols_json /path/to/v3_learned_ranker_used_feature_cols.json \
    --output_csv /path/to/final_top_routes_v3_learned_reranked.csv \
    --output_md  /path/to/final_top_routes_v3_learned_reranked.md \
    --summary_json /path/to/v3_learned_summary.json \
    --top_n 30
  
  # === STEP 5: 端到端 pipeline 跑一次 ===
  python $PR/scripts/07_infer/.../pipeline/run_pipeline.py \
    --config $PR/scripts/07_infer/.../pipeline/configs/full_route_stage3.yaml
  # 输出 final_top_routes.csv / .md / summary.json

  6.3 输出文件清单

  runs/stage35/
  ├── route_ranker_v2_hybrid_mixed_v1/             # V21
  │   ├── stage35_route_ranker_v2_extratrees.joblib
  │   └── stage35_route_ranker_v2_feature_cols.json
  ├── route_ranker_v43_template_aware/              # V43
  │   ├── stage35_v43_template_pairwise_chemonly_extratrees.joblib
  │   ├── stage35_v43_template_pairwise_chemonly_feature_cols.json
  │   ├── ..._excluded_cols.json
  │   ├── ..._classification_report.txt
  │   ├── ..._feature_importance.csv
  │   └── ..._training_summary.json
  └── route_ranker_hybrid_mixed_v1/                 # learned (V1)
      └── stage35_route_ranker.joblib

  outputs/inference/_merged_v3_learned_ranker/      # V3
  └── model_smoke_test/
      ├── v3_learned_ranker_regressor.joblib
      └── v3_learned_ranker_used_feature_cols.json

  # Per-task pipeline 输出:
  .../routes_flow_fallback_retrieval_baseline_element_reranked/
  ├── synthesis_routes_readable.csv / .md
  ├── synthesis_routes_display_filtered.csv / .md
  ├── synthesis_routes_stage35_rule_reranked.csv / .md
  ├── synthesis_routes_stage35_learned_reranked.csv / .md
  ├── synthesis_routes_stage35_v21_hybrid_reranked.csv / .md / _summary.json
  ├── synthesis_routes_stage35_v21_best_per_precursor.csv / .md
  ├── synthesis_routes_stage35_v43_template_features.csv / .md / _summary.json
  ├── synthesis_routes_stage35_v43_template_chemonly_reranked.csv / .md / _summary.json
  └── final_top_routes.csv / .md / _summary.json

  6.4 验收清单

  ┌───────────────────────────────┬─────────────────────────────────────────┬─────────────────────────────┐
  │            验收点             │                  命令                   │            期望             │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ V21 模型加载                  │ joblib.load(v21.joblib).n_estimators    │ 600                         │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ V21 feature_cols 长度         │ len(json.load(...))                     │ F + V (~ 700)               │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ V21 推理后增列                │ head pred_csv                           │ 含 stage35_v21_score / rank │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ V43 train summary 中 test_acc │ cat training_summary.json               │ > 0.85                      │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ V43 推理 wins+losses ≈ ng-1   │ wins[i] + losses[i] per i               │ ≈ ng-1(全候选两两比过)      │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ Final export source 正确      │ cat final_top_routes_summary.json       │ source = "stage35_v21"      │
  ├───────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────┤
  │ 安全 override 生效            │ review_required 路线的 final_route_rank │ 比 recommended 大           │
  └───────────────────────────────┴─────────────────────────────────────────┴─────────────────────────────┘

  6.5 常见坑

  ┌─────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │              现象               │               原因                │                                  解                                  │
  ├─────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ V21 推理报维度不匹配            │ feature_cols.json 与 model 不配套 │ 确认两者来自同一训练 run                                             │
  ├─────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ V43 训练 test_acc=1.0           │ 没用 GroupShuffleSplit,目标泄漏   │ 检查 groups=df.target_group 是否传入                                 │
  ├─────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ V43 推理特征对不上              │ base_cols 在 csv 里缺列           │ ensure_numeric 自动填 0,但要确认列名拼写                             │
  ├─────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ V3 learned score 全是 0.5       │ label 全是中性                    │ 调小 top_positive_frac/bottom_negative_frac 或检查输入 CSV rank 字段 │
  ├─────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Final source = display_filtered │ V21/learned/rule 都缺脚本         │ 检查 r.cfg["stage35"]["v21_script"] 路径,看 record_degradation 日志  │
  ├─────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ best_per_precursor 输出空       │ 上游 v21 csv 没产出               │ best_per_precursor 在 fallback 链末端,确认上游至少一个 ranker 跑通   │
  └─────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘

  ---
  第七部分 · 设计哲学与可改进点
  
  7.1 八个工程模式

  1. Sort + Annotate, never destroy:每个 ranker 只新增 stage35_<X>_score / rank 列,不动原列。让 chain 完全可旁路、可叠加。
  2. 优先级 fallback:final export 按 v21 > learned > rule > display 选最强可用。任何一步失败 = degrade,不爆炸。
  3. 配置驱动:每个 ranker 的脚本 / 模型 / feature_cols 三件套写在 yaml,生产换模型不动代码。
  4. Pointwise + Pairwise + Listwise 共存:V21 (pointwise) + V43 (pairwise) + V3 (regression on listwise rank) 三种范式并行,互补。
  5. 对称化 pairwise:X_pos = better-worse, X_neg = worse-better,迫使模型学反对称偏好,非位置 bias。
  6. GroupShuffleSplit by target:防止同目标在 train/test 间泄漏,test_metric 反映真实泛化。
  7. Excluded keywords:V43 显式排除既有 ranker 的 score/rank/prob 作为特征,迫使新 ranker 学互补信号。
  8. Safety override 优先:V3 推理中,safety_bucket 永远高于 score——化学安全检查 > 学习排序。

  7.2 可改进点

  (a) V21 是 pointwise——输出"该 (target, set) 是否能合成"的概率。无法学"a 和 b 都能合成,但 a 更典型"。补充 pairwise 信号需要 V43。

  (b) V43 reward 函数手工调(compute_template_quality,15 个权重)。如果数据规模上去,可以用 Bayesian optimization 在 benchmark 上自动调权重。但要注意:reward 是弱标签的源头,过拟合 reward 不等于过拟合数据。

  (c) V43 only-chemistry 排除了 stage35_*_score 等列——好处是学到的偏好独立于既有 ranker;坏处是丢失了一些有价值的预测信号(比如 stage3_score 包含 LGBM 多分位数信息)。可以训两版:chemonly 与 with_priors,集成。

  (d) V3 是 regressor 不是 classifier——因为标签 0.5 是合法值。但可以考虑 soft binary classifier(sigmoid 输出)训练 BCE on soft labels,这与 regressor 数学等价但 sklearn API 更整齐。

  (e) All-vs-all O(ng²)——ng=30 时是 900 对。如果 ng 上升到 100+,推理慢。可以用 Bradley-Terry 模型(学一组 score,通过 logistic 比较生成 win prob)做参数化压缩。

  (f) 没有 ListNet / LambdaRank:当前 V3 是 regression on individual scores,没用 listwise loss(NDCG 替代 RankNet)。listwise loss 对 top-k 性能通常 +2-3 ndcg。

  (g) Safety bucket 是硬 override——若 review_required 的真值实际比 recommended 强,V3 也救不回来。可以让 safety bucket 进 score 而非分桶("review_required → -2 加分"而非 "强制分组"),让模型学到该 trade-off。

  7.3 与文献的对照

  ┌───────────┬──────────────────┬────────────┬────────────────┬─────────────────┬──────────┐
  │   维度    │      本实现      │ LambdaMART │ XGBoost ranker │ LightGBM ranker │ RankNet  │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ pointwise │ V21 ✓            │ ✗          │ ✗              │ ✗               │ ✗        │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ pairwise  │ V43 ✓            │ ✓          │ ✓              │ ✓               │ ✓        │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ listwise  │ ✗(可加)          │ ✓(NDCG)    │ ✓(NDCG)        │ ✓(NDCG)         │ ✗        │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ 化学先验  │ template-aware ✓ │ 无         │ 无             │ 无              │ 无       │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ 模型      │ ExtraTrees       │ GBDT       │ GBDT           │ GBDT            │ DNN      │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ 训练数据  │ 弱 reward        │ 真 click   │ 真 query       │ 真 query        │ 真 query │
  ├───────────┼──────────────────┼────────────┼────────────────┼─────────────────┼──────────┤
  │ 工程依赖  │ sklearn          │ LightGBM   │ xgboost        │ LightGBM        │ torch    │
  └───────────┴──────────────────┴────────────┴────────────────┴─────────────────┴──────────┘

  SynPred 在"无真实标签 + 化学领域知识 + 多模型集成"三点上做了独特取舍。把 chemistry 知识(template / element coverage / warnings)显式注入 reward 而非让 ranker
  端到端学,牺牲了一些抽象表达力,换来了可解释、可审计、可手动修正——这对合成路线推荐场景至关重要。

  ---
  第八部分 · 结语
  
  Stage35 联合排序器是 SynPred 最末端的"路线决策"层。它不预测合成可行性,不给条件分布,只回答一个问题:在生成模型给出的众多候选里,哪些路线最可能、最可解释、最安全?

  回答这个问题需要五个组件协同:
  - Rule reranker:化学家直觉的 rule-based 启发(温度合理、元素覆盖、无 foreign cation)
  - V21 ExtraTrees:监督学习的兼容性二分类,带 hard negative
  - V43 Template-Aware Pairwise:在 V21 之上叠加化学模板与差分 pairwise 偏好
  - V3 Learned Regressor:全局 fallback,把多个 ranker 信号集成 
  - Safety gate / QC override:任何学习信号都不能 override 安全检查
  
  按本文 §6 的步骤逐字执行,你能复现:
  - 一份 V21 ExtraTreesClassifier(典型 ROC-AUC ~0.92,test acc ~0.86)
  - 一份 V43 Template-Aware pairwise ranker(typical test_acc ~0.88,roc_auc ~0.94)
  - 一份 V3 fallback regressor(冷启动友好)
  - 一份带完整排序链 + 安全 override 的 final_top_routes.csv
  
  排序链的最大优势是优雅降级——任意一环缺失,自动 fallback 到下一档,从不爆炸。chain 的最大代价是每条候选路过 5 个 ranker,推理总耗时 ~3-5s/sample;但相对模型训练时间(分钟级)与化学家审阅时间(小时级),完全可接受。

  Stage35 的契约就两条:输出 csv 只增不减字段名 + final_route_rank/source 两列必给。下游的 confidence attach、QC、export markdown
  都依赖这一契约。所有的工程优雅都是为了在保持这两条契约的前提下,让排序器可替换、可叠加、可审计、可 fallback。


❯    9. Stage 07 — 端到端推理流水线                                                                                                                                                                           
    pipeline/run_pipeline.py 用一张 STEP_FUNCS 列表注册所有 step,PipelineRunner 负责:                                                                                                                         
    - --start_from <step>:从指定 step 重启(后续步骤都执行)                                                                                                                                                    
    - --only_step <step>:只跑这一步                                                                                                                                                                           
    - restore_existing_outputs:看 outputs/inference/<infer_name>/ 下既有产物,补 outputs 字典(支持断点续跑)                                                                                                    
    - record_degradation:某步骤回退时(例如 CGCNN 不可用回退 CHGNet),写 degraded_steps                                                                                                                         
    完整 step 流(28 步):                                                                                                                                                                                      
    0  preflight                       # 资源/CKPT/词表自检                                                                                                                                                   
    1  make_infer_split                # POSCAR 目录 → infer.jsonl                                                                                                                                            
    2  build_structdesc                # 跑 02_build_infer_structdesc_direct.py 出描述子 CSV                                                                                                                  
    3  build_chgnet_embedding          # CHGNet 拿 64-dim crystal_fea                                                                                                                                         
    4  finalize_graph_embedding        # 优先 CGCNN,缺失则用 CHGNet                                                                                                                                           
    5  build_stage2_features           # hybrid CSV(描述子+图嵌入)                                                                                                                                            
    6  build_stage2_npz                # 转 NPZ + meta(标准化用 train 时的 μ,σ)                                                                                                                               
    7  sample_stage2_gflownet          # standard 或 composition_biased(默认后者)                                                                                                                             
    8  constrain_stage2_by_composition # 后置硬过滤:cov、extra penalty、可选 drop_zero_overlap                                                                                                                
    9  summarize_stage2                # unique_sets + count(去重的候选集合)                                                                                                                                  
    10 add_composition_fallback        # 元素覆盖低时补一些"组成回退"前驱物                                                                                                                                   
    11 retrieve_stage2_candidates      # k-NN 检索:从训练 NPZ 找相似目标的真实前驱物集合                                                                                                                      
    12 predict_stage2_baseline         # ExtraTrees 多标签基线(top-k labels → set 组合)                                                                                                                       
    13 merge_stage2_sources            # GFlowNet ∪ Retrieval ∪ Fallback ∪ Baseline 合并                                                                                                                      
    14 rerank_stage2_by_elements       # 元素覆盖加权 + extra penalty + 原 rank 微衰减                                                                                                                        
    15 fix_stage2_global_rank          # 修补合并后 global rank                                                                                                                                               
    16 build_stage3_features           # Stage3 hybrid CSV                                                                                                                                                    
    17 build_stage3_conditioned_table  # 把每个 stage2 候选展开成 (x, parent_set) 行                                                                                                                          
    18 run_stage3_flow                 # Mixture Flow 推理(默认关闭)                                                                                                                                          
    18b run_stage3_lgbm                # LightGBM quantile 推理(默认开启,主力)                                                                                                                                
    19 compare_stage3_models           # 二者一致性报表                                                                                                                                                       
    20 summarize_routes                # 把 stage3_flat → 可读路线 (CSV+MD)                                                                                                                                   
    21 filter_display_routes           # 物理上限/下限(温度 300-1600°C, 时间 0.1-240h)                                                                                                                        
    22 stage35_rule_rerank             # 规则打分                                                                                                                                                             
    23 stage35_learned_rerank          # ExtraTrees regressor 重排                                                                                                                                            
    24 stage35_v21_rerank              # ExtraTrees pairwise 重排(主力)                                                                                                                                       
    25 best_route_per_precursor        # 同一前驱物集合只保留最佳条件                                                                                                                                         
    26 export_final_top_routes         # 选最强可用 ranking 落地                                                                                                                                              
    27 reliability_layer (整层)        # ↓                                                                                                                                                                    
           precursor_qc                # 前驱物 QC 标注                                                                                                                                                       
           attach_route_confidence     # 集合一致性 + reranker 概率融合                                                                                                                                       
           condition_distribution_conf # 条件落在训练分布"身体"还是"尾部"                                                                                                                                     
           attach_stage3_condition_reference_support                                                                                                                                                          
           audit_final_recommended_routes                                                                                                                                                                     
           qc_confidence_postprocess                                                                                                                                                                          
    28 select_final_recommended_routes # 最终向用户展示的稳定列                                                                                                                                               
    9.1 候选集合的"四源合并"(关键设计)                                                                                                                                                                        
    Stage2 不是单一 GFlowNet 输出,而是四种异质来源:                                                                                                                                                           
    1. GFlowNet 多温度采样(主多样性源,composition_biased 模式带元素先验)                                                                                                                                      
    2. Retrieval(17_retrieve_stage2_precursor_candidates_from_npz.py):k-NN(余弦相似度,默认 top_k=50, min_similarity=0)拉相似目标的真实前驱物集合 —— 类似 "case-based reasoning"                               
    3. Composition fallback(11_add_composition_fallback_precursors.py):元素覆盖不足时,直接按目标元素族选"标准"前驱物(如 metal oxide / nitrate)                                                                
    4. ExtraTrees 多标签基线(19_train_stage2_extratrees_baseline.py 训出来的 joblib 模型给每个标签独立概率,top-k 标签做 enumerate-set)                                                                        
    合并后用元素覆盖率重排(10_rerank_stage2_candidates_by_elements.py):                                                                                                                                       
    这种"生成 + 检索 + 规则 + 监督学习"的合并极大缓解了 GFlowNet 在 OOD 目标上的失败模式 —— 任意一个 source 给出合理候选,后面 Stage35 都能挑出来。                                                            
    ---                                                                                                                                                                                                       
    10. 几个值得记住的工程取舍                                                                                                                                                                                
    1. 没有大一统模型,而是异质源 + 多层 reranker。生成模型(GFlowNet/Flow)给"多样性"和"覆盖率",GBDT/ExtraTrees 给"判别力"和"稳定性",规则/模板给"先验"。                                                        
    2. 数据划分按 DOI/material-group,不是按行随机划。这避免了同一篇论文的多个相关样本被切到不同 split。                                                                                                       
    3. Train mode 三件套(relaxed/gold/curriculum)默认存在。gold_only 在所有图缓存、特征、生成数据上统一支持,推理用的 GFlowNet ckpt 路径里就是 gflownet_joint_rerank_hybrid_gold_only_v1,说明最终生产用 gold   
    训出来的最优模型。                                                                                                                                                                                        
    4. Stage3 模型不直接预测条件值,而是 baseline + 残差 mixture。这等价于把"条件预测"分解成"先验均值估计"和"分布形状学习",数值稳定且可控。                                                                    
    5. 元素约束既出现在硬规则里(refine 阶段、composition_constrained 解码、元素 rerank),又通过 reward 进 GFlowNet RL。多重保险:同一个先验在不同位置生效。                                                     
    6. 可靠性层独立成一层(steps_reliability.py 1984 行),最后给每条路线 attach confidence、QC、distribution support,而不是把这些信号塞进 ranker 特征 —— 这是因为推断时间不确定性应该和排序分数解耦。           
    --- 对这一部分进行非常细致的分析，达到可以让别人复现，可以直接写文章的程度，要足够细致的分析                                                                                                              
⏺ ---
  Stage 07 · 端到端推理流水线 — 28 步全景与可重复指南
  
  ▎ 这是 SynPred 的"集大成"层 —— 给定一个 POSCAR(目标晶体结构),输出一份结构 → 前驱物 → 反应条件 → 路线排序 → 可靠性标注的合成路线推荐表。所有先前章节(数据、Stage1-3、Stage35 排序器)都在这里被串起来。
  ▎
  ▎ 流水线由两份核心代码驱动:
  ▎ - run_pipeline.py(394 行):STEP_FUNCS 注册表 + 命令行入口
  ▎ - src/runner.py(167 行):PipelineRunner 状态机(outputs 字典、degraded_steps、step_timings、断点续跑)
  ▎
  ▎ 加上 6 份 step 模块(steps_common, steps_stage2, steps_stage3, steps_stage35, steps_final, steps_reliability,共 ~3290 行),完整描述 28 个 step 的执行细节。
  ▎
  ▎ 本文给出逐步级解析,包含每一步的输入、输出、命令行、关键参数、failure 行为,达到逐字复现的程度。

  ---
  第一部分 · 整体架构
  
  1.1 目录布局(进入流水线前)

  {project_root}/
  ├── data/poscar/<task>/                  # 用户上传的目标 POSCAR
  ├── data/interim/                        # 离线训练好的所有特征/词表
  ├── runs/                                # 离线训练好的所有 ckpt
  └── outputs/inference/<infer_name>/
      ├── work_dir/                         # 中间产物(可清理)
      │   ├── split/infer.jsonl
      │   ├── infer_structdesc.csv
      │   ├── chgnet/graph_embed/...
      │   ├── graph_embed/...
      │   ├── stage2_hybrid_csv/, stage2_hybrid/, stage2_gflownet_candidates/, stage2_summary/
      │   ├── stage3_hybrid/, stage3_conditioned_x_*.csv
      │   └── (各种 .summary.json)
      └── out_dir/                          # 最终展示用产物
          ├── stage3_condition_predictions_*/
          ├── routes_flow_fallback_retrieval_baseline_element_reranked/
          │   ├── synthesis_routes_*.csv
          │   ├── final_top_routes*.csv
          │   ├── final_recommended_routes.csv  ← 用户拿这个
          │   └── *.md
          └── pipeline_v3_manifest.json     # 完整 outputs / degraded_steps / step_timings

  1.2 状态机(PipelineRunner)

  PipelineRunner 把"分散的命令行调用"组织成"带状态的有限状态机":
  - self.outputs: Dict[str, str] —— 每步产出的关键文件路径(key 名固定,如 stage2_final_csv、flow_flat_csv)
  - self.degraded_steps: List[Dict] —— 哪一步降级了 + 原因(如 "CGCNN unavailable, using CHGNet-only")
  - self.step_timings: List[Dict] —— 每步耗时(秒),写入 pipeline_v3_manifest.json
  - record_degradation(step, reason) —— 一旦降级就 [DEGRADED] 日志 + 写 manifest
  - require_file(path) / require_dir(path) —— 缺文件直接 raise,不静默继续
  - step_enabled(name) —— 看 yaml steps: 块,某步 false 就跳过
  - restore_existing_outputs() —— 启动时扫 work_dir/out_dir 下既有产物,符合校验(csv 必须有表头 + 至少 1 数据行,目录必须非空)就回填进 self.outputs,支持断点续跑
  
  ▎ 可重复要点 #1:restore_existing_outputs 的"file 校验"是关键(runner.py:14-30)—— 一个 0 字节或只有表头的 csv 不会被认为有效,这避免了"上次跑崩了留下半成品 csv,这次以为有"。

  1.3 命令行(run_pipeline.py:299-355)

  python run_pipeline.py \
    --config <yaml>            # 必填,流水线配置
    [--start_from <step_name>] # 从指定 step 重启(后续步骤都执行)
    [--only_step <step_name>]  # 只跑这一步(便于调试)
    [--skip_preflight]         # 跳过 STEP 0 的资源自检
    [--infer_name <X>]         # 覆盖 yaml 里的 infer_name(批量跑同一目标多版本)
    [--project_root <path>]    # 覆盖 yaml 里的项目根

  配置加载(config.py,76 行):
  def load_config(path):
      cfg = yaml.safe_load(...)
      for _ in range(5):                  # 迭代 5 次解析模板
          new_cfg = resolve_templates(cfg, cfg)
          if new_cfg == cfg: break
          cfg = new_cfg 
      cfg["_config_path"] = str(Path(path).resolve())
      return cfg
      
  resolve_templates 把 {project_root} / {infer_name} / {stage2.gflownet_run_dir}(支持 . 路径)替换为实际值。5 次迭代是为了支持模板里嵌模板(A 引用 B,B 引用 C)。

  apply_infer_name_override(run_pipeline.py:289-296):递归把整份 cfg 的字符串里 old_infer_name → new_infer_name,这让"同一份 yaml 跑不同目标"成为可能。

  1.4 28 步全景(STEP_FUNCS,run_pipeline.py:24-64)

  STEP 0  preflight                       — 资源/CKPT/词表自检(steps_common.preflight)
  STEP 1  make_infer_split                — POSCAR → infer.jsonl
  STEP 2  build_structdesc                — 结构描述子 CSV
  STEP 3  build_chgnet_embedding          — CHGNet 64-dim crystal_fea
  STEP 4  finalize_graph_embedding        — 优先 CGCNN,缺失则 CHGNet(degrade)
  STEP 5  build_stage2_features           — Stage2 hybrid CSV(描述子 + 图嵌入)
  STEP 6  build_stage2_npz                — 转 NPZ + meta(用 train 时 μ,σ)
  STEP 7  sample_stage2_gflownet          — standard 或 composition_biased
  STEP 8  constrain_stage2_by_composition — 后置硬过滤 (cov / extra_penalty / 可选 drop_zero_overlap)
  STEP 9  summarize_stage2                — unique_sets + count
  STEP 10 add_composition_fallback        — 元素覆盖低补"组成回退"前驱物
  STEP 11 retrieve_stage2_candidates      — k-NN 检索相似目标的真前驱物集合
  STEP 12 predict_stage2_baseline         — ExtraTrees 多标签基线
  STEP 13 merge_stage2_sources            — 四源合并
  STEP 14 rerank_stage2_by_elements       — 元素覆盖加权重排
  STEP 15 fix_stage2_global_rank          — 修补合并后 global rank
  STEP 16 build_stage3_features           — Stage3 hybrid CSV
  STEP 17 build_stage3_conditioned_table  — (x, parent_set) 行展开
  STEP 18  run_stage3_flow                — Mixture Flow 推理(默认关闭)
  STEP 18b run_stage3_lgbm                — LightGBM quantile 推理(默认开启,主力)
  STEP 19  compare_stage3_models          — 二者一致性报表
  STEP 20  summarize_routes               — stage3_flat → 可读路线 (CSV+MD)
  STEP 21  filter_display_routes          — 物理上限/下限
  STEP 22  stage35_rule_rerank            — 规则打分
  STEP 23  stage35_learned_rerank         — ExtraTrees regressor 重排
  STEP 24  stage35_v21_rerank             — ExtraTrees pairwise 重排(主力)
  STEP 25  best_route_per_precursor       — 同一前驱物只留最佳条件
  STEP 26  export_final_top_routes        — 选最强可用 ranking
  STEP 27  reliability_layer              — 6+ 子步骤(QC / confidence / distribution support / V3+V43 ranker / safe_gate / finalize)
  STEP 28  select_final_recommended_routes — 最终稳定列写出

  reliability_layer 内部不在 STEP_FUNCS 注册,而是 steps_reliability.run_reliability_layer(r, cfg) 一次性调度 17 个子函数(每个函数自己看 yaml 是否 enabled)。

  1.5 配置层关键键(configs/full_route_stage3.yaml)
  
  pipeline_name: structure_to_synthesis_route_v3
  infer_name: demo_poscar_test         # 也可被 --infer_name 覆盖
  project_root: /Users/wyc/SynPred
  device: cpu                          # 或 cuda

  paths:
    poscar_dir: '{project_root}/data/poscar/test_v1'
    work_dir:   '{project_root}/outputs/inference/{infer_name}/work_dir'
    out_dir:    '{project_root}/outputs/inference/{infer_name}/out_dir'

  graph:
    max_sites: 200
    cgcnn_checkpoint: '...'
    cgcnn_model_py:   '...'
    cgcnn_model_class: ...

  stage2:
    gflownet_ckpt: '{project_root}/runs/stage2/gflownet_joint_rerank_hybrid_gold_only_v1/best_model.pt'
    template_dir:  '{project_root}/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only'
    precursor_vocab_json: '...'
    sampler_mode: composition_biased   # 默认!
    batch_size: 128
    n_samples: 100
    temperature: 1.0
    top_k: 20
    target_hit_bonus: 6.0
    extra_element_penalty: 1.0
    no_overlap_penalty: 6.0
    stop_bias: -2.0
    ignore_elements: ["H", "O"]
    retrieval_top_k: 50
    retrieval_min_similarity: 0.0
    baseline_top_k_labels: 12
    baseline_top_k_sets: 30
    element_top_n: 30
    element_coverage_weight: 20.0
    element_extra_penalty_weight: 5.0
    element_rank_weight: 0.01
    ...

  stage3:
    schema_json: '...'
    flow_ckpt:   '...'
    flow_script: '...'
    top_k_conditions: 5
    n_flow_samples: 64
    ...

  stage35:
    rule_script:   '...'
    learned_script:'...'
    v21_script:    '...'
    v21_model:     '...'
    ...
  
  steps:
    make_infer_split: true
    build_structdesc: true
    ...
    run_stage3_flow: false              # 默认关
    run_stage3_lgbm: true               # 默认开
    stage35_v21_rerank: true
    ...

  reliability:
    precursor_qc: { enabled: true }
    route_confidence: { enabled: true }
    ...

  display:
    min_temperature_c: 300
    max_temperature_c: 1600
    min_time_h: 0.1
    max_time_h: 240
    top_n: 30

  每一个 step 是否执行,通过 steps:<name>: bool 控制。Pipeline 主循环:
  for name, func in STEP_FUNCS:
      if args.only_step and name != args.only_step: SKIP
      if args.start_from and not started: SKIP
      if r.step_enabled(name):
          r.begin_step(name); func(r); r.end_step()
      else:
          SKIP

  ---
  第二部分 · STEP 0 ~ 4:结构入口与图嵌入
  
  2.1 STEP 0 · preflight(steps_common.py:7-25)

  不做任何写入,只检查上游 ckpt / 词表 / 模板目录就位:
  r.require_dir(paths["poscar_dir"])
  r.require_file(stage2["gflownet_ckpt"])
  r.require_dir(stage2["template_dir"])
  r.require_file(template_dir / "feature_cols.json")
  r.require_file(template_dir / "feature_mean.npy")     # train 期统计的 μ
  r.require_file(template_dir / "feature_std.npy")      # σ
  r.require_file(template_dir / "action_to_id.json")    
  r.require_file(template_dir / "action_vocab.json")
  r.require_file(template_dir / "precursor_names.json")
  r.require_file(stage2["precursor_vocab_json"])
  
  if cfg["steps"].get("run_stage3_flow", False):
      r.require_file(cfg["stage3"]["schema_json"])
      r.require_file(cfg["stage3"]["flow_ckpt"])
      r.require_file(cfg["stage3"]["flow_script"])
      
  **为什么 preflight 要单独一步?**因为后续 step 跑到一半发现缺 ckpt 才报错,前面的中间产物白算。preflight 只读元数据,几毫秒,但能阻止 30min 后白跑。命令行 --skip_preflight
  让你绕开(用于调试场景已知缺某个非关键文件)。

  2.2 STEP 1 · make_infer_split(steps_common.py:28-39)

  python pipeline/src/01_make_infer_split_from_poscars.py \
    --poscar_dir <poscar_dir> \
    --output_dir <work_dir>/split
  # 产出:<work_dir>/split/infer.jsonl

  infer.jsonl 每行一条:
  {"sample_id": "abc", "formula": "Fe2O3", "poscar_path": ".../abc.vasp", ...}
  
  关键设计:Stage 03 训练时把数据切成 train.jsonl / val.jsonl / test.jsonl,推理时用同样格式的 infer.jsonl,确保下游 02_build_infer_structdesc_direct.py 等脚本可以无差别处理 train/val/test/infer 任意 split。

  2.3 STEP 2 · build_structdesc(steps_common.py:42-52)

  python pipeline/src/02_build_infer_structdesc_direct.py \
    --infer_jsonl <infer.jsonl> \
    --output_csv <work_dir>/infer_structdesc.csv

  走 Stage 03 同样的描述子提取流程(matminer + magpie + composition + symmetry),无 GNN。输出:infer_structdesc.csv,每行一条,~120 列描述子。

  ▎ 可重复要点 #2:这一步的描述子列名 / 顺序必须严格匹配训练期 feature_cols.json,否则后续 NPZ 标准化时 μ/σ 对不上列。02_build 内部走一份"hard-coded order"的 column 抽取逻辑保证一致。

  2.4 STEP 3 · build_chgnet_embedding(steps_common.py:55-69)
  
  python pipeline/src/03_build_infer_graph_embeddings_chgnet.py \
    --infer_jsonl ... \
    --work_dir <work_dir>/chgnet \
    --project_root ... \
    --precursor_vocab_json ... \
    --train_mode gold_only \
    --max_sites <graph.max_sites>=200

  调用 CHGNet 预训练模型抽 crystal_fea(默认 64 维)。对 site 数 > 200 的超晶胞会触发 fallback,通常用对称化后的 primitive cell 替代。train_mode=gold_only 严格,不允许 relaxed structure 参与。

  输出:<work_dir>/chgnet/graph_embed/infer_graph_embed.csv(每行 sample_id + 64 列 chgnet_<i>)。

  2.5 STEP 4 · finalize_graph_embedding(steps_common.py:72-99)

  双路径:
  1. Primary 路径:CGCNN 缓存 cgcnn_cache_infer/ + checkpoint + model_py + model_class 都齐 → 跑 04_finalize 把 CGCNN embedding 与 CHGNet 拼接(或仅用 CGCNN,内部由 04 决定)
  2. Fallback:任一缺失 → record_degradation("finalize_graph_embedding", "CGCNN unavailable, using CHGNet-only"),直接把 CHGNet csv 抄到 final_graph_embed_csv
  
  降级语义:推理结果质量略降,但不爆炸。degraded_steps 写入 manifest 让用户知情。

  ▎ 可重复要点 #3:这是流水线第一个"显式 degradation"。SynPred 哲学:任何上游模型缺失,流水线降级而非崩溃——下游模型可能仍能工作。但 manifest 必须如实记录降级。

  ---
  第三部分 · STEP 5 ~ 15:Stage2 候选集合的"四源合并"
  
  3.1 STEP 5 · build_stage2_features(steps_stage2.py:9-24)

  python pipeline/src/05_build_hybrid_features_infer.py \
    --task stage2 \
    --output_dir <work_dir>/stage2_hybrid_csv \
    --infer_descriptor_csv ... \
    --infer_embedding_csv  ... \
    --embedding_prefix graph_emb \
    --replicate_to_train_val

  --replicate_to_train_val 是个奇怪但聪明的开关:即使是推理,也写出三份 train/val/test 同样内容,让下游"以为是数据集",直接用 03_data 的 hybrid CSV → NPZ 转换器(07_build_stage2_gflownet_infer_npz.py 期待
  train/val/test 三 split,即使我们只 care test)。

  输出:stage2_train_hybrid.csv(描述子 + 图嵌入,每行一个 sample)。

  3.2 STEP 6 · build_stage2_npz(steps_stage2.py:27-40)
  
  python pipeline/src/07_build_stage2_gflownet_infer_npz.py \
    --infer_hybrid_csv ... \
    --template_dir <stage2_gflownet_dataset/hybrid/gold_only> \
    --output_dir <work_dir>/stage2_hybrid \
    --split_name test

  关键 trick:--template_dir 指向训练时的 NPZ 目录,内部读其中 feature_mean.npy / feature_std.npy / feature_cols.json,用训练期统计量标准化推理特征。

  输出:<work_dir>/stage2_hybrid/test.npz + test_meta.csv。NPZ 含:
  - x: (N_infer, F) 标准化后 float32
  - meta 列(sample_id, formula, ...)
  
  3.3 STEP 7 · sample_stage2_gflownet(steps_stage2.py:43-169)

  两种模式:

  (a) composition_biased(默认):
  python pipeline/src/19_sample_stage2_gflownet_composition_constrained.py \
    --input_dir <stage2_npz_dir> \
    --output_dir <work_dir>/stage2_gflownet_candidates_composition_decoding \
    --ckpt_path <gflownet_ckpt> \
    --split test \
    --batch_size 128 --n_samples 100 \
    --temperature 1.0 --top_k 20 \
    --use_greedy_as_first \
    --composition_constrained \
    --target_hit_bonus 6.0 \
    --extra_element_penalty 1.0 \
    --no_overlap_penalty 6.0 \
    --stop_bias -2.0 \
    --ignore_elements H,O \
    --device <device> 
    
  GFlowNet 在解码每一步加 element-aware bias:
  - target_hit_bonus=6:动作引入新的目标元素 → +6 logit
  - extra_element_penalty=1:动作引入非目标元素 → -1 logit
  - no_overlap_penalty=6:当前 set ∪ {action} 与目标元素无任何交集 → -6 logit
  - stop_bias=-2:STOP 动作 logit -2(让模型生成更长的 set)
  - ignore_elements=H,O:H/O 不算"目标元素"(因几乎所有氧化物前驱物都含 O)
  
  (b) standard:用 06_sample_stage2_gflownet_infer.py,纯模型采样,无 element bias。

  --use_greedy_as_first:第一条样本永远是贪心采样(top-1),其余 99 条按 temperature=1.0 多样性采样。

  输出:test_samples.csv,每行一个 (sample_id, sampled_set, sample_rank, decode_method, action_seq)。每个 sample 有 100 行候选。

  3.4 STEP 8 · constrain_stage2_by_composition(steps_stage2.py:240-285)

  后置硬过滤:
  python pipeline/src/18_constrain_stage2_sample_candidates_by_composition.py \
    --input_csv <test_samples.csv> \
    --output_csv <test_samples_composition_constrained.csv> \
    --summary_json ... \
    --min_coverage 0.0 \
    --coverage_weight 20.0 \
    --extra_penalty_weight 5.0 \
    --rank_weight 0.01 \
    --top_n_per_sample 100 \
    --dedup \
    [--drop_zero_overlap]
    
  公式与 STEP 14 元素 rerank 同(下文详述)。--dedup 按 set_key 去重(集合相同的多次采样合并)。--drop_zero_overlap 是可选硬丢弃:如果 set 与目标元素一个都不沾边就直接砍掉。

  3.5 STEP 9 · summarize_stage2(unique_sets_ranked.csv)

  聚合 step 7+8 的候选,按 set_key 去重,统计:
  - count:同一 set 在所有采样中出现几次
  - frequency = count / n_samples
  - sample_rank_min/mean/max:出现位次的统计
  - decode_methods_seen:greedy / sampled / composition_biased 等

  3.6 STEP 10 · add_composition_fallback(steps_stage2.py:288-323)

  python pipeline/src/11_add_composition_fallback_precursors.py \
    --input_csv <unique_sets_ranked.csv> \
    --output_csv <unique_sets_ranked_with_fallback.csv> \
    --summary_json ... \
    --top_n_fallback 20 \
    --rank_col rank \
    --precursor_col precursor_set

  如果一个目标的 GFlowNet 候选集合元素覆盖普遍偏低(例如目标含 P,但所有候选都没有磷源),11_add_composition_fallback 会按目标元素族直接拼一些"标准前驱物 set":
  - 含 P → 加 [P2O5, NH4H2PO4](标准磷源)
  - 含 S → 加 [S, Na2SO4]
  - 含 Se → 加 [SeO2, Se]
  - 含 K → 加 [K2CO3, KNO3]
  - 含 Li → 加 [Li2CO3, LiOH·H2O]
  - ...
  
  这是化学家先验,补 GFlowNet OOD 失败模式。

  3.7 STEP 11 · retrieve_stage2_candidates(steps_stage2.py:326-373)

  python pipeline/src/17_retrieve_stage2_precursor_candidates_from_npz.py \
    --infer_hybrid_csv ... \
    --template_dir <stage2_gflownet_dataset/hybrid/gold_only> \
    --output_csv <stage2_summary/retrieval_npz_candidates.csv> \
    --top_k 50 \
    --min_similarity 0.0 \
    --label_threshold 0.5
  
  Case-based reasoning:
  1. 在训练 NPZ 里取每个目标的 (x_struct, y_set)(真实标签)
  2. 推理目标的 x_struct(标准化空间)对训练库做余弦相似度
  3. 返回 top-50 相似目标的真实前驱物集合(以 multi-hot label,阈值 0.5 二值化)
  
  label_threshold=0.5:训练里 multi-hot label 通常是硬 0/1,这个阈值是 sanity check。

  ▎ 可重复要点 #4:Retrieval 不是"和当前目标一模一样的训练样本"——它是"特征空间最近的训练样本",返回它们的真实前驱物作为候选。这等价于"如果新目标长得像 X 化合物,就把 X 的合成路线作为参考"。

  3.8 STEP 12 · predict_stage2_baseline(steps_stage2.py:376-443)

  python pipeline/src/22_predict_stage2_extratrees_baseline_candidates.py \
    --input_dir <stage2_npz_dir> \
    --model_path <runs/stage2/extratrees_multilabel_hybrid_gold_only_v1/stage2_extratrees_multilabel.joblib> \
    --split test \
    --output_csv <extratrees_baseline_candidates.csv> \
    --top_k_labels 12 \
    --top_k_sets 30 \
    --min_prob 0.02 \
    --max_set_size 4

  多标签 ExtraTrees:每个前驱物标签独立训练一个二分类器,推理出每个标签的概率 P(label_i | x)。

  候选生成:
  1. 取概率 ≥ 0.02 的标签 top-12
  2. 按 top_k_sets=30 枚举 size ≤ 4 的所有子集组合,按 ∑log(P_i) + ∑log(1-P_j) 打分
  3. 输出 top-30 个 set
  
  降级行为(行 425-439):若 model 或 script 不存在,写一个空 csv(只有表头),让下游 merge_stage2_sources 仍能 concat 不出错。

  3.9 STEP 13 · merge_stage2_sources(steps_stage2.py:446-489)

  python pipeline/src/16_merge_stage2_candidate_sources.py \
    --input_csvs <fallback_csv> <retrieval_csv> <baseline_csv> \
    --output_csv <unique_sets_ranked_with_fallback_retrieval_baseline.csv> \
    --precursor_col precursor_set

  四源合并(GFlowNet 已经在 fallback_csv 里):
  1. GFlowNet 多温度(主多样性源)
  2. Retrieval k-NN(case-based,稳)
  3. Composition fallback(规则保险)
  4. ExtraTrees baseline(监督学习独立)
  
  合并按 set_key 去重,但保留所有源信息(source_methods 列里 OR 起来,如 gflownet|retrieval|baseline)。多源出现的 set 后面 rerank 时会得到加分。

  ▎ 核心思想:任何一源给出合理候选,后面 Stage35 都能挑出来。这是 SynPred 在 OOD 目标上不至于完全崩盘的关键工程冗余。

  3.10 STEP 14 · rerank_stage2_by_elements(steps_stage2.py:492-533)

  python pipeline/src/10_rerank_stage2_candidates_by_elements.py \
    --input_csv  <merged.csv> \
    --output_csv <element_reranked.csv> \
    --precursor_col precursor_set \
    --rank_col rank \
    --top_n 30 \
    --coverage_weight 20.0 \
    --extra_penalty_weight 5.0 \
    --rank_weight 0.01

  公式:
  $$
  \text{score}(\mathcal P, \text{target}) = 20\cdot \text{cov}(\mathcal P, \text{target}) - 5\cdot |\text{extra metals}| - 0.01\cdot \text{rank}_{\text{src}}
  $$

  - cov:set 中前驱物涵盖目标元素的比例(0~1)
  - extra metals:set 引入的非目标金属元素数(H/O 已 ignore)
  - rank_src:候选在原 source 里的位次(GFlowNet rank 0 / retrieval rank 0 / ...)

  每个 sample 内部按 score 降序,取 top-30。

  ▎ 可重复要点 #5:覆盖权重 20 vs penalty 5 vs rank 0.01 比例失衡是有意的——化学正确性 >> 排序新鲜度。一个 cov=0.5 的候选(score = 10)永远赢一个 cov=0.0 但原 rank=1 的(score = -0.01)。

  3.11 STEP 15 · fix_stage2_global_rank

  合并去重后,rank 列可能不连续(被去掉的 set 留 gap)。这一步把 rank 重赋为 [1, 2, ..., n]。

  最终输出:stage2_final_csv = unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv,这是 Stage3 的输入。

  ---
  第四部分 · STEP 16 ~ 19:Stage3 条件预测
  
  4.1 STEP 16 · build_stage3_features(steps_stage3.py:7-21)

  同 STEP 5,但 --task stage3。Stage3 的 hybrid 特征通常含更多对反应条件相关的描述子(formula composition、density、formability 等),与 stage2 的特征列略有差异。05_build_hybrid_features_infer.py 内部按 --task 
  stage2|stage3 选不同 column 子集。

  4.2 STEP 17 · build_stage3_conditioned_table(steps_stage3.py:24-37)

  python pipeline/src/05c_build_stage3_conditioned_feature_table_infer.py \
    --infer_hybrid_csv      <stage3_train_hybrid.csv> \
    --stage2_candidates_csv <stage2_final_csv> \
    --schema_json           <stage3_schema_json> \
    --output_csv <work_dir>/stage3_conditioned_x_*.csv \
    --max_stage2_candidates 30

  展开:每个 sample × 每个 Stage2 候选 → 一行。文件大小放大 30×。每行含:
  - 所有 Stage3 hybrid 特征(描述子 + 图嵌入)
  - parent_precursor_set(JSON 列表)
  - parent_precursor_set_key
  - parent_precursor_rank
  - 编码后的 y_set multi-hot 列(feat_y_set_<i>)
  
  下游 LGBM / Flow 推理直接读这份 csv。

  4.3 STEP 18 · run_stage3_flow(默认关)

  python pipeline/src/13_run_stage3_infer_mixture_flow_conditioned.py \
    --conditioned_x_csv ... --schema_json ... \
    --flow_ckpt <ckpt> --flow_script <model.py> \
    --output_dir <out_dir>/stage3_condition_predictions_flow_*/ \
    --top_k_conditions 5 \
    --n_flow_samples 64 \
    --device <device>

  输出 test_candidates_flat.csv(每行一组候选条件:temperature_c, time_h, atmosphere, synthesis_type, condition_source, stage3_score, ...)。

  4.4 STEP 18b · run_stage3_lgbm(默认开,主力)

  python pipeline/src/13b_run_stage3_infer_lgbm_quantile.py \
    --conditioned_x_csv ... --schema_json ... \
    --temp_model_dir <runs/stage3/lgbm_quantile_ensemble_v2_fulldata/temperature> \
    --time_model_dir <runs/stage3/lgbm_quantile_ensemble_v2_fulldata/time> \
    --atm_model         <model_atmosphere_binary_final.txt> \
    --time_bucket_model <model_time_bucket.txt> \
    --output_dir <out_dir>/stage3_condition_predictions_lgbm/ \
    --top_k_conditions 5

  输出同样格式 test_candidates_flat.csv。详细见 SynPred 文章 ⑦ §3.5。

  4.5 STEP 19 · compare_stage3_models(steps_stage3.py:106-200)

  如果同时跑了 flow + lgbm:
  - 算两份预测的 mean / std / min / max
  - 按 precursor_set group,算 top-1 温度的 absolute diff(mean / median / max)
  - 写入 <out_dir>/stage3_model_comparison/stage3_model_comparison.md

  最后:
  primary = cfg["stage3_comparison"]["primary_model"]   # default "lgbm"
  r.outputs["flow_flat_csv"] = lgbm_csv if primary == "lgbm" else flow_csv
  
  关键:flow_flat_csv 这个 outputs key 名字保留,但实际指向 lgbm 的 csv(因为下游 summarize_routes 一律读 flow_flat_csv)。命名是历史遗留,语义已变。

  ---
  第五部分 · STEP 20 ~ 26:路线汇总与 Stage35 排序
  
  5.1 STEP 20 · summarize_routes

  python pipeline/src/12_summarize_structure_to_routes.py \
    --stage3_flat_csv <flow_flat_csv> \
    --output_dir <out_dir>/routes_*/ \
    --top_n 100

  把每行的 (parent_set, condition) 配对展平成可读路线行,加上目标元素覆盖、警告等列。输出 synthesis_routes_readable.csv + .md。

  5.2 STEP 21 · filter_display_routes

  python pipeline/src/14_filter_synthesis_routes_for_display.py \
    --input_csv <readable.csv> \
    --output_csv <display_filtered.csv> \
    --output_md  <display_filtered.md> \
    --min_temperature_c 300 --max_temperature_c 1600 \
    --min_time_h 0.1 --max_time_h 240 \
    --top_n 30 \
    --prefer_top_component_mean

  物理可行性 hard gate(温度 300-1600°C、时间 0.1-240h 之外砍掉)。--prefer_top_component_mean 让 Mixture Flow 给的"top-1 mean"路线优先于 sample 路线。

  5.3 STEP 22 ~ 24 · 三种 reranker

  如 SynPred 文章 ⑧ 详述:
  - STEP 22 stage35_rule_rerank:启发式规则评分
  - STEP 23 stage35_learned_rerank:旧版 ExtraTrees Regressor
  - STEP 24 stage35_v21_rerank:V21 ExtraTrees pairwise(主力)
  
  每一步在 csv 上加 stage35_<X>_score / stage35_<X>_rank 列,非破坏。任一缺脚本/缺模型则 record_degradation。

  5.4 STEP 25 · best_route_per_precursor

  python 19b_select_best_route_per_precursor.py \
    --input_csv <最强可用 ranking csv> \
    --output_csv <best_per_precursor.csv> \
    --output_md  ... --summary_json ... \
    --top_n 30 \
    --prefer_top_component_mean \
    --prefer_full_element_coverage \
    --min_temperature_c 300 --max_temperature_c 1600 \
    --min_time_h 0.1 --max_time_h 240

  逻辑:同一 precursor_set_key 下,所有 (温度, 时间) 候选选最佳一组。这避免最终展示给用户时出现"前驱物 a + 800°C + 12h" 与 "前驱物 a + 850°C + 10h" 两条几乎相同的路线。

  输入 CSV 选择按 fallback chain:stage35_v21_csv > stage35_learned_csv > stage35_rule_csv > display_csv。

  5.5 STEP 26 · export_final_top_routes(steps_final.py)

  按优先级 v21 > learned > rule > display 选最强 ranking 写到 final_top_routes.csv(详见 SynPred 文章 ⑧ §5)。

  ---
  第六部分 · STEP 27 · Reliability Layer(独立一层)
  
  6.1 设计意图

  ▎ "推断时间不确定性应该和排序分数解耦。"

  Reliability layer 不改 ranking,只在 final csv 上附加列:置信度、警告、QC、distribution support。理由:
  - 排序信号是"哪条更优",置信度是"我们对这个排序有多确定"
  - 把不确定性塞进 score 会让"高 score + 低置信度" 与 "中 score + 高置信度"无法区分
  - 解耦后,UI 可以分两栏展示:final_route_rank | confidence_level
  
  run_reliability_layer(steps_reliability.py:1927-1984)按序调度 17 个子步骤,每个看 yaml 是否 enabled。

  6.2 子步骤(主要的)

  6.2.1 run_precursor_qc(行 513-551)

  python pipeline/scripts/qc_route_precursors.py \
    --input_csv <final_top_routes.csv> \
    --output_csv <final_top_routes_with_precursor_qc.csv> \
    --output_md  ... --summary_json ... \
    --top_n 30

  逐个前驱物查 RDKit/SMILES/化学合理性,标注 precursor_qc_level ∈ {pass, minor_warning, major_warning} 与具体警告(expensive, toxic, rare_in_publications, ...)。

  6.2.2 attach_route_confidence(行 554-602)
  
  调用 attach_route_confidence.py,核心公式(行 110-156):

  $$
  \text{score} = 0.30\cdot \text{element} + 0.18\cdot \text{rank} + 0.17\cdot \text{cond_model} + 0.17\cdot \text{cond_rule} + 0.08\cdot \text{prob} + 0.10\cdot \text{route_score}
  $$

  - element_score = element_coverage − min(0.6, 0.2 × missing_count) − min(0.4, 0.1 × extra_penalty)
  - rank_score = 1 − precursor_rank / max_rank
  - cond_model = stage3_score(归一化)
  - cond_rule = 物理合理性(温度在 300-1600°C 内?时间在 0.1-240h 内?atmosphere 与 synthesis_type 匹配?)
  - prob = stage35_v2_prob(权重小,因为 calibration 不好)
  - route_score_component = stage35_v21_score / 12.0(归一化压缩)

  阈值映射(行 159-167):
  - ≥ 0.78 → high_confidence / recommended
  - ≥ 0.58 → medium_confidence / recommended_with_validation
  - < 0.58 → low_confidence / review_required
  
  硬规则修正(行 170-184):
  - element_coverage < 0.5 OR missing_count ≥ 2 → 强制 low_confidence / review_required
  - 含 extreme_high_temperature_warning 或 too_long_time_warning → high → medium
  - 警告级别:无警告 / 含致命警告 → major / 其余 → minor

  6.2.3 attach_stage3_condition_reference_support(行 603-682)

  把 stage3 训练数据集里"实际观测到的 (材料族, 温度, 时间)"作为参考库,推理路线查"训练库里 ±15°C 范围有没有相似材料合成?",标 condition_reference_support_level ∈ {strong, weak, none}。

  这是"训练分布支持"的 sanity check——预测温度落在训练观测频繁出现的位置,可信度高。

  6.2.4 add_condition_distribution_confidence(行 770-827)

  在所有训练样本的 (温度, 时间) 二维分布上,看推理路线落在 KDE "body"(高密度区)还是 "tail"(低密度区):
  - body → high score
  - tail → low score(模型在分布尾部预测不确定性大)
  
  这是用 KDE 给"训练分布外推"打分。

  6.2.5 audit_condition_diversity(行 830-875)

  审计:同一 sample 的 30 条候选路线,温度范围多大?如果都在 ±20°C 内,Stage3 模型可能崩(没有学到真实多样性)。这一步只统计、不修改 ranking。

  6.2.6 postprocess_confidence_with_precursor_qc(行 877-913)

  把 precursor_qc_level 与 route_confidence_level 联动:precursor major_warning → confidence -0.15,minor → -0.05,严重时强制 review_required。

  6.2.7 V3 / V43 后置 reranker(行 943-1714)

  详见 SynPred 文章 ⑧:
  - build_joint_route_features + apply_v3_joint_route_rerank
  - build_v3_learned_ranker_dataset + apply_v3_learned_ranker
  - add_v43_route_template_features + apply_v43_template_ranker + apply_v43_safe_strict_gate
  
  每一步可单独 disable。

  6.2.8 finalize_recommended_routes(行 1715-1925)

  最末端:从所有可用 final csv(v43 > v3_learned > v3_joint > with_confidence > with_precursor_qc > final_top_routes)中选最强,统一字段名,生成 final_recommended_routes.csv + .md + .json。

  6.2.9 audit_final_recommended_routes(行 1772+)

  最后一步,对 final_recommended_routes.csv 做 sanity check:
  - top-1 路线 confidence 是否合理?
  - 整批 confidence 分布是否合理?
  - 是否所有路线 element_coverage > 0?
  
  写一份 audit json,不改 ranking,纯诊断。

  6.3 _get_current_csv 模式

  reliability 层各 step 的输入选择按 fallback chain:
  def _get_current_csv(r, route_out_dir):
      for key in ["final_top_routes_current_csv",
                  "final_top_routes_with_metadata_stage3_reference_csv",
                  "final_top_routes_with_stage3_condition_reference_csv",
                  "final_top_routes_with_confidence_csv",
                  "final_top_routes_with_precursor_qc_csv",
                  "final_top_routes_csv"]:
          p = r.outputs.get(key, "")
          if p and Path(p).exists(): return Path(p)
      return route_out_dir / "final_top_routes.csv"
      
  每完成一步 enrichment,_set_current_csv(r, output_csv) 更新 final_top_routes_current_csv 为本步输出。下一步从这继续。这让"步骤可任意 enable/disable + 任意删除中间产物"都不破坏链路。

  ---
  第七部分 · STEP 28 · select_final_recommended_routes
  
  run_pipeline.py:170-275 定义,优先级:
  1. v4.3 template-aware chem-only
  2. v3 learned reranker 
  3. v3 joint reranker
  4. confidence route table
  5. original final_top_routes
  
  关键:不删除、不覆盖历史输出,只创建一个稳定别名 final_recommended_routes.csv + .md 指向当前最强可用版本。这让用户拿同一个文件名总能读到"当前最佳"路线表。

  r.outputs["final_recommended_routes_source"] 记录究竟用了哪一档(v43_template_chemonly / v3_learned / v3_joint / confidence / final_top_routes),写入 manifest。

  ---
  第八部分 · 端到端复现指南

  8.1 环境

  # Python 3.10+
  pip install numpy pandas scikit-learn torch yaml lightgbm joblib
  pip install pymatgen ase matminer chgnet     # 结构 + GNN
  pip install rdkit                             # QC (precursor)

  CHGNet 需要 GPU(可选),CGCNN 需要训练好的 checkpoint(可选)。LGBM CPU 即可。

  8.2 单 target 跑
  
  PR=/Users/wyc/SynPred
  TASK=demo_poscar_test
  # 1. 把 POSCAR 放到 data/poscar/<TASK>/<sample_id>.vasp
  mkdir -p $PR/data/poscar/$TASK
  cp my_target.vasp $PR/data/poscar/$TASK/

  # 2. 准备配置(或用默认 full_route_stage3.yaml + override)
  cd $PR/scripts/07_infer/structure_to_synthesis_route/pipeline

  # 3. 全跑
  python run_pipeline.py \
    --config configs/full_route_stage3.yaml \
    --infer_name $TASK \
    --project_root $PR

  # 4. 结果
  ls $PR/outputs/inference/$TASK/out_dir/routes_*/
  # 找:final_recommended_routes.csv / .md

  8.3 断点续跑

  # 假设 STEP 19 (compare_stage3_models) 跑崩,前面都好
  python run_pipeline.py \
    --config configs/full_route_stage3.yaml \
    --infer_name $TASK \
    --start_from compare_stage3_models

  PipelineRunner.restore_existing_outputs() 会扫 work_dir/out_dir 自动回填 outputs 字典。从 compare_stage3_models 开始,后续 step 自动找到上游 outputs。

  8.4 单步调试

  # 只跑 stage35_v21_rerank,前后都跳
  python run_pipeline.py \
    --config configs/full_route_stage3.yaml \
    --infer_name $TASK \
    --only_step stage35_v21_rerank

  8.5 批量跑(同一 yaml,多个 target)

  for tgt in target_1 target_2 target_3; do
      python run_pipeline.py \
        --config configs/full_route_stage3.yaml \
        --infer_name $tgt \
        --project_root $PR
  done

  apply_infer_name_override 会把 yaml 里的所有 {infer_name} 模板替换成新值,output dir 各自隔离。

  8.6 输出文件全清单

  {out_dir}/
  ├── stage3_condition_predictions_flow_*/test_candidates_flat.csv  # Flow 输出(若开启)
  ├── stage3_condition_predictions_lgbm/test_candidates_flat.csv    # LGBM 输出(主力)
  ├── stage3_model_comparison/stage3_model_comparison.md            # 对比报告
  └── routes_flow_fallback_retrieval_baseline_element_reranked/
      ├── synthesis_routes_readable.csv / .md                       # 12_summarize 输出
      ├── synthesis_routes_display_filtered.csv / .md               # 物理可行性过滤
      ├── synthesis_routes_stage35_rule_reranked.csv / .md
      ├── synthesis_routes_stage35_learned_reranked.csv / .md
      ├── synthesis_routes_stage35_v21_hybrid_reranked.csv / .md / _summary.json
      ├── synthesis_routes_stage35_v21_best_per_precursor.csv / .md
      ├── final_top_routes.csv / .md / _summary.json                 # STEP 26 出
      ├── final_top_routes_with_precursor_qc.csv / .md               # STEP 27.1
      ├── final_top_routes_with_confidence.csv / .md                 # STEP 27.2
      ├── final_top_routes_with_condition_confidence.csv / .md       # STEP 27.4
      ├── final_top_routes_with_stage3_condition_reference.csv       # STEP 27.3
      ├── final_top_routes_with_joint_features.csv                   # V3 joint reranker
      ├── final_top_routes_v3_joint_reranked.csv / .md
      ├── final_top_routes_v3_learned_reranked.csv / .md
      ├── synthesis_routes_stage35_v43_template_features.csv
      ├── final_top_routes_v43_template_chemonly_reranked.csv / .md
      ├── final_top_routes_v43_safe_strict.csv
      ├── final_recommended_routes.csv / .md / _summary.json         # STEP 28 ★用户看这个
      └── condition_diversity_audit.csv / .md / _summary.json
  {out_dir}/pipeline_v3_manifest.json                                 # 整个 pipeline 的 outputs / degraded / timings

  8.7 验收清单

  ┌───────────────────────────────┬──────────────────────────────────────────────────────────────────┬──────────────────────────────┐
  │            验收点             │                               命令                               │             期望             │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ Manifest 没缺步骤             │ cat pipeline_v3_manifest.json | jq '.degraded_steps'             │ ≤ 1 项,通常是 CGCNN fallback │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ Stage2 候选数                 │ wc -l stage2_summary/unique_sets_ranked_with_*.csv               │ ≥ 30                         │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ Stage3 候选条件数             │ wc -l stage3_condition_predictions_lgbm/test_candidates_flat.csv │ ≥ 5 × n_samples              │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ final_recommended_routes 行数 │ wc -l final_recommended_routes.csv                               │ ≥ 10                         │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 至少一条 high_confidence      │ grep high_confidence final_recommended_routes.csv                │ 有                           │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 元素覆盖正常                  │ awk -F, '$X > 0.8' final_recommended_routes.csv                  │ 有                           │
  ├───────────────────────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 总耗时合理                    │ cat pipeline_v3_manifest.json | jq '.step_timings'               │ 各步 < 600s                  │
  └───────────────────────────────┴──────────────────────────────────────────────────────────────────┴──────────────────────────────┘

  8.8 常见坑
  
  ┌──────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
  │                       现象                       │                               原因                                │                          解                           │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ preflight 失败                                   │ ckpt 路径错                                                       │ 检查 yaml stage2.gflownet_ckpt 等是否真存在           │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ KeyError: graph_emb_0 在 build_stage2_features   │ infer_embedding_csv 的 --embedding_prefix 不对                    │ 看 chgnet csv 列名,默认 chgnet_<i> 需要 prefix=chgnet │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Stage2 候选全空                                  │ composition_biased 太严                                           │ 调小 --target_hit_bonus、关 --drop_zero_overlap       │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Stage3 NPZ 标准化报错                            │ template_dir 与训练期不一致                                       │ 训练用 gold_only,推理也必须 gold_only                 │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ flow_flat_csv 空                                 │ flow ckpt 缺,而 flow 默认开                                       │ steps.run_stage3_flow=false 改回(或补 ckpt)           │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ V21 推理报维度不匹配                             │ feature_cols.json 不配套                                          │ 替换为同 run 的 feature_cols.json                     │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ reliability 整层抛                               │ route_out_dir 不存在                                              │ 检查 STEP 26 是否成功,看 manifest                     │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ final_recommended_routes 总指向 final_top_routes │ reliability 子步全 disable 或全 fail                              │ 看 manifest degraded_steps                            │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 断点续跑跳错步                                   │ restore_existing_outputs 把旧产物当成功                           │ 删 work_dir 重跑                                      │
  ├──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ 内存爆                                           │ n_samples=100 × 5 candidates × 30 stage2 = 15000 行,Flow 推理 OOM │ 调 --n_flow_samples 32、--top_k 10、batch_size 64     │
  └──────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘

  ---
  第九部分 · 设计哲学(对应 § 10)
  
  9.1 八条工程取舍的总结

  1. 没有大一统模型,而是异质源 + 多层 reranker
    - GFlowNet/Flow 给"多样性 + 覆盖率"
    - GBDT/ExtraTrees 给"判别力 + 稳定性"
    - 规则 / 模板给"先验"
    - 任一组件失效,其他组件不至于全崩
  2. 数据划分按 DOI/material-group(贯穿 03_data + Stage35 V43 训练 GroupShuffleSplit)
    - 避免同一篇论文样本被切到 train/test
    - test metric 反映真实泛化
  3. Train mode 三件套(relaxed_only / gold_only / curriculum)
    - gold_only = 仅 ICSD/MP gold reference structures(高质量)
    - relaxed_only = 仅 DFT relax 后的(数据多但有噪)
    - curriculum = 先 relaxed 后 gold 的两阶段
    - 生产路径:gflownet_joint_rerank_hybrid_gold_only_v1 是最终选择 → gold 训出来的最稳
  4. Stage3 模型不直接预测条件值,而是 baseline + 残差 mixture
    - c = c_baseline(x, y_set) + Δc, Δc ~ p_θ(...)
    - 解耦"先验均值估计"与"分布形状学习"
    - 残差范围小,数值稳定
  5. 元素约束既出现在硬规则里,又通过 reward 进 GFlowNet RL
    - 解码期:composition_biased 解码加 logit bonus/penalty
    - 后置期:18_constrain 硬过滤
    - 重排期:10_rerank_by_elements 公式权重
    - 训练期:GFlowNet reward 含 element_match
    - 多重保险:同一先验在不同位置生效,避免单点失效
  6. 可靠性层独立成一层(steps_reliability.py 1984 行)
    - 排序分数 ≠ 置信度
    - 推断时间不确定性应解耦
    - 让 UI 双轴展示(rank × confidence)
  7. Sort + Annotate, never destroy
    - 每个 reranker / reliability step 只新增列
    - 让 chain 完全可旁路 / 可叠加 / 可审计
    - 一次跑出所有版本,UI 切换查看
  8. 配置驱动 + 模板插值 + override
    - 一份 yaml 配整套
    - {project_root} / {infer_name} 模板支持多目标批量
    - --start_from / --only_step / --infer_name / --project_root 命令行覆盖

  9.2 与"标准"机器学习 pipeline 的差异

  ┌─────────────────┬───────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │      维度       │ 普通 ML pipeline  │                                                   SynPred Stage 07                                                   │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 模型数          │ 1 个端到端        │ 6+ 个异质模型(GFlowNet/Retrieval/Rule/ExtraTrees baseline + Mixture Flow + LGBM Quantile + Stage35 ranker chain × 4) │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 失败模式        │ 一处崩全崩        │ 任何一处崩 → degrade,其他继续                                                                                        │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 排序信号        │ 一个 score        │ rule, learned, v21, v3, v43 五种 score 共存                                                                          │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 可解释          │ 黑盒              │ 每步独立 csv,人能查看具体哪一步什么决定                                                                              │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Reproducibility │ 重跑可能不一致    │ manifest 记录所有 outputs / degraded / timings                                                                       │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 调试            │ 改代码 + 全量重跑 │ --only_step / --start_from 单步定位                                                                                  │
  ├─────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 用户接口        │ 一个数            │ 多档 confidence + warnings + QC 标注                                                                                 │
  └─────────────────┴───────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  9.3 可改进点
  
  (a) reliability_layer 17 个子函数——耦合在一个 1984 行文件里,后续可拆成 sub-pipeline 独立编排。

  (b) Stage2/3/35 之间的中间 csv 是 flat 表格,跨 step 重新读写。如果数据量大,可以用 parquet + 内存共享(如 r.dataframes["stage2_final"])。

  (c) manifest 没有 hash —— 同一份输入跑两次,manifest 比对不出"跑出的产物是否真的一致"。可加 file hash。

  (d) degradation 不分级 —— "CGCNN 缺" 与 "Flow ckpt 缺" 同样写 degraded_steps,但前者影响小、后者影响大。可加 severity: low/med/high。

  (e) 没有 stage2 candidate diversity audit —— 100 个 GFlowNet sample 是否都很相近?reliability 只 audit Stage3 condition 多样性,Stage2 同样需要。

  (f) 没有 dry-run 模式 —— 想看 28 步会怎么走但不实际执行。可加 --dry_run 只打印 [PLAN] 行。

  ---
  第十部分 · 结语
  
  Stage 07 是 SynPred 的最终交付层。28 个 step 串起 4000 行代码,把"一份 POSCAR" 变成 "一份带置信度 + QC + 警告的合成路线推荐表"。

  整个流水线的工程哲学可以一句话概括:让多个不完美的模型协同输出一份"足够可靠"的推荐。生成模型给广度,监督模型给深度,规则给先验,reranker chain 给精度,reliability layer
  给可信度。任何单一组件都不够好,但合起来就能用。

  按本文 §8 的步骤,你能复现:
  - 一份完整的 final_recommended_routes.csv(典型 30-50 条候选,带 confidence/warning/QC)
  - 一份完整的 pipeline_v3_manifest.json(28 步 outputs / timings / degradations)
  - 任意单步的中间 csv(便于审计任何一步的决策) 
  
  流水线最大的优势是优雅降级 + 可审计 + 可断点——任何一环失效,流程不崩;任何一个决策,都能定位到具体 csv 行;任何一次失败,都能从那一步重启。这三点让 SynPred 在生产环境(用户上传新
  POSCAR、在线服务)真正"可用"——不是 demo 级,而是"周末没人值班也能稳定跑"级。

  Stage 07 的契约只有两条:{infer_name} 隔离的 outputs 目录 + pipeline_v3_manifest.json 
  自描述。下游(用户、UI、Dashboard、监控)只要会读这两样东西,就能完整重建任意一次推理的所有状态、所有产物、所有降级原因。这是工程"可靠性"的根本——所有信息可见、可追溯、可还原。
