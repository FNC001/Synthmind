先说一句话总结

04_refine_strict_exact_for_structdesc.py 的核心任务是：

从严格精确匹配的数据 strict_exact_only.jsonl 中，筛出“目标公式明确、前驱体明确、文本污染少、条件信息可提取”的样本，并分别整理成
stage2_gold / stage2_train_relaxed / stage3_gold / stage3_train_relaxed
这四类数据集，供后续结构描述符、前驱体预测和条件预测使用。

⸻

这个脚本的输入输出

输入

默认输入：/Users/wyc/SynPred/data/raw/strict_exact_only.jsonl

输出

默认输出目录：/Users/wyc/SynPred/data/interim/refined/structdesc_refined

会生成 7 个文件：
stage2_gold.jsonl
stage2_train_relaxed.jsonl
stage3_gold.jsonl
stage3_train_relaxed.jsonl
dropped_records.jsonl
summary.json
reason_summary.json

这几个输出文件分别是什么

1. stage2_gold.jsonl这是给stage2高质量任务的金标准数据。stage2 在你这个项目里基本就是：结构 / 描述符 → 前驱体集合预测

所以这里要求样本必须：
	•	目标材料明确
	•	主前驱体明确
	•	没有明显变量配方
	•	文本不太脏
	•	没有明显冲突

⸻

2. stage2_train_relaxed.jsonl这是给 stage2 训练 的放宽版数据。它允许一些轻微问题存在，但不能有致命错误。
目的是：尽量保留更多训练样本，提高覆盖率。

⸻

3. stage3_gold.jsonl这是给 stage3 条件预测 的高质量数据。
stage3 基本对应：结构 / 前驱体 → 温度、时间、气氛、溶剂等工艺条件

所以不仅要 stage2 的质量过关，还必须：
	•	能从 operation 层提取出条件
	•	最好存在 heat-like 操作（加热、煅烧、退火、干燥等）

⸻

4. stage3_train_relaxed.jsonl这是给 stage3 训练 的放宽版数据。

允许条件不是从 operation 里直接提出来，而是从行级字段 fallback 得到，比如：
	•	max_temperature_c
	•	total_time_h
	•	atmosphere
	•	solvent

⸻

5. dropped_records.jsonl被彻底丢弃的样本。

也就是：
	•	不适合 stage2 relaxed
	•	也不适合 stage3 relaxed

这种样本基本就是问题太严重了。

⸻

6. summary.json

记录总数统计，比如：
	•	输入多少条
	•	stage2 gold 有多少
	•	stage3 relaxed 有多少
	•	drop 了多少
⸻

7. reason_summary.json记录原因统计，比如：
	•	哪些 severe reason 最常见
	•	哪些 mild reason 最常见
	•	stage3 relaxed 的条件来源是 operation 还是 row_fallback

这个文件很重要，能让你知道数据质量瓶颈在哪。

这个脚本内部拆成 6 个核心模块来看。

一、读取 strict_exact_only.jsonl
rows = read_jsonl(args.input_path)说明它不是从最原始数据开始，而是建立在：你已经做过 strict exact matching 的基础上。

二、识别公式、前驱体和文本中的问题
1. 识别目标材料候选公式
函数：target_formula_candidates(row)
它会从这些字段里找目标材料公式：
	•	mp_formula
	•	synth_formula
	•	raw_synthesis_record.target.material_formula
	•	parent_formula
也就是说，它允许目标公式来自不同来源，只要能找到一个合理的。

⸻

2. 提取前驱体列表
函数：precursor_formula_list(row)
会从：
	•	raw_synthesis_record.precursors
	•	或 row["precursors"]
里抽取前驱体公式。
它还能兼容前驱体是 dict 或字符串的情况。
⸻

3. 把前驱体分成主前驱体和辅助物种
函数：split_precursors(precursors)
它定义了一批辅助物种：
AUX_SPECIES = {
    "O2", "H2O", "CO2", "NH3", "N2", "H2", "Ar", "He", ...
}
所以像这些不会算作真正的主前驱体，而是归到 aux_precursors。
这个设计很合理，因为后面 stage2 预测的核心目标通常是：
真正参与组成构建的主要前驱体，而不是气氛或简单辅助分子。

三、判断样本是否“脏”或“不稳定”
这部分是脚本的质量控制核心。
⸻
1. 检查变量型配方
函数：split_variable_reasons(row)
它会找这些问题：
	•	一个记录里有多个目标材料
	•	target 里有 amount variables
	•	reaction string 里有 x/y/z/δ
	•	synthesis_text 里有 x/y/z/δ
	•	precursor 公式里有变量表达

比如这类都容易出问题：
	•	Li_xCoO2
	•	La1-xSrxMnO3
	•	RE = Y, Gd
	•	x = 0.1, 0.2
这些对结构描述符和监督学习都不友好，因为目标不唯一。

其中它把问题分成：
	•	severe
	•	mild
这很关键，后面 gold / relaxed 就按这个区分。
⸻

2. 检查文本污染和泛化描述
函数：paragraph_contamination_reasons(text)
会检测文本里有没有这些可疑词：
electrochemical
battery
coin cell
cathode
anode
separator
BET surface area
polymer-coated
这些通常说明这段文字可能已经不是“合成段落”本身，而混入了：
	•	电化学测试
	•	器件描述
	•	性能表征
	•	材料应用描述
也会检测过于泛泛的话，比如：
	•	according to the previous method
	•	according to the literature
	•	prepared under similar conditions
这类文本没有足够具体的合成信息。

另外如果文本长度太短：
len(text.strip()) < 30
会直接判成 severe 的 text_too_short。
3. 检查文本中的公式是否和目标冲突
函数：target_text_conflict(row, text)
它会看：
	•	文本里提到的化学式
	•	去掉前驱体和辅助物种之后
	•	是否还能找到与目标公式完全不一致的其他公式

如果有，说明这段话可能不是在描述当前目标材料，而是混进了别的材料。

这是一个很聪明的规则。

⸻

4. 检查文本中的前驱体是否冲突
函数：precursor_text_conflict(row, text)
当文本中出现类似：
	•	starting materials
	•	precursors
	•	used as starting materials
时，它会检查文本里提到的化学式和结构化前驱体列表是否一致。如果不一致，就说明抽取结果和原文可能对不上。

⸻

四、提取工艺条件

这部分是为 stage3 做准备的。
⸻
1. 从 operation 层提取
函数：extract_conditions_from_operations(row)
它会从：raw_synthesis_record.operations
里找各种操作类型：
	•	HeatingOperation
	•	DryingOperation
	•	AnnealingOperation
	•	CalciningOperation
然后提取：
	•	temperature_c
	•	time_h
	•	atmosphere
	•	solvent
还会保留：
	•	all_temps_clean
	•	all_times_clean
	•	all_atmos_clean
	•	all_solvents_clean
	•	n_heatlike_ops
这说明它不只提取单个值，也保留了清洗后的完整候选列表。

温度转换
如果原来是 K，会转成 ℃：v = v - 273.15
时间转换,如果原来是分钟或天，会转成小时：
	•	min → h
	•	day → h
这一步非常标准，说明脚本在做统一单位规范化。
⸻

2. 从行级字段 fallback 提取
函数：
extract_conditions_row_fallback(row)
如果 operation 里没有条件，就从这些字段兜底：
	•	max_temperature_c
	•	total_time_h
	•	atmosphere
	•	all_atmospheres
	•	solvent
	•	all_solvents
这个逻辑是为了尽量别浪费样本。
⸻
3. relaxed 条件合并
函数：merge_conditions_for_relaxed(op_cond, fallback_cond)
规则很简单：
	•	operation 里有就优先用 operation
	•	没有就用 fallback
这就是 stage3 relaxed 的来源。
⸻

五、给每条样本打标签：能不能进入 stage2 / stage3这是整个脚本最核心的判定逻辑。

⸻
1. 先定义一些硬性失败条件hard_fail_for_stage2_relaxed = {
    "missing_target_formula",
    "no_main_precursors",
    "multiple_targets_in_record",
    "target_has_amount_variables",
    "variable_pattern_in_reaction",
    "variable_precursor_formula",
    "text_too_short",
}
这些只要出现，stage2 relaxed 就直接不收。
说明这些问题属于“根本没法建模”的类型。
⸻

2. 定义 gold 额外不能接受的 mild 问题
gold_blocking_mild = {
    "too_generic_text",
    "paragraph_contamination",
    "target_text_conflict",
    "precursor_text_conflict",
    "variable_pattern_in_text",
}
这些问题 relaxed 可以忍，但 gold 不接受。这很合理，因为 gold 追求的是高纯度，不是高覆盖。
⸻

3. stage2 判定规则
stage2_relaxed_ok = not any(r in hard_fail_for_stage2_relaxed for r in severe)
只要没有 severe 的硬失败项就行。

stage2_gold_ok = stage2_relaxed_ok and not any(r in gold_blocking_mild for r in mild)
即：
	•	先满足 relaxed
	•	再没有 gold 阻断的 mild 问题
⸻

4. stage3 判定规则
stage3_gold_ok = stage2_gold_ok and op_has_any_condition and base_out.get("n_heatlike_ops", 0) > 0
也就是说 stage3 gold 必须：
	•	先是 stage2 gold
	•	operation 层有条件
	•	还要确实有 heating / annealing / calcining 这类操作
这个标准很严格，也很适合条件预测。

⸻

stage3_relaxed_ok = stage2_relaxed_ok and relaxed_has_any_condition
只要：
	•	stage2 relaxed 过关
	•	operation 或 fallback 至少能拿到一个条件
就可以进入 stage3 relaxed。
⸻

六、组装输出字段
函数：build_base_out_row(...) 会把每条样本整理成标准结构，包含：
基本身份信息
	•	id
	•	synth_uid
	•	source_dataset
	•	record_index
	•	material_id
	•	doi
	•	dois
	•	split_group
公式相关
	•	formula
	•	mp_formula
	•	synth_formula
	•	parent_formula
文件路径
	•	poscar_path
	•	summary_json_path
	•	provenance_json_path

反应和文本
	•	reaction_string
	•	synthesis_type
	•	synthesis_text

前驱体
	•	main_precursors
	•	aux_precursors
	•	n_main_precursors
	•	n_aux_precursors

operation 条件
	•	temperature_c_op
	•	time_h_op
	•	atmosphere_op
	•	solvent_op
	•	all_temps_clean
	•	all_times_clean
	•	all_atmos_clean
	•	all_solvents_clean
	•	n_heatlike_ops

fallback 条件
	•	temperature_c_fallback
	•	time_h_fallback
	•	atmosphere_fallback
	•	solvent_fallback
另外还会加上：
	•	_refine_severe_reasons
	•	_refine_mild_reasons

这个很有用，后续你能回溯每条样本为什么被收或被丢。
⸻
这个脚本在你的整个项目里扮演什么角色

如果放到全流程里，它的位置大概是：
原始 MP + synthesis 数据
    ↓
02_prepare_dataset.py
    ↓
strict_exact_only.jsonl
    ↓
04_refine_strict_exact_for_structdesc.py
    ↓
stage2/stage3 的 refined 数据
    ↓
structdesc / hybrid / graph 特征构建
    ↓
训练 stage2 / stage3 模型
所以它本质上是：
从“严格匹配后的原始样本”到“真正可训练的数据集”的桥梁。
这个脚本做了三件事：
1. 清洗
去掉明显不可靠的样本
2. 规范化
统一公式、前驱体、温度、时间、气氛、溶剂表达
3. 分层建库
根据质量分成：
	•	stage2 gold
	•	stage2 relaxed
	•	stage3 gold
	•	stage3 relaxed
⸻

因为你后面的模型很依赖输入标签质量：
对 stage2
如果前驱体抽得不准，模型学到的就是噪音。
对 stage3
如果温度时间来源混乱，回归模型会被严重污染。
所以这个脚本其实是你整个 SynPred 数据质量控制的关键节点之一。
⸻
这个脚本最值得注意的几个设计亮点
我觉得有 5 个点很不错：
1. gold / relaxed 双轨制
不是简单一刀切丢数据，而是保留高质量和高覆盖两个版本。
2. severe / mild 分级
让规则更有弹性，不会过度过滤。
3. operation 优先，row fallback 兜底
适合 stage3 条件预测，兼顾精度和样本量。
4. 主前驱体 / 辅助物种分开
更符合化学建模需求。
5. 明确记录每条样本的剔除原因
后期非常方便 debug 和写方法部分。
⸻

如果你要写论文/项目说明，可以这样描述它
We further refined the strictly matched synthesis-structure records by removing ambiguous targets, variable-composition formulas, contaminated or overly generic synthesis text, and records lacking identifiable main precursors. We then normalized precursor roles and synthesis conditions, and constructed four downstream datasets: stage2 gold, stage2 relaxed, stage3 gold, and stage3 relaxed, corresponding to different quality levels for precursor prediction and condition prediction tasks.
⸻

最后给你一个最简版结论
这个脚本不是普通清洗脚本，而是一个：
面向下游 stage2/stage3 任务的数据精修与分层构建脚本。
它完成了：
	•	公式和前驱体规范化
	•	文本冲突和变量配方过滤
	•	工艺条件抽取与单位统一
	•	gold / relaxed 数据集构建
	•	dropped 样本与原因统计输出
