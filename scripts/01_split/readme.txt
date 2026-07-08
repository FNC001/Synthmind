原始数据
  ↓
02_prepare_dataset.py
  ↓
strict_exact_only.jsonl
  ↓
04_refine_strict_exact_for_structdesc.py
  ↓
stage2_gold / stage2_relaxed / stage3_gold / stage3_relaxed
  ↓
01_make_group_split.py
  ↓
stage2_train/val/test, stage3_train/val/test

所以它不是做特征，不是做训练，而是做正式训练前的数据集划分。

因为如果你直接随机按“行”切分，很容易出现这种情况：
	•	同一篇 DOI 的不同记录，一部分进 train，一部分进 test
	•	同一个 synth_uid 的变体样本，分到不同集合
	•	同一个材料公式或同一个 material_id 的样本同时出现在 train 和 val/test

这样模型就会“见过很像的样本”，导致测试分数虚高。

所以这个脚本采取的是：按组切分，而不是按行切分

这就是它名字里 group_split 的意思。

输入

它读 4 个文件：
--stage2_gold
--stage2_relaxed
--stage3_gold
--stage3_relaxed

默认路径是：
/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage2_gold.jsonl
/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage2_train_relaxed.jsonl
/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage3_gold.jsonl
/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage3_train_relaxed.jsonl
这说明它默认接的是你上一步 refine 之后的数据。


输出

它会写出：
对 stage2：
	•	stage2_train.jsonl
	•	stage2_val.jsonl
	•	stage2_test.jsonl
	•	stage2_gold_train_holdout.jsonl

对 stage3：
	•	stage3_train.jsonl
	•	stage3_val.jsonl
	•	stage3_test.jsonl
	•	stage3_gold_train_holdout.jsonl

以及两个说明文件：
	•	group_manifest.json
	•	split_summary.json

⸻

核心逻辑是什么

可以分成 5 个部分来看。
⸻
一、先定义“分组键”是什么

函数：def get_group_key(row):
    for key in ["split_group", "doi", "synth_uid", "material_id", "formula", "id"]:
        ...
它会按这个优先级找分组键：
	1.	split_group
	2.	doi
	3.	synth_uid
	4.	material_id
	5.	formula
	6.	id

只要某个字段存在且非空，就拿它作为这个样本所属的组。

如果都没有，才会退化成：UNKNOWN_GROUP::{id}

这意味着什么

也就是说，这个脚本希望优先按更强的“共享来源”来分组：
	•	如果有 split_group，就用它
	•	没有的话，优先 DOI
	•	再不行用合成记录 ID
	•	再不行用材料 ID
	•	再不行公式
	•	最后才用样本自身 ID

这套优先级设计是合理的，因为：
	•	split_group 往往是前面就专门准备好的防泄漏字段
	•	doi 很适合防止同一篇文章同时出现在 train/test
	•	synth_uid 适合防止同一合成记录的多个变体泄漏

⸻

二、把 gold 数据先按组切分

函数：split_gold_groups(...)
这是整个脚本最重要的地方。

它的做法是：
	1.	先把 gold_rows 按 group 分组
	2.	打乱 group 顺序
	3.	按组依次分配到：
	•	val
	•	test
	•	holdout

直到满足目标比例为止。

⸻

为什么只用 gold 来决定 val/test 组？
因为它的设计思想是：
	•	验证集和测试集只从 gold 中取
	•	relaxed 更多是为了补充训练集样本量

这很合理。因为 val/test 应该尽量干净，才能真正反映模型泛化性能。

⸻

比例怎么算
默认参数：
--val_ratio 0.15
--test_ratio 0.15
它会根据 gold_rows 总行数计算目标行数：
target_val_rows = round(total_rows * val_ratio)
target_test_rows = round(total_rows * test_ratio)
注意这里是按“目标行数”控制，但真正分配单位是“组”。

所以最终：
	•	不是精确 15%
	•	而是尽量接近 15%

因为一个组可能有多条样本，不能拆开。
holdout_groups 是什么

剩下没进 val/test 的 gold 组，就进入：holdout_groups

这个其实就是：gold 中留给训练的组

后面会叫：gold_train_holdout
名字有点绕，但本质就是“训练可用的 gold 部分”。

三、如何构造 train / val / test

函数：split_one_task(...)
它对 stage2 和 stage3 各跑一遍。

逻辑如下：
1. val/test 只取 gold
val_rows = filter_rows_by_groups(gold_rows, val_group_set)
test_rows = filter_rows_by_groups(gold_rows, test_group_set)
gold_train_holdout_rows = filter_rows_by_groups(gold_rows, holdout_group_set)
也就是说：
	•	val 来自 gold
	•	test 来自 gold
	•	train 的 gold 部分来自 holdout 的 gold

这保证 val/test 数据质量较高。
2. relaxed 只用于训练，而且要避开 val/test 的 group
relaxed_train_rows = filter_rows_not_in_groups(relaxed_rows, blocked_groups)
这里的 blocked_groups = val_groups ∪ test_groups

意思是：任何属于val/test组的relaxed 样本，都不能进入训练集。

这是整个程序防泄漏的关键。

因为 relaxed 里可能包含 gold 的相近记录，必须挡掉。
⸻

3. train 怎么组成

这里有一个参数：--include_gold_train_in_relaxed_train
如果你开启它，那么：train_rows = relaxed_train_rows + gold_train_holdout_rows
也就是训练集 = relaxed 训练样本 + gold 训练保留样本
如果不开启：train_rows = relaxed_train_rows
也就是训练集只用 relaxed。

⸻

这代表两种训练策略

策略 A：不开启

训练集只用 relaxed
验证测试集用 gold
适合你想明确区分：
	•	训练可以稍脏一点
	•	验证测试必须干净
⸻

策略 B：开启
训练集 = relaxed + gold holdout
验证测试集 = gold
这样训练数据更多，而且 gold 的高质量训练样本也会被利用。
通常这个更常见，也更划算。
⸻

四、输出哪些文件
它会对 stage2 和 stage3 各自输出四类 split：
train
val
test
gold_train_holdout

所以最终有 8 个 jsonl 文件。
其中：
train 真正训练时用的集合

val 验证集，只来自 gold

test 测试集，只来自 gold

gold_train_holdout 训练侧保留下来的 gold 样本集合

这个文件主要是为了透明化和复现实验，方便你知道：
	•	哪些 gold 没进 val/test
	•	是否要把它们并到 train 里
⸻

五、写出两个说明文件
⸻

1. group_manifest.json

这个文件记录的是“组级别”的划分结果：
{
  "stage2": {
    "val_groups": [...],
    "test_groups": [...],
    "gold_train_holdout_groups": [...]
  },
  "stage3": {
    ...
  }
}
这个很重要，因为它记录了：
	•	哪些 group 被分到了 val
	•	哪些 group 被分到了 test
	•	哪些 group 属于 train-side holdout
以后你想复现实验或者做别的特征版本，只要沿用同一组划分，就能保证可比性。

⸻
2. split_summary.json

这个文件是划分统计摘要，包括：
	•	输入 gold/relaxed 各多少条
	•	train/val/test 各多少条
	•	每部分有多少 group
	•	有多少 DOI
	•	source_dataset 分布
	•	synthesis_type 分布

这是个非常实用的检查文件。

你能拿它快速看出：
	•	划分比例是不是差不多合理
	•	数据源是不是失衡
	•	某种 synthesis_type 是否只集中在一个 split
⸻

函数级分析

下面我把主要函数逐个翻成“人话”。
⸻
read_jsonl / write_jsonl / write_json基础 I/O，不复杂。
⸻

get_group_key(row)

给每条样本找“它属于哪一组”。
优先级很重要：split_group > doi > synth_uid > material_id > formula > id
group_rows(rows)
把所有样本按 group key 聚成字典：
{
  group1: [row1, row2, ...],
  group2: [row3, row4, ...]
}

split_gold_groups(...)

把 gold 数据的 group 打乱后分成：
	•	val_groups
	•	test_groups
	•	holdout_groups

这是分组切分的核心函数。
⸻

filter_rows_by_groups(rows, groups)
保留指定 group 的样本。
⸻
filter_rows_not_in_groups(rows, groups)
去掉指定 group 的样本。
这个主要用来从 relaxed 中删除 val/test 相关组。
⸻
summarize_rows(rows)
统计一个 split 的情况，包括：
	•	行数
	•	group 数
	•	有 DOI 的行数
	•	source_dataset 前十
	•	synthesis_type 前十
⸻

split_one_task(...)
对某一个任务（stage2 或 stage3）执行完整切分。
输出：
	•	train
	•	val
	•	test
	•	gold_train_holdout
	•	val_groups
	•	test_groups
	•	holdout_groups
⸻

main()

解析参数、读取 4 个输入文件、分别切分 stage2/stage3、写出所有结果。
⸻
这个程序的设计思路总结
它采用的是一种很合理的实验划分策略：
1. gold 决定验证/测试
保证评估集质量高。
2. relaxed 用来扩充训练
提升训练集规模。
3. 按组切分，防止泄漏
避免同 DOI / 同合成记录 / 同材料跨集合。
4. 提供 group manifest
方便后续不同特征版本复用同一切分。
⸻

这个程序和前一个 refine 脚本的关系
前一个脚本负责：
“什么样本能进 stage2/stage3”
这个脚本负责：
“进来的样本怎么分 train/val/test”
也就是：
	•	04_refine_strict_exact_for_structdesc.py：质量控制 + 数据分层
	•	01_make_group_split.py：防泄漏切分 + 训练验证测试集构建
两者配合起来，才构成完整的可训练数据准备流程。
⸻

你需要把参数改成你现在实际的路径，比如类似：
python 01_make_group_split.py \
  --stage2_gold /Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage2_gold.jsonl \
  --stage2_relaxed /Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage2_train_relaxed.jsonl \
  --stage3_gold /Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage3_gold.jsonl \
  --stage3_relaxed /Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage3_train_relaxed.jsonl \
  --output_dir /Users/wyc/SynPred/data/interim/splits/structdesc_splits \
  --include_gold_train_in_relaxed_train

最后给你一个最简版结论

这个程序的本质是：
把 refined 后的 stage2/stage3 数据按 group 做无泄漏切分，生成可直接用于模型训练和评估的 train/val/test 数据集。
它的关键价值有三点：
	1.	避免 DOI / 合成记录 / 材料级别的数据泄漏
	2.	让 val/test 保持 gold 高质量
	3.	让 relaxed 数据安全地用于训练扩容
