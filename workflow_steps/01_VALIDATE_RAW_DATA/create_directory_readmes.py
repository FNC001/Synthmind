#!/usr/bin/env python3
"""Create concise README files for every numbered data/model/evaluation directory."""

from __future__ import annotations

import argparse
from pathlib import Path


DESCRIPTIONS = {
    "00_OVERVIEW_AND_MANIFEST": ("总览与清单", "保存数据/模型血缘、环境、源压缩包、文件索引和全文件哈希。", "无", "整个发布包"),
    "01_SOURCE_CODE": ("编号化源码", "共享源码保持原导入结构；01–10 包装入口定义执行顺序。", "发布包数据", "训练、评估和推理工作目录"),
    "02_RAW_DATA": ("原始数据", "保存最初合成条目、完整本地结构归档、对齐结果、两套 strict 分支和新增数据。", "文献数据与本地 MP-labelled 结构归档（两种来源证据层级）", "03_CLEANED_AND_MERGED_DATA"),
    "02_RAW_DATA/01_ORIGINAL_SYNTHESIS_RECORDS": ("最初合成条目", "固相和溶液合成原始 JSON，共 67,457 条。", "原始文献数据库", "结构—合成对齐"),
    "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE": ("完整原子结构归档", "62,689 个可解析的本地 MP-labelled 结构：49,283 个有 metadata 支持，13,406 个 conventional 补充缺少包内上游来源元数据。", "历史本地结构快照", "结构对齐、特征与图缓存"),
    "02_RAW_DATA/03_STRUCTURE_SYNTHESIS_ALIGNMENT": ("结构—合成对齐", "38,505 个 direct-aligned、28,952 个 unmatched 和 83,577 个候选。", "原始合成条目 + 本地 MP-labelled 结构归档", "strict 分支与清洗"),
    "02_RAW_DATA/04_STRICT_FILTER_OUTPUTS": ("strict 分支", "01 是远程训练 alignment split；02 是 June merge 使用的 legacy 分支，禁止覆盖。", "direct-aligned", "基础 Stage2 或 June merge"),
    "02_RAW_DATA/05_NEW_20260608_SOURCE_TABLES": ("新增合成源表", "保存 2026-06-08 清洗后的关系表及完整性报告。", "新增数据库", "新增 direct alignment"),
    "02_RAW_DATA/06_NEW_20260608_DIRECT_ALIGNMENT": ("新增对齐结果", "58,593 条新增对齐记录及报告。", "新增合成源表 + 本地 MP-labelled 结构归档", "June merged 数据"),
    "03_CLEANED_AND_MERGED_DATA": ("清洗与合并", "保存未附结构/附结构 merge、refined、route unified、units normalized 和丢弃审计。", "02_RAW_DATA", "04_SPLITS"),
    "03_CLEANED_AND_MERGED_DATA/01_MERGED_20260609": ("June 未附结构合并", "旧 strict 分支与新增数据的合并 checkpoint。", "legacy strict + new aligned", "附结构 merge"),
    "03_CLEANED_AND_MERGED_DATA/02_MERGED_WITH_STRUCTURES": ("June 附结构合并", "55,241 条 train-ready，结构覆盖 100%。", "未附结构 merge + 本地 MP-labelled 结构归档", "merged refined"),
    "03_CLEANED_AND_MERGED_DATA/03_BASE_REFINED": ("基础与 merged refined", "gold/relaxed/dropped 等清洗产物。", "strict 或 merged-with-structures", "分组切分"),
    "03_CLEANED_AND_MERGED_DATA/04_ROUTE_UNIFIED": ("路线统一", "把不同来源的路线/任务字段统一到共同 schema。", "merged refined", "单位归一化"),
    "03_CLEANED_AND_MERGED_DATA/05_UNITS_NORMALIZED": ("条件单位归一化", "统一温度、时间等单位并保存 before/after 审计。", "route unified", "route split 与 Stage3 特征"),
    "03_CLEANED_AND_MERGED_DATA/06_DROPPED_AND_AUDITS": ("丢弃与审计索引", "本目录只放汇总索引；逐行 dropped/audit 文件保留在其产生步骤旁。", "所有清洗步骤", "发布审计"),
    "04_SPLITS": ("数据切分", "基础、路线和最终 family 分组清单。", "refined 数据", "特征与训练"),
    "04_SPLITS/00_PORTABLE_SPLITS": ("便携 split", "base 与 route 两套 Stage2/Stage3 split 的标准 JSONL 副本；活动结构路径均为包根相对路径。", "01/02 historical split checkpoints", "Stage2/Stage3 特征重建"),
    "04_SPLITS/00_PORTABLE_SPLITS/01_BASE_GROUP_SPLIT": ("便携 base split", "8 个 base split 标准 JSONL，共 50,409 行。", "01_BASE_GROUP_SPLIT", "Stage2 描述符与图特征"),
    "04_SPLITS/00_PORTABLE_SPLITS/02_ROUTE_GROUP_SPLIT": ("便携 route split", "8 个 route split 标准 JSONL，共 83,116 行。", "02_ROUTE_GROUP_SPLIT", "Stage3 路线特征"),
    "04_SPLITS/01_BASE_GROUP_SPLIT": ("基础分组切分", "seed 42 的历史中间切分，不是最终 family 切分。", "base refined", "基础描述符/Stage2 graph"),
    "04_SPLITS/02_ROUTE_GROUP_SPLIT": ("路线分组切分", "单位归一化路线数据的分组切分。", "units normalized", "Stage3 task views"),
    "04_SPLITS/03_FINAL_FAMILY_MANIFESTS": ("最终 family manifests", "保存 Stage2/Stage3 最终切分输入哈希和零交叉审计。", "最终数据 builder", "训练与评估"),
    "05_FEATURES_AND_EMBEDDINGS": ("特征与结构嵌入", "131 描述符、CHGNet graph/64维 embedding 和 195维 hybrid。", "04_SPLITS + 本地结构归档", "06_TRAIN_READY_DATA"),
    "05_FEATURES_AND_EMBEDDINGS/01_STRUCTURAL_DESCRIPTORS": ("结构描述符", "基础 Stage2 描述符及 Stage3 单位归一化 poscar geometry 特征。", "split JSONL + structures", "hybrid 或 Stage3 mixed"),
    "05_FEATURES_AND_EMBEDDINGS/02_CHGNET_GRAPH_CACHE": ("CHGNet 图缓存", "从 POSCAR 构图；82 个超大结构等丢弃项由原 summary 记录。", "Stage2 split + structures", "CHGNet embeddings"),
    "05_FEATURES_AND_EMBEDDINGS/03_CHGNET_EMBEDDINGS": ("CHGNet 嵌入", "与保留图缓存行对齐的 64 维嵌入。", "CHGNet graph cache", "hybrid features"),
    "05_FEATURES_AND_EMBEDDINGS/04_HYBRID_FEATURES": ("Stage2 混合特征", "131 descriptor + 64 CHGNet，共 195 维。", "descriptors + embeddings", "Stage2 GFlowNet-ready 数据"),
    "06_TRAIN_READY_DATA": ("训练就绪数据", "按生成顺序保存 Stage2/Stage3 的直接训练输入。", "特征、切分和化学检查", "07_BEST_MODELS"),
    "06_TRAIN_READY_DATA/01_STAGE2_GFLOWNET_RELAXED": ("Stage2 relaxed NPZ", "21,333/635/629 行、195 维、1,968 历史标签。", "hybrid relaxed mode", "full family split"),
    "06_TRAIN_READY_DATA/02_STAGE2_GFLOWNET_GOLD": ("Stage2 gold NPZ", "2,978 条 gold 训练 metadata 与对应验证/测试视图。", "hybrid gold mode", "quality tier 标记"),
    "06_TRAIN_READY_DATA/03_STAGE2_FAMILY_FULL": ("Stage2 全库 family 数据", "把 relaxed 三拆分合并后重新进行 family 分组。", "relaxed NPZ + gold meta", "canonical labels"),
    "06_TRAIN_READY_DATA/04_STAGE2_CANONICAL": ("Stage2 最终规范数据", "18,127/2,170/2,300 行；219 特征；1,780 规范标签。", "full family data", "Stage2 最终模型"),
    "06_TRAIN_READY_DATA/05_STAGE3_CONDITION_MIXED": ("Stage3 mixed 条件数据", "单位归一化后的 133 维历史条件数据。", "Stage3 features", "method stratification"),
    "06_TRAIN_READY_DATA/06_STAGE3_METHOD_STRATIFIED": ("Stage2/3 方法分层共同库", "36,530 个共同 ID 的方法分层数据。", "route-aligned Stage2/Stage3", "chem checked"),
    "06_TRAIN_READY_DATA/07_STAGE3_CHEM_CHECKED_CORE": ("Stage3 化学检查与 core methods", "保留 full method-stratified 与 solution/solid_state/melt_arc core 子集及 ontology/schema。", "method stratified + repaired candidates", "final Stage3 family split"),
    "06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL": ("Stage3 最终 family 数据", "19,788/2,422/2,421 行；155 特征。", "chem-checked core", "NF/CVAE/Diffusion"),
    "07_BEST_MODELS": ("最终入选模型", "只保存最终主链及其直接 checkpoint，不收录失败实验权重。", "06_TRAIN_READY_DATA", "08_GENERATED_OUTPUTS"),
    "07_BEST_MODELS/01_STAGE2_IMPORTED_UPSTREAM": ("Stage2 导入上游", "历史高覆盖候选与 MatSciBERT/template checkpoint；按不可变输入保存。", "历史候选训练链", "factorized/meta/gate"),
    "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS": ("Stage2 三个因子化专家", "A/B/C 三个互补集合专家及候选。", "Stage2 canonical", "meta scores"),
    "07_BEST_MODELS/03_STAGE2_META_AND_GATE": ("Stage2 元排序与最终门控", "s9141/s9144/s9156/s9161、五种子与复跑证据。", "基础候选 + 三专家", "最终 precursor candidates"),
    "07_BEST_MODELS/04_STAGE3_NF": ("Conditional NF", "29,239,356 参数，seed 8060。", "Stage3 final family", "component samples"),
    "07_BEST_MODELS/05_STAGE3_CVAE": ("Hybrid CVAE", "43,248,908 参数，seed 8040。", "Stage3 final family", "component samples"),
    "07_BEST_MODELS/06_STAGE3_DIFFUSION": ("Conditional Diffusion", "128,575,498 参数，seed 8320。", "Stage3 final family", "component samples"),
    "07_BEST_MODELS/07_STAGE3_ENSEMBLE": ("Stage3 集成配置", "保存历史 ensemble 组合与最终 NF+CVAE+Diffusion 选择。", "三个生成模型", "ranked conditions"),
    "08_GENERATED_OUTPUTS": ("训练后生成数据", "保存候选、score、样本、集成和条件排行。", "07_BEST_MODELS", "09_ACCURACY_EVALUATION"),
    "08_GENERATED_OUTPUTS/01_STAGE2_COMPONENT_CANDIDATES": ("Stage2 组件候选", "最终模型各组件候选/分数，以及 Stage3 化学检查使用的修复/校准候选。", "Stage2 components", "meta gate 与 Stage3 chem check"),
    "08_GENERATED_OUTPUTS/02_STAGE2_FINAL_CANDIDATES": ("Stage2 最终候选", "2,170 行冻结验证候选，SHA `ad7ddc...`。", "s9161", "precursor strict metric"),
    "08_GENERATED_OUTPUTS/03_STAGE3_COMPONENT_SAMPLES": ("Stage3 三模型样本", "NF、CVAE、Diffusion 各模型验证样本。", "Stage3 model checkpoints", "192-sample ensemble"),
    "08_GENERATED_OUTPUTS/04_STAGE3_ENSEMBLE_SAMPLES": ("Stage3 集成样本", "每模型 64 个样本，共 192 个/行。", "three component samples", "condition ranking"),
    "08_GENERATED_OUTPUTS/05_STAGE3_RANKED_CONDITIONS": ("最终条件候选", "2,422 行按分桶频率排序的完整条件候选。", "ensemble samples", "full condition/E2E metrics"),
    "09_ACCURACY_EVALUATION": ("精度测试", "只保留 precursor strict、full condition、strict end-to-end 三种口径。", "冻结候选与验证标签", "最终报告"),
    "09_ACCURACY_EVALUATION/01_VALIDATION_REFERENCE": ("验证参考与图", "验证集报告、图和历史结果说明。", "模型评估", "人工查看"),
    "09_ACCURACY_EVALUATION/02_TEST_LOCKBOX": ("测试分割披露", "保存当前最终模型未评估 test 的范围，以及历史非最终模型曾访问 test 的完整披露。", "final test split + historical results", "未来外部盲测策略"),
    "09_ACCURACY_EVALUATION/03_THREE_METRICS": ("三项权威指标", "机器可读 final_three_metrics.json。", "independent evaluator", "对外报告"),
    "09_ACCURACY_EVALUATION/04_RECOMPUTE_TOOLS": ("独立复算工具与结果", "评估器、环境检查、两套复现记录和最新全链复算输出。", "冻结数据/候选", "metric verification"),
    "09_ACCURACY_EVALUATION/05_STABILITY_AND_LEAKAGE": ("稳定性与泄漏审计", "五种子、候选哈希、Stage3 重建和 split 零交叉证据。", "final datasets/models", "release gate"),
    "10_METHODS_AND_GUIDES": ("方法与执行文档", "方法、数据血缘、执行顺序和指标限制。", "全部主链证据", "使用者"),
    "11_TESTS_AND_AUDITS": ("测试与审计", "结构审计、包完整性、代码编译、shell 语法和单元测试。", "全部发布文件", "freeze gate")
}


TEMPLATE = """# {title}\n\n功能：{purpose}\n\n- 上游：`{upstream}`\n- 下游：`{downstream}`\n- 状态：冻结主链资产；不要在本目录原位训练或覆盖文件。\n- 执行入口：参见 `01_SOURCE_CODE/` 对应编号步骤及 `10_METHODS_AND_GUIDES/03_EXECUTION_ORDER.md`。\n\n文件身份、行数和 SHA-256 以 `00_OVERVIEW_AND_MANIFEST/` 中的机器清单为准。\n"""


GENERIC_PURPOSES = {
    "00_OVERVIEW_AND_MANIFEST": "发布环境、不可变来源快照或机器清单的二级资产",
    "01_SOURCE_CODE": "共享源码、参考工具或历史复现资料",
    "02_RAW_DATA": "原始数据或原始对齐 checkpoint",
    "03_CLEANED_AND_MERGED_DATA": "清洗、合并或单位规范化 checkpoint",
    "04_SPLITS": "冻结的数据分组与切分 checkpoint",
    "05_FEATURES_AND_EMBEDDINGS": "描述符、结构图、嵌入或混合特征 checkpoint",
    "06_TRAIN_READY_DATA": "可直接交给相应训练器的冻结数据集",
    "07_BEST_MODELS": "入选模型的一次训练运行目录或权重依赖",
    "08_GENERATED_OUTPUTS": "由入选模型生成的候选、分数或样本",
    "09_ACCURACY_EVALUATION": "精度图、报告、复算工作记录或稳定性证据",
}


GENERIC_TEMPLATE = """# {name}\n\n位置：`{relative}/`  \n功能：{purpose}。\n\n## 正确使用顺序\n\n1. 先阅读上一级 `README.md` 和 `10_METHODS_AND_GUIDES/03_EXECUTION_ORDER.md`；\n2. 本目录是数据、模型或证据资产目录，不把其中的权重、缓存、CSV/NPZ/JSON 当作独立程序执行；\n3. 需要验证或重建时，从 `01_SOURCE_CODE/` 中与上级编号对应的 `run_step.sh` 进入，并把新输出写到包外 `WORK_ROOT`；\n4. 用 `00_OVERVIEW_AND_MANIFEST/FILE_INDEX.tsv` 和 `SHA256SUMS` 核对文件身份。\n\n状态：冻结资产；禁止原位覆盖。\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    created = 0
    for relative, (title, purpose, upstream, downstream) in DESCRIPTIONS.items():
        directory = root / relative
        if not directory.is_dir():
            raise SystemExit(f"Missing directory: {directory}")
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(
                TEMPLATE.format(title=title, purpose=purpose, upstream=upstream, downstream=downstream),
                encoding="utf-8",
            )
            created += 1

    # Cover every meaningful asset directory through depth three. Deeper
    # homogeneous stores (for example poscar/ and summary_json/) are documented
    # by their parent so README files cannot be mistaken for data records.
    generic_created = 0
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        relative_path = directory.relative_to(root)
        if len(relative_path.parts) > 3:
            continue
        readme = directory / "README.md"
        if readme.exists():
            continue
        top = relative_path.parts[0]
        purpose = GENERIC_PURPOSES.get(top, "冻结发布包的二级资产")
        readme.write_text(
            GENERIC_TEMPLATE.format(
                name=directory.name,
                relative=relative_path.as_posix(),
                purpose=purpose,
            ),
            encoding="utf-8",
        )
        generic_created += 1

    print(
        f"known_readmes={len(DESCRIPTIONS)} newly_created={created}; "
        f"generic_depth3_created={generic_created}"
    )


if __name__ == "__main__":
    main()
