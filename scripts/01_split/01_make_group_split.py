#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_group_key(row: Dict[str, Any]) -> str:
    for key in ["split_group", "doi", "synth_uid", "material_id", "formula", "id"]:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return f"UNKNOWN_GROUP::{row.get('id', 'NO_ID')}"


def group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[get_group_key(row)].append(row)
    return grouped


def split_gold_groups(
    gold_rows: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    grouped = group_rows(gold_rows)
    groups = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_rows = len(gold_rows)
    target_val_rows = max(1, round(total_rows * val_ratio))
    target_test_rows = max(1, round(total_rows * test_ratio))

    val_groups: List[str] = []
    test_groups: List[str] = []
    holdout_groups: List[str] = []

    val_rows = 0
    test_rows = 0

    for g in groups:
        g_rows = len(grouped[g])
        if val_rows < target_val_rows:
            val_groups.append(g)
            val_rows += g_rows
        elif test_rows < target_test_rows:
            test_groups.append(g)
            test_rows += g_rows
        else:
            holdout_groups.append(g)

    # 兜底，避免某一侧为空
    if not val_groups and holdout_groups:
        val_groups.append(holdout_groups.pop())
    if not test_groups and holdout_groups:
        test_groups.append(holdout_groups.pop())

    return val_groups, test_groups, holdout_groups


def filter_rows_by_groups(rows: List[Dict[str, Any]], groups: set) -> List[Dict[str, Any]]:
    return [r for r in rows if get_group_key(r) in groups]


def filter_rows_not_in_groups(rows: List[Dict[str, Any]], groups: set) -> List[Dict[str, Any]]:
    return [r for r in rows if get_group_key(r) not in groups]


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    src_counter = Counter()
    synth_counter = Counter()
    n_rows_with_doi = 0
    groups = set()

    for row in rows:
        groups.add(get_group_key(row))
        if row.get("doi"):
            n_rows_with_doi += 1
        src_counter[str(row.get("source_dataset", "UNKNOWN"))] += 1
        synth_counter[str(row.get("synthesis_type", "UNKNOWN"))] += 1

    return {
        "n_rows": len(rows),
        "n_groups": len(groups),
        "n_rows_with_doi": n_rows_with_doi,
        "source_dataset_top10": dict(src_counter.most_common(10)),
        "synthesis_type_top10": dict(synth_counter.most_common(10)),
    }


def split_one_task(
    gold_rows: List[Dict[str, Any]],
    relaxed_rows: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_gold_train_in_relaxed_train: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    val_groups, test_groups, holdout_groups = split_gold_groups(
        gold_rows=gold_rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    val_group_set = set(val_groups)
    test_group_set = set(test_groups)
    holdout_group_set = set(holdout_groups)
    blocked_groups = val_group_set | test_group_set

    val_rows = filter_rows_by_groups(gold_rows, val_group_set)
    test_rows = filter_rows_by_groups(gold_rows, test_group_set)
    gold_train_holdout_rows = filter_rows_by_groups(gold_rows, holdout_group_set)

    relaxed_train_rows = filter_rows_not_in_groups(relaxed_rows, blocked_groups)

    if include_gold_train_in_relaxed_train:
        train_rows = relaxed_train_rows + gold_train_holdout_rows
    else:
        train_rows = relaxed_train_rows

    return {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "gold_train_holdout": gold_train_holdout_rows,
        "val_groups": val_groups,
        "test_groups": test_groups,
        "gold_train_holdout_groups": holdout_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Group split structdesc refined datasets into train/val/test.")
    parser.add_argument(
        "--stage2_gold",
        type=str,
        default="/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage2_gold.jsonl",
    )
    parser.add_argument(
        "--stage2_relaxed",
        type=str,
        default="/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage2_train_relaxed.jsonl",
    )
    parser.add_argument(
        "--stage3_gold",
        type=str,
        default="/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage3_gold.jsonl",
    )
    parser.add_argument(
        "--stage3_relaxed",
        type=str,
        default="/Users/wyc/SynPred/data/interim/refined/structdesc_refined/stage3_train_relaxed.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/splits/structdesc_splits",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_gold_train_in_relaxed_train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2_gold = read_jsonl(args.stage2_gold)
    stage2_relaxed = read_jsonl(args.stage2_relaxed)
    stage3_gold = read_jsonl(args.stage3_gold)
    stage3_relaxed = read_jsonl(args.stage3_relaxed)

    stage2_split = split_one_task(
        gold_rows=stage2_gold,
        relaxed_rows=stage2_relaxed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_gold_train_in_relaxed_train=args.include_gold_train_in_relaxed_train,
    )
    stage3_split = split_one_task(
        gold_rows=stage3_gold,
        relaxed_rows=stage3_relaxed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_gold_train_in_relaxed_train=args.include_gold_train_in_relaxed_train,
    )

    # write files
    for split_name in ["train", "val", "test", "gold_train_holdout"]:
        write_jsonl(out_dir / f"stage2_{split_name}.jsonl", stage2_split[split_name])
        write_jsonl(out_dir / f"stage3_{split_name}.jsonl", stage3_split[split_name])

    group_manifest = {
        "stage2": {
            "val_groups": stage2_split["val_groups"],
            "test_groups": stage2_split["test_groups"],
            "gold_train_holdout_groups": stage2_split["gold_train_holdout_groups"],
        },
        "stage3": {
            "val_groups": stage3_split["val_groups"],
            "test_groups": stage3_split["test_groups"],
            "gold_train_holdout_groups": stage3_split["gold_train_holdout_groups"],
        },
    }
    write_json(out_dir / "group_manifest.json", group_manifest)

    summary = {
        "config": {
            "base_dir": "/Users/wyc/SynPred/data",
            "stage2_gold": args.stage2_gold,
            "stage2_relaxed": args.stage2_relaxed,
            "stage3_gold": args.stage3_gold,
            "stage3_relaxed": args.stage3_relaxed,
            "output_dir": args.output_dir,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "include_gold_train_in_relaxed_train": args.include_gold_train_in_relaxed_train,
        },
        "stage2": {
            "input_gold": len(stage2_gold),
            "input_relaxed": len(stage2_relaxed),
            "train": summarize_rows(stage2_split["train"]),
            "val": summarize_rows(stage2_split["val"]),
            "test": summarize_rows(stage2_split["test"]),
            "gold_train_holdout": summarize_rows(stage2_split["gold_train_holdout"]),
        },
        "stage3": {
            "input_gold": len(stage3_gold),
            "input_relaxed": len(stage3_relaxed),
            "train": summarize_rows(stage3_split["train"]),
            "val": summarize_rows(stage3_split["val"]),
            "test": summarize_rows(stage3_split["test"]),
            "gold_train_holdout": summarize_rows(stage3_split["gold_train_holdout"]),
        },
    }
    write_json(out_dir / "split_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
