#!/usr/bin/env python3
"""Create a label-free family-filtered view of a Stage-2 candidate JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--input_candidates", required=True)
    parser.add_argument("--families", default="")
    parser.add_argument("--anion_signature", default="")
    parser.add_argument("--source_dataset", default="")
    parser.add_argument("--synthesis_type", default="")
    parser.add_argument("--candidate_limit", type=int, default=0)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    allowed = {value.strip() for value in str(args.families).split(",") if value.strip()}
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    requested_anions = "+".join(
        sorted(value.strip() for value in str(args.anion_signature).split("+") if value.strip())
    )
    selected: set[int] = set()
    for index, family in enumerate(families):
        row = meta.iloc[int(index)]
        if allowed and str(family) not in allowed:
            continue
        if requested_anions:
            try:
                anions = "+".join(sorted(str(value) for value in json.loads(str(row.get("target_anion_elements", "[]")))))
            except Exception:
                anions = ""
            if anions != requested_anions:
                continue
        if str(args.source_dataset).strip() and str(row.get("source_dataset", "")) != str(args.source_dataset).strip():
            continue
        if str(args.synthesis_type).strip() and str(row.get("synthesis_type", "")) != str(args.synthesis_type).strip():
            continue
        selected.add(int(index))
    output = Path(args.output_candidates_jsonl).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    candidates_written = 0
    seen: set[int] = set()
    with Path(args.input_candidates).resolve().open(encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            record = json.loads(line)
            row_index = int(record["row_index"])
            if row_index not in selected:
                continue
            if int(args.candidate_limit) > 0:
                record["candidate_label_ids"] = record.get("candidate_label_ids", [
                ])[: int(args.candidate_limit)]
                if "scores" in record:
                    record["scores"] = record.get("scores", [])[: int(args.candidate_limit)]
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
            candidates_written += len(record.get("candidate_label_ids", []))
            seen.add(row_index)
    report = {
        "protocol": "label_free_family_filtered_candidate_view",
        "config": vars(args),
        "selected_rows": int(len(selected)),
        "rows_written": int(rows_written),
        "unique_rows_written": int(len(seen)),
        "missing_selected_rows": int(len(selected - seen)),
        "candidates_written": int(candidates_written),
        "allowed_families": sorted(allowed),
    }
    report_path = Path(args.output_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
