#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Stage2 candidate label IDs onto the canonical precursor-label vocabulary."
    )
    parser.add_argument("--canonicalization_json", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    mapping_record = json.loads(Path(args.canonicalization_json).read_text(encoding="utf-8"))
    old_to_new = [int(value) for value in mapping_record["old_to_canonical_label_id"]]
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    input_candidates = 0
    output_candidates = 0
    duplicate_candidates_removed = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            record: Dict[str, Any] = json.loads(line)
            candidates = list(record.get("candidate_label_ids", []))
            input_candidates += len(candidates)
            remapped: List[List[int]] = []
            kept_indices: List[int] = []
            seen = set()
            for candidate_index, candidate in enumerate(candidates):
                mapped = tuple(sorted({old_to_new[int(label)] for label in candidate}))
                if not mapped or mapped in seen:
                    duplicate_candidates_removed += 1
                    continue
                seen.add(mapped)
                remapped.append(list(mapped))
                kept_indices.append(int(candidate_index))
                if int(args.limit) > 0 and len(remapped) >= int(args.limit):
                    break
            record["candidate_label_ids"] = remapped
            for key, value in list(record.items()):
                if key == "candidate_label_ids" or not isinstance(value, list):
                    continue
                if len(value) == len(candidates):
                    record[key] = [value[index] for index in kept_indices]
            record["canonicalization_version"] = mapping_record["version"]
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
            output_candidates += len(remapped)

    print(json.dumps({
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "rows": rows,
        "input_candidates": input_candidates,
        "output_candidates": output_candidates,
        "duplicate_candidates_removed": duplicate_candidates_removed,
        "mean_candidates_before": input_candidates / max(1, rows),
        "mean_candidates_after": output_candidates / max(1, rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
