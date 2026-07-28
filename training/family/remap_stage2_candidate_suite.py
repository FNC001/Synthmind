#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_named(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize a complete Stage2 validation/OOF candidate suite."
    )
    parser.add_argument("--canonicalization_json", required=True)
    parser.add_argument(
        "--expert_metrics_json",
        required=True,
        help="Metrics JSON whose config.expert list contains NAME=PATH sources.",
    )
    parser.add_argument("--aux", action="append", default=[], help="Repeat NAME=PATH.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.expert_metrics_json).read_text(encoding="utf-8"))
    sources: Dict[str, List[tuple[str, str]]] = {
        "expert": [parse_named(value) for value in metrics["config"]["expert"]],
        "aux": [parse_named(value) for value in args.aux],
    }
    manifest: Dict[str, List[str]] = {"expert": [], "aux": []}
    converter = Path(__file__).with_name("remap_stage2_candidates.py")
    for kind, values in sources.items():
        for name, source_path in values:
            output_path = output_dir / Path(source_path).name
            subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    "--canonicalization_json",
                    args.canonicalization_json,
                    "--input_jsonl",
                    source_path,
                    "--output_jsonl",
                    str(output_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            manifest[kind].append(f"{name}={output_path}")
            print(f"{kind}\t{name}\t{output_path}", flush=True)

    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
