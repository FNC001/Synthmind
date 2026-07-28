#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the historical alignment algorithm with explicit paths.")
    parser.add_argument("--data-root", "--release-root", dest="data_root", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("SYNTHMIND_REPO_ROOT", Path(__file__).resolve().parents[2])),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if root == output or root in output.parents:
        raise SystemExit("Refusing to write alignment outputs inside the frozen release directory")

    script = source_root / "scripts/00_refine/02_prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("synthmind_historical_alignment", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mp = root / "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export"
    raw = root / "02_RAW_DATA/01_ORIGINAL_SYNTHESIS_RECORDS"
    module.CONFIG.update(
        {
            "MP_METADATA_CSV": str(mp / "mp_full_archive_metadata.csv"),
            "MP_POSCAR_DIR": str(mp / "poscar"),
            "SYN_FILES": {
                "solid_state": str(raw / "solid-state_dataset_20200713.json"),
                "solution_synthesis": str(raw / "solutionsynthesis_dataset_202185.json"),
            },
            "OUTPUT_DIR": str(output),
            "RESUME": bool(args.resume),
        }
    )
    module.main()


if __name__ == "__main__":
    main()
