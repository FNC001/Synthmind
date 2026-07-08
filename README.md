# SynPred Utilities

This repository contains lightweight utilities for preparing structure inputs, collecting SynPred route predictions, and building a DOCX synthesis-route report.

Training data, model checkpoints, intermediate inference outputs, generated reports, and local credentials are intentionally excluded from this repository.

## Contents

- `scripts/reporting/json_three_tasks_to_symmetry_poscar.py`  
  Converts `*_three_tasks.json` files with pyxtal-style structure descriptions into symmetry-expanded POSCAR files using `pymatgen`.

- `scripts/reporting/json_three_tasks_to_poscar.py`  
  Converts `*_three_tasks.json` files into reduced-formula POSCAR files without symmetry expansion. This is useful as a fallback when symmetry expansion is not desired.

- `scripts/reporting/build_synthesis_report.py`  
  Builds a Word report from three-task JSON files and SynPred `final_top_routes_with_condition_confidence.csv` outputs.

- `scripts/figures/make_figure3_heldout_performance.py`  
  Generates held-out performance figures from existing evaluation artifacts.

## Install

Use a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate POSCAR Inputs

Given one or more `*_three_tasks.json` files:

```bash
python scripts/reporting/json_three_tasks_to_symmetry_poscar.py \
  data/infer/my_batch/poscars \
  /path/to/A_three_tasks.json \
  /path/to/B_three_tasks.json
```

The output layout is:

```text
data/infer/my_batch/poscars/
  A/POSCAR
  B/POSCAR
```

If you do not want symmetry expansion:

```bash
python scripts/reporting/json_three_tasks_to_poscar.py \
  data/infer/my_batch/poscars \
  /path/to/A_three_tasks.json
```

## Run SynPred Inference

This repository does not include trained models or training data. Run the full SynPred route pipeline in the environment that contains the model checkpoints and inference code. The report builder expects one route CSV per structure:

```text
<key>_final_top_routes_with_condition_confidence.csv
```

The required columns include:

- `final_route_rank`
- `precursor_set`
- `temperature_c`
- `time_h`
- `pred_atmosphere`
- `route_confidence_score`
- `route_confidence_level`
- `condition_distribution_support_score`
- `condition_distribution_confidence_level`
- `condition_distribution_recommendation_status`
- `precursor_qc_level`
- `precursor_qc_status`
- `precursor_qc_warnings`

## Build a DOCX Report

Create a manifest CSV with three columns:

```csv
key,json_path,prediction_csv
ExampleMaterial,/absolute/path/to/ExampleMaterial_three_tasks.json,/absolute/path/to/ExampleMaterial_final_top_routes_with_condition_confidence.csv
```

Then run:

```bash
python scripts/reporting/build_synthesis_report.py \
  --manifest examples/report_manifest.example.csv \
  --output reports/synthesis_report.docx
```

You can also use directory discovery when the JSON and CSV files follow the expected naming convention:

```bash
python scripts/reporting/build_synthesis_report.py \
  --json-dir /path/to/jsons \
  --prediction-dir /path/to/prediction_csvs \
  --output reports/synthesis_report.docx
```

## Notes

- The generated routes are model suggestions, not experimental SOPs.
- Always perform safety review and phase validation before using a proposed synthesis route.
- Generated reports, route outputs, model checkpoints, and training data should remain outside Git unless there is an explicit release decision.
