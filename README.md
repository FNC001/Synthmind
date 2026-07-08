# Synthmind

Synthmind is a synthesis-route prediction utility suite for inorganic materials. It helps turn structure descriptions into inference-ready POSCAR inputs, organize SynPred route-prediction outputs, and generate polished synthesis-process reports for review.

This public repository is a lightweight release package. It includes reusable code, examples, and documentation, but intentionally excludes training data, model checkpoints, private inference outputs, generated reports, and local credentials.

## What It Does

Synthmind supports a practical structure-to-synthesis workflow:

1. Convert `*_three_tasks.json` structure/task outputs into POSCAR files.
2. Run route prediction in a SynPred model environment that has the trained checkpoints.
3. Collect `final_top_routes_with_condition_confidence.csv` route tables.
4. Build a Word report summarizing predicted precursors, conditions, confidence, QC warnings, and suggested experimental validation steps.

The generated routes are model suggestions, not experimental SOPs. Safety review, phase validation, and domain expert inspection are required before any laboratory use.

## Repository Contents

```text
.
├── examples/
│   └── report_manifest.example.csv
├── scripts/
│   ├── figures/
│   │   └── make_figure3_heldout_performance.py
│   └── reporting/
│       ├── build_synthesis_report.py
│       ├── json_three_tasks_to_poscar.py
│       └── json_three_tasks_to_symmetry_poscar.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Reporting Scripts

- `scripts/reporting/json_three_tasks_to_symmetry_poscar.py`  
  Converts pyxtal-style `*_three_tasks.json` structure descriptions into symmetry-expanded POSCAR files using `pymatgen`.

- `scripts/reporting/json_three_tasks_to_poscar.py`  
  Converts `*_three_tasks.json` files into reduced-formula POSCAR files without symmetry expansion. Use this as a fallback when symmetry expansion is not desired or `pymatgen` is unavailable.

- `scripts/reporting/build_synthesis_report.py`  
  Builds a DOCX synthesis-route report from three-task JSON files and SynPred route CSV outputs.

### Figure Script

- `scripts/figures/make_figure3_heldout_performance.py`  
  Generates held-out performance figures from existing evaluation artifacts. It reads metrics from local output directories and does not embed private performance data.

## Installation

Create a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the symmetry-expanded POSCAR converter, `pymatgen` is required. The reduced POSCAR converter and DOCX report builder use lighter dependencies.

## Input Format

The reporting utilities expect `*_three_tasks.json` files with this shape:

```json
{
  "structure_description": "14 |7.589,5.301,12.036,90.00,119.00,90.00| ...",
  "description_source": "pyxtal",
  "results": [
    {"task": "synthesis", "output": "True"},
    {"task": "method", "output": "solid_state"},
    {"task": "precursor", "output": "['SiO2', 'SO3']"}
  ]
}
```

The report builder also expects one prediction CSV per structure, usually exported by the SynPred route pipeline:

```text
<key>_final_top_routes_with_condition_confidence.csv
```

Required route CSV columns include:

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

## Generate POSCAR Inputs

Symmetry-expanded POSCAR generation:

```bash
python scripts/reporting/json_three_tasks_to_symmetry_poscar.py \
  data/infer/my_batch/poscars \
  /path/to/A_three_tasks.json \
  /path/to/B_three_tasks.json
```

Reduced-formula POSCAR generation:

```bash
python scripts/reporting/json_three_tasks_to_poscar.py \
  data/infer/my_batch/poscars \
  /path/to/A_three_tasks.json
```

Example output layout:

```text
data/infer/my_batch/poscars/
  A/POSCAR
  B/POSCAR
```

## Run Route Prediction

This repository does not include trained SynPred models. Run inference in the private or production environment that contains the route-prediction pipeline and checkpoints.

For best isolation, run each target structure as its own inference case when downstream candidate merging is known to de-duplicate globally by `precursor_set`. Then collect the final route CSV for each structure.

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

You can also use directory discovery when files follow the expected names:

```bash
python scripts/reporting/build_synthesis_report.py \
  --json-dir /path/to/jsons \
  --prediction-dir /path/to/prediction_csvs \
  --output reports/synthesis_report.docx
```

## Data and Model Policy

The following are intentionally excluded from Git:

- training datasets
- model checkpoints and serialized estimators
- intermediate inference outputs
- generated DOCX/PDF/PNG reports
- local credentials and machine-specific paths

The `.gitignore` is configured to keep those artifacts out of the public repository.

## Limitations

- Synthmind outputs are predictive recommendations, not validated recipes.
- Some precursor suggestions may trigger QC warnings, such as missing target elements, extra non-target elements, or elemental precursor use.
- Route confidence and condition support are internal model diagnostics, not experimental validation.
- Any route involving reactive, volatile, corrosive, toxic, oxidizing, reducing, or pressure-generating species must be reviewed before laboratory use.

## License

No license has been selected yet. Add a license file before distributing the code for broad reuse.
