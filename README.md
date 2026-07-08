# Synthmind

Synthmind is a cleaned public release of the current structure-to-synthesis workflow for inorganic materials. It keeps the best runnable code path for precursor prediction, synthesis-condition prediction, route assembly, confidence/QC post-processing, and final route ranking.

Training datasets, model checkpoints, generated inference outputs, reports, logs, and credentials are intentionally excluded.

## Repository Layout

```text
configs/                  # Public pipeline configs
pipeline/
  core/                   # End-to-end inference steps
  postprocess/            # QC, confidence, reporting, and final ranking helpers
  ranking/                # Route-ranker feature/ranking modules
  run_pipeline.py         # Main pipeline entry point
scripts/                  # Thin user-facing run wrappers
training/                 # Selected training entry points for current models
synthmind/                # Reusable research utilities
research/specs/           # Lightweight task, metric, and split specs
docs/                     # Pipeline/config/output documentation
tests/                    # Lightweight unit tests
```

Removed from the public tree: old staged experiments, paper/table/figure code, remote-machine helper scripts, historical benchmark batches, generated reports, and private data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install GPU-specific packages such as PyTorch/CHGNet/LightGBM according to the target machine CUDA stack.

## Required Artifacts

The configs expect model/data artifacts under ignored local directories:

```text
data/infer/<infer_name>/poscars/
data/interim/...
runs/stage2/...
runs/stage3/...
runs/stage35/...
outputs/inference/...
```

These directories are not tracked by Git. Copy artifacts from the training machine or regenerate them locally before full inference.

## Run Inference

Place POSCAR inputs like this:

```text
data/infer/demo_poscar_test/poscars/
  case_001/POSCAR
  case_002/POSCAR
```

Run the full structure-to-synthesis route pipeline:

```bash
scripts/run_full_route.sh demo_poscar_test
```

Equivalent direct command:

```bash
python pipeline/run_pipeline.py \
  --config configs/full_route.yaml \
  --project_root "$(pwd)" \
  --infer_name demo_poscar_test
```

Run precursor prediction only:

```bash
scripts/run_precursor_only.sh demo_poscar_test
```

Main outputs are written to:

```text
outputs/inference/<infer_name>/
```

The stable user-facing route files are:

```text
final_recommended_routes.csv
final_recommended_routes.md
pipeline_v3_manifest.json
```

## Training Entry Points

Current selected training scripts are grouped by model family:

```text
training/precursor/train_gflownet.py
training/precursor/train_mlp_baseline.py
training/conditions/train_lgbm_method_experts.py
training/conditions/train_lgbm_quantile_ensemble.py
training/ranking/train_route_reranker.py
```

Training scripts expect local data under `data/` and write artifacts under `runs/` or `outputs/`; those paths remain ignored.

## Verification

Lightweight checks:

```bash
python -m compileall -q pipeline training synthmind tests
python -m unittest discover -s tests/research -p 'test_*.py'
python pipeline/run_pipeline.py --help
```

Full prediction requires the excluded model and data artifacts.

## Safety

Synthmind produces predictive synthesis-route recommendations, not validated laboratory procedures. Suggested precursors, temperatures, times, atmospheres, solvents, or process routes must be reviewed by domain experts before experimental use.
