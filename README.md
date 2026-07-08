# Synthmind

Synthmind is the public code release of the SynPred structure-to-synthesis workflow for inorganic materials.  It contains the calculation code for precursor-set prediction, synthesis-condition/process-pool prediction, route assembly, confidence/QC post-processing, and final route ranking.

This repository intentionally excludes training datasets, model checkpoints, private inference outputs, generated reports, logs, and machine credentials.  To reproduce trained predictions you need to supply the corresponding data/model artifacts under the paths expected by the configs, or override those paths in your own config.

## What Is Included

```text
.
├── synpred/                         # Reusable research modules
│   └── research/                    # Candidate pools, schemas, splits, metrics, attribution
├── research/                        # YAML/JSON experiment specs and candidate-budget configs
├── scripts/
│   ├── 00_refine/                   # Raw synthesis-record refinement and normalization
│   ├── 01_split/                    # Group/task split construction
│   ├── 02_features/                 # Structure/condition feature tables
│   ├── 03_data/                     # Stage2/Stage3/Stage35 dataset and candidate builders
│   ├── 03_graph/                    # CGCNN/ALIGNN/CHGNet graph caches and embeddings
│   ├── 04_train/                    # Stage2, Stage3, Stage35 model training code
│   ├── 06_eval/                     # Candidate-pool, calibration, route-stack evaluation
│   ├── 07_infer/                    # Full structure-to-synthesis inference pipelines
│   ├── 08_auto_improve/             # Diagnostic and improvement experiments
│   ├── 09_remote_autodl/            # Remote execution helper scripts without credentials
│   ├── 10_autorun/                  # Long-run training/evaluation queues
│   ├── 11_paper/                    # Paper/table/figure generation helpers
│   ├── 12_research/                 # Additional research training entry points
│   ├── reporting/                   # DOCX/POSCAR reporting utilities
│   └── figures/                     # Figure utilities
├── docs/                            # Pipeline and method documentation
├── tests/                           # Lightweight research-module tests
├── examples/                        # Small manifest examples
└── requirements.txt
```

## Pipeline Overview

The main inference pipeline is:

1. Structure input preparation from POSCAR files.
2. Structural descriptors and graph embeddings.
3. Stage2 precursor candidate generation:
   GFlowNet sampling, composition constraints, retrieval augmentation, fallback completion, baseline ensemble candidates, and element-aware reranking.
4. Stage3 process/condition prediction:
   precursor-conditioned temperature, time, atmosphere, and condition-distribution estimates.
5. Stage35 route construction and ranking:
   readable route table generation, display filtering, rule/learned reranking where artifacts exist, route confidence, precursor QC, condition support, and final recommended-route export.

The primary entry point is:

```bash
python scripts/07_infer/structure_to_synthesis_route/pipeline/run_pipeline.py \
  --config scripts/07_infer/structure_to_synthesis_route/pipeline/configs/full_route_stage3.yaml \
  --project_root "$(pwd)" \
  --infer_name demo_poscar_test
```

For precursor-set prediction only:

```bash
python scripts/07_infer/structure_to_synthesis_route/pipeline/run_pipeline.py \
  --config scripts/07_infer/structure_to_synthesis_route/pipeline/configs/precursor_only.yaml \
  --project_root "$(pwd)" \
  --infer_name demo_poscar_test
```

Expected POSCAR layout:

```text
data/infer/<infer_name>/poscars/
  case_001/POSCAR
  case_002/POSCAR
```

Main outputs are written under:

```text
outputs/inference/<infer_name>/
```

including `final_recommended_routes.csv`, `final_recommended_routes.md`, precursor-only recommendations, route QC tables, confidence layers, and a pipeline manifest when the required model artifacts are present.

## Training And Evaluation Code

The repository includes the complete training/evaluation source code, but not the private training data or trained artifacts:

- Stage2 precursor models:
  `scripts/04_train/stage2/`
- Stage3 condition/process models:
  `scripts/04_train/stage3/`
- Stage35 route-ranker models:
  `scripts/04_train/stage35/`
- Candidate-pool and calibration evaluation:
  `scripts/06_eval/`
- End-to-end training orchestration:
  `scripts/run_prepare_training_data.sh`,
  `scripts/run_train_models.sh`,
  `scripts/run_stage2_stage3_benchmark.sh`,
  `scripts/run_full_pipeline.sh`

These scripts expect training data in `data/` and model outputs in `runs/`; both directories are ignored by Git.

## Installation

Create a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

GPU training and CHGNet/torch inference should be installed according to the CUDA version of the target machine.  CPU-only smoke checks can still validate imports and many data-processing scripts.

## Model/Data Artifacts

The public repository does not contain:

- raw or refined training datasets
- graph caches and feature matrices
- `runs/` model checkpoints or serialized estimators
- private `outputs/` inference/evaluation results
- generated DOCX/PDF/PNG/TGZ artifacts
- credentials, SSH keys, API keys, or machine-local config

Typical artifact paths referenced by the configs include:

```text
data/interim/...
runs/stage2/...
runs/stage3/...
runs/stage35/...
outputs/inference/...
```

Copy or regenerate those artifacts locally before running trained inference.  The pipeline records missing optional rankers as degraded steps; required model/data files must exist for the corresponding enabled stages.

## Utilities

The `scripts/reporting/` utilities remain available for the earlier three-structure report workflow:

- `json_three_tasks_to_symmetry_poscar.py`
- `json_three_tasks_to_poscar.py`
- `build_synthesis_report.py`

These utilities are separate from the main algorithm pipeline and are useful for preparing POSCAR inputs or writing DOCX summaries from route CSV outputs.

## Verification

Run lightweight checks:

```bash
python -m py_compile \
  scripts/07_infer/structure_to_synthesis_route/pipeline/run_pipeline.py \
  scripts/07_infer/structure_to_synthesis_route/pipeline/src/*.py \
  synpred/research/*.py

pytest tests
```

Full end-to-end prediction requires the excluded model/data artifacts.

## Safety Note

Synthmind produces predictive synthesis-route recommendations, not validated laboratory procedures.  Any suggested precursor, temperature, time, atmosphere, solvent, or process route must be reviewed by domain experts before experimental use.

## License

No license has been selected yet. Add a license file before distributing the code for broad reuse.
