# Synthmind V1.0

Synthmind V1.0 is the formal code release for predicting precursor sets and
synthesis conditions from inorganic crystal structures. The repository
contains the complete frozen source surface directly at the repository root:
data preparation, structure alignment, descriptors and graph features,
Stage2 precursor models, Stage3 generative condition models, evaluation,
inference, audits, tests, and historical reproducibility utilities.

This is not a nested snapshot under an older release directory. The V1.0 code
is the repository itself.

## What is included

```text
chemistry/               Historical chemistry compatibility modules
configs/                 Pipeline and family-routing configurations
pipeline/                Public end-to-end route pipeline
research/                Research schemas and reusable evaluation modules
scripts/                 Data, feature, training, evaluation, inference tools
synthmind/               Installable Python package and V1.0 CLI
tests/                   Chemistry, research, and pipeline tests
training/                Stage2/Stage3 final and experimental training code
workflow_steps/          Supported ordered V1.0 steps 01–10
```

Generated data, structures, embeddings, model checkpoints, and predictions
are deliberately not committed to the public Git repository. Point V1.0 at an
authorized artifact root with `--data-root`; the code never requires the
artifacts to be copied into the Git checkout.

## Install

Python 3.10 or newer is required. A CUDA-capable Linux environment is required
for the selected full training profiles.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[models,test]'
```

The frozen validation environment used Python 3.12, PyTorch 2.8.0+cu128,
LightGBM 4.6.0, CHGNet 0.4.2, pymatgen 2025.10.7, NumPy 2.3.2, and
scikit-learn 1.9.0.

## Supply an authorized data root

The expected artifact root follows the frozen full-lineage layout:

```text
/path/to/synthmind-data/
  00_OVERVIEW_AND_MANIFEST/
  02_RAW_DATA/
  03_CLEANED_AND_MERGED_DATA/
  04_SPLITS/
  05_FEATURES_AND_EMBEDDINGS/
  06_TRAIN_READY_DATA/
  07_BEST_MODELS/
  08_GENERATED_OUTPUTS/
  09_ACCURACY_EVALUATION/
  11_TESTS_AND_AUDITS/
```

Check it before running:

```bash
synthmind check-data \
  --data-root /path/to/synthmind-data \
  --profile validate-full
```

The data root is read-only. New outputs are written beneath `--work-root`.

## Run the supported V1.0 workflow

Exact frozen validation, including Stage2 candidate regeneration, Stage3
64+64+64 ensemble reconstruction, and all three metric reproductions:

```bash
synthmind workflow \
  --data-root /path/to/synthmind-data \
  --work-root /path/to/work/v1 \
  --profile validate-full
```

Equivalent shell entry:

```bash
bash scripts/run_v1_workflow.sh \
  --data-root /path/to/synthmind-data \
  --work-root /path/to/work/v1 \
  --profile validate-full
```

Other supported profiles:

```bash
# Check all prepared datasets and model dependencies without metric replay.
synthmind workflow --data-root /path/to/data --profile validate-fast

# Train selected Stage2 and Stage3 models from prepared data in 06_*.
synthmind workflow --data-root /path/to/data --profile train

# Rebuild final family datasets, then train both stages.
synthmind workflow --data-root /path/to/data --profile rebuild-train
```

Use `--dry-run` to inspect every ordered command without executing it.

## Ordered steps

| Step | Entry | Purpose |
|---|---|---|
| 01 | `workflow_steps/01_VALIDATE_RAW_DATA/` | Raw/package audit |
| 02 | `workflow_steps/02_ALIGN_SYNTHESIS_STRUCTURES/` | Structure–synthesis alignment |
| 03 | `workflow_steps/03_CLEAN_AND_STRATIFY/` | Clean, merge, route/unit normalization |
| 04 | `workflow_steps/04_BUILD_FEATURES/` | Splits, descriptors, CHGNet, hybrid features |
| 05 | `workflow_steps/05_BUILD_STAGE2_DATASET/` | Final Stage2 family/canonical data |
| 06 | `workflow_steps/06_TRAIN_STAGE2/` | Three experts, meta models, final gate |
| 07 | `workflow_steps/07_BUILD_STAGE3_DATASET/` | Final Stage3 family data |
| 08 | `workflow_steps/08_TRAIN_STAGE3/` | NF, CVAE, Diffusion, ensemble |
| 09 | `workflow_steps/09_EVALUATE_THREE_METRICS/` | Exact frozen metric replay |
| 10 | `workflow_steps/10_INFER_END_TO_END/` | V1.0 inference dependency gate |

All original historical scripts used to produce the frozen lineage are also
present under `scripts/` and `training/`. `workflow_steps/` is the supported
portable entry surface.

## Frozen validation facts

The complete authorized artifact package was independently revalidated before
this code release:

- 192,790/192,790 package files matched SHA-256 before and after validation.
- 470 V1.0 Python files compiled.
- 81 V1.0 shell scripts passed syntax checks.
- 86/86 V1.0 unit and regression tests passed.
- 62,689/62,689 structure files matched the frozen structure manifest.
- 16/16 final semantic checks and 33/33 expanded release checks passed.
- Frozen Stage2 candidate and Stage3 ensemble/metric replay passed.
- One-epoch GPU training smoke tests saved valid Stage2 and Stage3 models.
- A real-structure GPU inference smoke test passed all 23 output checks.

These checks prove code/artifact consistency and frozen metric replay. They do
not turn validation metrics into a pristine external-test claim.

## Metric scope

The three retained metrics are validation metrics. The historical test split
was accessed by non-final experiments and is not a pristine lockbox. The
metric historically called strict end-to-end uses frozen Stage3
chemistry-checked precursor inputs, including true-precursor fallback rows; it
is not a deployable online Stage2-to-Stage3 accuracy measurement.

Read [`docs/evaluation_protocol_v1.md`](docs/evaluation_protocol_v1.md) before
citing accuracy.

## Data and licensing

This public repository provides code, configuration, and documentation. It
does not grant redistribution rights for literature text, Materials
Project-labelled structures, third-party representations, or private model
artifacts. Obtain those assets through an authorized channel and mount them as
the external data root.

## Safety

Synthmind outputs research predictions, not validated laboratory procedures.
Every precursor, condition, atmosphere, and route must receive chemical,
thermodynamic, toxicological, regulatory, and experimental review.
