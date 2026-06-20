# Lightweight Drone Audio Classification POC

This repository is a proof of concept for detecting drone audio in short
window-level recordings.

The current POC includes product contracts, manifest validation, professor data
audit tooling, public dataset registry validation, public manifest assembly, and
a dependency-free v1 smoke baseline. It is not yet a product-quality model or
edge-export pipeline.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Professor Data Audit

`fwdronedata.zip` is private local data and is ignored by git. Audit it with:

```bash
python -m drone_sound.cli audit-professor-data \
  --zip fwdronedata.zip \
  --manifest-out data/manifests/professor_manifest.csv \
  --audit-out artifacts/audits/professor_data_audit.json
```

The audit reads the zip directly, writes a manifest plus redacted JSON report,
and does not extract raw audio into the repository.

## Validation

```bash
python -m pytest
```

If the local machine does not already have Python 3.13 and pytest installed,
the same suite can be run through `uv`:

```bash
uv run --no-project --with pytest --python 3.13 python -m pytest
```

Tests that exercise optional data conversion dependencies use:

```bash
uv run --extra data --with pytest --python 3.13 python -m pytest
```

V2 classical ML tests require the optional ML dependencies:

```bash
uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest
```

## V1 Training Smoke Baseline

Convert NASA MAT files into derived WAV files first, then build the local public
manifest from raw WAV folders. V1 research uses all datasets registered in
`configs/datasets.json`; license metadata is kept in the audit output but does
not block the v1 experiment.

```bash
python -m drone_sound.cli convert-nasa-mat \
  --mat-root data/raw/public/nasa_small_uas_flyover/data \
  --wav-out data/raw/public/nasa_small_uas_flyover/converted_wav \
  --audit-out artifacts/audits/nasa_mat_conversion_audit.json \
  --sample-rate 16000
```

```bash
python -m drone_sound.cli build-public-manifest \
  --registry configs/datasets.json \
  --raw-root data/raw/public \
  --manifest-out data/manifests/public_manifest.csv \
  --audit-out artifacts/audits/public_data_audit.json \
  --split-seed 20260619
```

Then train the smoke baseline. Professor data is evaluation-only and is
rejected for training.

```bash
python -m drone_sound.cli train-v1 \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --model-out artifacts/models/v1_baseline_model.json \
  --metrics-out artifacts/models/v1_metrics.json
```

`convert-nasa-mat` requires the optional `data` dependencies. With `uv`, run it
as `uv run --extra data --python 3.13 python -m drone_sound.cli convert-nasa-mat ...`.
The same `data` extra installs `tqdm`, so `train-v1` shows per-split progress
and ETA while extracting window features.

## V2A Classical ML Sweep

After extracting `spectral_mfcc_v1` features, run the classical ML sweep from
the feature bundle. Professor rows inside the bundle are evaluation-only and do
not affect training, threshold tuning, or model selection.

```bash
python -m drone_sound.cli sweep-v2-ml \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --model-types logistic_regression linear_svm random_forest extra_trees \
  --out-dir artifacts/models/v2a_sweep
```

Evaluate a frozen V2A model without retuning its stored threshold:

```bash
python -m drone_sound.cli eval-v2 \
  --model artifacts/models/v2a_sweep/models/logistic_regression.joblib \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --metrics-out artifacts/models/v2a_sweep/logistic_regression_eval.json \
  --export-errors artifacts/models/v2a_sweep/logistic_regression_errors
```

Use `uv run --extra data --extra ml --python 3.13 ...` if the local
environment does not already have the optional dependencies installed.

## V1 Frozen Model Evaluation

After training, evaluate the frozen model and validation-selected threshold on
`public_test` plus professor real-world positive data. Extract professor audio
locally first; `data/raw/private/` is ignored and must stay out of git.

```bash
mkdir -p data/raw/private/professor_real_world
unzip -o fwdronedata.zip -d data/raw/private/professor_real_world
```

```bash
python -m drone_sound.cli eval-v1 \
  --model artifacts/models/v1_baseline_model.json \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --metrics-out artifacts/models/v1_eval_metrics.json
```

Professor metrics are recall-oriented only because the professor data is
positive-only or near-positive-only. Combined metrics are supplementary and do
not replace the public test benchmark.
