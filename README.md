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

V2D YAMNet embedding tests require the optional embedding dependencies:

```bash
uv run --extra data --extra ml --extra embedding --with pytest --python 3.13 python -m pytest
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

Canonical reruns use full-recording mono/polyphase resampling at 16 kHz and
write only below `artifacts/v2_16khz/`. Start with
`init-v2-16khz-run`; it fails if the namespace exists and records SHA-256 for
all legacy artifacts. Build the test-only private manifest with
`build-private-real-manifest`, extract training and frozen-scaler evaluation
bundles with `extract-v2-features` and `extract-v2-eval-features`, then use
`eval-real-v2` after each model is frozen. Finish with
`finalize-v2-16khz-run` to verify legacy hashes and build comparison/provenance
files. See `docs/stories/US-017-canonical-16khz-v2/` for the locked protocol.

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

## V2B/V2C/V2D Experiment Tracks

After `spectral_mfcc_v1` extraction and V2A, the remaining V2 tracks can be
run from the same feature bundle. Tiny MLP uses `data` plus `ml`; Tiny CNN
requires the optional `dl` extra; V2D YAMNet embeddings require `data`, `ml`,
and `embedding`.

```bash
python -m drone_sound.cli train-v2-mlp \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --model-out artifacts/models/v2b_mlp/tiny_mlp.joblib \
  --metrics-out artifacts/models/v2b_mlp/tiny_mlp_metrics.json \
  --epochs 20 \
  --patience 3 \
  --min-delta 0.0005
```

```bash
python -m drone_sound.cli train-v2-cnn \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --model-out artifacts/models/v2c_cnn/tiny_cnn.pt \
  --metrics-out artifacts/models/v2c_cnn/tiny_cnn_metrics.json \
  --cache-dir artifacts/features/v2_logmel_cache \
  --epochs 20 \
  --patience 4 \
  --min-delta 0.0005
```

```bash
python -m drone_sound.cli train-v2-embedding \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --embedding-model yamnet \
  --head logistic_regression \
  --model-out artifacts/models/v2d_yamnet/yamnet_head.joblib \
  --metrics-out artifacts/models/v2d_yamnet/yamnet_head_metrics.json \
  --cache-dir artifacts/features/v2_yamnet_cache
```

With `uv`, run V2D on GPU by exposing the CUDA wheel libraries and selecting
the least-busy GPU:

```bash
TF_CUDA_LIBS=$(uv run --extra embedding --python 3.13 python - <<'PY'
import glob, site
paths = []
for root in site.getsitepackages():
    paths.extend(sorted(glob.glob(root + "/nvidia/*/lib")))
print(":".join(paths))
PY
)
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$TF_CUDA_LIBS:${LD_LIBRARY_PATH:-}" \
  uv run --extra data --extra ml --extra embedding --python 3.13 \
  python -m drone_sound.cli train-v2-embedding ...
```

YAMNet is the first spec-compliant V2D backend. V2D remains `reference_only`
unless a later deployment story proves the full pretrained extractor is
lightweight enough for the edge target.

The combined V1/V2 comparison artifacts are written under
`artifacts/models/v2_full_comparison/`. V2B/V2C metrics include
`learning_curve` and `early_stopping` blocks when trained with the current
commands.

Run the optional canonical Tiny MLP rescue sweep in a fresh, isolated output
directory. Candidate fitting and selection use public train/validation rows
only; professor and private-real rows remain post-freeze evaluation data.

```bash
python -m drone_sound.cli sweep-v2-mlp-rescue \
  --features artifacts/v2_16khz/features/public_professor_features.npz \
  --out-dir artifacts/v2_16khz/models/tiny_mlp_rescue \
  --epochs 20 --patience 4
```

Run the bounded lightweight-tree experiment in its own fail-fast namespace.
It trains three compressed Random Forests, two HistGradientBoosting variants,
and one shallow XGBoost candidate. One winner per family is selected from
validation metrics before frozen private-real evaluation:

```bash
uv run --extra data --extra ml --extra boosting --python 3.13 \
  python scripts/run_lightweight_tree_experiments.py
```

Artifacts are written below
`artifacts/v2_16khz/models/lightweight_tree_experiments/`. The current
canonical Random Forest artifacts remain unchanged.

Combined public-test plus professor-positive metrics are supplementary. Recompute
them from frozen models without retraining or retuning thresholds:

```bash
python -m drone_sound.cli eval-v2 \
  --model artifacts/models/v2b_mlp/tiny_mlp_es.joblib \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --metrics-out artifacts/models/v2_combined_supplementary/tiny_mlp_es_combined_eval.json
```

```bash
python -m drone_sound.cli eval-v2-cnn \
  --model artifacts/models/v2c_cnn/tiny_cnn_es.pt \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --cache-dir artifacts/features/v2_logmel_cache \
  --metrics-out artifacts/models/v2_combined_supplementary/tiny_cnn_es_combined_eval.json
```

```bash
python -m drone_sound.cli eval-v2-embedding \
  --model artifacts/models/v2d_yamnet/yamnet_head.joblib \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --metrics-out artifacts/models/v2_combined_supplementary/yamnet_head_combined_eval.json
```

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

## V3 Group-aware 16 kHz Run

V3 keeps the canonical 16 kHz feature contract but assigns whole correlated
recording-session groups to train, validation, and public test. Run artifacts
are isolated under `artifacts/v3_group_split_16khz/`; existing V1/V2 artifacts
are inventoried before and after the run. The main commands are
`init-v3-run`, `build-v3-grouped-manifest`, `repartition-v3-features`,
`rebind-v3-cache`, `eval-v3`, and `finalize-v3-run`.
