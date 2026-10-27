# Lightweight Drone Audio Classification POC

Proof of concept for detecting drone sound in 1-second audio windows
(0.5-second stride). The pipeline resamples each whole recording to 16 kHz,
extracts 100 `spectral_mfcc_v1` features per window, and scores them with a
frozen model and a validation-selected threshold.

This is a research POC, not a product-quality or edge-exported model.

## Current Model

The operational model is the canonical 16 kHz Random Forest
(`artifacts/v2_16khz/models/v2a/models/random_forest.joblib`, 14.1 MB). It uses
preprocessing profile `mono_resample_poly_16khz_v1` and frozen
`threshold_f2 = 0.4085`. See
[decision 0017](docs/decisions/0017-keep-canonical-random-forest-operational.md).

| Evaluation set | Precision | Recall | F1 | F2 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public test | 0.984 | 0.996 | 0.990 | 0.994 | 0.008 |
| Professor positive-only | n/a | 0.825 | n/a | n/a | n/a |
| Private real-only (11 recordings) | 0.947 | 0.825 | 0.882 | 0.847 | 0.086 |

The private real-only set contains only one positive recording and ten negative
recordings. Treat it as an external stress test, not a deployment estimate.

Full write-up:
[Random Forest preprocessing and evaluation report](docs/reports/random-forest-data-preprocessing-and-evaluation.md).

## Reproduce the Results (notebook)

[`notebooks/random_forest_16khz_inference_walkthrough.ipynb`](notebooks/random_forest_16khz_inference_walkthrough.ipynb)
re-runs the frozen Random Forest without importing `drone_sound` and without
retraining or retuning. It replays the saved feature bundles, then rebuilds the
private real-only set from the 11 raw WAV files and checks that the confusion
matrix is `TN=541, FP=51, FN=195, TP=918`.

It needs `numpy scipy joblib scikit-learn` and these local files, all ignored by
git:

- `artifacts/v2_16khz/models/v2a/models/random_forest.joblib`
- `artifacts/v2_16khz/features/scaler.json`
- `artifacts/v2_16khz/features/private_real_features.npz`
- `artifacts/v2_16khz/features/public_professor_features.npz` (optional; enables
  the public-test replay)
- `data/raw/private/professor_real_world/*.wav` and
  `data/raw/private/Recordings/*.wav` (optional; enables the raw rebuild)

The self-contained bundle
`artifacts/packages/rf_private_real_16khz_verification_20260806.zip` carries
everything except the public bundle. It includes private audio, so share it only
with authorized people.

Other notebooks are historical; see [legacy experiments](docs/reports/legacy-experiments.md).

## Setup

Requires Python 3.13 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,data,ml]"
```

Optional extras:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `data` | numpy, scipy, tqdm | NASA conversion, feature extraction, progress bars |
| `ml` | scikit-learn, joblib | Classical models, Tiny MLP, all evaluation |
| `dl` | torch, torchaudio | Tiny CNN |
| `embedding` | tensorflow, tensorflow-hub | YAMNet embedding head |
| `boosting` | xgboost | Lightweight tree experiment |
| `dev` | pytest | Tests |

With `uv`, prefix commands with
`uv run --extra data --extra ml --python 3.13` (add other extras as needed).

## Data

All raw data stays out of git (`data/raw/`, `fwdronedata.zip`, `artifacts/`).

1. Download and extract the six public datasets registered in
   [`configs/datasets.json`](configs/datasets.json) into `data/raw/public/`:

   ```bash
   python download.py
   ```

2. Convert NASA MAT files to WAV (requires the `data` extra):

   ```bash
   python -m drone_sound.cli convert-nasa-mat \
     --mat-root data/raw/public/nasa_small_uas_flyover/data \
     --wav-out data/raw/public/nasa_small_uas_flyover/converted_wav \
     --audit-out artifacts/audits/nasa_mat_conversion_audit.json \
     --sample-rate 16000
   ```

3. Build the public manifest with recording-level train/validation/test splits:

   ```bash
   python -m drone_sound.cli build-public-manifest \
     --registry configs/datasets.json \
     --raw-root data/raw/public \
     --manifest-out data/manifests/public_manifest.csv \
     --audit-out artifacts/audits/public_data_audit.json \
     --split-seed 20260619
   ```

   `validate-dataset-registry --registry configs/datasets.json` checks the
   registry, and `export-license-manifest --registry ... --out ...` exports its
   license metadata. License metadata is recorded but does not block research
   runs.

4. Audit the private professor recording (positive-only, evaluation-only). The
   audit reads the zip directly. Then extract the audio:

   ```bash
   python -m drone_sound.cli audit-professor-data \
     --zip fwdronedata.zip \
     --manifest-out data/manifests/professor_manifest.csv \
     --audit-out artifacts/audits/professor_data_audit.json
   mkdir -p data/raw/private/professor_real_world
   unzip -o fwdronedata.zip -d data/raw/private/professor_real_world
   ```

5. Place the ten private negative recordings (traffic and forest WAV files) in
   `data/raw/private/Recordings/`. They are evaluation-only; see
   [US-016](docs/stories/US-016-private-negative-audio-quality/overview.md).

Data rule: professor and private recordings are never used for training,
early stopping, threshold selection, or model selection.

## Canonical 16 kHz Pipeline

This is the workflow that produced the current model
([decision 0015](docs/decisions/0015-canonical-16khz-v2-preprocessing.md),
[US-017](docs/stories/US-017-canonical-16khz-v2/overview.md)). The recorded run
lives in `artifacts/v2_16khz/`. Use a new run root for reruns, because
`init-v2-16khz-run` refuses to overwrite an existing one.

```bash
RUN=artifacts/v2_16khz_rerun
```

Initialize the run. This records SHA-256 hashes for every existing file under
`artifacts/` so that the finalize step can prove nothing outside the run root
changed:

```bash
mkdir -p artifacts
python -m drone_sound.cli init-v2-16khz-run --run-root $RUN --legacy-root artifacts
```

Extract training features. The scaler is fitted on train rows only. Professor
rows are included for evaluation only.

```bash
python -m drone_sound.cli extract-v2-features \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --feature-set spectral_mfcc_v1 \
  --features-out $RUN/features/public_professor_features.npz \
  --scaler-out $RUN/features/scaler.json \
  --audit-out $RUN/features/public_professor_audit.json
```

Build the private real-only test set and extract it with the frozen scaler:

```bash
python -m drone_sound.cli build-private-real-manifest \
  --professor-manifest data/manifests/professor_manifest.csv \
  --private-audio-root data/raw/private \
  --negatives-dir data/raw/private/Recordings \
  --manifest-out $RUN/features/private_real_manifest.csv \
  --audit-out $RUN/features/private_real_manifest_audit.json

python -m drone_sound.cli extract-v2-eval-features \
  --manifest $RUN/features/private_real_manifest.csv \
  --audio-root data/raw/private \
  --feature-set spectral_mfcc_v1 \
  --scaler $RUN/features/scaler.json \
  --features-out $RUN/features/private_real_features.npz \
  --audit-out $RUN/features/private_real_features_audit.json
```

Train the four model tracks. Thresholds and model choice come from validation
rows only.

```bash
# V2A classical ML (includes the operational Random Forest)
python -m drone_sound.cli sweep-v2-ml \
  --features $RUN/features/public_professor_features.npz \
  --model-types logistic_regression linear_svm random_forest extra_trees \
  --out-dir $RUN/models/v2a

# V2B Tiny MLP
python -m drone_sound.cli train-v2-mlp \
  --features $RUN/features/public_professor_features.npz \
  --model-out $RUN/models/v2b/tiny_mlp.joblib \
  --metrics-out $RUN/models/v2b/tiny_mlp_metrics.json \
  --epochs 20 --patience 3

# V2C Tiny CNN (needs the dl extra)
python -m drone_sound.cli train-v2-cnn \
  --features $RUN/features/public_professor_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --model-out $RUN/models/v2c/tiny_cnn.pt \
  --metrics-out $RUN/models/v2c/tiny_cnn_metrics.json \
  --cache-dir $RUN/cache/logmel \
  --epochs 20 --patience 4

# V2D YAMNet embedding head (needs the embedding extra; reference_only)
python -m drone_sound.cli train-v2-embedding \
  --features $RUN/features/public_professor_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --embedding-model yamnet \
  --head logistic_regression \
  --model-out $RUN/models/v2d/yamnet_head.joblib \
  --metrics-out $RUN/models/v2d/yamnet_head_metrics.json \
  --cache-dir $RUN/cache/yamnet
```

Evaluate frozen models. These commands never refit and never retune thresholds.
`eval-v2` covers public test and professor positives; use `eval-v2-cnn` and
`eval-v2-embedding` for the CNN and YAMNet tracks.

```bash
python -m drone_sound.cli eval-v2 \
  --model $RUN/models/v2a/models/random_forest.joblib \
  --features $RUN/features/public_professor_features.npz \
  --metrics-out $RUN/evaluation/public_professor/random_forest.json

python -m drone_sound.cli eval-real-v2 \
  --model $RUN/models/v2a/models/random_forest.joblib \
  --features $RUN/features/private_real_features.npz \
  --audio-root data/raw/private \
  --cache-dir $RUN/cache/real_logmel \
  --metrics-out $RUN/evaluation/real_only/random_forest.json \
  --export-errors $RUN/evaluation/real_only/random_forest_errors
```

`finalize-v2-16khz-run` re-hashes `artifacts/` and writes comparison and
provenance files. It also requires `--legacy-feature-bundle`, the old
native-rate bundle (`artifacts/features/v2_spectral_mfcc_features.npz`), so it
only works in a workspace that still has that bundle.

YAMNet on GPU with `uv`: expose the CUDA wheel libraries and pick a GPU
explicitly with `CUDA_VISIBLE_DEVICES`:

```bash
TF_CUDA_LIBS=$(uv run --extra embedding --python 3.13 python - <<'PY'
import glob, site
paths = []
for root in site.getsitepackages():
    paths.extend(sorted(glob.glob(root + "/nvidia/*/lib")))
print(":".join(paths))
PY
)
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$TF_CUDA_LIBS:${LD_LIBRARY_PATH:-}" \
  uv run --extra data --extra ml --extra embedding --python 3.13 \
  python -m drone_sound.cli train-v2-embedding ...
```

## Follow-up Experiments

These experiments ran on top of the canonical features. None of them replaced
the operational model.

- **Tiny MLP rescue** ([US-018](docs/stories/US-018-tiny-mlp-rescue/overview.md)):
  a validation-selected hyperparameter sweep in an isolated directory.

  ```bash
  python -m drone_sound.cli sweep-v2-mlp-rescue \
    --features artifacts/v2_16khz/features/public_professor_features.npz \
    --out-dir artifacts/v2_16khz/models/tiny_mlp_rescue \
    --epochs 20 --patience 4
  ```

- **Lightweight trees** ([US-020](docs/stories/US-020-lightweight-tree-experiments.md)):
  three compressed Random Forests, two HistGradientBoosting variants, and one
  shallow XGBoost. One winner per family is chosen on validation, then evaluated
  on private real-only. HistGradientBoosting and XGBoost got under 1 MB but lost
  private positive recall, and the compressed Random Forest saved only about
  15%, so the canonical Random Forest was kept (decision 0017).

  ```bash
  uv run --extra data --extra ml --extra boosting --python 3.13 \
    python scripts/run_lightweight_tree_experiments.py
  ```

  Output goes to `artifacts/v2_16khz/models/lightweight_tree_experiments/`. The
  script refuses to overwrite an existing output directory.

- **V3 group-aware split**
  ([decision 0016](docs/decisions/0016-group-aware-v3-split.md),
  [US-019](docs/stories/US-019-v3-group-aware-retraining/overview.md)): reuses
  the canonical features but assigns whole recording-session groups to
  train/validation/test, which removes near-duplicate leakage across splits. All
  tracks were retrained with `--balance-groups` and
  `--primary-threshold threshold_f1`; V2A and V2B were also trained with natural
  weighting for comparison. The run in
  `artifacts/v3_group_split_16khz/` is complete. Validation F1 selected a Tiny
  MLP: public-test F1 0.986, but private real-only recall 0.41. The
  natural-weighting V3 Random Forest reached private real-only F1 0.911 with
  FPR 0.125. No V3 model has been promoted.

  Command sequence (`V3=artifacts/v3_group_split_16khz`):

  ```bash
  python -m drone_sound.cli init-v3-run --run-root $V3 --legacy-root artifacts
  python -m drone_sound.cli build-v3-grouped-manifest \
    --source-manifest data/manifests/public_manifest.csv \
    --manifest-out $V3/manifests/public_grouped_manifest.csv \
    --group-map-out $V3/manifests/group_map.csv \
    --audit-out $V3/manifests/split_audit.json
  python -m drone_sound.cli repartition-v3-features \
    --source-bundle artifacts/v2_16khz/features/public_professor_features.npz \
    --grouped-manifest $V3/manifests/public_grouped_manifest.csv \
    --features-out $V3/features/public_professor_features.npz \
    --scaler-out $V3/features/scaler.json \
    --audit-out $V3/features/public_professor_audit.json
  # rebind-v3-cache reuses the canonical log-mel/YAMNet caches after
  # row-identity checks; then train each track, for example:
  python -m drone_sound.cli sweep-v2-ml \
    --features $V3/features/public_professor_features.npz \
    --model-types logistic_regression linear_svm random_forest extra_trees \
    --out-dir $V3/models/v2a_group_balanced \
    --balance-groups --primary-threshold threshold_f1
  # build the private real-only bundle with the V3 scaler
  # (build-private-real-manifest, then extract-v2-eval-features
  #  --scaler $V3/features/scaler.json), evaluate with eval-v3 (public +
  # professor, with cluster-bootstrap CIs) and eval-real-v2, then verify with
  # finalize-v3-run.
  ```

Earlier experiments (V1 smoke baseline and native-rate V2 tracks) are recorded
in [docs/reports/legacy-experiments.md](docs/reports/legacy-experiments.md).

## Tests

```bash
python -m pytest
```

Tests that need an optional extra are skipped when it is missing. To run the
full suite:

```bash
uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest
```

## Project Docs and Harness

- [`SPEC.md`](SPEC.md): original project specification.
- [`docs/README.md`](docs/README.md): documentation map, covering product
  contracts, stories, decisions, and reports.
- This repo uses Harness for intake, stories, and proof tracking. Read
  [`AGENTS.md`](AGENTS.md) and [`docs/HARNESS.md`](docs/HARNESS.md). The CLI is
  `scripts/bin/harness-cli` (local binary, ignored by git).

Long-running data and training commands print progress (elapsed time, counts,
rate, ETA), as required by `AGENTS.md`.
