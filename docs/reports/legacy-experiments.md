# Legacy Experiments (V1 and native-rate V2)

This page preserves the commands used before the canonical 16 kHz pipeline
(decision `0015`). They are kept for provenance only. The current workflow is in
the root `README.md`.

## Status

- **V1 smoke baseline** still runs with the current code. It is a
  dependency-free sanity check, not a candidate model.
- **Native-rate V2 (`legacy_native_rate_v1`)** artifacts under
  `artifacts/features/` and `artifacts/models/` cannot be regenerated:
  `extract-v2-features` now only accepts `--target-sample-rate 16000`. Re-running
  extraction to these paths produces canonical 16 kHz features instead, and
  models trained on one profile are rejected when evaluated on the other.
- The canonical and V3 runs inventoried everything under `artifacts/` by
  SHA-256 and verified it was unchanged. Do not write new outputs into the
  legacy paths below.
- `docs/project-summary.md`, `notebooks/best_models_inference_walkthrough.ipynb`,
  and `notebooks/professor_negative_data_quality.ipynb` report or load these
  legacy models (`tiny_mlp_es`, native-rate `random_forest`).

## V1 Smoke Baseline

Requires the public and professor manifests from the README data steps.
Professor rows are rejected for training.

```bash
python -m drone_sound.cli train-v1 \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --model-out artifacts/models/v1_baseline_model.json \
  --metrics-out artifacts/models/v1_metrics.json
```

`train-v1` uses numpy only to speed up feature extraction and falls back to pure
Python without it. With the `data` extra, `tqdm` shows per-split progress.

Evaluate the frozen model and validation-selected threshold on `public_test`
plus professor positives:

```bash
python -m drone_sound.cli eval-v1 \
  --model artifacts/models/v1_baseline_model.json \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --metrics-out artifacts/models/v1_eval_metrics.json
```

Professor metrics are recall-only because that data is positive-only.

## Native-rate V2 Tracks (historical)

All commands read the legacy bundle
`artifacts/features/v2_spectral_mfcc_features.npz`.

V2A classical ML sweep (US-009) and frozen evaluation:

```bash
python -m drone_sound.cli sweep-v2-ml \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --model-types logistic_regression linear_svm random_forest extra_trees \
  --out-dir artifacts/models/v2a_sweep

python -m drone_sound.cli eval-v2 \
  --model artifacts/models/v2a_sweep/models/logistic_regression.joblib \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --metrics-out artifacts/models/v2a_sweep/logistic_regression_eval.json \
  --export-errors artifacts/models/v2a_sweep/logistic_regression_errors
```

V2B Tiny MLP and V2C Tiny CNN early-stopping reruns (US-011). These wrote the
`_es` artifacts; the earlier fixed-epoch runs (US-010) wrote `tiny_mlp.joblib`
and `tiny_cnn.pt`.

```bash
python -m drone_sound.cli train-v2-mlp \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --model-out artifacts/models/v2b_mlp/tiny_mlp_es.joblib \
  --metrics-out artifacts/models/v2b_mlp/tiny_mlp_es_metrics.json \
  --epochs 20 --batch-size 4096 --patience 3 --min-delta 0.0005

python -m drone_sound.cli train-v2-cnn \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --model-out artifacts/models/v2c_cnn/tiny_cnn_es.pt \
  --metrics-out artifacts/models/v2c_cnn/tiny_cnn_es_metrics.json \
  --cache-dir artifacts/features/v2_logmel_cache \
  --epochs 20 --batch-size 1024 --patience 4 --min-delta 0.0005
```

V2D YAMNet embedding head (US-013):

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

Combined public-test plus professor-positive metrics (US-014) are supplementary.
They were recomputed from frozen models with `eval-v2`, `eval-v2-cnn`, and
`eval-v2-embedding` into `artifacts/models/v2_combined_supplementary/`.

No CLI command or script in this repo writes the comparison tables in
`artifacts/models/v2_full_comparison/`; they exist only as recorded outputs of
US-010 through US-013.
