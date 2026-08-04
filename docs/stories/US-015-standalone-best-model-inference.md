# US-015 Standalone Best Model Inference

## Status

implemented

## Lane

normal

## Product Contract

External users can run the two current best V2 model candidates on extracted
DroneNoise/Salford WAV files from a self-contained notebook without importing
the local `drone_sound` package or an auxiliary script. The notebook must
reconstruct the same `spectral_mfcc_v1` feature set, apply the train-fitted
scaler, load both joblib model packages, and report thresholded detections with
progress.

## Relevant Product Docs

- `docs/product/preprocessing.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`
- `docs/product/v2-exploration.md`

## Acceptance Criteria

- A self-contained notebook exists for WAV-folder inference with the packaged
  `tiny_mlp_es` and `random_forest` models.
- The notebook does not import the local `drone_sound` package or
  `run_best_models_standalone.py`.
- The notebook extracts `spectral_mfcc_v1` features with the same 100-column
  feature schema used by the trained models.
- The notebook applies `v2_spectral_mfcc_scaler.json` before model scoring.
- The notebook reports progress for long-running feature extraction and
  scoring.
- The notebook writes per-window CSVs and a summary JSON.
- The direct local Salford/DroneNoise run detects at least one window in every
  non-calibration WAV file for both models.

## Design Notes

- Packaged bundle:
  `artifacts/packages/best_models_standalone_inference_20260702.zip`.
- Notebook walkthrough:
  `notebooks/best_models_inference_walkthrough.ipynb`.
- The notebook skips `Calib_*.wav` files by default.
- The notebook supports repo-relative defaults and same-directory model/scaler
  lookup so it works after extracting the standalone zip.
- Direct Salford WAV-file inference is not equivalent to replaying samples
  through speakers and recording them through a microphone.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id <id> --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `python3 -m json.tool notebooks/best_models_inference_walkthrough.ipynb /tmp/best_models_inference_walkthrough.validated.json` |
| Integration | Execute notebook code cells with `MAX_FILES=2` smoke override through `uv run --extra data --extra ml --python 3.13 ...`; full Salford evidence remains from the prior standalone run using the same feature/model logic. |
| E2E | Not applicable; no user-facing app flow. |
| Platform | Not run; no platform packaging or installer. |
| Release | Standalone zip contents and checksum verified. |

## Harness Delta

No harness policy change. This story captures a packaging and handoff gap:
weights alone are not enough for external inference without the exact feature
construction and scaler.

## Evidence

- Full Salford run excluding calibration files:
  - WAV files: 164
  - Windows: 6,936
  - `tiny_mlp_es`: 6,907/6,936 detected windows and 164/164 files with at
    least one detection.
  - `random_forest`: 6,904/6,936 detected windows and 164/164 files with at
    least one detection.
- Full self-contained notebook run:
  `artifacts/notebook_selfcontained_full/summary.json`
- Self-contained notebook smoke test with `MAX_FILES=2`:
  - `tiny_mlp_es`: 77/78 detected windows and 2/2 files with detection.
  - `random_forest`: 78/78 detected windows and 2/2 files with detection.
- Final unzip smoke test from `/tmp/best_models_review_bundle_20260702`:
  - Notebook found `tiny_mlp_es.joblib`, `random_forest.joblib`, and
    `v2_spectral_mfcc_scaler.json` in the extracted bundle folder via
    `ARTIFACT_DIR = Path(".")`.
  - Only `DATASET_ROOT` was overridden for the local dataset path.
- Standalone zip:
  `artifacts/packages/best_models_standalone_inference_20260702.zip`
- Zip SHA256:
  `0f0a9162c758e83bd2c13b83fe2746efa0de1038a6f6e3f803c09406c561fae0`
- User-facing `best_models.zip` SHA256:
  `353cbaee9888bb45f5784910e3a1bc9264b2d0e3853665996b3abd91418d49a1`
- Notebook JSON validation:
  `python3 -m json.tool notebooks/best_models_inference_walkthrough.ipynb /tmp/best_models_inference_walkthrough.validated.json`
