# US-003 V1 Training Smoke Baseline

## Status

implemented

## Lane

normal

## Product Contract

The POC should support a first runnable training attempt before adding every
planned hard-negative subtype. V1 is a smoke baseline, not a product-quality
model: it must train only from non-professor train/validation manifest rows,
choose its threshold from validation data, write model and metrics artifacts,
and report missing speech-only negative coverage as a known limitation.

## Relevant Product Docs

- `docs/product/datasets.md`
- `docs/product/preprocessing.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`

## Acceptance Criteria

- Add a `train-v1` CLI command that reads a manifest plus local WAV audio root.
- Exclude `professor_test` and `is_professor_data = true` rows from training,
  validation, and threshold selection.
- Require both `drone` and `no_drone` labels in train data.
- Require both `drone` and `no_drone` labels in validation data before
  selecting a threshold.
- Write a JSON model artifact and a JSON metrics report.
- Document that speech-only negative data is deferred for v1 and that
  `no_drone_speech` false-positive coverage is not available yet.
- Tests cover successful training, professor-only rejection, and one-class
  train-data rejection.

## Design Notes

- Commands:
  - `python -m drone_sound.cli train-v1 --manifest <csv> --audio-root <dir> --model-out artifacts/models/v1_baseline_model.json --metrics-out artifacts/models/v1_metrics.json`
- Queries: none.
- API: Python functions for v1 baseline feature extraction, training, threshold
  selection, and metric reporting.
- Tables: none.
- Domain rules:
  - Professor data remains evaluation-only.
  - V1 may proceed without speech-only negatives, but must not claim
    `no_drone_speech` coverage.
  - Thresholds are selected from validation rows only.
- UI surfaces: none.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-003 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `python -m pytest tests/test_training.py` |
| Integration | `python -m pytest` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | `python -m pytest` |

## Harness Delta

Story `US-003` should be registered in the durable Harness database with verify
command `python -m pytest`.

## Evidence

- `uv run --no-project --with pytest --python 3.13 python -m pytest` -> 18 passed.
- `. .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m drone_sound.cli train-v1 --manifest data/manifests/professor_manifest.csv --audio-root . --model-out artifacts/models/v1_baseline_model.json --metrics-out artifacts/models/v1_metrics.json` -> cleanly rejected current repo data because it only has professor `professor_test` rows and no non-professor train rows with both labels.
- `scripts/bin/harness-cli story verify-all` cannot be re-run in this checkout until the ignored prebuilt Harness CLI binary is restored.
