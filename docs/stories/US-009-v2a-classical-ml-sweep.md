# US-009 V2A Classical ML Sweep And Frozen Evaluation

## Status

implemented

## Lane

normal

## Product Contract

V2-M2 trains classical ML candidates on the `spectral_mfcc_v1` V2 feature
bundle, selects thresholds from validation rows only, evaluates the frozen
models on public test and professor positive-only rows, and writes comparison
artifacts. Professor rows remain evaluation-only and cannot affect training,
threshold tuning, or model selection.

## Relevant Product Docs

- `docs/product/v2-exploration.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`
- `docs/product/preprocessing.md`

## Acceptance Criteria

- `sweep-v2-ml` reads a V2 `.npz` feature bundle and trains V2A
  `logistic_regression`, `linear_svm`, `random_forest`, and `extra_trees`
  candidates.
- V2A training uses public `train` rows only; validation thresholds and model
  selection use public `val` rows only.
- `public_test` rows and professor rows are evaluation-only.
- The V2 feature bundle includes eval metadata: `clip_id`, `label_subtype`,
  `window_start_sec`, and `window_end_sec`.
- Old feature bundles missing required metadata fail with a clear regenerate
  features error.
- Every candidate reports `threshold_f1`, `threshold_f2`,
  `threshold_at_fpr_10`, `threshold_at_fpr_5`, `threshold_at_fpr_2`, and
  `threshold_at_fpr_1`.
- Sweep artifacts include per-model joblib models, per-model metrics JSON,
  `comparison.json`, `comparison.csv`, and `best_model.json`.
- `eval-v2` loads a frozen V2 model, reuses its stored `threshold_f2`, and
  writes metrics without retuning thresholds.
- `eval-v2 --export-errors` writes minimal public false-positive and professor
  missed-window CSVs.
- V2B/V2C/V2D train commands remain skeletons.

## Design Notes

- Commands:
  - `python -m drone_sound.cli sweep-v2-ml --features <features.npz> --model-types logistic_regression linear_svm random_forest extra_trees --out-dir <dir>`
  - `python -m drone_sound.cli eval-v2 --model <model.joblib> --features <features.npz> --metrics-out <metrics.json> [--export-errors <dir>]`
- Queries: none.
- API:
  - `drone_sound.v2_ml.load_feature_bundle`
  - `drone_sound.v2_ml.run_v2a_sweep`
  - `drone_sound.v2_ml.evaluate_v2_model`
- Tables: none.
- Domain rules:
  - Professor data is evaluation-only.
  - Validation F2 at `threshold_f2` is the primary model-selection metric.
  - Public FPR, professor recall, model size, and inference latency are
    reported guardrails, not threshold-selection sources.
- UI surfaces: none.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-009 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py -q` |
| Integration | `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest tests/test_v2_cli.py tests/test_v2_features.py tests/test_v2_ml.py -q` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest -q` |

## Harness Delta

Story `US-009` is registered in the durable Harness database with verify command
`uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest -q`.

## Evidence

- `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py tests/test_v2_cli.py tests/test_v2_features.py -q` -> 23 passed.
- `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest -q` -> 59 passed.
- Real local V2A sweep:
  `uv run --extra data --extra ml --python 3.13 python -m drone_sound.cli sweep-v2-ml --features artifacts/features/v2_spectral_mfcc_features.npz --model-types logistic_regression linear_svm random_forest extra_trees --out-dir artifacts/models/v2a_sweep` -> best model `random_forest`.
- Real frozen eval results:
  - `logistic_regression`: public F2 0.975609, public FPR 0.015855, professor recall 0.725966, longest missed 26.5s.
  - `linear_svm`: public F2 0.973310, public FPR 0.032683, professor recall 0.778077, longest missed 10.5s.
  - `random_forest`: public F2 0.996308, public FPR 0.006107, professor recall 0.888589, longest missed 7.5s.
  - `extra_trees`: public F2 0.989803, public FPR 0.017303, professor recall 0.533693, longest missed 35.0s.
