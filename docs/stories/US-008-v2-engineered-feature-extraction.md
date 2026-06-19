# US-008 V2 Engineered Feature Extraction

## Status

implemented

## Lane

normal

## Product Contract

V2-M1 extracts deterministic engineered features for V2A/V2B without training
models. Public train rows fit the scaler. Validation, public test, and
professor rows use the train-fitted scaler only.

## Relevant Product Docs

- `docs/product/preprocessing.md`
- `docs/product/v2-exploration.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`

## Acceptance Criteria

- `v1_basic`, `spectral_v1`, `mfcc_v1`, and `spectral_mfcc_v1` feature sets are
  implemented.
- Feature extraction is deterministic and uses existing WAV window metadata.
- NaN/Inf feature values are replaced with `0.0` and counted in the audit.
- Scaler is fit only on public `train` rows.
- Professor rows may be included only for later evaluation and do not
  contribute to scaler fitting.
- `extract-v2-features` writes `.npz`, scaler JSON, and audit JSON artifacts.
- The CLI prints configurable progress logs during full-dataset extraction.
- V2 feature extraction uses only the existing optional `data` dependencies.
- No V2 model training is implemented in this story.

## Design Notes

- Commands:
  - `python -m drone_sound.cli extract-v2-features ...`
- Queries: none.
- API:
  - `drone_sound.v2_features.feature_names`
  - `drone_sound.v2_features.extract_feature_vector`
  - `drone_sound.v2_features.fit_scaler`
  - `drone_sound.v2_features.apply_scaler`
  - `drone_sound.v2_features.build_v2_feature_bundle`
- Tables: none.
- Domain rules:
  - Professor data remains evaluation-only.
  - Public `train` is the only scaler-fitting split.
- UI surfaces: none.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-008 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `python -m pytest tests/test_v2_features.py` |
| Integration | `python -m pytest tests/test_v2_features.py tests/test_v2_cli.py` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | `uv run --extra data --with pytest --python 3.13 python -m pytest -q` |

## Harness Delta

Story `US-008` should be registered in the durable Harness database with verify
command `uv run --extra data --with pytest --python 3.13 python -m pytest -q`.

## Evidence

- `uv run --extra data --with pytest --python 3.13 python -m pytest tests/test_v2_contract.py tests/test_v2_cli.py tests/test_v2_features.py -q` -> 19 passed.
- `uv run --extra data --with pytest --python 3.13 python -m pytest -q` -> 52 passed.
