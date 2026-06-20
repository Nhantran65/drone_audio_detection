# US-010 V2 Remaining Experiment Tracks

## Status

partial

## Lane

normal

## Product Contract

V2-M3 and V2-M4 run the remaining implemented V2 experiment tracks after the
V2A classical ML sweep:

- V2B Tiny MLP on the existing `spectral_mfcc_v1` feature bundle.
- V2C Tiny CNN on `logmel_v1` windows.

V2D remains pending. It must use a spec-compliant pretrained audio embedding
family such as YAMNet, EfficientAT, VGGish, or PANNs. The prior non-spec
reference run was removed.

Professor rows remain evaluation-only and cannot affect training, threshold
tuning, or model selection.

## Relevant Product Docs

- `docs/product/v2-exploration.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`
- `docs/product/preprocessing.md`

## Acceptance Criteria

- `train-v2-mlp` trains a V2B Tiny MLP from a V2 feature bundle and writes a
  frozen model plus metrics JSON.
- `train-v2-cnn` trains a V2C Tiny CNN from public/professor manifests and
  writes a frozen model plus metrics JSON.
- `train-v2-embedding` does not run until a spec-compliant V2D backend is
  implemented.
- Long-running extraction/training/evaluation steps print progress with elapsed
  time, counts, and current phase/model.
- Validation thresholds are selected only from public `val` rows.
- Public test and professor metrics use frozen thresholds.
- Output metrics can be compared against V1 and V2A columns.
- V1 and V2A commands remain compatible.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-010 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py tests/test_v2_cli.py -q` |
| Integration | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest tests/test_v2_cli.py tests/test_v2_features.py tests/test_v2_ml.py -q` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q` |

## Harness Delta

Story `US-010` is registered in the durable Harness database with verify command
`uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q`.

## Evidence

- `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q` -> 59 passed.
- Real V2B Tiny MLP run:
  `uv run --extra data --extra ml --python 3.13 python -m drone_sound.cli train-v2-mlp --features artifacts/features/v2_spectral_mfcc_features.npz --model-out artifacts/models/v2b_mlp/tiny_mlp.joblib --metrics-out artifacts/models/v2b_mlp/tiny_mlp_metrics.json --epochs 8 --batch-size 4096`
  -> public F2 0.994734, public FPR 0.004772, professor recall 0.723270,
  longest missed 9.5s.
- Real V2C Tiny CNN run:
  `uv run --extra data --extra ml --extra dl --python 3.13 python -m drone_sound.cli train-v2-cnn --features artifacts/features/v2_spectral_mfcc_features.npz --public-audio-root data/raw/public --professor-audio-root data/raw/private/professor_real_world --model-out artifacts/models/v2c_cnn/tiny_cnn.pt --metrics-out artifacts/models/v2c_cnn/tiny_cnn_metrics.json --cache-dir artifacts/features/v2_logmel_cache --epochs 4 --batch-size 1024`
  -> public F2 0.941591, public FPR 0.102685, professor recall 0.446541,
  longest missed 21.0s.
- Unified comparison artifacts:
  `artifacts/models/v2_full_comparison/comparison.json` and
  `artifacts/models/v2_full_comparison/comparison.csv`.
- Cleanup note: non-spec V2D code, docs, artifacts, and comparison rows were
  removed. Next V2D work must follow the SPEC-preferred embedding families
  before it can be marked implemented.
