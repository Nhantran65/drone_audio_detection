# US-011 V2 Early-Stopping Rerun

## Status

implemented

## Lane

normal

## Product Contract

V2B Tiny MLP and V2C Tiny CNN should support bounded 10-20 epoch training with
validation-based early stopping. Training artifacts must keep the best
validation checkpoint and expose enough learning-curve evidence to decide
whether the model has flattened.

Professor rows remain evaluation-only and cannot affect training, threshold
tuning, checkpoint selection, or early-stopping decisions.

## Relevant Product Docs

- `docs/product/v2-exploration.md`
- `docs/product/modeling.md`

## Acceptance Criteria

- `train-v2-mlp` accepts `--patience` and `--min-delta`.
- `train-v2-cnn` accepts `--patience` and `--min-delta`.
- MLP and CNN metrics JSON include `learning_curve` and `early_stopping`.
- Early stopping uses validation rows only, with public test and professor rows
  evaluated after checkpoint selection.
- Long-running training still prints progress by batch and epoch, plus early
  stop status when triggered.
- Full-data rerun artifacts are produced for MLP and CNN without overwriting the
  previous baseline runs.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-011 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py tests/test_v2_cli.py -q` |
| Integration | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |

## Harness Delta

Story `US-011` is registered in the durable Harness database with verify command
`uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q`.

## Evidence

- Focused tests:
  `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py tests/test_v2_cli.py -q`
  -> 14 passed.
- Real V2B Tiny MLP early-stopping rerun:
  `uv run --extra data --extra ml --python 3.13 python -m drone_sound.cli train-v2-mlp --features artifacts/features/v2_spectral_mfcc_features.npz --model-out artifacts/models/v2b_mlp/tiny_mlp_es.joblib --metrics-out artifacts/models/v2b_mlp/tiny_mlp_es_metrics.json --epochs 20 --batch-size 4096 --patience 3 --min-delta 0.0005`
  -> ran 20/20 epochs, best epoch 19, public F2 0.997475, public FPR
  0.003234, professor recall 0.791554, longest missed 6.5s.
- Real V2C Tiny CNN early-stopping rerun:
  `uv run --extra data --extra ml --extra dl --python 3.13 python -m drone_sound.cli train-v2-cnn --features artifacts/features/v2_spectral_mfcc_features.npz --public-audio-root data/raw/public --professor-audio-root data/raw/private/professor_real_world --model-out artifacts/models/v2c_cnn/tiny_cnn_es.pt --metrics-out artifacts/models/v2c_cnn/tiny_cnn_es_metrics.json --cache-dir artifacts/features/v2_logmel_cache --epochs 20 --batch-size 1024 --patience 4 --min-delta 0.0005`
  -> ran 20/20 epochs, best epoch 18, public F2 0.966515, public FPR
  0.039830, professor recall 0.554358, longest missed 19.0s.
- Early-stopping comparison artifacts:
  `artifacts/models/v2_full_comparison/comparison_early_stopping.json` and
  `artifacts/models/v2_full_comparison/comparison_early_stopping.csv`.
