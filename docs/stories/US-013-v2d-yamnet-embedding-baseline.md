# US-013 V2D YAMNet Embedding Baseline

## Status

implemented

## Lane

normal

## Product Contract

V2D implements the first spec-compliant pretrained audio embedding baseline
using YAMNet from TensorFlow Hub. The command caches mean-pooled YAMNet
embeddings for every row in the existing V2 feature bundle, trains a Logistic
Regression head on public `train` rows only, selects thresholds on public `val`
rows only, and evaluates frozen public test and professor positive-only rows.

V2D remains `reference_only` unless a future deployment story proves the full
pretrained extractor and head meet the lightweight edge target.

## Relevant Product Docs

- `docs/product/v2-exploration.md`
- `docs/product/modeling.md`
- `docs/product/preprocessing.md`
- `docs/product/evaluation.md`

## Acceptance Criteria

- `train-v2-embedding --embedding-model yamnet --head logistic_regression`
  trains successfully and writes a joblib head plus metrics JSON.
- The command creates and reuses a YAMNet embedding cache when source bundle,
  shape, backend, and aggregation metadata match.
- Professor rows are excluded from training, validation, threshold tuning, and
  model selection.
- Metrics use `feature_set=pretrained_embedding_v1`,
  `model_type=yamnet_head`, `track=v2d`, and deployability
  `reference_only`.
- Long-running embedding extraction prints progress with elapsed time,
  processed/total rows, throughput, ETA, and current file.
- V2 comparison artifacts can include the V2D row without reintroducing any
  non-spec pretrained backend.

## Design Notes

- Commands: existing `train-v2-embedding` CLI.
- Dependency extras: `data`, `ml`, and `embedding`; `embedding` installs
  TensorFlow CUDA runtime packages via `tensorflow[and-cuda]`.
- Backend: TensorFlow Hub YAMNet via `hub.load("https://tfhub.dev/google/yamnet/1")`.
- Head: scikit-learn Logistic Regression saved with joblib.
- Domain rules: V2D is reference-only unless deployability proof changes.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-013 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `uv run --extra data --extra ml --extra embedding --with pytest --python 3.13 python -m pytest tests/test_v2_cli.py tests/test_v2_ml.py tests/test_v2_contract.py -q` |
| Integration | `uv run --extra data --extra ml --extra embedding --with pytest --python 3.13 python -m pytest -q` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | Real `CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH="$TF_CUDA_LIBS:${LD_LIBRARY_PATH:-}" uv run --extra data --extra ml --extra embedding --python 3.13 python -m drone_sound.cli train-v2-embedding ...` run on `artifacts/features/v2_spectral_mfcc_features.npz` when TensorFlow Hub download/runtime is available |

## Harness Delta

Story `US-013` is registered in the durable Harness database with verify command
`uv run --extra data --extra ml --extra embedding --with pytest --python 3.13 python -m pytest -q`.

## Evidence

Progress update 2026-06-20:

- GPU-backed YAMNet run completed with
  `CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$TF_CUDA_LIBS:${LD_LIBRARY_PATH:-}"`
  and `uv run --extra data --extra ml --extra embedding --python 3.13`.
- Long-running extraction logged phase, elapsed time, processed/total rows,
  file count, throughput, ETA, and current file. Final cache phase processed
  402383/402383 rows across 29442 files in 23m08s.
- Cache artifacts written:
  `artifacts/features/v2_yamnet_cache/yamnet_embedding_v1.npy` and
  `artifacts/features/v2_yamnet_cache/yamnet_embedding_v1_meta.json`.
- Model artifacts written:
  `artifacts/models/v2d_yamnet/yamnet_head.joblib` and
  `artifacts/models/v2d_yamnet/yamnet_head_metrics.json`.
- Frozen test metrics:
  `public_test_f2=0.9744354203077343`,
  `public_test_fpr=0.04188813244973198`, and
  `professor_recall=0.6199460916442049`.
- Comparison artifacts updated with row `yamnet_head`, `track=v2d`,
  `feature/input=pretrained_embedding_v1`, and
  `deployability_label=reference_only`.
- Validation passed:
  `uv run --extra data --extra ml --extra embedding --with pytest --python 3.13 python -m pytest tests/test_v2_cli.py tests/test_v2_ml.py tests/test_v2_contract.py -q`
  -> 20 passed, 4 warnings.
- Validation passed:
  `uv run --extra data --extra ml --extra embedding --with pytest --python 3.13 python -m pytest -q`
  -> 62 passed, 4 warnings.
- Harness verification passed: `scripts/bin/harness-cli story verify US-013`.
