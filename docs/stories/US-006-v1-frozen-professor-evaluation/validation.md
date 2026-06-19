# Validation

## Proof Strategy

Synthetic fixtures prove frozen model loading, scoring, public metrics, and
professor-positive metrics. A real local eval run checks that the current V1
artifacts and professor manifest work together.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | Frozen model schema validation, score path, professor metric calculations |
| Integration | Synthetic public and professor manifests feed `eval-v1` |
| E2E | Not applicable, no UI |
| Platform | Local extracted professor audio path is ignored data |

## Commands

```text
uv run --extra data --with pytest --python 3.13 python -m pytest
uv run --extra data --python 3.13 python -m drone_sound.cli eval-v1 \
  --model artifacts/models/v1_baseline_model.json \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --metrics-out artifacts/models/v1_eval_metrics.json
```

## Acceptance Evidence

- `uv run --extra data --with pytest --python 3.13 python -m pytest` -> 33 passed.
- Real `eval-v1` completed with frozen threshold `0.225143`.
- Real public test: `public_test_f2=0.871481`.
- Real professor test: 1113 rows, recall `0.752920`, missed windows `275`,
  longest missed segment `16.0` seconds.
- Combined supplementary: `f2=0.868144`.
