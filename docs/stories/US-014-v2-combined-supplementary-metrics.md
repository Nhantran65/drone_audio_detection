# US-014 V2 Combined Supplementary Metrics

## Status

implemented

## Lane

normal

## Product Contract

V2 frozen evaluation reports a combined supplementary metrics block over
`public_test + professor_test` after the model and threshold are frozen.
Combined metrics are informational only and do not replace the public held-out
test benchmark or professor positive-only recall reporting.

Professor rows remain excluded from training, validation, threshold tuning, and
model selection.

## Relevant Product Docs

- `docs/product/evaluation.md`
- `docs/product/v2-exploration.md`
- `docs/project-summary.md`

## Acceptance Criteria

- V2 metrics reports include `combined_supplementary.supplementary`,
  `combined_supplementary.note`, `combined_supplementary.counts`, and
  `combined_supplementary.metrics`.
- Combined metrics use public-test labels/scores plus professor-test
  labels/scores with the frozen stored `threshold_f2`.
- `eval-v2` writes combined supplementary metrics for V2A/V2B joblib packages.
- `eval-v2-cnn` loads a frozen V2C `.pt` checkpoint, reuses the log-mel cache,
  scores public/professor rows with progress logs, and writes combined
  supplementary metrics.
- `eval-v2-embedding` loads a frozen V2D YAMNet joblib head, reuses the
  pretrained embedding cache, scores public/professor rows with progress logs,
  and writes combined supplementary metrics.
- Real rerun artifacts are written under
  `artifacts/models/v2_combined_supplementary/`.
- `docs/project-summary.md` includes a compact combined supplementary metrics
  table and caveat.

## Design Notes

- Commands: `eval-v2`, `eval-v2-cnn`, `eval-v2-embedding`.
- Domain rules: no retraining, no threshold retuning, professor data remains
  evaluation-only.
- Combined set: public held-out test plus professor positive-only rows.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-014 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py tests/test_v2_cli.py -q` |
| Integration | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | Real frozen eval reruns for V2A/V2B/V2C/V2D into `artifacts/models/v2_combined_supplementary/` |

## Harness Delta

Story `US-014` is registered in the durable Harness database with verify command
`uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q`.

## Evidence

- `uv run --extra data --extra ml --with pytest --python 3.13 python -m pytest tests/test_v2_ml.py tests/test_v2_cli.py -q`
  -> 17 passed.
- `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q`
  -> 62 passed.
- Real frozen combined eval artifacts written under
  `artifacts/models/v2_combined_supplementary/` for V2A/V2B/V2C/V2D.
- V2D YAMNet frozen eval artifact:
  `artifacts/models/v2_combined_supplementary/yamnet_head_combined_eval.json`
  with `combined_f2=0.96162998703494`,
  `combined_roc_auc=0.9940048277610928`, and
  confusion matrix `TN/FP/FN/TP=42361/1852/706/23438`.
- Combined row counts are `drone=24144`, `no_drone=44213`.
