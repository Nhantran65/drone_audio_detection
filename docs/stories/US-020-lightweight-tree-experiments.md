# US-020 Lightweight Tree Experiments

## Status

implemented

## Lane

normal

## Product Contract

Benchmark a bounded set of compressed Random Forest, histogram gradient
boosting, and shallow XGBoost candidates on the canonical 16 kHz engineered
feature bundle. Candidate fitting, early stopping, threshold selection, and
model selection may use public train/validation rows only. Public test,
professor-positive, and private real-only results are post-freeze reports.

## Relevant Product Docs

- `docs/product/modeling.md`
- `docs/product/evaluation.md`
- `docs/product/v2-exploration.md`

## Acceptance Criteria

- The experiment writes to a fresh isolated directory and refuses overwrite.
- Three compressed RF, two HistGradientBoosting, and one shallow XGBoost
  candidate expose progress and deterministic configuration.
- Each group winner is selected by validation F2, then validation FPR, then
  serialized model size; test metrics cannot affect selection.
- Frozen packages retain the canonical preprocessing profile, bundle
  fingerprint, frozen validation thresholds, and runtime dependency metadata.
- Only the frozen winner of each group is evaluated on private real-only data.
- The report compares validation, public, professor, private real-only,
  artifact size, tree complexity, and inference latency against the existing
  canonical Random Forest baseline.

## Design Notes

- Command: `scripts/run_lightweight_tree_experiments.py`.
- Input: `artifacts/v2_16khz/features/public_professor_features.npz`.
- Output: `artifacts/v2_16khz/models/lightweight_tree_experiments/`.
- XGBoost is isolated in the optional `boosting` dependency extra.
- The already-observed private real-only set remains an exploratory stress
  test and cannot establish an independently validated improvement.

## Validation

| Layer | Expected proof |
| --- | --- |
| Unit | Candidate grid, validation-only selection, overwrite refusal, XGBoost label adapter |
| Integration | Synthetic canonical bundle trains sklearn candidates and emits frozen packages |
| E2E | Six real candidates train; three frozen group winners evaluate on 1,705 private windows |
| Platform | Not required |
| Release | Full pytest suite and artifact/result reconciliation |

## Harness Delta

Adds a reproducible bounded model-efficiency benchmark rather than extending
the primary V2A model family or changing its frozen baseline.

## Evidence

- `scripts/run_lightweight_tree_experiments.py` completed six candidates in
  177 seconds and wrote one validation-selected winner per family.
- Winners: compressed RF 80/depth 14/leaf 5, HistGradientBoosting 31 leaves,
  and shallow XGBoost depth 4.
- Private real-only winner evaluations reconciled 1,113 positive plus 592
  negative windows and recorded `fit_called=false`, `threshold_tuned=false`.
- Existing canonical RF model and metrics SHA-256 values still match the
  original `artifacts/v2_16khz/run_manifest.json` inventory.
- All new artifact checksums verified from the experiment run manifest.
- `.venv/bin/python -m pytest -q` -> 91 passed.
