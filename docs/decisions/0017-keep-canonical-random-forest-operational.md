# 0017 Keep the canonical Random Forest as the operational model

Date: 2026-08-05

## Status

Accepted

## Context

The canonical 16 kHz Random Forest is 14.1 MB and achieved private real-only
precision 0.947368, recall 0.824798, F1 0.881844, F2 0.846707, and FPR
0.086149. US-020 evaluated compressed Random Forest, HistGradientBoosting, and
shallow XGBoost alternatives to determine whether similar robustness could be
retained with a smaller artifact.

HistGradientBoosting and XGBoost reduced their serialized artifacts below 1 MB
and performed strongly on public validation/test data, but recall on the single
private positive recording fell to 0.760108 and 0.714286 respectively. The
compressed RF winner improved descriptive real-only F1/F2 slightly, but reduced
artifact size by only about 15% and increased FPR.

The private real-only benchmark has already informed research iterations and
contains only one independent positive recording, so these differences cannot
establish an independently validated replacement.

## Decision

Keep
`artifacts/v2_16khz/models/v2a/models/random_forest.joblib` as the operational
model package. It uses preprocessing profile `mono_resample_poly_16khz_v1` and
its frozen validation-selected `threshold_f2` of 0.4085200850214452.

The US-020 candidates remain research artifacts. They must not replace the
operational RF solely from the already-observed private stress test.

## Alternatives Considered

1. Promote `compressed_rf_80_d14_leaf5`: similar real-only behavior and an
   11.9 MB artifact, but the size reduction is modest and FPR is higher.
2. Promote `hist_gb_leaf31_iter200`: 920 KB and strongest validation F2, but
   materially lower private positive recall and higher private FPR.
3. Promote `xgb_shallow_d4_iter400`: 647 KB model artifact and fast scoring,
   but lowest private positive recall and an additional native runtime.

## Consequences

Positive:

- Preserves the strongest current balance of real-only precision, recall, F1,
  F2, and FPR without changing deployment dependencies.
- Keeps the canonical model, scaler, preprocessing metadata, and threshold
  contract unchanged.
- Avoids promoting a model from repeated observation of one positive recording.

Tradeoffs:

- The operational artifact remains 14.1 MB rather than below 1 MB.
- The result remains a POC stress-test conclusion, not a deployment-performance
  estimate.
- HGB may still be preferable if a future device imposes a hard sub-1-MB model
  budget and new independent data confirms its robustness.

## Follow-Up

- Collect independent positive recordings across devices, distances,
  environments, and recording sessions.
- Predeclare promotion guardrails and evaluate RF versus HGB once on a new
  recording-level holdout.
- Measure end-to-end feature extraction plus inference latency on target
  hardware before making an edge deployment claim.
