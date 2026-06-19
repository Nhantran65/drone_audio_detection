# 0012 V2 Lightweight ML/DL Exploration

Date: 2026-06-19

## Status

Accepted

## Context

V1 proved the full public-data training, validation threshold, public-test
evaluation, and professor positive-only evaluation loop with a dependency-free
centroid classifier. The result is reportable as a smoke baseline, but the
model uses only simple time-domain features and may miss spectral or
time-frequency drone patterns.

The next phase needs to improve performance enough for a professor report while
keeping professor data evaluation-only and preserving the V1 commands as a
stable reference.

## Decision

- Add a V2.1 exploration initiative with four tracks:
  - V2A classical ML on `spectral_mfcc_v1`.
  - V2B Tiny MLP on `spectral_mfcc_v1`.
  - V2C Tiny CNN on `logmel_v1`.
  - V2D pretrained audio embeddings as a reference or upper-bound baseline.
- Keep `train-v1` and `eval-v1` unchanged.
- Keep professor data out of training, validation, threshold tuning, and model
  selection.
- Use validation F2 as the main selection metric, with public FPR, professor
  recall, longest missed professor segment, model size, and inference latency
  as guardrails.
- Report fixed-FPR thresholds and recall at fixed FPR levels as the
  product-oriented view.
- Treat pretrained embedding models as `reference_only` unless the full
  inference path satisfies lightweight deployment constraints.

## Alternatives Considered

1. Jump directly to Tiny CNN. Rejected because an engineered-feature classical
   baseline is faster to train, easier to explain, and may be sufficient for
   the professor report.
2. Make pretrained embeddings the primary model. Rejected because they may
   exceed the lightweight deployment target if the full backbone is required at
   inference time.
3. Optimize only public-test F2. Rejected because a candidate could improve F2
   while worsening false positives or professor real-world recall.

## Consequences

Positive:

- V2 has a clear path from lightweight classical ML to compact neural models.
- V1 remains a stable reference baseline.
- The report can distinguish deployable lightweight candidates from
  reference-only transfer-learning results.
- Stop criteria make it clear when to pause experimentation and report.

Tradeoffs:

- V2 introduces optional ML/DL dependencies that are not part of V1.
- More artifacts are required: feature audits, model summaries, threshold
  reports, comparison tables, and error analysis.
- Fixed-FPR views require careful validation-only threshold handling.

## Follow-Up

- Create the first V2 story for setup and compatibility.
- Add V2 feature/model/metrics schema before implementing model training.
- Decide the exact dependency boundary for classical ML and DL packages.
- Generate V1 vs V2 comparison artifacts once V2 candidates complete.
