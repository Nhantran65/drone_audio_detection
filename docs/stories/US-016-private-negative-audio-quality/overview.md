# US-016 Private Negative Audio Quality Assessment

## Current Behavior

Professor-provided evaluation data contains one positive recording. Ten newly
provided negative recordings under `data/raw/private/Recordings/` are not yet
audited, represented in a manifest, or accepted for training/evaluation.

## Target Behavior

Produce a reproducible, non-destructive quality assessment of the new negative
recordings, including signal quality, label-risk proxies, sample-rate
sensitivity, domain shift, and frozen-model false-positive behavior. Conclude
whether the files are suitable for evaluation, training, or neither.

## Affected Users

- Researchers deciding whether to admit private field recordings into the ML
  pipeline.
- Reviewers auditing private-data handling and model-evaluation claims.

## Affected Product Docs

- `docs/product/datasets.md`
- `docs/product/preprocessing.md`
- `docs/product/evaluation.md`

## Non-Goals

- Do not modify, delete, rename, denoise, or redistribute raw private audio.
- Do not retrain or retune any model on the new recordings.
- Do not change the professor-data contract until the quality evidence is
  reviewed.

## Outcome

The recordings passed basic file/signal-integrity checks but exposed a severe
sample-rate confound and large model sensitivity. The intended role is now
explicitly a real-only external test: professor real positive plus private real
negative recordings, evaluated with frozen models and frozen thresholds. The
set is not approved for training, threshold tuning, or selecting a winning
model.
