# US-016 Exec Plan

## Goal

Decide, with reproducible evidence, whether the newly supplied noisy negative
recordings are usable for model evaluation or future training.

## Scope

In scope:

- All ten WAV files under `data/raw/private/Recordings/`.
- Existing professor positive audio as a local comparison.
- Frozen Tiny MLP and Random Forest behavior.
- Native-rate versus in-memory 16 kHz sensitivity.
- File-level independence, label-confidence limitations, and leakage risk.

Out of scope:

- Model retraining, threshold tuning, raw-file edits, or external upload.
- Claims that automated signal/model checks replace expert listening.

## Risk Classification

Risk flags:

- Audit/security: private research recordings.
- Data model: potential change from positive-only to mixed-label professor data.
- Existing behavior: future evaluation semantics may change.
- Weak proof: no human annotation timestamps or independent listening labels.

Hard gates:

- Private-data handling and audit/security.

## Work Phases

1. Inventory and metadata audit.
2. Non-destructive signal preprocessing and quality profiling.
3. Feature/domain/model sensitivity analysis.
4. Reproducible notebook and report generation.
5. Validation and contract recommendation.
6. Harness evidence update.

## Stop Conditions

Pause for human confirmation if raw data would need mutation, external upload,
or the analysis would require treating inferred labels as verified truth.
