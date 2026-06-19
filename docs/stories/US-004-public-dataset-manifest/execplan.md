# Exec Plan

## Goal

Implement a local public dataset manifest builder that includes every registered
dataset for v1 research training while preserving license metadata in audit
outputs.

## Scope

In scope:

- Add `build-public-manifest` CLI.
- Scan `data/raw/public/<dataset_id>/**/*.wav`.
- Write `data/manifests/public_manifest.csv`.
- Write `artifacts/audits/public_data_audit.json`.
- Split by `original_file` into train, val, and public_test.
- Validate that each split contains both `drone` and `no_drone`.

Out of scope:

- Dataset downloads.
- Resampling, feature caching, or processed audio outputs.
- K-fold cross-validation.
- Professor data inclusion.

## Risk Classification

Risk flags:

- Data model.
- Public contracts.
- Existing behavior.
- Weak proof.

Hard gates:

- Public training/evaluation contract changes.

## Work Phases

1. Add public manifest builder.
2. Add CLI and error handling.
3. Add synthetic six-dataset tests.
4. Update product docs and decision record.
5. Register Harness story and proof.

## Stop Conditions

Pause for human confirmation if:

- The user asks to include professor data in training.
- The split contract needs to weaken public_test isolation.
- The implementation would require downloading datasets.
