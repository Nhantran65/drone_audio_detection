# Exec Plan

## Goal

Add an auditable NASA MAT-to-WAV conversion step so the real local public
manifest can include all registered datasets for v1 research.

## Scope

In scope:

- Optional `data` dependencies for numpy/scipy.
- `convert-nasa-mat` CLI.
- Batch-global PCM scaling.
- Conversion audit JSON.
- Synthetic unit and integration tests.
- Product docs, story, decision, and test matrix updates.

Out of scope:

- Product-quality preprocessing.
- Direct MAT support in `build-public-manifest`.
- NASA license clearance.
- Professor data changes.

## Risk Classification

Risk flags:

- Data model.
- Public contracts.
- Existing behavior.
- Weak proof.

Hard gates:

- Public training/evaluation data boundary changes.

## Work Phases

1. Add converter module and CLI.
2. Add optional data dependencies.
3. Add synthetic MAT tests and manifest/training integration proof.
4. Update product docs, decision, and matrix.
5. Run validation and record any Harness friction.

## Stop Conditions

Pause for human confirmation if:

- NASA conversion would require deleting or rewriting raw MAT files.
- Public manifest rules would need to include professor data.
- Conversion quality issues require a different scaling policy.
