# US-017 Execution Plan

## Goal

Build and verify a canonical 16-kHz V2 pipeline, retrain all four experiment
tracks, and evaluate frozen models on real-only private data without changing
legacy artifacts.

## Work Phases

1. Implement shared full-recording mono/polyphase preprocessing.
2. Add bundle, scaler, model, and cache provenance plus compatibility guards.
3. Build the private real-only manifest, evaluation bundle, and frozen evaluator.
4. Add unit, integration, cache, model, real-only, and end-to-end tests.
5. Inventory legacy artifacts, execute V2A/V2B/V2C/V2D with progress, and report.
6. Re-inventory legacy artifacts, build comparisons, and record Harness evidence.

## Hard Gates

- The final artifact root must not exist at run start.
- No private row may enter training, validation, fitting, threshold selection,
  or model selection.
- Legacy artifact SHA-256 inventories must match before and after the run.
- Every long-running command must expose phase, count, elapsed time, rate, and
  ETA where the total is known.

## Stop Conditions

Stop if a required model dependency cannot execute, raw-data labels are
inconsistent with the locked manifest contract, or legacy artifacts change.
