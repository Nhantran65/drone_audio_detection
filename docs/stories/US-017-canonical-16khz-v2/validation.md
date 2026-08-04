# US-017 Validation

## Proof Strategy

Validate signal-level resampling, artifact compatibility, evaluation isolation,
cache identity, full-track execution, and preservation of legacy files.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | 16-kHz identity; 48/44.1/22.05-kHz duration; anti-alias; determinism |
| Integration | Mixed-rate bundle preserves rows/splits/labels and has 100 features |
| Model | legacy/legacy works; mixed profiles fail; thresholds come from validation |
| Cache | reuse only for matching bundle fingerprint and preprocessing profile |
| Real-only | 1,113 positive and 592 negative windows; inference-only metrics reconcile |
| E2E | V2A/B/C/D train, freeze, evaluate, compare, and emit progress/audit evidence |
| Audit | Legacy artifact SHA-256 inventories are identical before and after |

## Acceptance Evidence

- Full test suite: `77 passed, 4 warnings` with data, ML, DL, and embedding extras.
- Canonical feature bundle: 402,383 rows, 100 `spectral_mfcc_v1` features,
  all processed at 16 kHz, with source rates spanning 8 kHz to 192 kHz.
- Legacy/canonical window contract: window IDs, split assignments, and labels all
  match for all 402,383 rows.
- Private real-only bundle: 1,705 rows from 11 recordings, comprising exactly
  1,113 positive and 592 negative windows.
- All seven frozen models were evaluated without fit or threshold tuning. The
  strongest real-only result was Random Forest: F2 0.846707, balanced accuracy
  0.869325, and FPR 0.086149.
- `artifacts/v2_16khz/run_manifest.json` is complete and records checksums for
  the new namespace, dataset hashes, environment versions, and commands.
- Legacy inventory verification: all 85 files under the pre-existing artifact
  namespace had identical before/after SHA-256 inventory
  `eb81bd949cb1ac3bdc5e094920b42759e83b240da7c6ed9fb1ed3762d278a62b`.
- Comparison outputs:
  `artifacts/v2_16khz/comparison/all_tracks_and_legacy.{json,csv}` include seven
  canonical and nine legacy rows, while explicitly treating regression as a
  reportable result rather than a pipeline failure.
