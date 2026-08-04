# US-017 Canonical 16 kHz V2 Retraining

## Current Behavior

V2 engineered features are extracted at each file's native sample rate while
the CNN and YAMNet tracks resample individual windows. The resulting artifacts
do not carry a shared preprocessing identity, so mixed-rate data can leak
dataset provenance and incompatible bundles, caches, scalers, and models can be
combined accidentally.

## Target Behavior

All V2 tracks use one preprocessing contract:
`mono_resample_poly_16khz_v1`. Each complete recording is downmixed to mono,
resampled once with `scipy.signal.resample_poly`, and then divided into
one-second windows with a 0.5-second stride. New artifacts live only under
`artifacts/v2_16khz/`; legacy artifacts remain byte-identical.

The four frozen V2 tracks are evaluated on a real-only external test containing
the professor positive recording and ten private negative recordings. No
training, threshold tuning, or model selection may use this external set.

## Non-Goals

- No denoising, loudness normalization, new feature, or architecture change.
- No use of private recordings for fitting or selecting models.
- No deletion, migration, or overwrite of legacy V2 artifacts.

## Acceptance Summary

- Metadata and compatibility checks prevent legacy/16-kHz artifact mixing.
- Cache reuse requires both preprocessing profile and bundle fingerprint.
- All V2A/V2B/V2C/V2D artifacts and reports are rebuilt in the new namespace.
- A before/after SHA-256 inventory proves that legacy artifacts did not change.
