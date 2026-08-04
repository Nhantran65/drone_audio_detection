# 0015 Canonical 16 kHz V2 Preprocessing

Date: 2026-08-04

## Status

Accepted

## Context

V2 feature extraction currently preserves native sample rates while model
tracks apply different resampling behavior. Sample rate is strongly correlated
with public dataset source and label, making it a potential shortcut and making
cross-track comparisons difficult to interpret.

## Decision

- Adopt `mono_resample_poly_16khz_v1` as the canonical preprocessing profile.
- Downmix and resample each complete recording once with
  `scipy.signal.resample_poly` before applying unchanged one-second windows.
- Keep `spectral_mfcc_v1`, log-mel settings, MFCC/band definitions, model
  architectures, splits, and strides unchanged.
- Treat artifacts lacking preprocessing metadata as
  `legacy_native_rate_v1`; reject legacy/canonical mixing.
- Write all canonical artifacts below `artifacts/v2_16khz/` and preserve legacy
  artifact bytes.
- Use private real-only data only after model and validation-derived threshold
  are frozen.

## Consequences

The new artifacts remove sample-rate availability as a dataset shortcut and
share one preprocessing contract across all tracks. Results may decline and
training must be repeated, but that decline is evidence rather than a pipeline
failure. Full-recording resampling adds temporary memory and compute cost and
requires explicit provenance checks for every reusable cache.

## Alternatives Considered

1. Preserve native rates for engineered features. Rejected because frequency
   resolution and available bandwidth remain dataset-dependent.
2. Resample each window independently. Rejected because filter boundaries are
   repeated and cross-track preprocessing remains inconsistent.
3. Add denoising or loudness normalization simultaneously. Rejected because it
   would introduce additional experimental variables.
