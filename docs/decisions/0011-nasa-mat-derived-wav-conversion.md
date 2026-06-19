# 0011 NASA MAT Derived WAV Conversion

Date: 2026-06-19

## Status

Accepted

## Context

The NASA Small UAS source is registered for v1 research, but the local archive
contains MATLAB v5 `.mat` files rather than WAV files. The current public
manifest builder intentionally scans WAV files only, and the v1 model contract
expects mono 16 kHz audio windows.

## Decision

- Keep the original NASA `.mat` files as raw source data.
- Add a data-prep command that writes derived mono 16 kHz WAV files.
- Read `acoustics.incident_pascals` from each MAT file and infer source sample
  rate from `median(diff(acoustics.utc_time))`.
- Downmix microphone channels to mono with a channel mean.
- Use batch-global peak scaling across the NASA conversion batch when writing
  PCM int16 WAV files.
- Write a conversion audit JSON with source/output checksums, source sample
  rates, channel counts, timestamp jitter, output frames, scale factor, and
  clipping counts.

## Alternatives Considered

1. Read MAT directly in `build-public-manifest`. Rejected for v1 because the
   existing manifest builder and training baseline already operate on WAV
   paths, and pre-conversion keeps the boundary simpler.
2. Per-file peak normalization. Rejected because it hides relative amplitude
   differences between NASA recordings.
3. Fixed Pascal-to-int16 mapping. Rejected for v1 because outlier risk is
   higher without first auditing the full local batch.

## Consequences

Positive:

- The real local registry can feed the existing WAV manifest path.
- NASA derived WAVs are reproducible and auditable.
- Relative amplitude is better preserved within the NASA batch than with
  per-file normalization.

Tradeoffs:

- The v1 experiment now has optional data dependencies for conversion.
- Derived WAVs are research artifacts and should not be treated as replacements
  for raw MAT files.
- Batch-global scaling preserves within-batch relationships but not physical
  units in the PCM WAV itself.

## Follow-Up

- Run the real NASA conversion before building the all-dataset public manifest.
- Review v1 metrics with and without NASA if conversion artifacts appear to
  dominate results.
