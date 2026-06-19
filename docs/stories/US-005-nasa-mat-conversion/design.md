# Design

## Domain Model

Each NASA MAT file is treated as one source clip. The converter reads
`acoustics.incident_pascals`, infers the source sample rate from
`acoustics.utc_time`, downmixes microphone channels to mono, resamples to 16 kHz,
and writes a derived PCM int16 WAV file.

## Application Flow

`convert-nasa-mat` performs two passes:

1. Audit every MAT file and compute the batch-global mono peak.
2. Reload each MAT file, resample, apply the shared scale factor, write WAV, and
   record per-file audit metadata.

## Interface Contract

```bash
python -m drone_sound.cli convert-nasa-mat \
  --mat-root data/raw/public/nasa_small_uas_flyover/data \
  --wav-out data/raw/public/nasa_small_uas_flyover/converted_wav \
  --audit-out artifacts/audits/nasa_mat_conversion_audit.json \
  --sample-rate 16000
```

Errors are printed as `NASA MAT conversion error: ...` and exit with status `1`.

## Data Model

No database changes. The audit JSON records conversion policy, global scale
factor, source/output checksums, source rates, channel counts, duration, output
frames, timestamp jitter, max mono amplitude, and clipped sample counts.

## UI / Platform Impact

CLI only.

## Observability

The conversion audit JSON is the durable data-prep report. Harness durable rows
could not be updated in this checkout because `scripts/bin/harness-cli` is
absent.

## Alternatives Considered

1. Direct MAT reader in manifest builder. Deferred to keep v1 manifest logic
   WAV-only.
2. Per-file normalization. Rejected because it loses within-NASA amplitude
   comparability.
3. Fixed Pascal mapping. Rejected for v1 because clipping risk is higher before
   batch audit.
