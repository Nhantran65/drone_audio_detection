# Validation

## Proof Strategy

Synthetic MAT fixtures prove conversion behavior deterministically. The real
local conversion command is then run against NASA MAT files to produce derived
WAVs and a conversion audit.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | MAT parsing, sample-rate inference, batch-global scaling, WAV output |
| Integration | Converted NASA WAVs feed public manifest and `train-v1` with synthetic datasets |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Performance | Real conversion should complete locally without loading all MAT files at once |
| Logs/Audit | Audit JSON includes scale policy, checksums, timestamp jitter, and clipping counts |

## Fixtures

Tests create MATLAB files with `scipy.io.savemat`, four-channel
`acoustics.incident_pascals`, and deterministic `acoustics.utc_time`.

## Commands

```text
uv run --extra data --with pytest --python 3.13 python -m pytest
uv run --extra data --python 3.13 python -m drone_sound.cli convert-nasa-mat \
  --mat-root data/raw/public/nasa_small_uas_flyover/data \
  --wav-out data/raw/public/nasa_small_uas_flyover/converted_wav \
  --audit-out artifacts/audits/nasa_mat_conversion_audit.json \
  --sample-rate 16000
uv run --extra data --python 3.13 python -m drone_sound.cli train-v1 \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --model-out artifacts/models/v1_baseline_model.json \
  --metrics-out artifacts/models/v1_metrics.json
```

## Acceptance Evidence

- `uv run --extra data --with pytest --python 3.13 python -m pytest` -> 28 passed.
- Real NASA conversion: 62 MAT files converted to
  `data/raw/public/nasa_small_uas_flyover/converted_wav`.
- NASA conversion audit: `scale_policy=batch_global_peak`,
  `source_sample_rate_hz=20000`, `clipped_samples=0`.
- Real public manifest build: 6 datasets, 401270 windows, 9 unsupported
  UrbanSound8K WAV files skipped with audit entries.
- Real `train-v1` completed after `tqdm` extraction progress:
  `val_f2=0.878716`, threshold `0.225143`,
  `public_test_f2=0.871481`.
