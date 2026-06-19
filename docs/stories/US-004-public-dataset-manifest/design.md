# Design

## Domain Model

The builder treats each WAV file as the split group. Each group has dataset
metadata from `configs/datasets.json`, inferred subtype, checksum, duration,
estimated windows, and one split.

## Application Flow

`build-public-manifest` loads and validates the dataset registry, scans every
registered dataset folder, audits WAV metadata, assigns deterministic splits,
writes manifest rows, validates the manifest, and writes an audit JSON report.

## Interface Contract

```bash
python -m drone_sound.cli build-public-manifest \
  --registry configs/datasets.json \
  --raw-root data/raw/public \
  --manifest-out data/manifests/public_manifest.csv \
  --audit-out artifacts/audits/public_data_audit.json \
  --split-seed 20260619
```

Errors are printed as `public dataset error: ...` and exit with status `1`.

## Data Model

No database changes. Manifest rows use the existing required manifest fields.
The audit JSON stores dataset license metadata, split policy, per-dataset
counts, and per-file audio metadata.

## UI / Platform Impact

CLI only.

## Observability

The audit JSON is the durable public dataset assembly report. Harness records
the story, decision, and validation trace.

## Alternatives Considered

1. License-gated training manifest. Rejected by user direction for v1 research.
2. Window-level split. Rejected because it violates leakage rules.
3. Automatic dataset download. Deferred because source pages are not stable
   direct artifact URLs.
