# Lightweight Drone Audio Classification POC

This repository is a proof of concept for detecting drone audio in short
window-level recordings.

Milestone 1 focuses on product contracts, manifest validation, and professor
data audit tooling. It does not train a model yet.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Professor Data Audit

`fwdronedata.zip` is private local data and is ignored by git. Audit it with:

```bash
python -m drone_sound.cli audit-professor-data \
  --zip fwdronedata.zip \
  --manifest-out data/manifests/professor_manifest.csv \
  --audit-out artifacts/audits/professor_data_audit.json
```

The audit reads the zip directly, writes a manifest plus redacted JSON report,
and does not extract raw audio into the repository.

## Validation

```bash
python -m pytest
```
