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

## V1 Training Smoke Baseline

Build the local public manifest from raw WAV folders first. V1 research uses
all datasets registered in `configs/datasets.json`; license metadata is kept in
the audit output but does not block the v1 experiment.

```bash
python -m drone_sound.cli build-public-manifest \
  --registry configs/datasets.json \
  --raw-root data/raw/public \
  --manifest-out data/manifests/public_manifest.csv \
  --audit-out artifacts/audits/public_data_audit.json \
  --split-seed 20260619
```

Then train the smoke baseline. Professor data is evaluation-only and is
rejected for training.

```bash
python -m drone_sound.cli train-v1 \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --model-out artifacts/models/v1_baseline_model.json \
  --metrics-out artifacts/models/v1_metrics.json
```
