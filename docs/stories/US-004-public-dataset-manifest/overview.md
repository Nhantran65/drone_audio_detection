# Overview

## Status

implemented

## Lane

high-risk

## Current Behavior

The repo has a dataset registry and a `train-v1` command, but no public
train/validation/test manifest. Running `train-v1` against the current
professor-only manifest fails correctly because professor data is not training
data.

## Target Behavior

Build a local public manifest from all datasets registered in
`configs/datasets.json`, using local WAV folders under
`data/raw/public/<dataset_id>/`. The manifest can feed `train-v1` for a v1
research model.

## Current Local Data Caveat

The implementation supports local WAV folders. NASA Small UAS `.mat` files are
handled by the separate US-005 derived WAV conversion step before building the
real all-registered-dataset manifest.

## Affected Users

- Researcher running v1 dataset assembly and training locally.
- Future agent validating dataset splits and model evidence.

## Affected Product Docs

- `docs/product/datasets.md`
- `docs/product/dataset-sources.md`
- `docs/product/preprocessing.md`
- `docs/product/evaluation.md`

## Non-Goals

- Download public datasets automatically.
- Resample or write processed audio copies.
- Clear datasets for product or commercial training.
- Add k-fold training.
