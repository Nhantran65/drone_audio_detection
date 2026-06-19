# Overview

## Status

implemented

## Lane

high-risk

## Current Behavior

The NASA Small UAS source is present locally as MATLAB `.mat` files, while the
public manifest builder scans local WAV files.

## Target Behavior

Convert NASA MAT audio into derived mono 16 kHz WAV files with batch-global
scaling and a JSON audit report, so the existing public manifest builder can
include NASA in v1 research manifests.

## Affected Users

- Researcher assembling the local v1 public manifest.
- Future agent validating data provenance and conversion quality.

## Affected Product Docs

- `docs/product/datasets.md`
- `docs/product/dataset-sources.md`
- `docs/product/preprocessing.md`
- `docs/product/modeling.md`

## Non-Goals

- Replace or delete raw NASA MAT files.
- Claim NASA is cleared for product or commercial training.
- Add direct MAT support to `build-public-manifest`.
- Add production-quality feature extraction or model training.
