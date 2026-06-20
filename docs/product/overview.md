# Drone Audio Classification POC Overview

## Objective

Build a lightweight binary audio classifier that answers whether a 1 second
audio window contains drone sound.

## Initial Scope

- Input: mono 16 kHz audio windows.
- Window length: 1.0 second.
- Window stride: 0.5 second.
- Output: `P(drone)` and a thresholded `drone` or `no_drone` decision.
- Primary deployment proxy: generic CPU edge hardware.
- Optional deployment lane: edge GPU benchmark if suitable hardware appears.

## Out Of Scope For Milestone 1

- Public dataset download.
- Model training.
- Threshold tuning.
- Real-time product UI.
- Cloud deployment.
- Event-level smoothing or tracking.

## Implemented Tooling After Milestone 1

- Professor data audit to manifest plus redacted audit JSON.
- Public dataset registry and license manifest export.
- Public WAV manifest builder for local raw folders.
- V1 dependency-free smoke baseline that trains from non-professor manifest
  rows and selects its threshold from validation data.
- V1 frozen evaluation tooling for public test, professor positive test, and
  caveated combined supplementary metrics.

## V2.1 Exploration Scope

V2.1 prepares the next reportable model phase. It keeps the binary
`drone`/`no_drone` task, preserves V1 commands unchanged, and explores:

- Classical ML on richer engineered spectral/MFCC features.
- Tiny MLP on the same engineered features.
- Tiny CNN on log-mel spectrograms.
- Pretrained audio embeddings as a reference or upper-bound baseline.

The V2.1 product contract is maintained in `docs/product/v2-exploration.md`.
Professor real-world data remains evaluation-only and cannot be used for
training, validation, threshold tuning, or model selection.
V2A classical ML sweep and frozen V2 evaluation are implemented from V2 feature
bundles.

## Living Contract

`SPEC.md` remains input material. Current project truth should be maintained in
the smaller product docs under `docs/product/`, story packets, decisions, and
executable validation.
