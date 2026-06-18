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

## Living Contract

`SPEC.md` remains input material. Current project truth should be maintained in
the smaller product docs under `docs/product/`, story packets, decisions, and
executable validation.
