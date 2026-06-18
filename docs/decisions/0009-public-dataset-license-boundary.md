# 0009 Public Dataset License Boundary

Date: 2026-06-19

## Status

Accepted

## Context

The first public dataset list mixes permissive CC BY 4.0 sources,
non-commercial Creative Commons sources, a no-derivatives source, and a NASA
source whose portal currently reports no specified license. The POC needs a
clear boundary before acquisition and training scripts are added.

## Decision

- Store dataset source and license truth in `configs/datasets.json`.
- Treat DroneNoise and Kielce UAV as `commercial_ok` public positives.
- Treat NASA Small UAS as `license_unspecified` and exclude it from training
  until license is clarified.
- Treat AeroSonicDB conservatively as `non_commercial` because its page has
  conflicting license signals.
- Treat IDMT-TRAFFIC as `non_commercial` with additional no-derivatives review
  needed before transformed-data redistribution or product training.
- Treat UrbanSound8K as `non_commercial`.

## Alternatives Considered

1. Use all datasets for training immediately. Rejected because it would blur
   research-only and commercial/productization boundaries.
2. Exclude all restricted datasets. Rejected for the POC because hard negatives
   are necessary, and research-only evaluation remains useful when documented.

## Consequences

Positive:

- Dataset acquisition can proceed without hiding license risk.
- Training and reporting code can use one registry for label and license rules.

Tradeoffs:

- Some useful hard-negative sources remain research-only.
- Productization will require license clarification or replacement datasets.

## Follow-Up

- Add a speech-only negative dataset with compatible license.
- Clarify NASA and AeroSonicDB license details before final training claims.
- Review IDMT no-derivatives constraints before distributing transformed
  windows or features.
