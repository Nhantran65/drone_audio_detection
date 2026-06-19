# Overview

## Status

implemented

## Lane

high-risk

## Current Behavior

`train-v1` reports validation and public test metrics while training. Professor
data is audited into a manifest but is not evaluated by the frozen model.

## Target Behavior

Evaluate an existing V1 model and stored threshold on `public_test`,
`professor_test`, and a clearly caveated combined supplementary report without
retraining or retuning.

## Affected Users

- Researcher checking whether the public-trained V1 baseline generalizes to
  professor real-world data.
- Future agent comparing frozen model reports across data revisions.

## Affected Product Docs

- `docs/product/evaluation.md`
- `docs/product/overview.md`

## Non-Goals

- Train with professor data.
- Select thresholds with professor data.
- Report professor-only false positive rate, specificity, accuracy, or
  precision.
