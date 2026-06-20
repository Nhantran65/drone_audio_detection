# 0013 V2A Scikit-Learn ML Extra

Date: 2026-06-19

## Status

Accepted

## Context

V2-M2 needs classical ML baselines for Logistic Regression, Linear SVM, Random
Forest, and ExtraTrees. The repository already keeps V1 dependency-free and
uses optional extras for heavier data tooling.

## Decision

- Add a separate optional `ml` dependency extra containing scikit-learn and
  joblib.
- Keep V1 training and evaluation dependency-free.
- Keep V2 feature extraction under the existing `data` extra.
- Run V2A validation with both `data` and `ml` extras.

## Alternatives Considered

1. Add scikit-learn to the base dependencies. Rejected because V1 should remain
   lightweight and dependency-free.
2. Add scikit-learn to the `data` extra. Rejected because data conversion and
   model training are separate capabilities.
3. Hand-roll classical models. Rejected because tree ensembles and calibrated
   linear SVM would add unnecessary implementation risk.

## Consequences

Positive:

- V2A can use standard, tested classical ML implementations.
- V1 users do not install ML dependencies unless they opt in.
- Future V2 model extras can remain explicit by capability.

Tradeoffs:

- V2A commands require installing `.[ml]` or using `uv --extra ml`.
- Model artifacts are joblib files tied to scikit-learn compatibility.

## Follow-Up

- Revisit model export once a final candidate is selected for edge deployment.
