# 0010 V1 Research All Dataset Manifest

Date: 2026-06-19

## Status

Accepted

## Context

The v1 training smoke baseline needs a public train/validation/test manifest
before it can produce a real model artifact. The current registry includes
permissive, non-commercial, no-derivatives, and unspecified-license datasets.
The human explicitly chose to use every registered dataset for the current v1
research experiment instead of blocking on license clarification.

## Decision

- Build the v1 public manifest from every dataset in `configs/datasets.json`.
- Keep license metadata in audit reports and manifest notes.
- Do not use license status to block v1 research manifest creation or training.
- Keep this as a research-only override.
- Keep professor data out of public train, validation, public test, and
  threshold selection.

## Alternatives Considered

1. License-gated manifest creation. Rejected for v1 because it would delay the
   first model feedback loop.
2. Drop license metadata entirely. Rejected because future productization still
   needs source and license traceability.

## Consequences

Positive:

- V1 can train on all currently identified positive and hard-negative sources.
- Audit output still preserves license risk for later review.

Tradeoffs:

- V1 metrics are research evidence only.
- Product or commercial training claims still require license review.

## Follow-Up

- Revisit license-gated productization before release or commercial use.
- Add proper preprocessing/resampling in a later story.
