# Product Docs

This directory contains the living product contract for the lightweight drone
audio classification POC. `SPEC.md` is input material; the smaller files here
are the current operating truth.

Current contract areas:

- `overview.md`: POC scope and implemented tooling.
- `datasets.md`: dataset roles, manifest fields, splits, labels, and privacy
  rules.
- `dataset-sources.md`: public source registry and license status summary.
- `preprocessing.md`: audio standardization, windowing, feature plan, and the
  NASA MAT-to-derived-WAV conversion boundary used for v1.
- `modeling.md`: baseline and planned model stack.
- `evaluation.md`: public, hard-negative, and professor-data metrics rules.

## Update Rule

When behavior changes:

1. Update the affected product doc.
2. Update or create the story packet.
3. Update durable proof status with `scripts/bin/harness-cli story add` or
   `scripts/bin/harness-cli story update`.
4. Record a decision if the change affects architecture, scope, risk, or a
   previously settled product rule.
