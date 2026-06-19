# Documentation Map

This directory holds the project harness plus the living product contract for
the drone audio classification POC.

## Main Files

- `HARNESS.md`: how humans and agents collaborate.
- `FEATURE_INTAKE.md`: how prompts become tiny, normal, or high-risk work.
- `ARCHITECTURE.md`: architecture discovery and boundary rules.
- `TEST_MATRIX.md`: proof map mirrored in markdown; current durable proof
  status is queried with `scripts/bin/harness-cli query matrix` when the local
  ignored CLI binary is installed.
- `HARNESS_BACKLOG.md`: legacy improvement list; current improvement records
  are stored with `scripts/bin/harness-cli backlog`.
- `GLOSSARY.md`: shared terms.

## Folders

- `product/`: current product truth for datasets, preprocessing, modeling, and
  evaluation.
- `stories/`: implemented feature packets and backlog.
- `decisions/`: durable decisions and tradeoffs.
- `demo/`: concrete walkthroughs that show how the harness transforms input
  into agent-ready work.
- `templates/`: reusable spec-intake, story, plan, decision, and validation
  formats.

## Current State

The repo has Python POC code, tests, product docs, decisions, and implemented
story packets. Local raw data, generated manifests, generated artifacts,
`fwdronedata.zip`, `harness.db`, and the Harness CLI binary remain ignored by
git.
