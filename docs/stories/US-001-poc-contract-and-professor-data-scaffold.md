# US-001 POC Contract And Professor Data Scaffold

## Status

implemented

## Lane

normal

## Product Contract

Milestone 1 turns the supplied drone audio POC spec into living product docs,
records the initial ML stack and data-boundary decision, and adds executable
scaffolding to audit professor data without extracting private raw audio into
the repository.

## Relevant Product Docs

- `docs/product/overview.md`
- `docs/product/datasets.md`
- `docs/product/preprocessing.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`

## Acceptance Criteria

- Product docs capture binary scope, dataset roles, preprocessing, modeling,
  evaluation, and professor-data restrictions.
- Professor data is documented as `professor_test` and is excluded from
  training, validation, and threshold tuning.
- Hardware defaults are documented as generic CPU edge proxy, with edge GPU
  optional.
- Manifest constants and validator enforce required fields, allowed labels,
  allowed subtypes, allowed splits, unique `window_id`, and original-file split
  isolation.
- Professor audit CLI reads `fwdronedata.zip` directly, writes manifest and
  audit outputs, redacts exact GPS coordinates by default, and does not extract
  raw audio into repo data folders.
- Unit and integration tests prove the manifest validator and professor audit
  behavior.

## Design Notes

- Commands: `python -m drone_sound.cli audit-professor-data`.
- Queries: none.
- API: Python functions for manifest validation and professor zip auditing.
- Tables: none.
- Domain rules: professor data is positive-only or near-positive-only test data.
- UI surfaces: none.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-001 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `python -m pytest tests/test_manifest.py` |
| Integration | `python -m pytest tests/test_professor_audit.py` |
| E2E | Not applicable, no UI |
| Platform | Not applicable in Milestone 1 |
| Release | `python -m pytest` |

## Harness Delta

Story `US-001` is registered in the durable Harness database with verify command
`python -m pytest`.

## Evidence

- `. .venv/bin/activate && python -m pytest` -> 7 passed.
- `. .venv/bin/activate && python -m drone_sound.cli audit-professor-data --zip fwdronedata.zip --manifest-out data/manifests/professor_manifest.csv --audit-out artifacts/audits/professor_data_audit.json` -> 1113 windows.
- Audit report check confirmed exact GPS coordinates are redacted.
