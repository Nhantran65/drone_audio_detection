# US-002 Public Dataset Registry And License Audit

## Status

implemented

## Lane

normal

## Product Contract

The POC must maintain a source-controlled dataset registry that records public
dataset roles, license status, binary label mapping, subtype mapping, and
training/evaluation usage boundaries before acquisition or model training.

## Relevant Product Docs

- `docs/product/datasets.md`
- `docs/product/dataset-sources.md`

## Acceptance Criteria

- Registry includes the three supplied positive datasets.
- Registry includes the three supplied negative/hard-negative datasets.
- Each entry records source URL, DOI when known, license, license status, role,
  binary label, subtype mapping, training use, evaluation use, and notes.
- Non-commercial, no-derivatives, and unspecified-license sources are not marked
  as commercial training allowed.
- Registry validation is executable with `python -m drone_sound.cli validate-dataset-registry`.
- License manifest export is executable with `python -m drone_sound.cli export-license-manifest`.
- Tests cover valid registry, duplicate ids, restricted-license guardrails, and
  license manifest export.

## Design Notes

- Commands:
  - `python -m drone_sound.cli validate-dataset-registry --registry configs/datasets.json`
  - `python -m drone_sound.cli export-license-manifest --registry configs/datasets.json --out data/manifests/license_manifest.csv`
- Queries: none.
- API: Python functions for dataset registry validation and CSV export.
- Tables: none.
- Domain rules: license status gates productization and commercial training.
- UI surfaces: none.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-002 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `python -m pytest tests/test_dataset_registry.py` |
| Integration | CLI validation and license manifest export tests |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | `python -m pytest` |

## Harness Delta

Story `US-002` is registered in the durable Harness database with verify command
`python -m pytest`.

## Evidence

- `. .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m drone_sound.cli validate-dataset-registry --registry configs/datasets.json` -> 6 entries validated.
- `. .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m drone_sound.cli export-license-manifest --registry configs/datasets.json --out data/manifests/license_manifest.csv` -> 6 license rows exported.
- `. .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m pytest` -> 12 passed.
