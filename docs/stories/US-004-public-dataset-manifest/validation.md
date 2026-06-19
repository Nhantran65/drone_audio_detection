# Validation

## Proof Strategy

Synthetic local WAV fixtures prove that all six registered datasets are scanned,
split by original file, included regardless of license status, and can feed
`train-v1`.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | Split labels, subtype inference, missing dataset errors |
| Integration | Build public manifest from all registry entries |
| E2E | Generated manifest feeds `train-v1` and writes model/metrics JSON |
| Platform | Not applicable |
| Performance | Not applicable for v1 |
| Logs/Audit | Audit JSON includes license status and research override policy |

## Fixtures

Tests create deterministic synthetic WAV files for all six dataset IDs under a
temporary `raw/public` root.

## Commands

```text
.venv/bin/python -m pytest
scripts/bin/harness-cli story verify-all
```

## Acceptance Evidence

- `.venv/bin/python -m pytest` -> 18 passed.
- `scripts/bin/harness-cli story verify-all` -> 4 stories verified: 4 passed, 0 failed, 0 skipped.
