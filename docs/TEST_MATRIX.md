# Test Matrix

This file maps product behavior to proof.

Do not mark a row implemented until tests or validation evidence exist. The
durable proof matrix should be queried with `scripts/bin/harness-cli query
matrix` when the local ignored Harness CLI binary is installed.

## Status Values

| Status | Meaning |
| --- | --- |
| planned | Accepted as intended behavior, not implemented |
| in_progress | Actively being built |
| implemented | Implemented and proof exists |
| changed | Contract changed after earlier implementation |
| retired | No longer part of the product contract |

## Matrix

| Story | Contract | Unit | Integration | E2E | Platform | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US-001 | POC contract and professor data scaffold | yes | yes | no | no | implemented | `python -m pytest`; professor audit generated 1113 windows with redacted GPS |
| US-002 | Public dataset registry and license audit | yes | yes | no | no | implemented | `python -m pytest`; registry validates 6 public sources and exports license manifest |
| US-003 | V1 training smoke baseline | yes | yes | no | no | implemented | `python -m pytest`; `train-v1` writes model/metrics and rejects professor-only training data |
| US-004 | Public dataset manifest builder | yes | yes | no | no | implemented | `python -m pytest`; synthetic all-registry WAV manifest feeds `train-v1` |
| US-005 | NASA MAT derived WAV conversion | yes | yes | no | no | implemented | `uv run --extra data --with pytest --python 3.13 python -m pytest`; synthetic MAT conversion feeds manifest and `train-v1` |
| US-006 | Frozen V1 professor evaluation | yes | yes | no | no | implemented | `uv run --extra data --with pytest --python 3.13 python -m pytest`; synthetic and real public/professor manifests feed `eval-v1` |

## Evidence Rules

- Unit proof covers pure domain and application rules.
- Integration proof covers backend enforcement, data integrity, provider
  behavior, jobs, or service contracts.
- E2E proof covers user-visible browser flows.
- Platform proof covers only shell, deployment, mobile, desktop, or runtime
  behavior that cannot be proven in lower layers.
- A story can be implemented without every proof column if the story packet
  explains why.
