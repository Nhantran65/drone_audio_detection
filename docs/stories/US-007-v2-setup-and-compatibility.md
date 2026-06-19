# US-007 V2 Setup And Compatibility

## Status

implemented

## Lane

normal

## Product Contract

V2-M0 adds the V2 experiment contract surface without changing V1 behavior.
The repo exposes canonical V2 feature/model/threshold/deployability names,
validates V2 CLI inputs, and provides command skeletons for later V2 work. The
skeletons do not train, evaluate, or write placeholder artifacts yet.

## Relevant Product Docs

- `docs/product/v2-exploration.md`
- `docs/product/preprocessing.md`
- `docs/product/modeling.md`
- `docs/product/evaluation.md`

## Acceptance Criteria

- V2 feature set, model type, track, deployability, and threshold names are
  centralized in code.
- V2 validation rejects unknown feature sets, model types, and deployability
  labels.
- `sweep-v2-ml`, `train-v2-mlp`, `train-v2-cnn`, `train-v2-embedding`, and
  `eval-v2` parse the documented arguments.
- V2 skeleton commands validate manifest/audio-root or evaluation input paths.
- Valid V2 skeleton commands exit cleanly with `v2 not implemented: ...`.
- V2 skeleton commands do not write model or metrics artifacts.
- V1 `train-v1` and `eval-v1` behavior remains unchanged.

## Design Notes

- Commands:
  - `python -m drone_sound.cli sweep-v2-ml ...`
  - `python -m drone_sound.cli train-v2-mlp ...`
  - `python -m drone_sound.cli train-v2-cnn ...`
  - `python -m drone_sound.cli train-v2-embedding ...`
  - `python -m drone_sound.cli eval-v2 ...`
- Queries: none.
- API: `drone_sound.v2` contract constants and validation helpers.
- Tables: none.
- Domain rules:
  - Professor data remains evaluation-only.
  - V2-M0 intentionally stops before feature extraction or model training.
- UI surfaces: none.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-007 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `python -m pytest tests/test_v2_contract.py` |
| Integration | `python -m pytest tests/test_v2_cli.py` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |
| Release | `uv run --extra data --with pytest --python 3.13 python -m pytest -q` |

## Harness Delta

Story `US-007` should be registered in the durable Harness database with verify
command `uv run --extra data --with pytest --python 3.13 python -m pytest -q`.

## Evidence

- `uv run --extra data --with pytest --python 3.13 python -m pytest tests/test_v2_contract.py tests/test_v2_cli.py -q` -> 10 passed.
- `uv run --extra data --with pytest --python 3.13 python -m pytest -q` -> 43 passed.
