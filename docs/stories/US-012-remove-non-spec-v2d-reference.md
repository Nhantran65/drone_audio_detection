# US-012 Remove Non-Spec V2D Reference

## Status

implemented

## Lane

normal

## Product Contract

V2D must follow the SPEC-preferred pretrained audio embedding families:
YAMNet, EfficientAT, VGGish, or PANNs. Non-spec pretrained backbones must not be
introduced, documented as accepted V2D candidates, or included in comparison
artifacts without an explicit product/spec update first.

## Relevant Product Docs

- `docs/product/v2-exploration.md`
- `docs/product/modeling.md`
- `docs/product/preprocessing.md`
- `docs/product/overview.md`

## Acceptance Criteria

- Non-spec V2D code paths are removed.
- Non-spec V2D model names and embedding backends are removed from the V2
  contract.
- Comparison artifacts no longer include non-spec V2D rows.
- Non-spec V2D model/cache artifacts are deleted from local artifacts.
- Docs and stories state that V2D remains pending until a spec-compliant
  pretrained audio embedding backend is implemented.
- Future V2D work is explicitly noted as SPEC-first.

## Validation

When updating durable proof status, use numeric booleans:
`scripts/bin/harness-cli story update --id US-012 --unit 1 --integration 1 --e2e 0 --platform 0`.

| Layer | Expected proof |
| --- | --- |
| Unit | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest tests/test_v2_contract.py tests/test_v2_cli.py tests/test_v2_ml.py -q` |
| Integration | `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest -q` |
| E2E | Not applicable, no UI |
| Platform | Not applicable |

## Evidence

- Focused tests:
  `uv run --extra data --extra ml --extra dl --with pytest --python 3.13 python -m pytest tests/test_v2_contract.py tests/test_v2_cli.py tests/test_v2_ml.py -q`
  -> 17 passed.
- Cleanup grep over the known non-spec backend tokens in README, docs, source,
  tests, and comparison artifacts -> no matches.
- Deleted local non-spec V2D model and embedding-cache artifacts.
- Updated comparison artifacts:
  `artifacts/models/v2_full_comparison/comparison.json`,
  `artifacts/models/v2_full_comparison/comparison.csv`,
  `artifacts/models/v2_full_comparison/comparison_early_stopping.json`, and
  `artifacts/models/v2_full_comparison/comparison_early_stopping.csv`.

## SPEC Note

Next V2D work must start from the SPEC and use a SPEC-preferred pretrained
audio embedding family. If a different backend is desired, update product/spec
docs first and only then implement it.
