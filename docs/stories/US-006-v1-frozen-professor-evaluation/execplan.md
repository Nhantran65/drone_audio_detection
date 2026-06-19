# Execution Plan

## Steps

1. Add a frozen V1 evaluation module.
2. Add `eval-v1` CLI arguments and error handling.
3. Add tests for model validation, scoring, professor-only metrics, and CLI
   integration.
4. Document professor extraction and frozen evaluation commands.
5. Validate with full pytest and a real local eval run.

## Safety Rules

- Do not alter `v1_baseline_model.json`.
- Do not use professor rows for train, validation, or threshold tuning.
- Keep extracted professor audio under ignored private data paths.
