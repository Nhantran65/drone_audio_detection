# US-018 Canonical Tiny MLP Rescue

## Current Behavior

The canonical 16-kHz Tiny MLP has excellent public validation/test metrics but
only 42.59% recall on the one-recording private positive stress test. The
existing frozen model and every other V2 artifact must remain unchanged.

## Target Behavior

Run a bounded, reproducible Tiny MLP-only sweep using the existing canonical
feature bundle. Candidate selection uses public validation data only. The
selected frozen candidate is then reported on public test, professor-positive,
and private real-only data without fitting or threshold tuning.

## Affected Users

- Researchers comparing lightweight V2B candidates.

## Affected Product Docs

- `docs/product/modeling.md`
- `docs/product/evaluation.md`

## Non-Goals

- No changes to preprocessing, feature extraction, scalers, splits, or other models.
- No use of professor or private-real rows for fitting, early stopping,
  threshold selection, or candidate selection.
- No overwrite of existing canonical or legacy artifacts.
