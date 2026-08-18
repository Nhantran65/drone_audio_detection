# Validation

- Unit and integration tests cover group parsing, zero overlap, row preservation,
  train-only scaler fitting, preprocessing mismatch rejection, and legacy compatibility.
- Split audit must contain both labels in every split and zero pairwise group overlap.
- Frozen evaluators must report `fit_called=false`, `threshold_tuned=false`, and
  `selection_performed=false`.
- Selection ranks validation F1 only; public/professor/private metrics are post-freeze.
- Finalization fails if any artifact existing before V3 changes or disappears.
