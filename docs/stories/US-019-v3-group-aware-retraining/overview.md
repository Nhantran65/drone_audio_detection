# US-019: Group-aware V3 retraining

Rebuild train, validation, and public-test assignments at correlated recording-session
group granularity, then retrain every model family at canonical 16 kHz. Professor and
private-real audio remain external evaluation-only data.

Acceptance requires zero group overlap, train-only scaler fitting, validation-only
threshold/model selection, frozen evaluation, per-dataset diagnostics, and unchanged
SHA-256 inventory for all pre-V3 artifacts.
