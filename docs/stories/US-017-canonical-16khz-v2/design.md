# US-017 Design

## Preprocessing Contract

`mono_resample_poly_16khz_v1` reads and downmixes each WAV once, resamples the
whole waveform to 16,000 Hz with a rational polyphase filter, normalizes the
result length deterministically, and only then applies existing manifest window
boundaries at 16 kHz. The identity path for 16-kHz mono input is lossless.

## Artifact Contract

Feature bundles, scaler metadata, model packages, and caches carry the
preprocessing profile and source feature-bundle fingerprint. Missing metadata
is interpreted as `legacy_native_rate_v1`. Consumers reject profile mismatch;
legacy artifacts remain readable only with other legacy artifacts.

All new artifacts are namespaced below `artifacts/v2_16khz/`. Final-run setup
fails if that root already exists, preventing accidental overwrite.

## Evaluation Contract

Model selection uses validation F2, then validation FPR and model size. Frozen
models and frozen validation-derived `threshold_f2` are used for public,
professor, and real-only evaluation. The real-only evaluator is inference-only
and reports confusion matrix, balanced accuracy, precision, recall, F1/F2,
ROC-AUC, PR-AUC, FPR/FNR, environment summaries, and per-recording FPR.

## Privacy and Statistical Limits

Raw private audio is never copied into artifacts. The real-only set has eleven
independent recordings, including only one positive recording, so it is an
external stress test rather than a deployment-performance estimate. Eight
questionable negative timestamps remain a review queue and are not excluded or
used for tuning.
