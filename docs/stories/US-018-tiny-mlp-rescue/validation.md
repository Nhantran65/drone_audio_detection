# US-018 Validation

## Proof Strategy

Prove isolation, deterministic configuration, validation-only selection,
artifact compatibility, frozen-threshold evaluation, and preservation of all
pre-existing model files.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | Invalid architecture/weights/noise rejected; source weights deterministic |
| Integration | Rescue sweep writes new candidate packages and selects from validation fields only |
| E2E | Canonical bundle trains candidates with progress; winner evaluates frozen on 1,705 real-only windows |
| Platform | Existing V2 model loader/evaluator accepts the selected joblib package |
| Performance | Parameter count and serialized size remain lightweight |
| Logs/Audit | Config, bundle fingerprint, selection rule, and no-private-selection statement recorded |

## Fixtures

- Existing synthetic V2 feature fixture.
- `artifacts/v2_16khz/features/public_professor_features.npz`.
- `artifacts/v2_16khz/features/private_real_features.npz`.

## Commands

```text
uv run --extra data --extra ml --with pytest --python 3.13 \
  python -m pytest -q tests/test_v2_mlp_rescue.py tests/test_v2_ml.py tests/test_v2_cli.py

uv run --extra data --extra ml --python 3.13 \
  python -m drone_sound.cli sweep-v2-mlp-rescue \
  --features artifacts/v2_16khz/features/public_professor_features.npz \
  --out-dir artifacts/v2_16khz/models/tiny_mlp_rescue \
  --epochs 20 --batch-size 4096 --patience 4 --min-delta 0.0

uv run --extra data --extra ml --python 3.13 \
  python -m drone_sound.cli eval-real-v2 \
  --model artifacts/v2_16khz/models/tiny_mlp_rescue/models/feature_noise_0p02.joblib \
  --features artifacts/v2_16khz/features/private_real_features.npz \
  --audio-root data/raw/private \
  --cache-dir artifacts/v2_16khz/cache/tiny_mlp_rescue \
  --metrics-out artifacts/v2_16khz/evaluation/tiny_mlp_rescue/feature_noise_0p02_real_only.json
```

## Acceptance Evidence

- Targeted suite: 24 passed; full suite: 83 passed with five harmless
  scikit-learn batch-size warnings from tiny fixtures.
- Six real candidates completed with per-batch/per-epoch progress in 3m26s.
- Validation-only winner: `feature_noise_0p02`, validation F2 0.996372,
  validation FPR 0.006611, 7,521 parameters, 279.3 KB.
- Frozen winner public test: F2 0.994651 and FPR 0.006265.
- Frozen winner private real-only: precision 0.950640, recall 0.467206,
  F1 0.626506, F2 0.520104, balanced accuracy 0.710799, FPR 0.045608.
- Validation-defined `threshold_at_fpr_1` sensitivity on the frozen winner:
  recall 0.523810, F1 0.672822, F2 0.574724, and private FPR 0.062500. This is
  an exploratory operating point and does not replace primary `threshold_f2`.
- Existing canonical Tiny MLP model/metrics SHA-256 remained
  `cc29e7adc054d1ce0762c7bb532c38831ea037f9d2cf741348df91d648aa8a09`
  and `506d5cde3ae95d2ccb6524c2035d8e33717ace0abec7ad337a3443c08e029001`.
- The private set still has one independent positive recording, so the observed
  gain is a stress-test result rather than independent generalization proof.
