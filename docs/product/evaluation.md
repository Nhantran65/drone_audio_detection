# Evaluation Contract

## Public Test Evaluation

The public merged test set is the main binary benchmark. It must include both
`drone` and `no_drone` samples.

Metrics:

- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- F1.
- F2.
- ROC-AUC.
- PR-AUC.
- False positive rate.
- False negative rate.
- Confusion matrix.

## Hard-Negative Evaluation

Report false positive rate by negative subtype:

- `no_drone_aircraft`
- `no_drone_traffic`
- `no_drone_urban_industrial`
- `no_drone_speech`
- `no_drone_background`

V1 may omit `no_drone_speech` if no speech-only negative source has been
selected yet. In that case the metrics report must state that speech-only
false-positive coverage is unavailable rather than treating it as zero.

## Professor Test Evaluation

Professor data is evaluated only after the model and threshold are frozen.

Allowed metrics:

- Recall.
- Missed window count.
- Longest missed segment.
- Mean `P(drone)`.
- Median `P(drone)`.
- Prediction stability over time.

Do not use professor data alone to report false positive rate, specificity,
accuracy, or precision unless real-world no-drone professor data is added.

## Combined Supplementary Evaluation

Combined public-test plus professor-positive metrics are supplementary only.
They may be reported for communication, but they do not replace the public
held-out test benchmark, and they must not be used for threshold tuning or model
selection.

The combined supplementary block reports the same binary metrics as public
test, using public-test rows plus professor rows and the frozen stored
threshold.

## Private Real-Only External Test

US-017 adds one professor-positive recording (1,113 windows) and ten asserted
private-negative recordings (592 windows) as an external stress test. All rows
use `mono_resample_poly_16khz_v1`; the model and `threshold_f2` must already be
frozen. This set cannot be used for fitting, threshold tuning, early stopping,
or model selection.

`eval-real-v2` auto-detects V2A/V2B joblib packages, V2C PyTorch checkpoints,
and V2D YAMNet heads. It reports confusion matrix, balanced accuracy,
precision, recall, F1/F2, ROC-AUC, PR-AUC, FPR/FNR, per-environment metrics,
and per-negative-recording FPR. Accuracy alone is not a conclusion. Because
there is only one independent positive recording and ten negative recordings,
the result is not a deployment-performance estimate.

Repeated post-freeze rescue diagnostics on this same set are explicitly
exploratory. They may describe operating-point sensitivity, but they cannot be
used to claim an independently validated model improvement or to select a
production checkpoint.

## Threshold Rule

Thresholds must be selected with validation data only. Professor data must not
be used for threshold selection.

For v1, `public_test` is a fixed split produced by the public manifest builder.
It must not be used for threshold selection or k-fold training.

For canonical V2, validation F2 is the only primary selection metric; ties use
lower validation FPR and then smaller model size. Public test, professor test,
and private real-only data are excluded from selection.

Every V2 candidate should report these validation-selected thresholds:

```text
threshold_f1
threshold_f2
threshold_at_fpr_10
threshold_at_fpr_5
threshold_at_fpr_2
threshold_at_fpr_1
```

The primary report threshold is `threshold_f2`. Recall at fixed FPR levels is
the secondary product-oriented view.

## V1 Frozen Evaluation Tool

`eval-v1` evaluates an already-trained V1 model without retraining or changing
the stored threshold:

```text
python -m drone_sound.cli eval-v1 \
  --model artifacts/models/v1_baseline_model.json \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --metrics-out artifacts/models/v1_eval_metrics.json
```

Professor audio may be extracted locally to `data/raw/private/` for this
evaluation. That folder remains private ignored data and is not a training
source.

## V2.1 Efficiency Evaluation

For each V2 candidate, report:

- Feature count.
- Model type.
- Model file size.
- Parameter count when applicable.
- Average inference time per 1-second window.
- Runtime dependency.
- Deployability note.

For pretrained embedding baselines, separately report the embedding extractor
size, classifier head size, and whether the full backbone is required at
inference time.

## V2 Frozen Evaluation Tool

`eval-v2` evaluates an already-trained V2A/V2B joblib model package from a V2
feature bundle without retraining or changing the stored `threshold_f2`:

```text
python -m drone_sound.cli eval-v2 \
  --model artifacts/models/v2a_sweep/models/logistic_regression.joblib \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --metrics-out artifacts/models/v2a_sweep/logistic_regression_eval.json \
  --export-errors artifacts/models/v2a_sweep/logistic_regression_errors
```

The feature bundle must include public test and professor rows plus eval
metadata. Professor rows remain positive-only evaluation data.

`eval-v2-cnn` does the same for frozen V2C `.pt` checkpoints using the reusable
log-mel cache. `eval-v2-embedding` does the same for frozen V2D YAMNet joblib
heads using the reusable pretrained embedding cache. Neither command retrains
the model or retunes thresholds.

`eval-v2-cnn` evaluates an already-trained V2C Tiny CNN checkpoint from the same
feature bundle and log-mel cache:

```text
python -m drone_sound.cli eval-v2-cnn \
  --model artifacts/models/v2c_cnn/tiny_cnn_es.pt \
  --features artifacts/features/v2_spectral_mfcc_features.npz \
  --public-audio-root data/raw/public \
  --professor-audio-root data/raw/private/professor_real_world \
  --cache-dir artifacts/features/v2_logmel_cache \
  --metrics-out artifacts/models/v2_combined_supplementary/tiny_cnn_es_combined_eval.json
```

Both frozen V2 evaluators write a `combined_supplementary` block for
public-test plus professor-positive metrics.

## V2.1 Comparison Output

The final V2 comparison table should include:

```text
model
track
feature/input
public_f2
public_f1
public_precision
public_recall
public_fpr
professor_recall
longest_missed_segment
model_size
params
deployability_label
```
