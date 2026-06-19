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

## Threshold Rule

Thresholds must be selected with validation data only. Professor data must not
be used for threshold selection.

For v1, `public_test` is a fixed split produced by the public manifest builder.
It must not be used for threshold selection or k-fold training.

For V2.1, validation F2 remains the main selection metric, with FPR, precision,
professor recall, longest missed professor segment, model size, and inference
latency as guardrails. A candidate should not be selected if it improves F2 only
by significantly increasing false positives.

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
