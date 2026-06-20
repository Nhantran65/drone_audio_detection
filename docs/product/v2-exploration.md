# V2.1 ML/DL Exploration Contract

## Objective

V2.1 explores stronger lightweight candidates beyond the V1 smoke baseline.
The task remains binary window-level classification:

```text
drone / no_drone
```

Professor real-world data remains evaluation-only. It must not be used for
training, validation, threshold tuning, or model selection.

## V1 Reference

Current V1 smoke baseline:

```text
public_test_f2: 0.871481
public_fpr: 0.168729
professor_recall: 0.752920
longest_professor_missed_segment: 16.0s
```

V1 validates the end-to-end loop, but it uses simple time-domain features and
is not the final model candidate.

## Experiment Tracks

| Track | Input | Model family | Role |
| --- | --- | --- | --- |
| V2A | `spectral_mfcc_v1` | Logistic Regression, Linear SVM/SGD, Random Forest Small, ExtraTrees Small | Main classical ML baseline |
| V2B | `spectral_mfcc_v1` | Tiny MLP | Lightest neural candidate on engineered features |
| V2C | `logmel_v1` | Tiny CNN | Main lightweight DL candidate |
| V2D | `pretrained_embedding_v1` | YAMNet/EfficientAT/VGGish/PANNs embedding plus small head | Reference or upper-bound baseline |

V2D is reported separately unless the full embedding extractor plus classifier
meets the lightweight deployment target.

Canonical V2 model type names:

```text
logistic_regression
linear_svm
random_forest
extra_trees
tiny_mlp
tiny_cnn
yamnet_head
```

## Feature Sets

| Name | Contents | Purpose |
| --- | --- | --- |
| `v1_basic` | mean, mean absolute amplitude, RMS, peak absolute amplitude, zero-crossing rate | V1 compatibility baseline |
| `spectral_v1` | spectral centroid, bandwidth, rolloff, flatness, contrast, dominant frequency, low/mid/high band energy, band ratios | Frequency distribution and band-energy cues |
| `mfcc_v1` | 13 or 20 MFCC coefficients, mean/std, optional delta mean/std | Compact timbre representation |
| `spectral_mfcc_v1` | `v1_basic` + `spectral_v1` + `mfcc_v1` | Main V2A/V2B engineered feature set |
| `logmel_v1` | 64 mel bins, 1-second window, about 100 frames | V2C Tiny CNN input |
| `pretrained_embedding_v1` | YAMNet/EfficientAT/VGGish/PANNs embedding | V2D transfer-learning reference |

V2D must use the spec-preferred pretrained audio embedding families above. Do
not substitute unrelated pretrained backbones without an explicit spec update.
US-013 implements YAMNet first. For each 1-second window, the command
standardizes audio to 16 kHz mono, runs the TensorFlow Hub YAMNet extractor,
mean-pools the 1024-dimensional frame embeddings, caches them as
`pretrained_embedding_v1`, and trains a small Logistic Regression head.

Implemented V2-M1 engineered dimensions:

```text
v1_basic: 5
spectral_v1: 15
mfcc_v1: 80
spectral_mfcc_v1: 100
```

The `extract-v2-features` command writes a compressed `.npz` feature bundle, a
train-fitted scaler JSON, and a feature audit JSON. Scaler fitting uses public
`train` rows only; `val`, `public_test`, and professor rows use that frozen
scaler. The bundle also carries evaluation metadata: `clip_id`,
`label_subtype`, `window_start_sec`, and `window_end_sec`.

Full-dataset extraction prints progress every 5,000 rows by default. Use
`--progress-every N` to change the interval or `--progress-every 0` to suppress
row progress logs.

## V2-M2 Classical ML Sweep

V2-M2 implements `sweep-v2-ml` for V2A candidates on `spectral_mfcc_v1`.
The command reads the V2 feature bundle directly and writes joblib model
artifacts, per-model metrics, a comparison JSON/CSV, and `best_model.json`.
Model selection uses validation F2 at `threshold_f2`, with lower validation FPR
and smaller model size as tie-breakers.

`eval-v2` loads one frozen V2A model package, reuses the stored `threshold_f2`,
and evaluates public test and professor rows from the same feature bundle
without threshold retuning.

## Selection Rule

Primary model-selection metric:

```text
validation F2
```

Guardrails:

- Validation precision, F1, and FPR.
- Public test FPR.
- Professor recall.
- Longest missed professor segment.
- Model size and inference latency.

A candidate should not be selected if it improves F2 only by significantly
increasing false positives.

## Thresholds

Thresholds are tuned on validation data only. Report:

```text
threshold_f1
threshold_f2
threshold_at_fpr_10
threshold_at_fpr_5
threshold_at_fpr_2
threshold_at_fpr_1
```

The primary reportable threshold is `threshold_f2`. Also report recall at fixed
FPR levels as the product-oriented view.

## Evaluation

Public test remains the main binary benchmark and reports the metrics defined
in `docs/product/evaluation.md`.

Professor positive-only evaluation reports:

- Professor recall.
- Missed window count.
- Longest missed segment.
- Mean and median `P(drone)`.
- Score standard deviation.
- Decision flip count and rate.
- Score quantiles.

Combined public-test plus professor-positive metrics are supplementary only.
For V2, `eval-v2`, `eval-v2-cnn`, and `eval-v2-embedding` write a
`combined_supplementary` block from frozen public/professor scores without
retraining or threshold retuning.

Efficiency reporting for every candidate:

- Feature count.
- Model type.
- Model file size.
- Parameter count when applicable.
- Average inference time per 1-second window.
- Runtime dependency.
- Deployability note.

## V2B/V2C Early Stopping

`train-v2-mlp` and `train-v2-cnn` support validation-based early stopping with
`--patience` and `--min-delta`. Both commands keep the best validation
checkpoint by `threshold_f2`, with validation FPR as the near-tie breaker, and
write `learning_curve` plus `early_stopping` blocks into the metrics JSON.

Current full-data rerun artifacts:

- `artifacts/models/v2b_mlp/tiny_mlp_es_metrics.json`: 20/20 epochs, best epoch
  19, public F2 0.997475, public FPR 0.003234, professor recall 0.791554,
  longest missed professor segment 6.5s.
- `artifacts/models/v2c_cnn/tiny_cnn_es_metrics.json`: 20/20 epochs, best epoch
  18, public F2 0.966515, public FPR 0.039830, professor recall 0.554358,
  longest missed professor segment 19.0s.

Neither run stopped before the 20-epoch cap, so the curves are near plateau but
not flat enough to trigger early stopping under `min_delta=0.0005`.

## V2D YAMNet Embedding Baseline

`train-v2-embedding` implements the first spec-compliant V2D backend with:

```text
--embedding-model yamnet
--head logistic_regression
```

The command reads row metadata from the existing V2 feature bundle, caches
YAMNet embeddings under the requested cache directory, trains only on public
`train` rows, tunes thresholds only on public `val` rows, and evaluates public
test plus positive-only professor rows with the frozen `threshold_f2`.

YAMNet V2D metrics use `feature_set=pretrained_embedding_v1` and
`model_type=yamnet_head`. The deployability label is `reference_only` unless a
later deployment story proves the full pretrained extractor plus head meets
the lightweight target.

Current full-data YAMNet artifact:

- `artifacts/models/v2d_yamnet/yamnet_head_metrics.json`: public F2 0.974435,
  public FPR 0.041888, professor recall 0.619946, longest missed professor
  segment 12.0s, deployability `reference_only`.

## Stop Criteria

Stop and report if any deployable lightweight candidate from V2A, V2B, or V2C
achieves:

```text
public_test_f2 >= 0.88
professor_recall >= 0.85
longest_professor_missed_segment_sec <= 8.0
public_fpr <= 0.168729
model remains lightweight
```

If no candidate reaches all targets, report if one candidate clearly improves
at least two of:

- Professor recall.
- Longest missed professor segment.
- Public F2.
- Public FPR.
- Model complexity.

If V2D wins but is not lightweight deployable, label it `reference_only` and
select the best V2A/V2B/V2C candidate as the main lightweight candidate.

## Final Report Columns

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

Allowed deployability labels:

```text
deployable_lightweight
maybe_deployable
reference_only
```
