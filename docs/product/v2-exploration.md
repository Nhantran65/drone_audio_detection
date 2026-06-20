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

Efficiency reporting for every candidate:

- Feature count.
- Model type.
- Model file size.
- Parameter count when applicable.
- Average inference time per 1-second window.
- Runtime dependency.
- Deployability note.

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
