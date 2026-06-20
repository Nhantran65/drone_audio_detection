# Modeling Contract

## Default Stack

- Research and training stack: PyTorch.
- V2A classical ML stack: optional scikit-learn plus joblib under the `ml`
  dependency extra.
- Edge export direction: ONNX first.
- TensorFlow Lite may be considered later if target hardware requires it.

No ML framework dependency is installed in Milestone 1.

## Baselines

Planned baselines:

- V1 smoke baseline: dependency-free centroid classifier over simple WAV window
  features, used only to prove the training/evaluation loop before heavier ML
  dependencies are added.
- Classical MFCC baseline with Logistic Regression, Linear SVM, or Random
  Forest.
- Tiny depthwise-separable CNN over log-mel spectrograms.
- Optional teacher/reference model such as YAMNet or PANNs.

## V2.1 Model Tracks

V2.1 model exploration is organized into four tracks:

| Track | Candidate | Input | Deployability label |
| --- | --- | --- | --- |
| V2A | Logistic Regression, Linear SVM/SGD, Random Forest Small, ExtraTrees Small | `spectral_mfcc_v1` | `deployable_lightweight` or `maybe_deployable` |
| V2B | Tiny MLP, Dense 64 -> Dense 16 -> Sigmoid | `spectral_mfcc_v1` | `deployable_lightweight` if size and latency stay small |
| V2C | Tiny CNN with small convolutional blocks and global average pooling | `logmel_v1` | `deployable_lightweight` or `maybe_deployable` |
| V2D | YAMNet/EfficientAT/VGGish/PANNs embedding plus small head | `pretrained_embedding_v1` | `reference_only` unless the full inference path is lightweight |

Preferred final candidate:

```text
Best V2A/V2B/V2C model that improves professor recall and public F2 without
increasing public FPR.
```

V2D answers whether transfer learning helps, but it does not automatically
become the reportable lightweight model if the full backbone is too heavy.

## V2.1 Stop Criteria

Stop and report if any deployable lightweight V2A/V2B/V2C candidate achieves:

```text
public_test_f2 >= 0.88
professor_recall >= 0.85
longest_professor_missed_segment_sec <= 8.0
public_fpr <= 0.168729
model remains lightweight
```

If no candidate reaches all targets, still report if one candidate clearly
improves at least two of professor recall, longest missed professor segment,
public F2, public FPR, or model complexity.

## Edge Targets

Preferred targets:

- Parameter count below 100k.
- INT8 model below 500 KB when quantization is implemented.
- Real-time factor below 1.0 on generic CPU proxy.

Edge GPU is an optional benchmark lane and must be reported separately.
