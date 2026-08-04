# Modeling Contract

## Default Stack

- Research and training stack: PyTorch.
- V2A classical ML stack: optional scikit-learn plus joblib under the `ml`
  dependency extra.
- V2C stack: optional PyTorch plus torchaudio under the `dl` dependency extra.
- V2D YAMNet stack: optional TensorFlow plus TensorFlow Hub under the
  `embedding` dependency extra, with scikit-learn/joblib under `ml` for the
  classifier head.
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
V2D must follow the spec-preferred pretrained audio embedding families:
YAMNet, EfficientAT, VGGish, or PANNs.
US-013 implements YAMNet first by mean-pooling the 1024-dimensional pretrained
embedding over each 1-second window and training a Logistic Regression head.

V2B and V2C training keeps the best validation checkpoint and can stop early
with `patience`/`min_delta`. Metrics JSON stores the epoch-level learning curve
so later model selection can distinguish true improvement from plateau noise.

All canonical V2 model packages store the preprocessing profile, processed
sample rate, and training-bundle fingerprint. A model can evaluate a different
bundle only when feature names and preprocessing contract match. Model
selection is validation-only: highest validation F2, then lower validation FPR,
then smaller serialized model size. Public, professor, and private real-only
results are post-freeze reports and never selection inputs.

US-018 adds an optional, isolated Tiny MLP rescue sweep over positive weighting,
public-source balancing, mild feature-noise regularization, and a modest width
increase. It preserves the current Tiny MLP defaults and all existing artifacts.
The rescue winner follows the same validation-only selection rule; stress-test
results cannot override that winner.

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
