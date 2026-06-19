# Modeling Contract

## Default Stack

- Research and training stack: PyTorch.
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

## Edge Targets

Preferred targets:

- Parameter count below 100k.
- INT8 model below 500 KB when quantization is implemented.
- Real-time factor below 1.0 on generic CPU proxy.

Edge GPU is an optional benchmark lane and must be reported separately.
