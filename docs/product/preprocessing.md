# Preprocessing Contract

## Audio Standardization

Audio entering the model pipeline is standardized to:

- Sample rate: 16 kHz.
- Channels: mono.
- Window length: 1.0 second.
- Stride: 0.5 second.

Milestone 1 only audits source files and writes manifests. It does not write
processed audio copies.

## Windowing

Complete windows are counted with:

```text
floor((duration_sec - window_length_sec) / stride_sec) + 1
```

Durations shorter than the window length produce zero complete windows.

For each window row:

- `window_start_sec` is the zero-based start time.
- `window_end_sec` is `window_start_sec + 1.0`.
- `window_id` is unique and deterministic.

## Feature Extraction

Feature extraction is planned but not implemented in Milestone 1.

Default feature plan:

- Primary: log-mel spectrogram.
- Ablation: MFCC statistics for classical baselines.
- Normalization must be fit on train split only.
