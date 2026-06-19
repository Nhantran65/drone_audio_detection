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

The US-004 public manifest builder audits local raw WAV files and does not
write processed audio copies. Until a preprocessing story adds resampling,
`sample_rate_processed` mirrors the audited source sample rate and sample-rate
or channel mismatches are visible in the audit JSON.

## Feature Extraction

Milestone 1 did not implement feature extraction.

The v1 training smoke baseline extracts simple dependency-free WAV window
features directly from manifest rows:

- mean amplitude
- mean absolute amplitude
- RMS amplitude
- peak absolute amplitude
- zero-crossing rate

Default feature plan:

- Primary: log-mel spectrogram.
- Ablation: MFCC statistics for classical baselines.
- Normalization must be fit on train split only.
