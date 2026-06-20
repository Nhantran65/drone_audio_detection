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

NASA Small UAS local `.mat` files are converted into derived mono 16 kHz WAV
files before they feed the public manifest builder. Conversion reads
`acoustics.incident_pascals`, infers source sample rate from
`acoustics.utc_time`, downmixes channels by mean, resamples to 16 kHz, and uses
batch-global peak scaling to write PCM int16 WAV files.

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

## V2.1 Feature Sets

V2.1 keeps `v1_basic` for compatibility and adds richer planned feature sets:

| Name | Contents | Used by |
| --- | --- | --- |
| `v1_basic` | mean, mean absolute amplitude, RMS, peak absolute amplitude, zero-crossing rate | V1 compatibility and ablation |
| `spectral_v1` | spectral centroid, bandwidth, rolloff, flatness, contrast, dominant frequency, low/mid/high band energy, band ratios | Engineered-feature models |
| `mfcc_v1` | 13 or 20 MFCC coefficients, mean/std, optional delta mean/std | Engineered-feature models |
| `spectral_mfcc_v1` | `v1_basic` + `spectral_v1` + `mfcc_v1` | V2A classical ML and V2B Tiny MLP |
| `logmel_v1` | 64 mel bins, 1-second window, about 100 frames | V2C Tiny CNN |
| `pretrained_embedding_v1` | YAMNet/EfficientAT/VGGish/PANNs embedding | V2D reference baseline |

Feature extraction must be deterministic. Scalers must be fit on train split
only, then reused unchanged for validation, public test, and professor test.
NaN and Inf values must be handled before model training or evaluation, and V2
feature runs should write an audit JSON with feature dimensions and scaler
metadata.

US-008 implements the engineered V2-M1 feature sets:

| Name | Feature count |
| --- | ---: |
| `v1_basic` | 5 |
| `spectral_v1` | 15 |
| `mfcc_v1` | 80 |
| `spectral_mfcc_v1` | 100 |

Spectral features use a full-window FFT with a Hann window. Band definitions:

- Low band: 20-500 Hz.
- Mid band: 500-2000 Hz.
- High band: 2000 Hz-Nyquist.

MFCC features use 25 ms frames, 10 ms hop, 40 mel filters, 20 coefficients,
DCT-II, and mean/std plus delta mean/std summaries across frames.

`extract-v2-features` writes:

- `.npz` feature bundle with raw and scaled features.
- Train-fitted scaler JSON.
- Feature audit JSON with row counts, dimensions, and non-finite replacement
  counts.
- Eval metadata arrays: `clip_id`, `label_subtype`, `window_start_sec`, and
  `window_end_sec`.
- Progress logs every 5,000 rows by default; override with `--progress-every N`
  or use `--progress-every 0` to disable row progress logs.
