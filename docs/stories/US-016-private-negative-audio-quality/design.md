# US-016 Design

## Domain Model

The unit of independence is an original recording, not an overlapping window.
Each recording has an asserted environment label (`traffic` or `forest`) and a
binary label (`no_drone`). One-second windows with 0.5-second stride are used
only for signal profiling and frozen-model evaluation.

## Application Flow

1. Read raw WAV files without altering them.
2. Validate metadata, hashes, duration, signal ranges, clipping, silence, and
   discontinuity proxies.
3. Extract the existing 100-feature `spectral_mfcc_v1` representation at the
   native sample rate and after in-memory resampling to 16 kHz.
4. Apply the existing train-fitted scaler and frozen Tiny MLP/Random Forest
   models without retraining or threshold tuning.
5. Compare the new negatives with public train/test classes and professor
   positive data, then emit bounded CSV/JSON evidence, a notebook, and a
   technical report.

## Interface Contract

The analysis command takes explicit private-negative, professor-positive,
feature-bundle, scaler, and frozen-model paths. It writes only derived evidence
under `artifacts/analysis/private_negative_quality/`.

## Data Model

No database or raw-data migration. Derived records use file-level and
window-level CSV/JSON outputs. Paths are repository-relative and exact location
metadata is never emitted.

## UI / Platform Impact

None. This is offline research tooling and a portable HTML report.

## Observability

The analysis prints current phase, file/window counts, elapsed time, rate, and
ETA for long-running extraction/scoring.

## Alternatives Considered

1. Listen to a few files and decide qualitatively. Rejected because it is not
   reproducible and cannot quantify model/domain risk.
2. Add all negative windows directly to training. Rejected because only ten
   independent recordings exist and 48 kHz sample rate may leak the label.
3. Aggressively denoise before analysis. Rejected because environmental noise
   is the target hard-negative content and denoising can destroy field realism.
