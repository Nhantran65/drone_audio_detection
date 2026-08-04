# US-018 Design

## Domain Model

Each rescue candidate is a named Tiny MLP training configuration. The sweep
records architecture, regularization, learning rate, positive weighting,
source balancing, feature-noise augmentation, seed, and early-stopping state.

## Application Flow

1. Load the existing canonical 16-kHz feature bundle.
2. Fit each candidate on public `train` rows only.
3. Select checkpoints and thresholds on public `val` rows only.
4. Rank candidates by validation F2, validation FPR, then serialized size.
5. Freeze the winner and evaluate it on private real-only data.

## Interface Contract

The existing `train-v2-mlp` defaults remain unchanged. Optional parameters
make candidate training reproducible. A separate rescue command writes a new,
fail-if-present output directory.

## Data Model

No dataset or schema changes. Rescue artifacts carry the canonical
preprocessing profile and source bundle fingerprint.

## UI / Platform Impact

CLI and local research artifacts only.

## Observability

Candidate, epoch, batch, elapsed-time, and completion progress are emitted via
the existing V2 progress logger.

## Alternatives Considered

1. Tune directly against private real-only recall. Rejected because it would
   turn the only external stress set into validation data.
2. Change features or resampling. Rejected because it would affect other tracks
   and confound the canonical comparison.
3. Add private positives to training. Rejected because only one recording
   exists and it is reserved for evaluation.
