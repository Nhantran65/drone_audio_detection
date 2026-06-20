# 0014 V2D YAMNet TensorFlow Embedding

Date: 2026-06-20

## Status

Accepted

## Context

V2D must use a spec-preferred pretrained audio embedding family. The previous
non-spec reference was removed, leaving the `train-v2-embedding` command as an
intentional pending path. The product backlog recommends YAMNet first.

## Decision

- Implement V2D first with YAMNet from TensorFlow Hub.
- Add a dedicated optional `embedding` dependency extra containing
  `tensorflow[and-cuda]` and TensorFlow Hub.
- Keep V2C dependencies under `dl`; do not require TensorFlow for V2A, V2B, or
  V2C.
- Train a scikit-learn Logistic Regression head on cached YAMNet embeddings.
- Label V2D output `reference_only` until the full extractor deployment path is
  proven lightweight.

## Alternatives Considered

1. Use PANNs first. Rejected because the accepted backlog recommends YAMNet as
   the first V2D baseline.
2. Put TensorFlow in the existing `dl` extra. Rejected because V2C only needs
   PyTorch/torchaudio and should not pull TensorFlow.
3. Keep V2D pending. Rejected because US-013 explicitly selects the
   spec-compliant YAMNet backend.

## Consequences

Positive:

- V2D now uses a spec-compliant pretrained embedding family.
- TensorFlow installation remains opt-in by capability.
- Cached embeddings make repeated head training and evaluation cheaper.

Tradeoffs:

- First-run YAMNet extraction may be slow and writes a large cache.
- The V2D artifact is not automatically a lightweight deployable candidate.
- TensorFlow Hub model download requires network access on first use.
- GPU use depends on selecting an available local GPU, for example with
  `CUDA_VISIBLE_DEVICES`.

## Follow-Up

- Add deployability proof only after a target hardware/export story exists.
- Consider EfficientAT, VGGish, or PANNs only after YAMNet evidence is recorded.
