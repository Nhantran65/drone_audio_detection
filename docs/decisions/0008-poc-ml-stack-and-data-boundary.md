# 0008 POC ML Stack And Data Boundary

Date: 2026-06-19

## Status

Accepted

## Context

The POC needs an implementation direction before dataset tooling and validation
can become executable. The first available professor dataset is a private zip
with one WAV file and one summary text file. The professor data should support
real-world recall analysis, but it cannot estimate false alarms by itself.

## Decision

- Use Python for the initial data and research tooling.
- Use PyTorch as the planned training framework.
- Use ONNX as the first planned edge export format.
- Treat generic CPU edge hardware as the primary benchmark proxy.
- Treat edge GPU as optional and report it separately if available.
- Treat `fwdronedata.zip` as private professor data in `professor_test`.
- Redact exact professor-data GPS coordinates in generated audit reports by
  default.

## Alternatives Considered

1. TensorFlow plus TFLite first. Rejected for Milestone 1 because data contract
   and validation should stabilize before choosing a more deployment-specific
   runtime.
2. Framework-neutral indefinitely. Rejected because future stories need a clear
   training/export direction.
3. Use professor data for model selection. Rejected because it is positive-only
   or near-positive-only and would make false-alarm claims misleading.

## Consequences

Positive:

- The first implementation can stay lightweight and standard-library focused.
- Future model stories have a clear PyTorch and ONNX direction.
- Professor data is protected from training leakage and privacy exposure.

Tradeoffs:

- TFLite compatibility is deferred.
- Edge GPU behavior remains unknown until hardware exists.

## Follow-Up

- Add public dataset acquisition and license audit stories.
- Add training dependencies only when the first model story is selected.
- Revisit export runtime after the target hardware is confirmed.
