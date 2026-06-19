# Harness Backlog

Use this file when an agent discovers a missing harness capability but should
not change the operating model immediately.

## Template

```md
## Missing Harness Capability

### Title

Short name.

### Discovered While

Task or story that exposed the gap.

### Current Pain

What was hard, repeated, ambiguous, or unsafe?

### Suggested Improvement

What should be added or changed?

### Risk

Tiny, normal, or high-risk.

CLI value: `--risk tiny`, `--risk normal`, or `--risk high-risk`.

### Status

proposed | accepted | implemented | rejected
```

## Items

## Missing Harness Capability

### Title

Restore local Harness CLI binary.

### Discovered While

Reconciling POC progress and generated artifacts after US-004.

### Current Pain

`scripts/bin/harness-cli` is ignored by git and is absent in this checkout, so
the agent cannot record intake, query the durable matrix, run story
verification, or record a trace through the required Harness CLI.

### Suggested Improvement

Restore the prebuilt binary with the Harness installer or document a local
bootstrap command that installs `scripts/bin/harness-cli` for this project.

### Risk

Tiny.

CLI value: `--risk tiny`.

### Status

proposed
