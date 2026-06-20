# Agent Instructions

## Long-Running Process Visibility

Any dataset, preprocessing, feature extraction, training, sweep, evaluation, or
benchmark command that may take more than a few seconds must expose progress.
Use `tqdm` or explicit logs that show enough runtime context to tell whether the
job is healthy:

- elapsed time
- processed count and total count when known
- throughput/rate when useful
- ETA when useful
- current phase or current file/model when useful

Do not leave long-running data or ML jobs silent. If an existing command is
silent, add progress reporting before relying on it for real local runs.

<!-- HARNESS:BEGIN -->
## Harness

This repo uses Harness. Before work, read:

- `README.md`
- `docs/HARNESS.md`
- `docs/FEATURE_INTAKE.md`
- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_RULES.md`
- `docs/TOOL_REGISTRY.md`
- `scripts/bin/harness-cli query matrix` on macOS/Linux, or `.\scripts\bin\harness-cli.exe query matrix` on Windows

Use the Rust Harness CLI at `scripts/bin/harness-cli` on macOS/Linux or
`scripts/bin/harness-cli.exe` on Windows as the main operational tool. Before a
step that could use an external tool, run `scripts/bin/harness-cli query tools
--capability <name> --status present` to see what is equipped; an absent
capability is a clean skip.
<!-- HARNESS:END -->
