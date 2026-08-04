# US-018 Exec Plan

## Goal

Test whether regularization, source balancing, modest capacity changes, or
feature-noise augmentation improve the canonical Tiny MLP without touching any
other model or existing artifact.

## Scope

In scope:

- Parameterize Tiny MLP training while preserving current defaults.
- Add an isolated rescue sweep and deterministic candidate registry.
- Train/select on public train/validation only.
- Evaluate only the frozen selected candidate on private real-only data.

Out of scope:

- New recordings, feature definitions, resampling profiles, CNN/YAMNet/tree retraining.
- Claims that one private positive recording estimates deployment performance.

## Risk Classification

Risk flags:

- Existing behavior: training code is parameterized, with regression coverage
  ensuring defaults remain unchanged.
- Weak proof: private positive diversity is unavailable.

Hard gates:

- No private/professor leakage into training or selection.
- No overwrite of existing artifacts.

## Work Phases

1. Inspect current V2B contract and canonical evidence.
2. Add candidate configuration and isolated sweep.
3. Add leakage/compatibility/default-regression tests.
4. Run bounded real training with progress.
5. Freeze and evaluate the validation-selected winner.
6. Record limitations and Harness evidence.

## Stop Conditions

Pause if existing artifacts would be overwritten, bundle compatibility fails,
or any candidate-selection path reads professor/private metrics.
