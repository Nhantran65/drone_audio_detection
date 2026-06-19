"""V2 experiment contract and CLI validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

FEATURE_SETS = (
    "v1_basic",
    "spectral_v1",
    "mfcc_v1",
    "spectral_mfcc_v1",
    "logmel_v1",
    "pretrained_embedding_v1",
)

ENGINEERED_FEATURE_SETS = (
    "v1_basic",
    "spectral_v1",
    "mfcc_v1",
    "spectral_mfcc_v1",
)

MODEL_TYPES = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "tiny_mlp",
    "tiny_cnn",
    "yamnet_head",
)

V2A_MODEL_TYPES = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "extra_trees",
)

PRETRAINED_EMBEDDING_MODELS = ("yamnet",)
EMBEDDING_HEAD_MODEL_TYPES = ("logistic_regression",)

TRACKS = ("v2a", "v2b", "v2c", "v2d")

DEPLOYABILITY_LABELS = (
    "deployable_lightweight",
    "maybe_deployable",
    "reference_only",
)

THRESHOLD_NAMES = (
    "threshold_f1",
    "threshold_f2",
    "threshold_at_fpr_10",
    "threshold_at_fpr_5",
    "threshold_at_fpr_2",
    "threshold_at_fpr_1",
)


class V2InputError(ValueError):
    """Raised when V2 CLI input is invalid before execution."""


class V2NotImplementedError(RuntimeError):
    """Raised by V2 skeleton commands that are intentionally not implemented yet."""

    def __init__(self, capability: str) -> None:
        super().__init__(f"v2 not implemented: {capability}")


def validate_feature_set(value: str) -> str:
    """Validate and return a V2 feature-set name."""

    return _validate_choice(value, FEATURE_SETS, "feature set")


def validate_model_type(value: str) -> str:
    """Validate and return a V2 model-type name."""

    return _validate_choice(value, MODEL_TYPES, "model type")


def validate_model_types(values: Iterable[str]) -> tuple[str, ...]:
    """Validate and return multiple V2 model-type names."""

    return tuple(validate_model_type(value) for value in values)


def validate_deployability_label(value: str) -> str:
    """Validate and return a V2 deployability label."""

    return _validate_choice(value, DEPLOYABILITY_LABELS, "deployability label")


def validate_existing_file(path: str | Path, field: str) -> Path:
    """Require an existing file path for V2 skeleton commands."""

    resolved = Path(path)
    if not resolved.exists():
        raise V2InputError(f"{field} does not exist: {resolved}")
    if not resolved.is_file():
        raise V2InputError(f"{field} is not a file: {resolved}")
    return resolved


def validate_existing_directory(path: str | Path, field: str) -> Path:
    """Require an existing directory path for V2 skeleton commands."""

    resolved = Path(path)
    if not resolved.exists():
        raise V2InputError(f"{field} does not exist: {resolved}")
    if not resolved.is_dir():
        raise V2InputError(f"{field} is not a directory: {resolved}")
    return resolved


def validate_manifest_and_audio_root(manifest: str | Path, audio_root: str | Path) -> None:
    """Validate common manifest/audio-root inputs for V2 train commands."""

    validate_existing_file(manifest, "manifest")
    validate_existing_directory(audio_root, "audio root")


def validate_v2_evaluation_inputs(
    model: str | Path,
    public_manifest: str | Path,
    public_audio_root: str | Path,
    professor_manifest: str | Path,
    professor_audio_root: str | Path,
) -> None:
    """Validate common inputs for the V2 evaluation skeleton."""

    validate_existing_file(model, "model")
    validate_existing_file(public_manifest, "public manifest")
    validate_existing_directory(public_audio_root, "public audio root")
    validate_existing_file(professor_manifest, "professor manifest")
    validate_existing_directory(professor_audio_root, "professor audio root")


def _validate_choice(value: str, allowed: tuple[str, ...], field: str) -> str:
    if value not in allowed:
        joined = ", ".join(allowed)
        raise V2InputError(f"unknown {field}: {value}; expected one of: {joined}")
    return value
