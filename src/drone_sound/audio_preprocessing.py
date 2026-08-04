"""Canonical audio preprocessing and provenance shared by all V2 tracks."""

from __future__ import annotations

import math
from typing import Any

from drone_sound.constants import TARGET_SAMPLE_RATE_HZ

CANONICAL_PREPROCESSING_PROFILE = "mono_resample_poly_16khz_v1"
LEGACY_PREPROCESSING_PROFILE = "legacy_native_rate_v1"
FEATURE_BUNDLE_SCHEMA_VERSION = "v2_feature_bundle_v2"


class AudioPreprocessingError(ValueError):
    """Raised when canonical preprocessing cannot be applied safely."""


def canonicalize_audio(
    samples: object,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> Any:
    """Return deterministic mono float64 audio at the requested sample rate.

    WAV decoding performs the channel downmix. This function deliberately
    resamples the complete decoded recording, before any window is selected.
    """

    import numpy as np

    if source_sample_rate_hz <= 0 or target_sample_rate_hz <= 0:
        raise AudioPreprocessingError("sample rates must be positive integers")
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise AudioPreprocessingError("audio contains non-finite samples")
    if source_sample_rate_hz == target_sample_rate_hz:
        return array.copy()

    try:
        from scipy.signal import resample_poly
    except ImportError as error:  # pragma: no cover - dependency guard
        raise AudioPreprocessingError(
            "canonical 16 kHz preprocessing requires scipy"
        ) from error

    divisor = math.gcd(source_sample_rate_hz, target_sample_rate_hz)
    up = target_sample_rate_hz // divisor
    down = source_sample_rate_hz // divisor
    resampled = np.asarray(resample_poly(array, up, down), dtype=np.float64)
    expected_length = int(round(len(array) * target_sample_rate_hz / source_sample_rate_hz))
    if len(resampled) > expected_length:
        resampled = resampled[:expected_length]
    elif len(resampled) < expected_length:
        resampled = np.pad(resampled, (0, expected_length - len(resampled)))
    return np.ascontiguousarray(resampled, dtype=np.float64)


def require_compatible_preprocessing(
    left_profile: str,
    right_profile: str,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject combinations of artifacts produced by different profiles."""

    if left_profile != right_profile:
        raise AudioPreprocessingError(
            "preprocessing profile mismatch: "
            f"{left_name}={left_profile}, {right_name}={right_profile}"
        )
