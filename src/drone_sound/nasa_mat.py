"""Convert NASA Small UAS MATLAB audio files into derived WAV files."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

TARGET_INT16_PEAK = 32767
PCM_HEADROOM = 0.95
SCALE_POLICY = "batch_global_peak"


class NasaMatConversionError(ValueError):
    """Raised when NASA MAT files cannot be converted safely."""


@dataclass(frozen=True)
class NasaMatAudio:
    path: Path
    mono: Any
    utc_time: Any
    source_sample_rate_hz: int
    source_channels: int
    timestamp_jitter_seconds_max: float
    duration_sec: float
    max_abs_mono: float


@dataclass(frozen=True)
class NasaMatSummary:
    path: Path
    source_sample_rate_hz: int
    source_channels: int
    timestamp_jitter_seconds_max: float
    duration_sec: float
    max_abs_mono: float


def convert_nasa_mat_directory(
    mat_root: str | Path,
    wav_out: str | Path,
    audit_out: str | Path,
    target_sample_rate_hz: int,
) -> dict[str, Any]:
    """Convert NASA MAT files under a directory to derived mono WAV files."""

    np, _, _ = _require_data_dependencies()
    mat_root = Path(mat_root)
    wav_out = Path(wav_out)
    audit_out = Path(audit_out)

    if target_sample_rate_hz <= 0:
        raise NasaMatConversionError("target sample rate must be positive")
    if not mat_root.exists() or not mat_root.is_dir():
        raise NasaMatConversionError(f"MAT root is not a directory: {mat_root}")

    mat_paths = sorted(mat_root.rglob("*.mat"))
    if not mat_paths:
        raise NasaMatConversionError(f"MAT root has no .mat files: {mat_root}")

    output_paths = _output_paths(mat_paths, wav_out)
    summaries: list[NasaMatSummary] = []
    global_peak = 0.0
    for path in mat_paths:
        audio = _load_nasa_audio(path)
        summaries.append(
            NasaMatSummary(
                path=path,
                source_sample_rate_hz=audio.source_sample_rate_hz,
                source_channels=audio.source_channels,
                timestamp_jitter_seconds_max=audio.timestamp_jitter_seconds_max,
                duration_sec=audio.duration_sec,
                max_abs_mono=audio.max_abs_mono,
            )
        )
        global_peak = max(global_peak, audio.max_abs_mono)

    if not math.isfinite(global_peak) or global_peak <= 0:
        raise NasaMatConversionError("NASA MAT audio has no non-zero finite mono samples")

    scale_factor = (TARGET_INT16_PEAK * PCM_HEADROOM) / global_peak

    wav_out.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for summary in summaries:
        audio = _load_nasa_audio(summary.path)
        converted = _resample(audio.mono, audio.source_sample_rate_hz, target_sample_rate_hz)
        pcm, clipped_samples = _to_pcm_int16(converted, scale_factor)
        output_path = output_paths[summary.path]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(output_path, pcm, target_sample_rate_hz)
        files.append(
            {
                "source_mat": str(summary.path),
                "source_mat_sha256": _sha256_path(summary.path),
                "output_wav": str(output_path),
                "output_wav_sha256": _sha256_path(output_path),
                "source_sample_rate_hz": summary.source_sample_rate_hz,
                "target_sample_rate_hz": target_sample_rate_hz,
                "source_channels": summary.source_channels,
                "duration_sec": round(summary.duration_sec, 6),
                "output_frames": int(len(pcm)),
                "max_abs_mono": summary.max_abs_mono,
                "timestamp_jitter_seconds_max": summary.timestamp_jitter_seconds_max,
                "clipped_samples": clipped_samples,
            }
        )

    report = {
        "mat_root": str(mat_root),
        "wav_out": str(wav_out),
        "target_sample_rate_hz": target_sample_rate_hz,
        "scale_policy": SCALE_POLICY,
        "global_peak_pascals_mono": global_peak,
        "scale_factor_int16_per_pascal": scale_factor,
        "files": files,
    }
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_nasa_audio(path: Path) -> NasaMatAudio:
    np, loadmat, _ = _require_data_dependencies()
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as error:  # pragma: no cover - scipy owns parser details.
        raise NasaMatConversionError(f"failed to read MAT file {path}: {error}") from error

    acoustics = mat.get("acoustics")
    if acoustics is None:
        raise NasaMatConversionError(f"{path}: missing acoustics struct")
    if not hasattr(acoustics, "incident_pascals"):
        raise NasaMatConversionError(f"{path}: missing acoustics.incident_pascals")
    if not hasattr(acoustics, "utc_time"):
        raise NasaMatConversionError(f"{path}: missing acoustics.utc_time")

    incident = np.asarray(acoustics.incident_pascals, dtype=np.float64)
    utc_time = np.asarray(acoustics.utc_time, dtype=np.float64).reshape(-1)
    if incident.size == 0:
        raise NasaMatConversionError(f"{path}: acoustics.incident_pascals is empty")
    if utc_time.size < 2:
        raise NasaMatConversionError(f"{path}: acoustics.utc_time must contain at least two samples")

    samples_by_channel = _samples_by_channel(incident, utc_time.size, path)
    if not np.all(np.isfinite(samples_by_channel)):
        raise NasaMatConversionError(f"{path}: acoustics.incident_pascals contains non-finite values")
    if not np.all(np.isfinite(utc_time)):
        raise NasaMatConversionError(f"{path}: acoustics.utc_time contains non-finite values")

    diffs = np.diff(utc_time)
    if diffs.size == 0 or np.any(diffs <= 0):
        raise NasaMatConversionError(f"{path}: acoustics.utc_time must be strictly increasing")
    median_dt = float(np.median(diffs))
    if median_dt <= 0 or not math.isfinite(median_dt):
        raise NasaMatConversionError(f"{path}: cannot infer source sample rate")
    source_sample_rate_hz = int(round(1.0 / median_dt))
    if source_sample_rate_hz <= 0:
        raise NasaMatConversionError(f"{path}: invalid inferred source sample rate")

    mono = np.mean(samples_by_channel, axis=1)
    max_abs_mono = float(np.max(np.abs(mono)))
    if not math.isfinite(max_abs_mono):
        raise NasaMatConversionError(f"{path}: mono audio has non-finite peak")

    return NasaMatAudio(
        path=path,
        mono=mono,
        utc_time=utc_time,
        source_sample_rate_hz=source_sample_rate_hz,
        source_channels=int(samples_by_channel.shape[1]),
        timestamp_jitter_seconds_max=float(np.max(np.abs(diffs - median_dt))),
        duration_sec=float(mono.shape[0] / source_sample_rate_hz),
        max_abs_mono=max_abs_mono,
    )


def _samples_by_channel(incident: Any, expected_samples: int, path: Path) -> Any:
    np, _, _ = _require_data_dependencies()
    if incident.ndim == 1:
        if incident.shape[0] != expected_samples:
            raise NasaMatConversionError(
                f"{path}: incident_pascals length does not match utc_time length"
            )
        return incident.reshape(-1, 1)
    if incident.ndim != 2:
        raise NasaMatConversionError(f"{path}: incident_pascals must be 1D or 2D")
    if incident.shape[0] == expected_samples:
        return incident
    if incident.shape[1] == expected_samples:
        return incident.T
    raise NasaMatConversionError(
        f"{path}: incident_pascals shape does not match utc_time length"
    )


def _resample(mono: Any, source_sample_rate_hz: int, target_sample_rate_hz: int) -> Any:
    _, _, resample_poly = _require_data_dependencies()
    if source_sample_rate_hz == target_sample_rate_hz:
        return mono
    fraction = Fraction(target_sample_rate_hz, source_sample_rate_hz).limit_denominator()
    return resample_poly(mono, fraction.numerator, fraction.denominator)


def _to_pcm_int16(samples: Any, scale_factor: float) -> tuple[Any, int]:
    np, _, _ = _require_data_dependencies()
    scaled = samples * scale_factor
    rounded = np.rint(scaled)
    clipped_samples = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    clipped = np.clip(rounded, -32768, 32767).astype("<i2")
    return clipped, clipped_samples


def _write_wav(path: Path, samples: Any, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(samples.tobytes())


def _output_paths(mat_paths: list[Path], wav_out: Path) -> dict[Path, Path]:
    output_paths = {path: wav_out / f"{path.stem}.wav" for path in mat_paths}
    reverse: dict[Path, list[Path]] = {}
    for source, output in output_paths.items():
        reverse.setdefault(output, []).append(source)
    conflicts = [output for output, sources in reverse.items() if len(sources) > 1]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts[:5])
        raise NasaMatConversionError(f"multiple MAT files map to the same WAV path: {joined}")
    return output_paths


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_data_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from scipy.io import loadmat
        from scipy.signal import resample_poly
    except ImportError as error:  # pragma: no cover - exercised without optional deps.
        raise NasaMatConversionError(
            "NASA MAT conversion requires the optional data dependencies: "
            'install with ".[data]" or run with "uv run --extra data ..."'
        ) from error
    return np, loadmat, resample_poly
