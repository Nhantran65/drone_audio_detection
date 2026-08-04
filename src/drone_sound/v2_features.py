"""V2 engineered audio feature extraction."""

from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.audio_preprocessing import (
    CANONICAL_PREPROCESSING_PROFILE,
    FEATURE_BUNDLE_SCHEMA_VERSION,
    canonicalize_audio,
)
from drone_sound.constants import TARGET_SAMPLE_RATE_HZ
from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL
from drone_sound.v2 import ENGINEERED_FEATURE_SETS, V2InputError, validate_feature_set
from drone_sound.wav_audio import WavReadError, read_wav_mono_array

V1_BASIC_FEATURE_NAMES = (
    "mean",
    "mean_abs",
    "rms",
    "peak_abs",
    "zero_crossing_rate",
)

SPECTRAL_FEATURE_NAMES = (
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
    "spectral_contrast_db",
    "dominant_frequency_hz",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
    "low_total_ratio",
    "mid_total_ratio",
    "high_total_ratio",
    "low_mid_ratio",
    "low_high_ratio",
    "mid_high_ratio",
)

MFCC_COEFFICIENT_COUNT = 20
MEL_FILTER_COUNT = 40
MFCC_FEATURE_NAMES = (
    tuple(f"mfcc_{index:02d}_mean" for index in range(1, MFCC_COEFFICIENT_COUNT + 1))
    + tuple(f"mfcc_{index:02d}_std" for index in range(1, MFCC_COEFFICIENT_COUNT + 1))
    + tuple(f"delta_mfcc_{index:02d}_mean" for index in range(1, MFCC_COEFFICIENT_COUNT + 1))
    + tuple(f"delta_mfcc_{index:02d}_std" for index in range(1, MFCC_COEFFICIENT_COUNT + 1))
)

LOW_BAND_HZ = (20.0, 500.0)
MID_BAND_HZ = (500.0, 2_000.0)
EPSILON = 1e-12
TRAIN_SPLIT = "train"
PUBLIC_SPLITS = frozenset({"train", "val", "public_test"})
PROFESSOR_SPLIT = "professor_test"
SUPPORTED_LABELS = frozenset({POSITIVE_LABEL, NEGATIVE_LABEL})
ProgressCallback = Callable[[dict[str, Any]], None]


class V2FeatureError(ValueError):
    """Raised when V2 feature extraction cannot complete."""


def feature_names(feature_set: str) -> tuple[str, ...]:
    """Return stable feature names for a V2 engineered feature set."""

    validate_feature_set(feature_set)
    if feature_set not in ENGINEERED_FEATURE_SETS:
        raise V2InputError(f"feature set is not implemented for V2-M1 extraction: {feature_set}")
    if feature_set == "v1_basic":
        return V1_BASIC_FEATURE_NAMES
    if feature_set == "spectral_v1":
        return SPECTRAL_FEATURE_NAMES
    if feature_set == "mfcc_v1":
        return MFCC_FEATURE_NAMES
    if feature_set == "spectral_mfcc_v1":
        return V1_BASIC_FEATURE_NAMES + SPECTRAL_FEATURE_NAMES + MFCC_FEATURE_NAMES
    raise V2InputError(f"feature set is not implemented for V2-M1 extraction: {feature_set}")


def extract_feature_vector(
    samples: object,
    sample_rate_hz: int,
    feature_set: str,
) -> tuple[tuple[float, ...], int]:
    """Extract one raw feature vector and return non-finite replacement count."""

    np, _ = _require_data_dependencies()
    validate_feature_set(feature_set)
    if feature_set not in ENGINEERED_FEATURE_SETS:
        raise V2InputError(f"feature set is not implemented for V2-M1 extraction: {feature_set}")
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise V2FeatureError("cannot extract features from an empty audio window")
    if sample_rate_hz <= 0:
        raise V2FeatureError(f"invalid sample rate: {sample_rate_hz}")

    values: tuple[float, ...]
    if feature_set == "v1_basic":
        values = _v1_basic_features(array)
    elif feature_set == "spectral_v1":
        values = _spectral_features(array, sample_rate_hz)
    elif feature_set == "mfcc_v1":
        values = _mfcc_features(array, sample_rate_hz)
    elif feature_set == "spectral_mfcc_v1":
        values = (
            _v1_basic_features(array)
            + _spectral_features(array, sample_rate_hz)
            + _mfcc_features(array, sample_rate_hz)
        )
    else:  # pragma: no cover - validate_feature_set and ENGINEERED gate above.
        raise V2InputError(f"feature set is not implemented for V2-M1 extraction: {feature_set}")
    return _finite_tuple(values)


def fit_scaler(train_features: object) -> dict[str, list[float]]:
    """Fit a standard scaler on train features only."""

    np, _ = _require_data_dependencies()
    matrix = np.asarray(train_features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise V2FeatureError("cannot fit scaler without train feature rows")
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std = np.where(std > EPSILON, std, 1.0)
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }


def apply_scaler(features: object, scaler: Mapping[str, list[float]]) -> object:
    """Apply a train-fitted standard scaler to a feature matrix."""

    np, _ = _require_data_dependencies()
    matrix = np.asarray(features, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if matrix.ndim != 2:
        raise V2FeatureError("features must be a 2D matrix")
    if matrix.shape[1] != len(mean) or len(mean) != len(std):
        raise V2FeatureError("scaler dimensions do not match features")
    return (matrix - mean) / std


def build_v2_feature_bundle(
    public_manifest_path: str | Path,
    public_audio_root: str | Path,
    feature_set: str,
    features_out: str | Path,
    scaler_out: str | Path,
    audit_out: str | Path,
    professor_manifest_path: str | Path | None = None,
    professor_audio_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 0,
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    """Extract V2 features, write bundle/scaler/audit artifacts, and return audit."""

    np, _ = _require_data_dependencies()
    if target_sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        raise V2FeatureError(
            f"canonical V2 preprocessing requires {TARGET_SAMPLE_RATE_HZ} Hz"
        )
    names = feature_names(feature_set)
    public_manifest = Path(public_manifest_path)
    public_root = Path(public_audio_root)
    features_out = Path(features_out)
    scaler_out = Path(scaler_out)
    audit_out = Path(audit_out)
    _require_outputs_absent(features_out, scaler_out, audit_out)

    _validate_existing_file(public_manifest, "public manifest")
    _validate_existing_directory(public_root, "public audio root")
    if (professor_manifest_path is None) != (professor_audio_root is None):
        raise V2InputError(
            "professor manifest and professor audio root must be provided together"
        )

    public_rows_all = load_manifest_csv(public_manifest)
    validate_manifest_rows(public_rows_all)
    rows = _eligible_public_rows(public_rows_all)

    professor_manifest: Path | None = None
    professor_root: Path | None = None
    if professor_manifest_path is not None and professor_audio_root is not None:
        professor_manifest = Path(professor_manifest_path)
        professor_root = Path(professor_audio_root)
        _validate_existing_file(professor_manifest, "professor manifest")
        _validate_existing_directory(professor_root, "professor audio root")
        professor_rows_all = load_manifest_csv(professor_manifest)
        validate_manifest_rows(professor_rows_all)
        rows.extend(_eligible_professor_rows(professor_rows_all))

    if not rows:
        raise V2FeatureError("no eligible rows found for V2 feature extraction")

    feature_rows, non_finite_replacements = _extract_rows(
        rows,
        public_root,
        professor_root,
        feature_set,
        progress_callback=progress_callback,
        progress_every=progress_every,
        target_sample_rate_hz=target_sample_rate_hz,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "scaling",
                "rows": len(feature_rows),
                "feature_count": len(names),
            }
        )
    raw_features = np.asarray([row["features"] for row in feature_rows], dtype=np.float64)
    splits = np.asarray([row["split"] for row in feature_rows])
    professor_flags = np.asarray([row["is_professor_data"] for row in feature_rows], dtype=bool)
    train_mask = (splits == TRAIN_SPLIT) & (~professor_flags)
    train_features = raw_features[train_mask]
    scaler = fit_scaler(train_features)
    scaled_features = apply_scaler(raw_features, scaler)
    bundle_fingerprint = _bundle_fingerprint(
        raw_features,
        feature_rows,
        names,
        feature_set,
        CANONICAL_PREPROCESSING_PROFILE,
        target_sample_rate_hz,
    )

    features_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "writing",
                "features_out": str(features_out),
                "scaler_out": str(scaler_out),
                "audit_out": str(audit_out),
            }
        )
    np.savez_compressed(
        features_out,
        raw_features=raw_features,
        scaled_features=scaled_features,
        labels=np.asarray([row["label"] for row in feature_rows]),
        splits=splits,
        window_ids=np.asarray([row["window_id"] for row in feature_rows]),
        clip_ids=np.asarray([row["clip_id"] for row in feature_rows]),
        original_files=np.asarray([row["original_file"] for row in feature_rows]),
        source_datasets=np.asarray([row["source_dataset"] for row in feature_rows]),
        label_subtypes=np.asarray([row["label_subtype"] for row in feature_rows]),
        environments=np.asarray([row["environment"] for row in feature_rows]),
        window_start_sec=np.asarray(
            [row["window_start_sec"] for row in feature_rows],
            dtype=np.float64,
        ),
        window_end_sec=np.asarray(
            [row["window_end_sec"] for row in feature_rows],
            dtype=np.float64,
        ),
        is_professor_data=professor_flags,
        source_sample_rates_hz=np.asarray(
            [row["source_sample_rate_hz"] for row in feature_rows], dtype=np.int32
        ),
        processed_sample_rates_hz=np.full(
            len(feature_rows), target_sample_rate_hz, dtype=np.int32
        ),
        feature_names=np.asarray(names),
        feature_set=np.asarray(feature_set),
        bundle_schema_version=np.asarray(FEATURE_BUNDLE_SCHEMA_VERSION),
        preprocessing_profile=np.asarray(CANONICAL_PREPROCESSING_PROFILE),
        processed_sample_rate_hz=np.asarray(target_sample_rate_hz, dtype=np.int32),
        bundle_fingerprint=np.asarray(bundle_fingerprint),
    )

    scaler_report = {
        "feature_set": feature_set,
        "feature_names": list(names),
        "fitted_split": TRAIN_SPLIT,
        "train_rows": int(train_mask.sum()),
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": target_sample_rate_hz,
        "source_bundle_fingerprint": bundle_fingerprint,
        "mean": scaler["mean"],
        "std": scaler["std"],
    }
    scaler_out.write_text(
        json.dumps(scaler_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = {
        "feature_set": feature_set,
        "feature_count": len(names),
        "feature_names": list(names),
        "bundle_schema_version": FEATURE_BUNDLE_SCHEMA_VERSION,
        "bundle_fingerprint": bundle_fingerprint,
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": target_sample_rate_hz,
        "source_sample_rates_hz": sorted(
            {int(row["source_sample_rate_hz"]) for row in feature_rows}
        ),
        "features": {
            "path": str(features_out),
            "rows": len(feature_rows),
        },
        "scaler": {
            "path": str(scaler_out),
            "fitted_split": TRAIN_SPLIT,
            "train_rows": int(train_mask.sum()),
        },
        "public_manifest": str(public_manifest),
        "public_audio_root": str(public_root),
        "professor_manifest": str(professor_manifest) if professor_manifest else None,
        "professor_audio_root": str(professor_root) if professor_root else None,
        "counts": _counts(feature_rows),
        "non_finite_replacements": non_finite_replacements,
    }
    audit_out.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def build_v2_evaluation_bundle(
    manifest_path: str | Path,
    audio_root: str | Path,
    feature_set: str,
    scaler_path: str | Path,
    features_out: str | Path,
    audit_out: str | Path,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Build an evaluation-only bundle with a frozen train-fitted scaler."""

    np, _ = _require_data_dependencies()
    manifest = Path(manifest_path)
    root = Path(audio_root)
    output = Path(features_out)
    audit_path = Path(audit_out)
    frozen_scaler_path = Path(scaler_path)
    _require_outputs_absent(output, audit_path)
    for path, label in (
        (manifest, "evaluation manifest"),
        (frozen_scaler_path, "frozen scaler"),
    ):
        _validate_existing_file(path, label)
    _validate_existing_directory(root, "evaluation audio root")
    rows_all = load_manifest_csv(manifest)
    validate_manifest_rows(rows_all)
    rows = _eligible_professor_rows(rows_all)
    if not rows:
        raise V2FeatureError("evaluation manifest contains no professor_test rows")
    names = feature_names(feature_set)
    scaler = json.loads(frozen_scaler_path.read_text(encoding="utf-8"))
    if scaler.get("feature_set") != feature_set:
        raise V2FeatureError("frozen scaler feature_set does not match request")
    if tuple(scaler.get("feature_names", ())) != names:
        raise V2FeatureError("frozen scaler feature_names do not match request")
    if scaler.get("preprocessing_profile", "legacy_native_rate_v1") != CANONICAL_PREPROCESSING_PROFILE:
        raise V2FeatureError("frozen scaler preprocessing profile is not canonical 16 kHz")
    if scaler.get("processed_sample_rate_hz") != TARGET_SAMPLE_RATE_HZ:
        raise V2FeatureError("frozen scaler processed sample rate is not 16000 Hz")

    feature_rows, replacements = _extract_rows(
        rows,
        root,
        root,
        feature_set,
        progress_callback=progress_callback,
        progress_every=progress_every,
        target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
    )
    raw_features = np.asarray([row["features"] for row in feature_rows], dtype=np.float64)
    scaled_features = apply_scaler(raw_features, scaler)
    fingerprint = _bundle_fingerprint(
        raw_features,
        feature_rows,
        names,
        feature_set,
        CANONICAL_PREPROCESSING_PROFILE,
        TARGET_SAMPLE_RATE_HZ,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        raw_features=raw_features,
        scaled_features=scaled_features,
        labels=np.asarray([row["label"] for row in feature_rows]),
        splits=np.asarray([row["split"] for row in feature_rows]),
        window_ids=np.asarray([row["window_id"] for row in feature_rows]),
        clip_ids=np.asarray([row["clip_id"] for row in feature_rows]),
        original_files=np.asarray([row["original_file"] for row in feature_rows]),
        source_datasets=np.asarray([row["source_dataset"] for row in feature_rows]),
        label_subtypes=np.asarray([row["label_subtype"] for row in feature_rows]),
        environments=np.asarray([row["environment"] for row in feature_rows]),
        window_start_sec=np.asarray([row["window_start_sec"] for row in feature_rows], dtype=np.float64),
        window_end_sec=np.asarray([row["window_end_sec"] for row in feature_rows], dtype=np.float64),
        is_professor_data=np.ones(len(feature_rows), dtype=bool),
        source_sample_rates_hz=np.asarray([row["source_sample_rate_hz"] for row in feature_rows], dtype=np.int32),
        processed_sample_rates_hz=np.full(len(feature_rows), TARGET_SAMPLE_RATE_HZ, dtype=np.int32),
        feature_names=np.asarray(names),
        feature_set=np.asarray(feature_set),
        bundle_schema_version=np.asarray(FEATURE_BUNDLE_SCHEMA_VERSION),
        preprocessing_profile=np.asarray(CANONICAL_PREPROCESSING_PROFILE),
        processed_sample_rate_hz=np.asarray(TARGET_SAMPLE_RATE_HZ, dtype=np.int32),
        bundle_fingerprint=np.asarray(fingerprint),
    )
    audit = {
        "schema_version": "v2_evaluation_feature_audit_v1",
        "evaluation_only": True,
        "fit_or_tune_performed": False,
        "manifest": str(manifest),
        "audio_root": str(root),
        "feature_bundle": str(output),
        "frozen_scaler": str(frozen_scaler_path),
        "scaler_source_bundle_fingerprint": scaler.get("source_bundle_fingerprint"),
        "bundle_fingerprint": fingerprint,
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "source_sample_rates_hz": sorted({row["source_sample_rate_hz"] for row in feature_rows}),
        "feature_set": feature_set,
        "feature_count": len(names),
        "counts": _counts(feature_rows),
        "non_finite_replacements": replacements,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _v1_basic_features(samples: object) -> tuple[float, ...]:
    np, _ = _require_data_dependencies()
    array = np.asarray(samples, dtype=np.float64)
    abs_samples = np.abs(array)
    signs = array >= 0
    crossings = np.count_nonzero(signs[1:] != signs[:-1]) if len(array) > 1 else 0
    zero_crossing_rate = crossings / (len(array) - 1) if len(array) > 1 else 0.0
    return (
        float(np.mean(array)),
        float(np.mean(abs_samples)),
        float(np.sqrt(np.mean(array * array))),
        float(np.max(abs_samples)),
        float(zero_crossing_rate),
    )


def _spectral_features(samples: object, sample_rate_hz: int) -> tuple[float, ...]:
    np, _ = _require_data_dependencies()
    array = np.asarray(samples, dtype=np.float64)
    windowed = array * np.hanning(len(array))
    magnitudes = np.abs(np.fft.rfft(windowed))
    power = magnitudes * magnitudes
    freqs = np.fft.rfftfreq(len(array), d=1.0 / sample_rate_hz)
    total_power = float(np.sum(power))
    if total_power <= EPSILON:
        return (0.0,) * len(SPECTRAL_FEATURE_NAMES)

    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    cumulative = np.cumsum(power)
    rolloff_index = int(np.searchsorted(cumulative, 0.85 * total_power, side="left"))
    rolloff_index = min(rolloff_index, len(freqs) - 1)
    rolloff = float(freqs[rolloff_index])
    flatness = float(np.exp(np.mean(np.log(power + EPSILON))) / (np.mean(power) + EPSILON))
    positive = magnitudes[magnitudes > EPSILON]
    if positive.size:
        low_percentile = float(np.percentile(positive, 5))
        high_percentile = float(np.percentile(positive, 95))
        spectral_contrast = 20.0 * math.log10((high_percentile + EPSILON) / (low_percentile + EPSILON))
    else:
        spectral_contrast = 0.0
    dominant_frequency = float(freqs[int(np.argmax(power))])

    low_energy = _band_energy(freqs, power, LOW_BAND_HZ[0], LOW_BAND_HZ[1])
    mid_energy = _band_energy(freqs, power, MID_BAND_HZ[0], MID_BAND_HZ[1])
    high_energy = _band_energy(freqs, power, MID_BAND_HZ[1], sample_rate_hz / 2.0)
    return (
        centroid,
        bandwidth,
        rolloff,
        flatness,
        spectral_contrast,
        dominant_frequency,
        low_energy,
        mid_energy,
        high_energy,
        low_energy / (total_power + EPSILON),
        mid_energy / (total_power + EPSILON),
        high_energy / (total_power + EPSILON),
        low_energy / (mid_energy + EPSILON),
        low_energy / (high_energy + EPSILON),
        mid_energy / (high_energy + EPSILON),
    )


def _mfcc_features(samples: object, sample_rate_hz: int) -> tuple[float, ...]:
    np, scipy_fft = _require_data_dependencies()
    array = np.asarray(samples, dtype=np.float64)
    frames = _framed_audio(array, sample_rate_hz)
    frame_length = frames.shape[1]
    n_fft = _next_power_of_two(frame_length)
    window = np.hanning(frame_length)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / n_fft
    filters = _mel_filterbank(sample_rate_hz, n_fft, MEL_FILTER_COUNT)
    mel_energy = np.maximum(power @ filters.T, EPSILON)
    log_mel = np.log(mel_energy)
    coefficients = scipy_fft.dct(log_mel, type=2, norm="ortho", axis=1)[
        :, :MFCC_COEFFICIENT_COUNT
    ]
    if coefficients.shape[0] > 1:
        deltas = np.diff(coefficients, axis=0)
    else:
        deltas = np.zeros_like(coefficients)
    return tuple(
        [float(value) for value in np.mean(coefficients, axis=0)]
        + [float(value) for value in np.std(coefficients, axis=0)]
        + [float(value) for value in np.mean(deltas, axis=0)]
        + [float(value) for value in np.std(deltas, axis=0)]
    )


def _extract_rows(
    rows: list[dict[str, Any]],
    public_audio_root: Path,
    professor_audio_root: Path | None,
    feature_set: str,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 0,
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> tuple[list[dict[str, Any]], int]:
    np, _ = _require_data_dependencies()
    by_audio_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root = professor_audio_root if row["is_professor_data"] else public_audio_root
        if root is None:
            raise V2FeatureError("professor audio root is required for professor rows")
        audio_path = _resolve_audio_path(root, row["original_file"])
        by_audio_path[audio_path].append(row)

    feature_rows = []
    non_finite_replacements = 0
    processed_rows = 0
    progress_interval = max(0, progress_every)
    total_rows = len(rows)
    total_files = len(by_audio_path)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "start",
                "feature_set": feature_set,
                "total_rows": total_rows,
                "total_files": total_files,
            }
        )
    for file_index, (audio_path, file_rows) in enumerate(by_audio_path.items(), start=1):
        try:
            samples, metadata = read_wav_mono_array(audio_path)
        except (ImportError, WavReadError) as error:
            raise V2FeatureError(str(error)) from error
        samples = canonicalize_audio(
            samples,
            metadata.sample_rate_hz,
            target_sample_rate_hz,
        )
        for row in file_rows:
            start_sec = float(str(row["window_start_sec"]).strip())
            end_sec = float(str(row["window_end_sec"]).strip())
            window_samples = _slice_window(
                samples, target_sample_rate_hz, start_sec, end_sec, audio_path
            )
            features, replacements = extract_feature_vector(
                np.asarray(window_samples, dtype=np.float64),
                target_sample_rate_hz,
                feature_set,
            )
            non_finite_replacements += replacements
            processed_rows += 1
            feature_rows.append(
                {
                    "features": features,
                    "label": row["label_binary"],
                    "split": row["split"],
                    "window_id": row["window_id"],
                    "clip_id": row["clip_id"],
                    "original_file": row["original_file"],
                    "source_dataset": row["source_dataset"],
                    "label_subtype": row["label_subtype"],
                    "environment": row["environment"],
                    "window_start_sec": start_sec,
                    "window_end_sec": end_sec,
                    "is_professor_data": row["is_professor_data"],
                    "source_sample_rate_hz": metadata.sample_rate_hz,
                }
            )
            should_report = processed_rows == total_rows or (
                progress_interval > 0 and processed_rows % progress_interval == 0
            )
            if progress_callback is not None and should_report:
                progress_callback(
                    {
                        "event": "row_progress",
                        "processed_rows": processed_rows,
                        "total_rows": total_rows,
                        "file_index": file_index,
                        "total_files": total_files,
                        "current_file": str(audio_path),
                        "non_finite_replacements": non_finite_replacements,
                    }
                )
    return feature_rows, non_finite_replacements


def _eligible_public_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        split = str(row.get("split", "")).strip()
        label = str(row.get("label_binary", "")).strip()
        if split not in PUBLIC_SPLITS or label not in SUPPORTED_LABELS:
            continue
        if _truthy(row.get("is_professor_data", "")):
            continue
        eligible.append(_normalized_row(row, is_professor_data=False))
    if not any(row["split"] == TRAIN_SPLIT for row in eligible):
        raise V2FeatureError("V2 feature extraction requires public train rows")
    return eligible


def _eligible_professor_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        split = str(row.get("split", "")).strip()
        label = str(row.get("label_binary", "")).strip()
        if split != PROFESSOR_SPLIT or label not in SUPPORTED_LABELS:
            continue
        eligible.append(_normalized_row(row, is_professor_data=True))
    return eligible


def _normalized_row(row: Mapping[str, str], *, is_professor_data: bool) -> dict[str, Any]:
    return {
        "label_binary": str(row["label_binary"]).strip(),
        "split": str(row["split"]).strip(),
        "clip_id": str(row["clip_id"]).strip(),
        "window_id": str(row["window_id"]).strip(),
        "original_file": str(row["original_file"]).strip(),
        "source_dataset": str(row["source_dataset"]).strip(),
        "label_subtype": str(row["label_subtype"]).strip(),
        "environment": str(row.get("environment", "")).strip(),
        "window_start_sec": str(row["window_start_sec"]).strip(),
        "window_end_sec": str(row["window_end_sec"]).strip(),
        "is_professor_data": is_professor_data,
    }


def _bundle_fingerprint(
    raw_features: object,
    rows: list[dict[str, Any]],
    names: tuple[str, ...],
    feature_set: str,
    preprocessing_profile: str,
    processed_sample_rate_hz: int,
) -> str:
    """Hash bundle content and row identity for cache/model provenance."""

    np, _ = _require_data_dependencies()
    digest = hashlib.sha256()
    header = {
        "schema_version": FEATURE_BUNDLE_SCHEMA_VERSION,
        "feature_set": feature_set,
        "feature_names": list(names),
        "preprocessing_profile": preprocessing_profile,
        "processed_sample_rate_hz": processed_sample_rate_hz,
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for row in rows:
        digest.update(str(row["window_id"]).encode())
        digest.update(b"\0")
        digest.update(str(row["label"]).encode())
        digest.update(b"\0")
        digest.update(str(row["split"]).encode())
        digest.update(b"\0")
    matrix = np.ascontiguousarray(raw_features, dtype="<f8")
    digest.update(memoryview(matrix).cast("B"))
    return digest.hexdigest()


def _slice_window(
    samples: object,
    sample_rate_hz: int,
    start_sec: float,
    end_sec: float,
    audio_path: Path,
) -> object:
    if end_sec <= start_sec:
        raise V2FeatureError(f"invalid window bounds for {audio_path}: {start_sec}..{end_sec}")
    start_index = int(round(start_sec * sample_rate_hz))
    sample_count = int(round((end_sec - start_sec) * sample_rate_hz))
    end_index = start_index + sample_count
    if start_index < 0 or end_index > len(samples):
        raise V2FeatureError(f"manifest window exceeds audio duration: {audio_path}")
    if sample_count <= 0:
        raise V2FeatureError(f"empty audio window: {audio_path}")
    return samples[start_index:end_index]


def _framed_audio(samples: object, sample_rate_hz: int) -> object:
    np, _ = _require_data_dependencies()
    array = np.asarray(samples, dtype=np.float64)
    frame_length = max(1, int(round(0.025 * sample_rate_hz)))
    hop_length = max(1, int(round(0.010 * sample_rate_hz)))
    if len(array) < frame_length:
        array = np.pad(array, (0, frame_length - len(array)))
    frame_starts = range(0, len(array) - frame_length + 1, hop_length)
    frames = [array[start : start + frame_length] for start in frame_starts]
    if not frames:
        frames = [np.pad(array, (0, frame_length - len(array)))]
    return np.vstack(frames)


def _mel_filterbank(sample_rate_hz: int, n_fft: int, mel_count: int) -> object:
    np, _ = _require_data_dependencies()
    min_mel = _hz_to_mel(0.0)
    max_mel = _hz_to_mel(sample_rate_hz / 2.0)
    mel_points = np.linspace(min_mel, max_mel, mel_count + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate_hz).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((mel_count, n_fft // 2 + 1), dtype=np.float64)
    for index in range(1, mel_count + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center == left:
            center = min(center + 1, n_fft // 2)
        if right == center:
            right = min(right + 1, n_fft // 2)
        for bin_index in range(left, center):
            filters[index - 1, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, right):
            filters[index - 1, bin_index] = (right - bin_index) / max(right - center, 1)
    return filters


def _hz_to_mel(value: object) -> object:
    np, _ = _require_data_dependencies()
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: object) -> object:
    np, _ = _require_data_dependencies()
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def _band_energy(freqs: object, power: object, low_hz: float, high_hz: float) -> float:
    np, _ = _require_data_dependencies()
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.sum(power[mask]))


def _finite_tuple(values: tuple[float, ...]) -> tuple[tuple[float, ...], int]:
    sanitized = []
    replacements = 0
    for value in values:
        if math.isfinite(value):
            sanitized.append(float(value))
        else:
            sanitized.append(0.0)
            replacements += 1
    return tuple(sanitized), replacements


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _resolve_audio_path(audio_root: Path, original_file: str) -> Path:
    path = Path(original_file)
    if not path.is_absolute():
        path = audio_root / path
    if not path.exists():
        raise V2FeatureError(f"audio file not found for manifest row: {path}")
    return path


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    labels: dict[str, int] = {}
    for row in rows:
        split = row["split"]
        label = row["label"]
        split_entry = splits.setdefault(split, {"rows": 0, "labels": {}})
        split_entry["rows"] += 1
        split_entry["labels"][label] = split_entry["labels"].get(label, 0) + 1
        labels[label] = labels.get(label, 0) + 1
    return {
        "rows": len(rows),
        "splits": splits,
        "labels": labels,
    }


def _validate_existing_file(path: Path, field: str) -> None:
    if not path.exists():
        raise V2InputError(f"{field} does not exist: {path}")
    if not path.is_file():
        raise V2InputError(f"{field} is not a file: {path}")


def _validate_existing_directory(path: Path, field: str) -> None:
    if not path.exists():
        raise V2InputError(f"{field} does not exist: {path}")
    if not path.is_dir():
        raise V2InputError(f"{field} is not a directory: {path}")


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise V2FeatureError(
            "refusing to overwrite existing V2 artifact(s): " + ", ".join(existing)
        )


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _require_data_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        from scipy import fft as scipy_fft
    except ImportError as error:  # pragma: no cover - exercised without data extra.
        raise V2InputError(
            "V2 feature extraction requires the optional data dependencies; "
            "run with `uv run --extra data --python 3.13 ...`"
        ) from error
    return np, scipy_fft
