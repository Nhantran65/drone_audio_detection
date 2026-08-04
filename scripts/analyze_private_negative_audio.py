#!/usr/bin/env python3
"""Non-destructive quality assessment for private professor negative audio.

The analysis intentionally keeps the ten source WAV files unchanged. It
profiles signal quality, extracts the existing spectral_mfcc_v1 features at
the native rate and after in-memory 16 kHz resampling, then reuses frozen model
thresholds to quantify false-positive and domain-shift risk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from scipy.signal import resample_poly

from drone_sound.v2_features import extract_feature_vector, feature_names
from drone_sound.v2_ml import _score_estimator
from drone_sound.wav_audio import read_wav_mono_array


WINDOW_SECONDS = 1.0
STRIDE_SECONDS = 0.5
TARGET_SAMPLE_RATE = 16_000
FEATURE_SET = "spectral_mfcc_v1"
PROGRESS_EVERY = 100
DB_FLOOR = -120.0
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260804


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--negative-root",
        type=Path,
        default=Path("data/raw/private/Recordings"),
    )
    parser.add_argument(
        "--positive-wav",
        type=Path,
        default=Path("data/raw/private/professor_real_world/SMU08662_20260608_135349.wav"),
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        default=Path("artifacts/features/v2_spectral_mfcc_features.npz"),
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=Path("artifacts/features/v2_spectral_mfcc_scaler.json"),
    )
    parser.add_argument(
        "--tiny-mlp",
        type=Path,
        default=Path("artifacts/models/v2b_mlp/tiny_mlp_es.joblib"),
    )
    parser.add_argument(
        "--random-forest",
        type=Path,
        default=Path("artifacts/models/v2a_sweep/models/random_forest.joblib"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/analysis/private_negative_quality"),
    )
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        negative_root=args.negative_root,
        positive_wav=args.positive_wav,
        feature_bundle=args.feature_bundle,
        scaler_path=args.scaler,
        model_paths={
            "tiny_mlp_es": args.tiny_mlp,
            "random_forest": args.random_forest,
        },
        out_dir=args.out_dir,
        progress_every=args.progress_every,
    )
    return 0


def run_analysis(
    *,
    negative_root: Path,
    positive_wav: Path,
    feature_bundle: Path,
    scaler_path: Path,
    model_paths: Mapping[str, Path],
    out_dir: Path,
    progress_every: int = PROGRESS_EVERY,
) -> dict[str, Any]:
    started_at = time.monotonic()
    paths = _validate_inputs(
        negative_root,
        positive_wav,
        feature_bundle,
        scaler_path,
        model_paths,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"phase=inventory negative_files={len(paths)} elapsed=0s",
        flush=True,
    )

    scaler = _load_scaler(scaler_path)
    models = _load_models(model_paths)
    public_reference = _load_public_reference(feature_bundle, models)

    file_profiles: list[dict[str, Any]] = []
    feature_sets: dict[str, list[np.ndarray]] = defaultdict(list)
    row_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_expected = sum(_expected_windows(path) for path in paths) * 2
    total_expected += _expected_windows(positive_wav)
    processed = 0

    for file_index, path in enumerate(paths, start=1):
        samples, metadata = read_wav_mono_array(path)
        environment = _environment(path)
        file_profile = _signal_profile(path, samples, metadata, environment)
        file_profiles.append(file_profile)
        modes = {
            "native_48k": (np.asarray(samples, dtype=np.float64), metadata.sample_rate_hz),
            "resampled_16k": (
                _resample(np.asarray(samples, dtype=np.float64), metadata.sample_rate_hz),
                TARGET_SAMPLE_RATE,
            ),
        }
        for mode, (mode_samples, sample_rate) in modes.items():
            rows, features = _extract_windows(
                path=path,
                samples=mode_samples,
                sample_rate_hz=sample_rate,
                label="no_drone",
                environment=environment,
                mode=mode,
                processed_before=processed,
                total_expected=total_expected,
                started_at=started_at,
                progress_every=progress_every,
            )
            processed += len(rows)
            row_sets[mode].extend(rows)
            feature_sets[mode].append(features)
        print(
            "phase=negative_file_complete "
            f"file={file_index}/{len(paths)} name={path.name} "
            f"duration={metadata.duration_sec:.2f}s windows={file_profile['windows']} "
            f"elapsed={_duration(time.monotonic() - started_at)}",
            flush=True,
        )

    positive_samples, positive_metadata = read_wav_mono_array(positive_wav)
    positive_rows, positive_features = _extract_windows(
        path=positive_wav,
        samples=np.asarray(positive_samples, dtype=np.float64),
        sample_rate_hz=positive_metadata.sample_rate_hz,
        label="drone",
        environment="professor_positive",
        mode="positive_reference_16k",
        processed_before=processed,
        total_expected=total_expected,
        started_at=started_at,
        progress_every=progress_every,
    )
    processed += len(positive_rows)

    negative_results: dict[str, dict[str, Any]] = {}
    all_window_rows: list[dict[str, Any]] = []
    file_model_rows: list[dict[str, Any]] = []
    mode_scaled: dict[str, np.ndarray] = {}
    for mode in ("native_48k", "resampled_16k"):
        raw = np.vstack(feature_sets[mode])
        scaled = _scale(raw, scaler)
        mode_scaled[mode] = scaled
        scored_rows = _score_rows(row_sets[mode], scaled, models)
        all_window_rows.extend(scored_rows)
        negative_results[mode] = _negative_summary(
            scored_rows,
            scaled,
            models,
            public_reference,
        )
        file_model_rows.extend(_file_model_rows(scored_rows, models))

    positive_scaled = _scale(positive_features, scaler)
    positive_scored = _score_rows(positive_rows, positive_scaled, models)
    all_window_rows.extend(positive_scored)
    positive_summary = _positive_summary(positive_scored, models)

    resampling_sensitivity = _resampling_sensitivity(
        row_sets["native_48k"],
        row_sets["resampled_16k"],
        mode_scaled["native_48k"],
        mode_scaled["resampled_16k"],
        models,
    )
    drift = _feature_drift(
        mode_scaled,
        row_sets,
        public_reference,
        feature_names(FEATURE_SET),
    )
    duplicates = _duplicate_checks(paths, feature_bundle)
    suspect_rows = _suspect_windows(
        [row for row in all_window_rows if row["mode"] == "resampled_16k"],
        models,
    )
    quality_flags = _quality_flags(file_profiles)

    summary = {
        "schema_version": "private_negative_quality_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "negative_files": len(paths),
            "negative_environments": _counts(item["environment"] for item in file_profiles),
            "negative_duration_sec": sum(item["duration_sec"] for item in file_profiles),
            "negative_windows_per_mode": len(row_sets["native_48k"]),
            "positive_files": 1,
            "positive_duration_sec": positive_metadata.duration_sec,
            "positive_windows": len(positive_rows),
            "independent_negative_recordings": len(paths),
            "window_length_sec": WINDOW_SECONDS,
            "window_stride_sec": STRIDE_SECONDS,
        },
        "audio_format": {
            "negative_sample_rates_hz": sorted({item["sample_rate_hz"] for item in file_profiles}),
            "negative_channels": sorted({item["channels"] for item in file_profiles}),
            "negative_bits_per_sample": sorted({item["bits_per_sample"] for item in file_profiles}),
            "positive_sample_rate_hz": positive_metadata.sample_rate_hz,
            "sample_rate_confound": len({item["sample_rate_hz"] for item in file_profiles}) == 1
            and file_profiles[0]["sample_rate_hz"] != positive_metadata.sample_rate_hz,
        },
        "signal_quality": {
            "files": file_profiles,
            "flags": quality_flags,
            "note": (
                "No noise-free reference exists, so true SNR cannot be estimated. "
                "Level spread and spectral statistics are descriptive proxies only."
            ),
        },
        "frozen_model_evaluation": {
            "public_test_negative_reference": public_reference["model_reference"],
            "new_negative": negative_results,
            "professor_positive_reference": positive_summary,
            "resampling_sensitivity": resampling_sensitivity,
            "threshold_source": "stored validation threshold_f2; no retuning",
        },
        "domain_shift": drift,
        "duplicate_checks": duplicates,
        "label_review": {
            "consensus_suspect_windows": len(suspect_rows),
            "suspect_output": str(out_dir / "suspect_windows.csv"),
            "status": "manual listening/annotation required",
        },
        "decision": {
            "evaluation_use": "conditional_yes",
            "training_use": "not_yet",
            "recommended_role": "held-out private hard-negative evaluation after manual label review",
            "rationale": [
                "Only ten independent negative recordings are available despite many overlapping windows.",
                "Negative and current professor-positive recordings have a 48 kHz versus 16 kHz sample-rate confound.",
                "Automated frozen-model scores can identify review targets but cannot verify labels.",
                "Raw environmental noise is useful hard-negative content and should not be aggressively denoised.",
            ],
        },
        "limitations": [
            "No human listening or timestamp-level annotation was available in this automated run.",
            "Traffic and forest recordings come from only two dates/sessions, so generalization is unproven.",
            "Overlapping one-second windows are correlated and are not independent samples.",
            "The existing public feature pipeline does not uniformly enforce 16 kHz despite the product contract.",
        ],
        "elapsed_seconds": time.monotonic() - started_at,
    }

    _write_csv(out_dir / "file_profile.csv", file_profiles)
    _write_csv(out_dir / "file_model_summary.csv", file_model_rows)
    _write_csv(out_dir / "window_scores.csv", all_window_rows)
    _write_csv(out_dir / "suspect_windows.csv", suspect_rows)
    _write_csv(out_dir / "feature_drift_top.csv", drift["top_shifted_features"])
    _write_json(out_dir / "summary.json", summary)
    print(
        "phase=complete "
        f"negative_windows={len(row_sets['native_48k'])} "
        f"positive_windows={len(positive_rows)} "
        f"elapsed={_duration(time.monotonic() - started_at)} out={out_dir}",
        flush=True,
    )
    return summary


def _validate_inputs(
    negative_root: Path,
    positive_wav: Path,
    feature_bundle: Path,
    scaler_path: Path,
    model_paths: Mapping[str, Path],
) -> list[Path]:
    if not negative_root.is_dir():
        raise ValueError(f"negative root is not a directory: {negative_root}")
    paths = sorted(path for path in negative_root.glob("*.wav") if path.is_file())
    if not paths:
        raise ValueError(f"no negative WAV files found: {negative_root}")
    for label, path in {
        "positive WAV": positive_wav,
        "feature bundle": feature_bundle,
        "scaler": scaler_path,
        **{f"model {name}": path for name, path in model_paths.items()},
    }.items():
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
    return paths


def _expected_windows(path: Path) -> int:
    _, metadata = read_wav_mono_array(path)
    if metadata.duration_sec < WINDOW_SECONDS:
        return 0
    return int(math.floor((metadata.duration_sec - WINDOW_SECONDS) / STRIDE_SECONDS)) + 1


def _load_scaler(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(data.get("feature_names", ()))
    if data.get("feature_set") != FEATURE_SET or names != feature_names(FEATURE_SET):
        raise ValueError("scaler does not match spectral_mfcc_v1 feature contract")
    return {
        "mean": np.asarray(data["mean"], dtype=np.float64),
        "std": np.asarray(data["std"], dtype=np.float64),
    }


def _load_models(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    expected_names = feature_names(FEATURE_SET)
    for name, path in paths.items():
        package = joblib.load(path)
        if not isinstance(package, dict):
            raise ValueError(f"model package is not a mapping: {path}")
        if package.get("feature_set") != FEATURE_SET:
            raise ValueError(f"model feature set mismatch: {path}")
        if tuple(package.get("feature_names", ())) != expected_names:
            raise ValueError(f"model feature names mismatch: {path}")
        if package.get("threshold_name") != "threshold_f2":
            raise ValueError(f"model threshold is not frozen threshold_f2: {path}")
        models[name] = package
    return models


def _load_public_reference(
    path: Path,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    print(f"phase=public_reference_load path={path}", flush=True)
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["scaled_features"], dtype=np.float64)
        labels = np.asarray(data["labels"]).astype(str)
        splits = np.asarray(data["splits"]).astype(str)
        professor = np.asarray(data["is_professor_data"], dtype=bool)
        original_files = np.asarray(data["original_files"]).astype(str)
        names = tuple(np.asarray(data["feature_names"]).astype(str))
    if names != feature_names(FEATURE_SET):
        raise ValueError("feature bundle names do not match analysis feature contract")

    train_negative = (splits == "train") & (labels == "no_drone") & (~professor)
    train_positive = (splits == "train") & (labels == "drone") & (~professor)
    test_negative = (splits == "public_test") & (labels == "no_drone") & (~professor)
    model_reference: dict[str, Any] = {}
    for model_name, model in models.items():
        scores = np.asarray(
            _score_estimator(model["estimator"], features[test_negative]),
            dtype=np.float64,
        )
        threshold = float(model["threshold"])
        model_reference[model_name] = {
            "rows": int(scores.size),
            "threshold": threshold,
            "false_positive_windows": int(np.count_nonzero(scores >= threshold)),
            "false_positive_rate": float(np.mean(scores >= threshold)),
            "score_median": float(np.median(scores)),
            "score_p95": float(np.quantile(scores, 0.95)),
        }
    return {
        "features": features,
        "labels": labels,
        "splits": splits,
        "professor": professor,
        "original_files": original_files,
        "train_negative_mask": train_negative,
        "train_positive_mask": train_positive,
        "model_reference": model_reference,
        "negative_centroid": features[train_negative].mean(axis=0),
        "positive_centroid": features[train_positive].mean(axis=0),
    }


def _signal_profile(path: Path, samples: np.ndarray, metadata: Any, environment: str) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values * values)))
    peak = float(np.max(np.abs(values)))
    one_second = max(1, metadata.sample_rate_hz)
    window_rms = []
    for start in range(0, max(len(values) - one_second + 1, 1), one_second):
        window = values[start : start + one_second]
        if window.size:
            window_rms.append(float(np.sqrt(np.mean(window * window))))
    exact_zero = values == 0
    longest_zero = _longest_true_run(exact_zero) / metadata.sample_rate_hz
    return {
        "file": path.name,
        "environment": environment,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "sample_rate_hz": metadata.sample_rate_hz,
        "channels": metadata.channels,
        "bits_per_sample": metadata.bits_per_sample,
        "frames": metadata.frames,
        "duration_sec": metadata.duration_sec,
        "windows": _window_count(len(values), metadata.sample_rate_hz),
        "dc_offset": float(np.mean(values)),
        "rms": rms,
        "rms_dbfs": _db(rms),
        "peak_abs": peak,
        "peak_dbfs": _db(peak),
        "crest_factor_db": 20.0 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)),
        "clipped_sample_fraction": float(np.mean(np.abs(values) >= (32767.0 / 32768.0))),
        "exact_zero_fraction": float(np.mean(exact_zero)),
        "longest_zero_run_sec": longest_zero,
        "one_sec_rms_dbfs_p10": _db(float(np.quantile(window_rms, 0.10))),
        "one_sec_rms_dbfs_p50": _db(float(np.quantile(window_rms, 0.50))),
        "one_sec_rms_dbfs_p90": _db(float(np.quantile(window_rms, 0.90))),
        "one_sec_level_spread_db": 20.0
        * math.log10(
            max(float(np.quantile(window_rms, 0.90)), 1e-12)
            / max(float(np.quantile(window_rms, 0.10)), 1e-12)
        ),
    }


def _extract_windows(
    *,
    path: Path,
    samples: np.ndarray,
    sample_rate_hz: int,
    label: str,
    environment: str,
    mode: str,
    processed_before: int,
    total_expected: int,
    started_at: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    window_size = int(round(WINDOW_SECONDS * sample_rate_hz))
    stride_size = int(round(STRIDE_SECONDS * sample_rate_hz))
    rows: list[dict[str, Any]] = []
    vectors: list[tuple[float, ...]] = []
    for local_index, start in enumerate(
        range(0, len(samples) - window_size + 1, stride_size),
        start=0,
    ):
        end = start + window_size
        window = samples[start:end]
        vector, replacements = extract_feature_vector(window, sample_rate_hz, FEATURE_SET)
        rms = float(np.sqrt(np.mean(window * window)))
        rows.append(
            {
                "file": path.name,
                "environment": environment,
                "label": label,
                "mode": mode,
                "sample_rate_hz": sample_rate_hz,
                "window_index": local_index,
                "window_start_sec": start / sample_rate_hz,
                "window_end_sec": end / sample_rate_hz,
                "rms_dbfs": _db(rms),
                "peak_dbfs": _db(float(np.max(np.abs(window)))),
                "clipped_sample_fraction": float(
                    np.mean(np.abs(window) >= (32767.0 / 32768.0))
                ),
                "non_finite_replacements": replacements,
            }
        )
        vectors.append(vector)
        processed = processed_before + len(rows)
        if progress_every > 0 and (
            processed % progress_every == 0 or end + stride_size > len(samples)
        ):
            elapsed = max(time.monotonic() - started_at, 1e-9)
            rate = processed / elapsed
            eta = max(total_expected - processed, 0) / rate if rate > 0 else None
            print(
                "phase=feature_extraction "
                f"processed={processed}/{total_expected} rate={rate:.1f}_windows/s "
                f"elapsed={_duration(elapsed)} eta={_duration(eta)} "
                f"file={path.name} mode={mode}",
                flush=True,
            )
    return rows, np.asarray(vectors, dtype=np.float64)


def _scale(features: np.ndarray, scaler: Mapping[str, np.ndarray]) -> np.ndarray:
    return (features - scaler["mean"]) / scaler["std"]


def _score_rows(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    models: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for name, model in models.items():
        scores = _score_estimator(model["estimator"], features)
        threshold = float(model["threshold"])
        for row, score in zip(output, scores, strict=True):
            row[f"{name}_score"] = float(score)
            row[f"{name}_detected"] = int(float(score) >= threshold)
    return output


def _negative_summary(
    rows: list[dict[str, Any]],
    scaled: np.ndarray,
    models: Mapping[str, Mapping[str, Any]],
    public_reference: Mapping[str, Any],
) -> dict[str, Any]:
    by_environment: dict[str, Any] = {}
    for environment in sorted({str(row["environment"]) for row in rows}):
        selected = [row for row in rows if row["environment"] == environment]
        by_environment[environment] = _score_summary(selected, models)
    negative_distance = np.linalg.norm(scaled - public_reference["negative_centroid"], axis=1)
    positive_distance = np.linalg.norm(scaled - public_reference["positive_centroid"], axis=1)
    return {
        "rows": len(rows),
        "files": len({str(row["file"]) for row in rows}),
        "models": _score_summary(rows, models),
        "recording_level_models": _recording_level_summary(rows, models),
        "by_environment": by_environment,
        "nearest_centroid": {
            "closer_to_public_negative_fraction": float(np.mean(negative_distance < positive_distance)),
            "median_negative_distance": float(np.median(negative_distance)),
            "median_positive_distance": float(np.median(positive_distance)),
        },
        "non_finite_replacements": int(
            sum(int(row["non_finite_replacements"]) for row in rows)
        ),
    }


def _positive_summary(
    rows: list[dict[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    model_summary = _score_summary(rows, models)
    for values in model_summary.values():
        values["recall"] = values.pop("false_positive_rate")
        values["detected_windows"] = values.pop("false_positive_windows")
    return {"rows": len(rows), "models": model_summary}


def _score_summary(
    rows: list[dict[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, model in models.items():
        scores = np.asarray([float(row[f"{name}_score"]) for row in rows])
        detected = scores >= float(model["threshold"])
        report[name] = {
            "threshold": float(model["threshold"]),
            "false_positive_windows": int(np.count_nonzero(detected)),
            "false_positive_rate": float(np.mean(detected)),
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_p95": float(np.quantile(scores, 0.95)),
            "score_max": float(np.max(scores)),
        }
    return report


def _file_model_rows(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    files = sorted({str(row["file"]) for row in rows})
    for file_name in files:
        selected = [row for row in rows if str(row["file"]) == file_name]
        for name, model in models.items():
            detected = np.asarray(
                [int(row[f"{name}_detected"]) for row in selected],
                dtype=np.int64,
            )
            output.append(
                {
                    "mode": str(selected[0]["mode"]),
                    "file": file_name,
                    "environment": str(selected[0]["environment"]),
                    "model": name,
                    "windows": int(detected.size),
                    "false_positive_windows": int(detected.sum()),
                    "false_positive_rate": float(detected.mean()),
                    "threshold": float(model["threshold"]),
                }
            )
    return output


def _recording_level_summary(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize FPR across independent recordings, not correlated windows."""

    file_rows = _file_model_rows(rows, models)
    report: dict[str, Any] = {}
    for model_name in models:
        rates = np.asarray(
            [
                float(row["false_positive_rate"])
                for row in file_rows
                if row["model"] == model_name
            ],
            dtype=np.float64,
        )
        lower, upper = _bootstrap_mean_ci(rates)
        report[model_name] = {
            "recordings": int(rates.size),
            "macro_false_positive_rate": float(rates.mean()),
            "bootstrap_95pct_ci_lower": lower,
            "bootstrap_95pct_ci_upper": upper,
            "recording_fpr_min": float(rates.min()),
            "recording_fpr_max": float(rates.max()),
            "bootstrap_unit": "recording",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
    return report


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty 1D array")
    if values.size == 1:
        value = float(values[0])
        return value, value
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        values.size,
        size=(BOOTSTRAP_REPLICATES, values.size),
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _resampling_sensitivity(
    native_rows: list[dict[str, Any]],
    resampled_rows: list[dict[str, Any]],
    native_features: np.ndarray,
    resampled_features: np.ndarray,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    native_keys = [(row["file"], row["window_index"]) for row in native_rows]
    resampled_keys = [(row["file"], row["window_index"]) for row in resampled_rows]
    if native_keys != resampled_keys:
        raise ValueError("native and resampled window identities differ")
    report: dict[str, Any] = {
        "feature_mean_absolute_difference": float(
            np.mean(np.abs(native_features - resampled_features))
        )
    }
    for name, model in models.items():
        native_scores = np.asarray(_score_estimator(model["estimator"], native_features))
        resampled_scores = np.asarray(_score_estimator(model["estimator"], resampled_features))
        threshold = float(model["threshold"])
        report[name] = {
            "mean_absolute_score_change": float(
                np.mean(np.abs(native_scores - resampled_scores))
            ),
            "decision_flip_fraction": float(
                np.mean((native_scores >= threshold) != (resampled_scores >= threshold))
            ),
            "native_fpr": float(np.mean(native_scores >= threshold)),
            "resampled_fpr": float(np.mean(resampled_scores >= threshold)),
        }
    return report


def _feature_drift(
    mode_scaled: Mapping[str, np.ndarray],
    row_sets: Mapping[str, list[dict[str, Any]]],
    public_reference: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    public_features = public_reference["features"]
    mask = public_reference["train_negative_mask"]
    public_file_means = _group_feature_means(
        public_features[mask],
        public_reference["original_files"][mask],
    )
    mode_reports: dict[str, Any] = {}
    all_top: list[dict[str, Any]] = []
    for mode, features in mode_scaled.items():
        keys = np.asarray([row["file"] for row in row_sets[mode]])
        private_file_means = _group_feature_means(features, keys)
        public_mean = public_file_means.mean(axis=0)
        private_mean = private_file_means.mean(axis=0)
        public_var = public_file_means.var(axis=0, ddof=1)
        private_var = private_file_means.var(axis=0, ddof=1)
        pooled = np.sqrt((public_var + private_var) / 2.0)
        smd = (private_mean - public_mean) / np.where(pooled > 1e-9, pooled, 1.0)
        order = np.argsort(np.abs(smd))[::-1]
        top_rows = [
            {
                "mode": mode,
                "rank": rank,
                "feature": names[index],
                "standardized_mean_difference": float(smd[index]),
                "absolute_smd": float(abs(smd[index])),
                "private_file_mean": float(private_mean[index]),
                "public_negative_file_mean": float(public_mean[index]),
            }
            for rank, index in enumerate(order[:15], start=1)
        ]
        all_top.extend(top_rows)
        mode_reports[mode] = {
            "private_files": int(private_file_means.shape[0]),
            "public_negative_files": int(public_file_means.shape[0]),
            "features_abs_smd_ge_1": int(np.count_nonzero(np.abs(smd) >= 1.0)),
            "features_abs_smd_ge_2": int(np.count_nonzero(np.abs(smd) >= 2.0)),
            "median_absolute_smd": float(np.median(np.abs(smd))),
            "max_absolute_smd": float(np.max(np.abs(smd))),
        }
    return {"by_mode": mode_reports, "top_shifted_features": all_top}


def _group_feature_means(features: np.ndarray, keys: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(keys, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, features.shape[1]), dtype=np.float64)
    counts = np.zeros(sums.shape[0], dtype=np.int64)
    np.add.at(sums, inverse, features)
    np.add.at(counts, inverse, 1)
    return sums / counts[:, None]


def _duplicate_checks(paths: Sequence[Path], bundle_path: Path) -> dict[str, Any]:
    hashes = [_sha256(path) for path in paths]
    duplicate_private = len(hashes) - len(set(hashes))
    with np.load(bundle_path, allow_pickle=False) as data:
        public_files = np.asarray(data["original_files"]).astype(str)
    public_manifest = Path("data/manifests/public_manifest.csv")
    public_hashes: set[str] = set()
    if public_manifest.is_file():
        with public_manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                value = str(row.get("original_file_checksum", "")).strip()
                if value:
                    public_hashes.add(value)
    return {
        "exact_duplicate_private_files": duplicate_private,
        "hash_matches_public_manifest": sum(value in public_hashes for value in hashes),
        "public_bundle_distinct_original_files": int(len(set(public_files))),
    }


def _suspect_windows(
    rows: list[dict[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    limit: int = 50,
) -> list[dict[str, Any]]:
    suspects = []
    for row in rows:
        detections = [int(row[f"{name}_detected"]) for name in models]
        if all(detections):
            candidate = dict(row)
            candidate["consensus_score_min"] = min(
                float(row[f"{name}_score"]) for name in models
            )
            suspects.append(candidate)
    suspects.sort(key=lambda row: float(row["consensus_score_min"]), reverse=True)
    return suspects[:limit]


def _quality_flags(file_profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "files_with_clipping_over_0_1pct": sum(
            float(row["clipped_sample_fraction"]) > 0.001 for row in file_profiles
        ),
        "files_with_dc_offset_over_1pct": sum(
            abs(float(row["dc_offset"])) > 0.01 for row in file_profiles
        ),
        "files_with_zero_dropout_over_100ms": sum(
            float(row["longest_zero_run_sec"]) > 0.1 for row in file_profiles
        ),
        "files_with_median_level_below_minus_50_dbfs": sum(
            float(row["one_sec_rms_dbfs_p50"]) < -50.0 for row in file_profiles
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == TARGET_SAMPLE_RATE:
        return samples
    divisor = math.gcd(source_rate, TARGET_SAMPLE_RATE)
    return resample_poly(
        samples,
        TARGET_SAMPLE_RATE // divisor,
        source_rate // divisor,
    ).astype(np.float64, copy=False)


def _window_count(sample_count: int, sample_rate: int) -> int:
    window = int(round(WINDOW_SECONDS * sample_rate))
    stride = int(round(STRIDE_SECONDS * sample_rate))
    if sample_count < window:
        return 0
    return ((sample_count - window) // stride) + 1


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(path: Path) -> str:
    lowered = path.stem.lower()
    if "traffic" in lowered:
        return "traffic"
    if "forest" in lowered:
        return "forest"
    return "unknown"


def _counts(values: Iterable[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return output


def _db(value: float) -> float:
    if value <= 0:
        return DB_FLOOR
    return max(DB_FLOOR, 20.0 * math.log10(value))


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
