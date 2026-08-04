#!/usr/bin/env python3
"""Evaluate every frozen legacy-native V2 model on the private real-only set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drone_sound.audio_preprocessing import LEGACY_PREPROCESSING_PROFILE
from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL
from drone_sound.v2_features import apply_scaler, extract_feature_vector, feature_names
from drone_sound.v2_real_eval import evaluate_real_v2
from drone_sound.wav_audio import read_wav_mono_array


FEATURE_SET = "spectral_mfcc_v1"
EXPECTED_POSITIVES = 1_113
EXPECTED_NEGATIVES = 592
EXPECTED_RECORDINGS = 11
MODEL_SPECS = (
    ("logistic_regression", Path("artifacts/models/v2a_sweep/models/logistic_regression.joblib")),
    ("linear_svm", Path("artifacts/models/v2a_sweep/models/linear_svm.joblib")),
    ("random_forest", Path("artifacts/models/v2a_sweep/models/random_forest.joblib")),
    ("extra_trees", Path("artifacts/models/v2a_sweep/models/extra_trees.joblib")),
    ("tiny_mlp", Path("artifacts/models/v2b_mlp/tiny_mlp.joblib")),
    ("tiny_mlp_es", Path("artifacts/models/v2b_mlp/tiny_mlp_es.joblib")),
    ("tiny_cnn", Path("artifacts/models/v2c_cnn/tiny_cnn.pt")),
    ("tiny_cnn_es", Path("artifacts/models/v2c_cnn/tiny_cnn_es.pt")),
    ("yamnet_head", Path("artifacts/models/v2d_yamnet/yamnet_head.joblib")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/v2_16khz/features/private_real_manifest.csv"),
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/private"))
    parser.add_argument(
        "--scaler",
        type=Path,
        default=Path("artifacts/features/v2_spectral_mfcc_scaler.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/evaluation/private_real_legacy_native_all"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    _validate_inputs(args)
    args.output_root.mkdir(parents=True)
    (args.output_root / "metrics").mkdir()
    (args.output_root / "errors").mkdir()
    (args.output_root / "cache").mkdir()

    bundle_path = args.output_root / "legacy_native_features.npz"
    audit_path = args.output_root / "legacy_native_features_audit.json"
    print("legacy real-only: phase=feature_bundle start", flush=True)
    audit = _build_legacy_bundle(
        args.manifest,
        args.audio_root,
        args.scaler,
        bundle_path,
        audit_path,
        args.progress_every,
    )
    print(
        "legacy real-only: phase=feature_bundle complete "
        f"rows={audit['counts']['rows']} rates={audit['processed_sample_rates_hz']} "
        f"elapsed={_elapsed(started)}",
        flush=True,
    )

    reports: list[dict[str, Any]] = []
    for index, (artifact_name, model_path) in enumerate(MODEL_SPECS, start=1):
        print(
            "legacy real-only: phase=evaluation start "
            f"model={artifact_name} model_index={index}/{len(MODEL_SPECS)} "
            f"elapsed={_elapsed(started)}",
            flush=True,
        )
        cache_name = "logmel" if model_path.suffix == ".pt" else (
            "yamnet" if artifact_name == "yamnet_head" else "engineered"
        )
        report = evaluate_real_v2(
            model_path,
            bundle_path,
            args.audio_root,
            args.output_root / "cache" / cache_name,
            args.output_root / "metrics" / f"{artifact_name}.json",
            args.output_root / "errors" / artifact_name,
            progress_callback=_progress_logger(artifact_name, started),
            batch_size=args.batch_size,
        )
        _reconcile_report(report)
        reports.append(
            {
                "artifact_name": artifact_name,
                "model_path": str(model_path),
                "model_size_bytes": model_path.stat().st_size,
                **report,
            }
        )
        metrics = report["metrics"]
        print(
            "legacy real-only: phase=evaluation complete "
            f"model={artifact_name} f2={metrics['f2']:.6f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f} "
            f"fpr={metrics['false_positive_rate']:.6f} elapsed={_elapsed(started)}",
            flush=True,
        )

    summary = _summary(args, bundle_path, audit_path, reports)
    summary_path = args.output_root / "summary.json"
    summary_csv_path = args.output_root / "summary.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_csv(summary_csv_path, reports)
    print(
        "legacy real-only: phase=complete "
        f"models={len(reports)} output={args.output_root} elapsed={_elapsed(started)}",
        flush=True,
    )
    return 0


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {args.output_root}")
    if args.batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("batch size and progress interval must be positive")
    for path in (args.manifest, args.scaler, args.audio_root):
        if not path.exists():
            raise ValueError(f"required input does not exist: {path}")
    for _, path in MODEL_SPECS:
        if not path.is_file():
            raise ValueError(f"legacy model does not exist: {path}")


def _build_legacy_bundle(
    manifest_path: Path,
    audio_root: Path,
    scaler_path: Path,
    bundle_path: Path,
    audit_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    rows = load_manifest_csv(manifest_path)
    validate_manifest_rows(rows)
    rows = [row for row in rows if row["split"] == "professor_test"]
    labels = [row["label_binary"] for row in rows]
    if labels.count(POSITIVE_LABEL) != EXPECTED_POSITIVES:
        raise ValueError("private real-only manifest does not have 1113 positive windows")
    if labels.count(NEGATIVE_LABEL) != EXPECTED_NEGATIVES:
        raise ValueError("private real-only manifest does not have 592 negative windows")
    if len({row["original_file"] for row in rows}) != EXPECTED_RECORDINGS:
        raise ValueError("private real-only manifest does not have 11 recordings")

    names = feature_names(FEATURE_SET)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    if scaler.get("feature_set") != FEATURE_SET:
        raise ValueError("legacy scaler feature_set mismatch")
    if tuple(scaler.get("feature_names", ())) != names:
        raise ValueError("legacy scaler feature names mismatch")

    raw_features = np.empty((len(rows), len(names)), dtype=np.float64)
    source_rates = np.empty(len(rows), dtype=np.int32)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[row["original_file"]].append(row_index)

    processed = 0
    replacements = 0
    started = time.monotonic()
    for file_index, (original_file, indices) in enumerate(grouped.items(), start=1):
        audio_path = _resolve_audio_path(audio_root, original_file)
        samples, metadata = read_wav_mono_array(audio_path)
        for row_index in indices:
            row = rows[row_index]
            start_sec = float(row["window_start_sec"])
            end_sec = float(row["window_end_sec"])
            start_frame = int(round(start_sec * metadata.sample_rate_hz))
            frame_count = int(round((end_sec - start_sec) * metadata.sample_rate_hz))
            end_frame = start_frame + frame_count
            if start_frame < 0 or frame_count <= 0 or end_frame > len(samples):
                raise ValueError(f"invalid manifest window for {audio_path}: {start_sec}..{end_sec}")
            vector, replaced = extract_feature_vector(
                samples[start_frame:end_frame], metadata.sample_rate_hz, FEATURE_SET
            )
            raw_features[row_index] = vector
            source_rates[row_index] = metadata.sample_rate_hz
            replacements += replaced
            processed += 1
            if processed == len(rows) or processed % progress_every == 0:
                elapsed = time.monotonic() - started
                rate = processed / elapsed if elapsed else 0.0
                eta = (len(rows) - processed) / rate if rate else math.inf
                print(
                    "legacy real-only: phase=feature_extraction "
                    f"rows={processed}/{len(rows)} files={file_index}/{len(grouped)} "
                    f"rate={rate:.1f}rows/s elapsed={_duration(elapsed)} "
                    f"eta={_duration(eta)} file={audio_path.name}",
                    flush=True,
                )

    scaled_features = apply_scaler(raw_features, scaler)
    np.savez_compressed(
        bundle_path,
        raw_features=raw_features,
        scaled_features=scaled_features,
        labels=np.asarray(labels),
        splits=np.asarray([row["split"] for row in rows]),
        window_ids=np.asarray([row["window_id"] for row in rows]),
        clip_ids=np.asarray([row["clip_id"] for row in rows]),
        original_files=np.asarray([row["original_file"] for row in rows]),
        source_datasets=np.asarray([row["source_dataset"] for row in rows]),
        label_subtypes=np.asarray([row["label_subtype"] for row in rows]),
        environments=np.asarray([row.get("environment", "") for row in rows]),
        window_start_sec=np.asarray([float(row["window_start_sec"]) for row in rows]),
        window_end_sec=np.asarray([float(row["window_end_sec"]) for row in rows]),
        is_professor_data=np.ones(len(rows), dtype=bool),
        source_sample_rates_hz=source_rates,
        processed_sample_rates_hz=source_rates,
        feature_names=np.asarray(names),
        feature_set=np.asarray(FEATURE_SET),
        preprocessing_profile=np.asarray(LEGACY_PREPROCESSING_PROFILE),
    )
    audit = {
        "schema_version": "legacy_native_real_only_feature_audit_v1",
        "evaluation_only": True,
        "fit_called": False,
        "threshold_tuned": False,
        "manifest": str(manifest_path),
        "audio_root": str(audio_root),
        "frozen_scaler": str(scaler_path),
        "feature_bundle": str(bundle_path),
        "feature_set": FEATURE_SET,
        "feature_count": len(names),
        "preprocessing_profile": LEGACY_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": None,
        "processed_sample_rates_hz": sorted({int(value) for value in source_rates}),
        "non_finite_replacements": replacements,
        "counts": {
            "rows": len(rows),
            POSITIVE_LABEL: labels.count(POSITIVE_LABEL),
            NEGATIVE_LABEL: labels.count(NEGATIVE_LABEL),
            "recordings": len(grouped),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _resolve_audio_path(root: Path, original_file: str) -> Path:
    relative = Path(original_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe original_file path: {original_file}")
    candidate = root / relative
    if not candidate.is_file():
        raise ValueError(f"audio file does not exist: {candidate}")
    return candidate


def _progress_logger(model_name: str, started: float):
    def log(event: Mapping[str, Any]) -> None:
        event_name = event.get("event", "progress")
        fields = " ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key != "event" and key not in {"model"}
        )
        print(
            f"legacy real-only: phase={event_name} model={model_name} {fields} "
            f"elapsed={_elapsed(started)}",
            flush=True,
        )

    return log


def _reconcile_report(report: Mapping[str, Any]) -> None:
    if report.get("fit_called") is not False or report.get("threshold_tuned") is not False:
        raise ValueError("real-only evaluator unexpectedly fit or tuned a model")
    metrics = report["metrics"]
    confusion = metrics["confusion_matrix"]
    tp, fn = int(confusion["tp"]), int(confusion["fn"])
    tn, fp = int(confusion["tn"]), int(confusion["fp"])
    if tp + fn != EXPECTED_POSITIVES or tn + fp != EXPECTED_NEGATIVES:
        raise ValueError("confusion matrix does not reconcile to real-only labels")
    expected_recall = tp / (tp + fn)
    expected_fpr = fp / (tn + fp)
    if not math.isclose(float(metrics["recall"]), expected_recall, abs_tol=1e-12):
        raise ValueError("reported recall does not reconcile to confusion matrix")
    if not math.isclose(float(metrics["false_positive_rate"]), expected_fpr, abs_tol=1e-12):
        raise ValueError("reported FPR does not reconcile to confusion matrix")


def _summary(
    args: argparse.Namespace,
    bundle_path: Path,
    audit_path: Path,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(
        reports,
        key=lambda report: (
            -float(report["metrics"]["f2"]),
            float(report["metrics"]["false_positive_rate"]),
        ),
    )
    return {
        "schema_version": "legacy_native_real_only_all_models_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_only": True,
        "fit_called": False,
        "threshold_tuned": False,
        "selection_performed": False,
        "preprocessing_profile": LEGACY_PREPROCESSING_PROFILE,
        "feature_bundle": str(bundle_path),
        "feature_audit": str(audit_path),
        "population": {
            POSITIVE_LABEL: EXPECTED_POSITIVES,
            NEGATIVE_LABEL: EXPECTED_NEGATIVES,
            "positive_recordings": 1,
            "negative_recordings": 10,
        },
        "ranking_is_descriptive_only": True,
        "rows": [
            {
                "rank_by_f2": index,
                "artifact_name": report["artifact_name"],
                "model_type": report["model_type"],
                "track": report["track"],
                "model_path": report["model_path"],
                "model_size_bytes": report["model_size_bytes"],
                "threshold": report["threshold"],
                "metrics": report["metrics"],
                "per_environment": report["per_environment"],
                "per_recording_fpr": report["per_recording_fpr"],
            }
            for index, report in enumerate(ranked, start=1)
        ],
        "limitations": [
            "The test contains one independent positive and ten independent negative recordings.",
            "Overlapping windows are correlated and are not independent observations.",
            "Native sample rate is confounded with label: positive audio is 16 kHz and negative audio is 48 kHz.",
            "The ranking is post-hoc reporting only and must not be used to tune models or thresholds.",
        ],
    }


def _write_summary_csv(path: Path, reports: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "artifact_name",
        "track",
        "model_type",
        "model_size_bytes",
        "threshold",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
        "false_negative_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    )
    ranked = sorted(reports, key=lambda report: -float(report["metrics"]["f2"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in ranked:
            metrics = report["metrics"]
            confusion = metrics["confusion_matrix"]
            writer.writerow(
                {
                    "artifact_name": report["artifact_name"],
                    "track": report["track"],
                    "model_type": report["model_type"],
                    "model_size_bytes": report["model_size_bytes"],
                    "threshold": report["threshold"],
                    **{field: metrics[field] for field in fields[5:14]},
                    **{field: confusion[field] for field in fields[14:]},
                }
            )


def _elapsed(started: float) -> str:
    return _duration(time.monotonic() - started)


def _duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    total = max(0, int(round(seconds)))
    minutes, remainder = divmod(total, 60)
    return f"{minutes}m{remainder:02d}s" if minutes else f"{remainder}s"


if __name__ == "__main__":
    raise SystemExit(main())
