"""Command line interface for POC tooling."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

from drone_sound.dataset_registry import (
    export_license_manifest,
    load_dataset_registry,
    validate_dataset_registry,
)
from drone_sound.nasa_mat import NasaMatConversionError, convert_nasa_mat_directory
from drone_sound.professor_data import audit_professor_zip
from drone_sound.public_data import PublicDatasetError, build_public_manifest
from drone_sound.training import TrainingDataError, train_v1_baseline
from drone_sound.v2 import (
    EMBEDDING_HEAD_MODEL_TYPES,
    ENGINEERED_FEATURE_SETS,
    FEATURE_SETS,
    PRETRAINED_EMBEDDING_MODELS,
    V2A_MODEL_TYPES,
    V2InputError,
    V2NotImplementedError,
    validate_manifest_and_audio_root,
    validate_v2_evaluation_inputs,
)
from drone_sound.v1_evaluation import V1EvaluationError, evaluate_frozen_v1_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drone-sound")
    subcommands = parser.add_subparsers(dest="command", required=True)

    audit_parser = subcommands.add_parser(
        "audit-professor-data",
        help="Audit a professor-provided zip and write manifest plus JSON report.",
    )
    audit_parser.add_argument("--zip", required=True, type=Path, help="Professor data zip path.")
    audit_parser.add_argument(
        "--manifest-out",
        required=True,
        type=Path,
        help="Output CSV manifest path.",
    )
    audit_parser.add_argument(
        "--audit-out",
        required=True,
        type=Path,
        help="Output redacted JSON audit report path.",
    )

    registry_parser = subcommands.add_parser(
        "validate-dataset-registry",
        help="Validate dataset source and license registry JSON.",
    )
    registry_parser.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="Dataset registry JSON path.",
    )

    license_parser = subcommands.add_parser(
        "export-license-manifest",
        help="Export a license manifest CSV from dataset registry JSON.",
    )
    license_parser.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="Dataset registry JSON path.",
    )
    license_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output license manifest CSV path.",
    )

    train_parser = subcommands.add_parser(
        "train-v1",
        help="Train the dependency-free v1 smoke baseline from a manifest and WAV root.",
    )
    train_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Manifest CSV containing train and validation rows.",
    )
    train_parser.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve relative original_file paths.",
    )
    train_parser.add_argument(
        "--model-out",
        required=True,
        type=Path,
        help="Output JSON model path.",
    )
    train_parser.add_argument(
        "--metrics-out",
        required=True,
        type=Path,
        help="Output JSON metrics report path.",
    )

    eval_parser = subcommands.add_parser(
        "eval-v1",
        help="Evaluate a frozen V1 model on public_test and professor_test data.",
    )
    eval_parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Frozen V1 model JSON path.",
    )
    eval_parser.add_argument(
        "--public-manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing public_test rows.",
    )
    eval_parser.add_argument(
        "--public-audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve public manifest original_file paths.",
    )
    eval_parser.add_argument(
        "--professor-manifest",
        required=True,
        type=Path,
        help="Professor manifest CSV containing professor_test rows.",
    )
    eval_parser.add_argument(
        "--professor-audio-root",
        required=True,
        type=Path,
        help="Directory containing extracted professor WAV files.",
    )
    eval_parser.add_argument(
        "--metrics-out",
        required=True,
        type=Path,
        help="Output JSON evaluation metrics report path.",
    )

    public_manifest_parser = subcommands.add_parser(
        "build-public-manifest",
        help="Build a public dataset manifest from local registered WAV folders.",
    )
    public_manifest_parser.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="Dataset registry JSON path.",
    )
    public_manifest_parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="Root containing one folder per dataset_id.",
    )
    public_manifest_parser.add_argument(
        "--manifest-out",
        required=True,
        type=Path,
        help="Output CSV manifest path.",
    )
    public_manifest_parser.add_argument(
        "--audit-out",
        required=True,
        type=Path,
        help="Output public dataset audit JSON path.",
    )
    public_manifest_parser.add_argument(
        "--split-seed",
        required=True,
        type=int,
        help="Deterministic seed for group split assignment.",
    )

    nasa_parser = subcommands.add_parser(
        "convert-nasa-mat",
        help="Convert NASA Small UAS MATLAB audio files to derived mono WAV files.",
    )
    nasa_parser.add_argument(
        "--mat-root",
        required=True,
        type=Path,
        help="Directory containing NASA .mat files.",
    )
    nasa_parser.add_argument(
        "--wav-out",
        required=True,
        type=Path,
        help="Output directory for derived WAV files.",
    )
    nasa_parser.add_argument(
        "--audit-out",
        required=True,
        type=Path,
        help="Output JSON conversion audit path.",
    )
    nasa_parser.add_argument(
        "--sample-rate",
        required=True,
        type=int,
        help="Target WAV sample rate in Hz.",
    )

    extract_v2_parser = subcommands.add_parser(
        "extract-v2-features",
        help="Extract V2 engineered features and write bundle/scaler/audit artifacts.",
    )
    extract_v2_parser.add_argument(
        "--public-manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing train/val/public_test rows.",
    )
    extract_v2_parser.add_argument(
        "--public-audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve public manifest original_file paths.",
    )
    extract_v2_parser.add_argument(
        "--feature-set",
        required=True,
        choices=ENGINEERED_FEATURE_SETS,
        help="Implemented V2 engineered feature set name.",
    )
    extract_v2_parser.add_argument(
        "--features-out",
        required=True,
        type=Path,
        help="Output .npz feature bundle path.",
    )
    extract_v2_parser.add_argument(
        "--scaler-out",
        required=True,
        type=Path,
        help="Output train-fitted scaler JSON path.",
    )
    extract_v2_parser.add_argument(
        "--audit-out",
        required=True,
        type=Path,
        help="Output feature extraction audit JSON path.",
    )
    extract_v2_parser.add_argument(
        "--professor-manifest",
        type=Path,
        help="Optional professor manifest CSV containing professor_test rows.",
    )
    extract_v2_parser.add_argument(
        "--professor-audio-root",
        type=Path,
        help="Optional directory used to resolve professor manifest original_file paths.",
    )
    extract_v2_parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
        help="Print feature extraction progress every N rows; use 0 to disable row progress.",
    )

    sweep_v2_parser = subcommands.add_parser(
        "sweep-v2-ml",
        help="Run the V2A classical ML sweep skeleton.",
    )
    sweep_v2_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing train/val/public_test rows.",
    )
    sweep_v2_parser.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve manifest original_file paths.",
    )
    sweep_v2_parser.add_argument(
        "--feature-set",
        required=True,
        choices=FEATURE_SETS,
        help="V2 feature set name.",
    )
    sweep_v2_parser.add_argument(
        "--model-types",
        required=True,
        nargs="+",
        choices=V2A_MODEL_TYPES,
        help="One or more V2A classical model types to include in the sweep.",
    )
    sweep_v2_parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for future V2A sweep artifacts.",
    )

    mlp_parser = subcommands.add_parser(
        "train-v2-mlp",
        help="Train the V2B Tiny MLP skeleton.",
    )
    mlp_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing train/val/public_test rows.",
    )
    mlp_parser.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve manifest original_file paths.",
    )
    mlp_parser.add_argument(
        "--feature-set",
        required=True,
        choices=FEATURE_SETS,
        help="V2 feature set name.",
    )
    mlp_parser.add_argument(
        "--model-out",
        required=True,
        type=Path,
        help="Output model path for future V2B artifacts.",
    )
    mlp_parser.add_argument(
        "--metrics-out",
        required=True,
        type=Path,
        help="Output metrics path for future V2B artifacts.",
    )

    cnn_parser = subcommands.add_parser(
        "train-v2-cnn",
        help="Train the V2C Tiny CNN skeleton.",
    )
    cnn_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing train/val/public_test rows.",
    )
    cnn_parser.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve manifest original_file paths.",
    )
    cnn_parser.add_argument(
        "--feature-set",
        required=True,
        choices=FEATURE_SETS,
        help="V2 feature set name.",
    )
    cnn_parser.add_argument(
        "--model-out",
        required=True,
        type=Path,
        help="Output model path for future V2C artifacts.",
    )
    cnn_parser.add_argument(
        "--metrics-out",
        required=True,
        type=Path,
        help="Output metrics path for future V2C artifacts.",
    )

    embedding_parser = subcommands.add_parser(
        "train-v2-embedding",
        help="Train the V2D pretrained embedding skeleton.",
    )
    embedding_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing train/val/public_test rows.",
    )
    embedding_parser.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve manifest original_file paths.",
    )
    embedding_parser.add_argument(
        "--embedding-model",
        required=True,
        choices=PRETRAINED_EMBEDDING_MODELS,
        help="Pretrained embedding model name.",
    )
    embedding_parser.add_argument(
        "--head",
        required=True,
        choices=EMBEDDING_HEAD_MODEL_TYPES,
        help="Classifier head model type.",
    )
    embedding_parser.add_argument(
        "--model-out",
        required=True,
        type=Path,
        help="Output model path for future V2D artifacts.",
    )
    embedding_parser.add_argument(
        "--metrics-out",
        required=True,
        type=Path,
        help="Output metrics path for future V2D artifacts.",
    )

    eval_v2_parser = subcommands.add_parser(
        "eval-v2",
        help="Evaluate a frozen V2 model skeleton.",
    )
    eval_v2_parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Frozen V2 model path.",
    )
    eval_v2_parser.add_argument(
        "--public-manifest",
        required=True,
        type=Path,
        help="Public manifest CSV containing public_test rows.",
    )
    eval_v2_parser.add_argument(
        "--public-audio-root",
        required=True,
        type=Path,
        help="Directory used to resolve public manifest original_file paths.",
    )
    eval_v2_parser.add_argument(
        "--professor-manifest",
        required=True,
        type=Path,
        help="Professor manifest CSV containing professor_test rows.",
    )
    eval_v2_parser.add_argument(
        "--professor-audio-root",
        required=True,
        type=Path,
        help="Directory containing extracted professor WAV files.",
    )
    eval_v2_parser.add_argument(
        "--metrics-out",
        required=True,
        type=Path,
        help="Output V2 evaluation metrics report path.",
    )
    eval_v2_parser.add_argument(
        "--export-errors",
        type=Path,
        help="Optional output directory for future V2 error analysis artifacts.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit-professor-data":
        report = audit_professor_zip(args.zip, args.manifest_out, args.audit_out)
        manifest = report["manifest"]
        print(
            "audited professor data: "
            f"{manifest['rows']} windows -> {manifest['path']}"
        )
        return 0

    if args.command == "validate-dataset-registry":
        entries = load_dataset_registry(args.registry)
        validate_dataset_registry(entries)
        print(f"validated dataset registry: {len(entries)} entries")
        return 0

    if args.command == "export-license-manifest":
        entries = load_dataset_registry(args.registry)
        validate_dataset_registry(entries)
        export_license_manifest(entries, args.out)
        print(f"exported license manifest: {len(entries)} entries -> {args.out}")
        return 0

    if args.command == "train-v1":
        try:
            report = train_v1_baseline(
                args.manifest,
                args.audio_root,
                args.model_out,
                args.metrics_out,
            )
        except TrainingDataError as error:
            print(f"training data error: {error}", file=sys.stderr)
            return 1
        val_metrics = report["metrics"]["val"]
        print(
            "trained v1 baseline: "
            f"val_f2={val_metrics['f2']:.6f} "
            f"threshold={report['threshold']:.6f} -> {args.model_out}"
        )
        return 0

    if args.command == "eval-v1":
        try:
            report = evaluate_frozen_v1_model(
                args.model,
                args.public_manifest,
                args.public_audio_root,
                args.professor_manifest,
                args.professor_audio_root,
                args.metrics_out,
            )
        except V1EvaluationError as error:
            print(f"v1 evaluation error: {error}", file=sys.stderr)
            return 1
        public_f2 = report["public_test"]["metrics"]["f2"]
        professor_recall = report["professor_test"]["metrics"]["recall"]
        print(
            "evaluated frozen v1 model: "
            f"public_test_f2={public_f2:.6f} "
            f"professor_recall={professor_recall:.6f} -> {args.metrics_out}"
        )
        return 0

    if args.command == "build-public-manifest":
        try:
            report = build_public_manifest(
                args.registry,
                args.raw_root,
                args.manifest_out,
                args.audit_out,
                args.split_seed,
            )
        except PublicDatasetError as error:
            print(f"public dataset error: {error}", file=sys.stderr)
            return 1
        total_windows = sum(
            dataset["windows"]
            for dataset in report["datasets"]
        )
        print(
            "built public manifest: "
            f"{len(report['datasets'])} datasets, "
            f"{total_windows} windows -> {args.manifest_out}"
        )
        return 0

    if args.command == "convert-nasa-mat":
        try:
            report = convert_nasa_mat_directory(
                args.mat_root,
                args.wav_out,
                args.audit_out,
                args.sample_rate,
            )
        except NasaMatConversionError as error:
            print(f"NASA MAT conversion error: {error}", file=sys.stderr)
            return 1
        print(
            "converted NASA MAT: "
            f"{len(report['files'])} files -> {args.wav_out}"
        )
        return 0

    if args.command == "extract-v2-features":
        from drone_sound.v2_features import V2FeatureError, build_v2_feature_bundle

        if args.progress_every < 0:
            print("v2 input error: progress-every must be >= 0", file=sys.stderr)
            return 1
        try:
            report = build_v2_feature_bundle(
                args.public_manifest,
                args.public_audio_root,
                args.feature_set,
                args.features_out,
                args.scaler_out,
                args.audit_out,
                args.professor_manifest,
                args.professor_audio_root,
                progress_callback=_v2_feature_progress_logger(),
                progress_every=args.progress_every,
            )
        except V2InputError as error:
            print(f"v2 input error: {error}", file=sys.stderr)
            return 1
        except V2FeatureError as error:
            print(f"v2 feature error: {error}", file=sys.stderr)
            return 1
        print(
            "extracted v2 features: "
            f"{report['features']['rows']} rows, "
            f"{report['feature_count']} features -> {args.features_out}"
        )
        return 0

    if args.command == "sweep-v2-ml":
        return _run_v2_skeleton(
            "classical ML sweep",
            lambda: validate_manifest_and_audio_root(args.manifest, args.audio_root),
        )

    if args.command == "train-v2-mlp":
        return _run_v2_skeleton(
            "Tiny MLP training",
            lambda: validate_manifest_and_audio_root(args.manifest, args.audio_root),
        )

    if args.command == "train-v2-cnn":
        return _run_v2_skeleton(
            "Tiny CNN training",
            lambda: validate_manifest_and_audio_root(args.manifest, args.audio_root),
        )

    if args.command == "train-v2-embedding":
        return _run_v2_skeleton(
            "pretrained embedding training",
            lambda: validate_manifest_and_audio_root(args.manifest, args.audio_root),
        )

    if args.command == "eval-v2":
        return _run_v2_skeleton(
            "frozen V2 evaluation",
            lambda: validate_v2_evaluation_inputs(
                args.model,
                args.public_manifest,
                args.public_audio_root,
                args.professor_manifest,
                args.professor_audio_root,
            ),
        )

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_v2_skeleton(capability: str, validate_inputs: Callable[[], None]) -> int:
    try:
        validate_inputs()
        raise V2NotImplementedError(capability)
    except V2InputError as error:
        print(f"v2 input error: {error}", file=sys.stderr)
        return 1
    except V2NotImplementedError as error:
        print(str(error), file=sys.stderr)
        return 1


def _v2_feature_progress_logger() -> Callable[[dict[str, Any]], None]:
    started_at = time.monotonic()

    def log(event: dict[str, Any]) -> None:
        event_name = event.get("event")
        elapsed = max(time.monotonic() - started_at, 0.0)
        if event_name == "start":
            print(
                "v2 feature progress: "
                f"start rows={event['total_rows']} files={event['total_files']} "
                f"feature_set={event['feature_set']}",
                file=sys.stderr,
                flush=True,
            )
            return
        if event_name == "row_progress":
            processed = int(event["processed_rows"])
            total = int(event["total_rows"])
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = max(total - processed, 0)
            eta = remaining / rate if rate > 0 else None
            percent = (processed / total * 100.0) if total else 100.0
            current_file = Path(str(event["current_file"])).name
            print(
                "v2 feature progress: "
                f"{processed}/{total} rows ({percent:.1f}%), "
                f"files={event['file_index']}/{event['total_files']}, "
                f"rate={rate:.1f} rows/s, elapsed={_format_duration(elapsed)}, "
                f"eta={_format_duration(eta)}, file={current_file}, "
                f"non_finite_replacements={event['non_finite_replacements']}",
                file=sys.stderr,
                flush=True,
            )
            return
        if event_name == "scaling":
            print(
                "v2 feature progress: "
                f"scaling {event['rows']} rows x {event['feature_count']} features",
                file=sys.stderr,
                flush=True,
            )
            return
        if event_name == "writing":
            print(
                "v2 feature progress: "
                f"writing artifacts -> {event['features_out']}",
                file=sys.stderr,
                flush=True,
            )

    return log


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remainder:02d}s"
    if minutes:
        return f"{minutes}m{remainder:02d}s"
    return f"{remainder}s"


if __name__ == "__main__":
    raise SystemExit(main())
