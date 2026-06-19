"""Command line interface for POC tooling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drone_sound.dataset_registry import (
    export_license_manifest,
    load_dataset_registry,
    validate_dataset_registry,
)
from drone_sound.professor_data import audit_professor_zip
from drone_sound.public_data import PublicDatasetError, build_public_manifest
from drone_sound.training import TrainingDataError, train_v1_baseline


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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
