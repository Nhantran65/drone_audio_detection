"""Command line interface for POC tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_sound.dataset_registry import (
    export_license_manifest,
    load_dataset_registry,
    validate_dataset_registry,
)
from drone_sound.professor_data import audit_professor_zip


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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
