"""Command line interface for POC tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
