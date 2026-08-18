"""Lifecycle, validation-only selection, and provenance for the V3 run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from drone_sound.audio_preprocessing import CANONICAL_PREPROCESSING_PROFILE
from drone_sound.v2_run import (
    _artifact_checksums,
    _environment,
    _inventory,
    _sha256_file,
    _utc_now,
    _write_json,
)
from drone_sound.v3_group_split import (
    V3_GROUP_PARSER_VERSION,
    V3_SPLIT_SEED,
    V3_SPLIT_VERSION,
)

ProgressCallback = Callable[[dict[str, Any]], None]
RUN_SUBDIRECTORIES = (
    "manifests",
    "features",
    "cache",
    "models",
    "evaluation",
    "comparison",
)


class V3RunError(ValueError):
    """Raised when V3 isolation, provenance, or selection checks fail."""


def initialize_v3_run(
    run_root: str | Path,
    legacy_root: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    legacy = Path(legacy_root)
    if root.exists():
        raise V3RunError(f"V3 run root already exists; refusing overwrite: {root}")
    if not legacy.is_dir():
        raise V3RunError(f"legacy artifact root does not exist: {legacy}")
    before = _inventory(
        legacy, exclude=root, progress_callback=progress_callback, phase="v3_legacy_before"
    )
    root.mkdir(parents=True)
    for directory in RUN_SUBDIRECTORIES:
        (root / directory).mkdir()
    manifest = {
        "schema_version": "v3_group_split_run_manifest_v1",
        "status": "initialized",
        "created_at_utc": _utc_now(),
        "completed_at_utc": None,
        "run_root": str(root),
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": 16_000,
        "split_version": V3_SPLIT_VERSION,
        "group_parser_version": V3_GROUP_PARSER_VERSION,
        "split_seed": V3_SPLIT_SEED,
        "selection_contract": (
            "validation F1 only; ties use worst validation-dataset F1, lower validation "
            "FPR, then smaller model size; public/professor/private tests excluded"
        ),
        "legacy_artifacts": {
            "root": str(legacy),
            "before": before,
            "after": None,
            "unchanged": None,
        },
        "dataset_hashes": {},
        "environment": _environment(),
        "commands": [],
        "artifact_checksums": {},
    }
    _write_json(root / "run_manifest.json", manifest)
    return manifest


def finalize_v3_run(
    run_root: str | Path,
    legacy_root: str | Path,
    source_manifest: str | Path,
    source_bundle: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise V3RunError(f"V3 run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    after = _inventory(
        Path(legacy_root),
        exclude=root,
        progress_callback=progress_callback,
        phase="v3_legacy_after",
    )
    unchanged = manifest["legacy_artifacts"]["before"]["files"] == after["files"]
    if not unchanged:
        raise V3RunError("legacy artifact SHA-256 inventory changed during V3 run")

    bundle_contract = _bundle_contract(
        Path(source_bundle), root / "features" / "public_professor_features.npz"
    )
    comparison = build_v3_comparison(root)
    _write_json(root / "comparison" / "all_models.json", comparison)
    _write_rows(root / "comparison" / "all_models.csv", comparison["models"])
    manifest["legacy_artifacts"].update({"after": after, "unchanged": True})
    paths = {
        "source_manifest": Path(source_manifest),
        "v3_manifest": root / "manifests" / "public_grouped_manifest.csv",
        "source_bundle": Path(source_bundle),
        "v3_bundle": root / "features" / "public_professor_features.npz",
        "private_manifest": root / "features" / "private_real_manifest.csv",
    }
    manifest["dataset_hashes"] = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
        if path.is_file()
    }
    manifest["bundle_contract"] = bundle_contract
    manifest["selection"] = comparison["selection"]
    manifest["commands"] = _commands(root)
    manifest["artifact_checksums"] = _artifact_checksums(root, exclude=manifest_path)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    return manifest


def build_v3_comparison(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((root / "models").rglob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        validation = report.get("validation", {})
        primary = validation.get("primary", {})
        if not report.get("model_type") or primary.get("f1") is None:
            continue
        per_dataset = validation.get("per_dataset", {})
        source_f1 = {
            name: value.get("metrics", {}).get("f1")
            for name, value in per_dataset.items()
            if value.get("metrics", {}).get("f1") is not None
        }
        model_path = Path(str(report.get("model_path", "")))
        size = report.get("efficiency", {}).get("model_size_bytes")
        if size is None and model_path.is_file():
            size = model_path.stat().st_size
        rows.append(
            {
                "track": report.get("track"),
                "model_type": report.get("model_type"),
                "variant": report.get("training_config", {}).get(
                    "weighting", report.get("head_training", {}).get("weighting")
                ),
                "threshold_name": report.get("threshold_name"),
                "validation_f1": primary.get("f1"),
                "validation_f2": primary.get("f2"),
                "validation_precision": primary.get("precision"),
                "validation_recall": primary.get("recall"),
                "validation_fpr": primary.get("false_positive_rate"),
                "worst_validation_dataset_f1": min(source_f1.values()) if source_f1 else None,
                "model_size_bytes": size,
                "model_path": str(model_path),
                "metrics_path": str(path),
            }
        )
    if not rows:
        raise V3RunError("no completed V3 model metrics found")
    eligible = [row for row in rows if row["threshold_name"] == "threshold_f1"]
    if not eligible:
        raise V3RunError("no V3 model uses a validation-frozen F1 threshold")
    winner = max(
        eligible,
        key=lambda row: (
            float(row["validation_f1"]),
            float(row["worst_validation_dataset_f1"] or -1.0),
            -float(row["validation_fpr"] if row["validation_fpr"] is not None else 1.0),
            -int(row["model_size_bytes"] or 2**63),
        ),
    )
    return {
        "schema_version": "v3_model_comparison_v1",
        "selection_uses_test_data": False,
        "selection_rule": (
            "highest validation F1; ties use worst validation-dataset F1, lower "
            "validation FPR, then smaller model size"
        ),
        "selection": dict(winner),
        "models": rows,
        "public_evaluations": _reports(root / "evaluation" / "public_professor"),
        "private_real_evaluations": _reports(root / "evaluation" / "private_real"),
    }


def _bundle_contract(source: Path, v3: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise V3RunError("V3 bundle verification requires numpy") from error
    if not source.is_file() or not v3.is_file():
        raise V3RunError("source and V3 feature bundles are required")
    with np.load(source, allow_pickle=False) as old, np.load(v3, allow_pickle=False) as new:
        preserved = {
            field: bool(np.array_equal(old[field], new[field]))
            for field in ("window_ids", "labels", "original_files", "raw_features")
        }
        changed_splits = int(np.sum(old["splits"] != new["splits"]))
        group_ids = [str(value) for value in new["group_ids"]]
        splits = [str(value) for value in new["splits"]]
        professor = np.asarray(new["is_professor_data"], dtype=bool)
    if not all(preserved.values()) or changed_splits == 0:
        raise V3RunError("V3 must preserve rows/raw features and change split assignments")
    group_splits: dict[str, set[str]] = {}
    for group, split, is_professor in zip(group_ids, splits, professor):
        if is_professor:
            continue
        group_splits.setdefault(group, set()).add(split)
    overlap = {group: sorted(value) for group, value in group_splits.items() if len(value) > 1}
    if overlap:
        raise V3RunError(f"V3 group overlap detected for {len(overlap)} groups")
    return {
        "preserved": preserved,
        "changed_split_rows": changed_splits,
        "public_groups": len(group_splits),
        "group_overlap_count": 0,
        "valid": True,
    }


def _reports(directory: Path) -> dict[str, Any]:
    output = {}
    if not directory.is_dir():
        return output
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        output[path.stem] = {
            "model_type": data.get("model_type"),
            "track": data.get("track"),
            "threshold_name": data.get("threshold_name"),
            "metrics": data.get("metrics", data.get("public_test", {}).get("metrics")),
            "path": str(path),
        }
    return output


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = (
        "track",
        "model_type",
        "variant",
        "threshold_name",
        "validation_f1",
        "validation_f2",
        "validation_precision",
        "validation_recall",
        "validation_fpr",
        "worst_validation_dataset_f1",
        "model_size_bytes",
        "model_path",
        "metrics_path",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def _commands(root: Path) -> list[str]:
    return [
        f"drone-sound init-v3-run --run-root {root} --legacy-root artifacts",
        "drone-sound build-v3-grouped-manifest ...",
        "drone-sound repartition-v3-features ...",
        "drone-sound rebind-v3-cache (log-mel and YAMNet after exact identity checks)",
        "drone-sound sweep-v2-ml --balance-groups --primary-threshold threshold_f1 ...",
        "drone-sound train-v2-mlp --balance-groups --primary-threshold threshold_f1 ...",
        "drone-sound train-v2-cnn --balance-groups --primary-threshold threshold_f1 ...",
        "drone-sound train-v2-embedding --balance-groups --primary-threshold threshold_f1 ...",
        "drone-sound eval-v3 (all frozen models)",
        "drone-sound eval-real-v2 (all frozen models; no fit/tune)",
        f"drone-sound finalize-v3-run --run-root {root} ...",
    ]
