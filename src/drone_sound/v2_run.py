"""Lifecycle, provenance, and comparison helpers for the canonical V2 run."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from drone_sound.audio_preprocessing import CANONICAL_PREPROCESSING_PROFILE
from drone_sound.v2_ml import RANDOM_SEED

ProgressCallback = Callable[[dict[str, Any]], None]
RUN_SUBDIRECTORIES = ("features", "cache", "models", "evaluation", "comparison")


class V2RunError(ValueError):
    """Raised when final-run provenance or artifact isolation fails."""


def initialize_v2_run(
    run_root: str | Path,
    legacy_root: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Fail-fast, inventory legacy files, and create the isolated run namespace."""

    root = Path(run_root)
    legacy = Path(legacy_root)
    if root.exists():
        raise V2RunError(f"final run root already exists; refusing overwrite: {root}")
    if not legacy.is_dir():
        raise V2RunError(f"legacy artifact root does not exist: {legacy}")
    before = _inventory(legacy, exclude=root, progress_callback=progress_callback, phase="legacy_before")
    root.mkdir(parents=True)
    for name in RUN_SUBDIRECTORIES:
        (root / name).mkdir()
    manifest = {
        "schema_version": "v2_16khz_run_manifest_v1",
        "status": "initialized",
        "created_at_utc": _utc_now(),
        "completed_at_utc": None,
        "run_root": str(root),
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": 16_000,
        "seed": RANDOM_SEED,
        "legacy_artifacts": {
            "root": str(legacy),
            "before": before,
            "after": None,
            "unchanged": None,
        },
        "commands": [],
        "dataset_hashes": {},
        "environment": _environment(),
        "artifact_checksums": {},
    }
    _write_json(root / "run_manifest.json", manifest)
    return manifest


def finalize_v2_run(
    run_root: str | Path,
    legacy_root: str | Path,
    public_manifest: str | Path,
    professor_manifest: str | Path,
    legacy_feature_bundle: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Verify preservation, compare generations, and finalize run provenance."""

    root = Path(run_root)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise V2RunError(f"run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    after = _inventory(Path(legacy_root), exclude=root, progress_callback=progress_callback, phase="legacy_after")
    before = manifest["legacy_artifacts"]["before"]
    unchanged = before["files"] == after["files"]
    if not unchanged:
        raise V2RunError("legacy artifact SHA-256 inventory changed during canonical run")

    canonical_bundle = root / "features" / "public_professor_features.npz"
    window_contract = _compare_window_contract(Path(legacy_feature_bundle), canonical_bundle)
    comparison = _build_comparison(root, window_contract)
    comparison_root = root / "comparison"
    _write_json(comparison_root / "all_tracks_and_legacy.json", comparison)
    _write_comparison_csv(comparison_root / "all_tracks_and_legacy.csv", comparison["rows"])

    dataset_paths = {
        "public_manifest": Path(public_manifest),
        "professor_manifest": Path(professor_manifest),
        "private_real_manifest": root / "features" / "private_real_manifest.csv",
    }
    manifest["dataset_hashes"] = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in dataset_paths.items()
    }
    manifest["legacy_artifacts"]["after"] = after
    manifest["legacy_artifacts"]["unchanged"] = unchanged
    manifest["commands"] = _canonical_commands(root)
    manifest["artifact_checksums"] = _artifact_checksums(root, exclude=manifest_path)
    manifest["window_contract"] = window_contract
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    return manifest


def _inventory(
    root: Path,
    *,
    exclude: Path,
    progress_callback: ProgressCallback | None,
    phase: str,
) -> dict[str, Any]:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _is_relative_to(path, exclude)
    ]
    started = time.monotonic()
    values: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(files, start=1):
        values[str(path.relative_to(root))] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if progress_callback is not None:
            elapsed = max(time.monotonic() - started, 0.0)
            rate = index / elapsed if elapsed else 0.0
            progress_callback(
                {
                    "event": "inventory_progress",
                    "phase": phase,
                    "processed_files": index,
                    "total_files": len(files),
                    "current_file": str(path),
                    "rate": rate,
                    "eta_seconds": (len(files) - index) / rate if rate else None,
                }
            )
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"file_count": len(files), "inventory_sha256": digest, "files": values}


def _compare_window_contract(legacy_path: Path, canonical_path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise V2RunError("comparison requires numpy") from error
    if not legacy_path.is_file() or not canonical_path.is_file():
        raise V2RunError("legacy and canonical feature bundles are required for comparison")
    with np.load(legacy_path, allow_pickle=False) as old, np.load(canonical_path, allow_pickle=False) as new:
        fields = ("window_ids", "splits", "labels")
        matches = {field: bool(np.array_equal(old[field], new[field])) for field in fields}
        old_rows = len(old["labels"])
        new_rows = len(new["labels"])
    if not all(matches.values()) or old_rows != new_rows:
        raise V2RunError("legacy and canonical bundles do not share identical rows/splits/labels")
    return {
        "legacy_bundle": str(legacy_path),
        "canonical_bundle": str(canonical_path),
        "rows": new_rows,
        "matches": matches,
        "valid": True,
    }


def _build_comparison(root: Path, window_contract: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_files = sorted((root / "models").rglob("*.json"))
    legacy_metric_files = sorted(Path("artifacts/models").glob("v2*/**/*.json"))
    for generation, paths in (("16khz", metric_files), ("legacy_native", legacy_metric_files)):
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "model_type" not in data or "validation" not in data:
                continue
            validation = data.get("validation", {}).get("primary", {}) or {}
            public = data.get("public_test", {}).get("metrics", {}) or {}
            professor = data.get("professor_test", {}).get("metrics", {}) or {}
            rows.append(
                {
                    "generation": generation,
                    "track": data.get("track"),
                    "model_type": data.get("model_type"),
                    "preprocessing_profile": data.get(
                        "preprocessing_profile",
                        "legacy_native_rate_v1" if generation == "legacy_native" else CANONICAL_PREPROCESSING_PROFILE,
                    ),
                    "validation_f2": validation.get("f2"),
                    "validation_fpr": validation.get("false_positive_rate"),
                    "public_f2": public.get("f2"),
                    "public_fpr": public.get("false_positive_rate"),
                    "professor_recall": professor.get("recall"),
                    "metrics_path": str(path),
                }
            )
    real_only: dict[str, Any] = {}
    for path in sorted((root / "evaluation" / "real_only").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        real_only[path.stem] = {
            "track": data.get("track"),
            "model_type": data.get("model_type"),
            "metrics": data.get("metrics"),
            "per_environment": data.get("per_environment"),
            "per_recording_fpr": data.get("per_recording_fpr"),
            "path": str(path),
        }
    return {
        "schema_version": "v2_16khz_comparison_v1",
        "selection_contract": "validation F2, then validation FPR, then model size; test sets excluded",
        "performance_regression_is_failure": False,
        "window_contract": dict(window_contract),
        "rows": rows,
        "real_only": real_only,
    }


def _canonical_commands(root: Path) -> list[str]:
    return [
        "drone-sound init-v2-16khz-run --run-root artifacts/v2_16khz --legacy-root artifacts",
        "drone-sound build-private-real-manifest ...",
        "drone-sound extract-v2-features --target-sample-rate 16000 ...",
        "drone-sound extract-v2-eval-features ...",
        "drone-sound sweep-v2-ml --model-types logistic_regression linear_svm random_forest extra_trees ...",
        "drone-sound train-v2-mlp --epochs 20 ...",
        "drone-sound train-v2-cnn --epochs 20 ...",
        "drone-sound train-v2-embedding --embedding-model yamnet --head logistic_regression ...",
        "drone-sound eval-real-v2 (all frozen V2A/B/C/D models) ...",
        f"drone-sound finalize-v2-16khz-run --run-root {root} ...",
    ]


def _artifact_checksums(root: Path, *, exclude: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == exclude:
            continue
        output[str(path.relative_to(root))] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return output


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "torchaudio", "tensorflow", "tensorflow-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "packages": packages,
    }


def _write_comparison_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = (
        "generation",
        "track",
        "model_type",
        "preprocessing_profile",
        "validation_f2",
        "validation_fpr",
        "public_f2",
        "public_fpr",
        "professor_recall",
        "metrics_path",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
