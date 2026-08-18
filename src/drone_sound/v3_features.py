"""Repartition canonical 16-kHz features under the V3 group-aware split."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from drone_sound.audio_preprocessing import CANONICAL_PREPROCESSING_PROFILE
from drone_sound.constants import TARGET_SAMPLE_RATE_HZ
from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.v2_features import apply_scaler, fit_scaler
from drone_sound.v3_group_split import V3_GROUP_PARSER_VERSION, V3_SPLIT_VERSION

ProgressCallback = Callable[[dict[str, Any]], None]


class V3FeatureError(ValueError):
    """Raised when canonical features cannot be safely repartitioned."""


def repartition_v3_features(
    source_bundle_path: str | Path,
    grouped_manifest_path: str | Path,
    features_out: str | Path,
    scaler_out: str | Path,
    audit_out: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Reuse identical raw features, refit the scaler, and write a V3 bundle."""

    np = _require_numpy()
    source_path = Path(source_bundle_path)
    manifest_path = Path(grouped_manifest_path)
    output_path = Path(features_out)
    scaler_path = Path(scaler_out)
    audit_path = Path(audit_out)
    _require_outputs_absent(output_path, scaler_path, audit_path)
    if not source_path.is_file():
        raise V3FeatureError(f"source feature bundle does not exist: {source_path}")
    if not manifest_path.is_file():
        raise V3FeatureError(f"V3 grouped manifest does not exist: {manifest_path}")

    manifest_rows = load_manifest_csv(manifest_path)
    validate_manifest_rows(manifest_rows)
    by_window: dict[str, Mapping[str, str]] = {}
    for row in manifest_rows:
        window_id = str(row["window_id"])
        if not row.get("group_id"):
            raise V3FeatureError(f"V3 manifest row lacks group_id: {window_id}")
        if window_id in by_window:
            raise V3FeatureError(f"duplicate V3 manifest window_id: {window_id}")
        by_window[window_id] = row

    if progress_callback is not None:
        progress_callback(
            {
                "event": "v3_repartition_start",
                "source_bundle": str(source_path),
                "manifest_rows": len(manifest_rows),
            }
        )
    try:
        with np.load(source_path, allow_pickle=False) as source:
            required = (
                "raw_features",
                "labels",
                "splits",
                "window_ids",
                "clip_ids",
                "original_files",
                "source_datasets",
                "label_subtypes",
                "environments",
                "window_start_sec",
                "window_end_sec",
                "is_professor_data",
                "source_sample_rates_hz",
                "processed_sample_rates_hz",
                "feature_names",
                "feature_set",
                "preprocessing_profile",
                "processed_sample_rate_hz",
                "bundle_fingerprint",
            )
            missing = [key for key in required if key not in source.files]
            if missing:
                raise V3FeatureError(
                    "source feature bundle lacks canonical arrays: " + ", ".join(missing)
                )
            profile = str(source["preprocessing_profile"].item())
            rate = int(source["processed_sample_rate_hz"].item())
            if profile != CANONICAL_PREPROCESSING_PROFILE or rate != TARGET_SAMPLE_RATE_HZ:
                raise V3FeatureError("V3 repartition requires a canonical 16-kHz source bundle")
            arrays = {key: np.asarray(source[key]) for key in source.files}
    except V3FeatureError:
        raise
    except Exception as error:
        raise V3FeatureError(f"invalid source feature bundle: {source_path}") from error

    window_ids = [str(value) for value in arrays["window_ids"]]
    professor_flags = np.asarray(arrays["is_professor_data"], dtype=bool)
    splits: list[str] = []
    group_ids: list[str] = []
    matched_public = set()
    for index, window_id in enumerate(window_ids):
        row = by_window.get(window_id)
        if professor_flags[index]:
            if row is not None:
                raise V3FeatureError("public V3 manifest unexpectedly contains professor rows")
            splits.append(str(arrays["splits"][index]))
            group_ids.append(f"professor:{str(arrays['clip_ids'][index])}")
            continue
        if row is None:
            raise V3FeatureError(f"public source bundle row is missing from V3 manifest: {window_id}")
        if str(arrays["labels"][index]) != str(row["label_binary"]):
            raise V3FeatureError(f"label mismatch for V3 window: {window_id}")
        splits.append(str(row["split"]))
        group_ids.append(str(row["group_id"]))
        matched_public.add(window_id)
    if matched_public != set(by_window):
        missing = len(set(by_window) - matched_public)
        raise V3FeatureError(f"V3 manifest has {missing} windows absent from source bundle")

    raw_features = np.asarray(arrays["raw_features"], dtype=np.float64)
    split_array = np.asarray(splits)
    train_mask = (split_array == "train") & (~professor_flags)
    if not train_mask.any():
        raise V3FeatureError("V3 feature bundle has no public train rows")
    scaler = fit_scaler(raw_features[train_mask])
    scaled_features = apply_scaler(raw_features, scaler)
    fingerprint = _fingerprint(
        raw_features,
        window_ids,
        [str(value) for value in arrays["labels"]],
        splits,
        group_ids,
        str(arrays["feature_set"].item()),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "v3_repartition_write",
                "rows": len(window_ids),
                "feature_count": int(raw_features.shape[1]),
                "features_out": str(output_path),
            }
        )
    passthrough = {
        key: value
        for key, value in arrays.items()
        if key
        not in {
            "scaled_features",
            "splits",
            "bundle_fingerprint",
            "bundle_schema_version",
            "group_ids",
            "split_version",
            "group_parser_version",
        }
    }
    np.savez_compressed(
        output_path,
        **passthrough,
        scaled_features=scaled_features,
        splits=split_array,
        group_ids=np.asarray(group_ids),
        bundle_schema_version=np.asarray("v3_grouped_feature_bundle_v1"),
        split_version=np.asarray(V3_SPLIT_VERSION),
        group_parser_version=np.asarray(V3_GROUP_PARSER_VERSION),
        bundle_fingerprint=np.asarray(fingerprint),
    )

    scaler_report = {
        "schema_version": "v3_grouped_scaler_v1",
        "feature_set": str(arrays["feature_set"].item()),
        "feature_names": [str(value) for value in arrays["feature_names"]],
        "fitted_split": "train",
        "train_rows": int(train_mask.sum()),
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "split_version": V3_SPLIT_VERSION,
        "group_parser_version": V3_GROUP_PARSER_VERSION,
        "source_bundle": str(source_path),
        "source_bundle_fingerprint": str(arrays["bundle_fingerprint"].item()),
        "bundle_fingerprint": fingerprint,
        "mean": scaler["mean"],
        "std": scaler["std"],
    }
    scaler_path.write_text(
        json.dumps(scaler_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "v3_feature_repartition_audit_v1",
        "source_bundle": str(source_path),
        "source_bundle_sha256": _sha256_file(source_path),
        "source_bundle_fingerprint": str(arrays["bundle_fingerprint"].item()),
        "grouped_manifest": str(manifest_path),
        "features": str(output_path),
        "scaler": str(scaler_path),
        "bundle_fingerprint": fingerprint,
        "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
        "processed_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "split_version": V3_SPLIT_VERSION,
        "group_parser_version": V3_GROUP_PARSER_VERSION,
        "raw_feature_values_reused": True,
        "feature_engineering_recomputed": False,
        "scaler_refit_on_v3_train_only": True,
        "counts": _counts(
            splits,
            [str(value) for value in arrays["labels"]],
            [str(value) for value in arrays["source_datasets"]],
            group_ids,
            professor_flags,
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "v3_repartition_complete",
                "rows": len(window_ids),
                "train_rows": int(train_mask.sum()),
                "bundle_fingerprint": fingerprint,
            }
        )
    return audit


def _fingerprint(
    raw_features: Any,
    window_ids: list[str],
    labels: list[str],
    splits: list[str],
    group_ids: list[str],
    feature_set: str,
) -> str:
    np = _require_numpy()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": "v3_grouped_feature_bundle_v1",
                "feature_set": feature_set,
                "preprocessing_profile": CANONICAL_PREPROCESSING_PROFILE,
                "processed_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                "split_version": V3_SPLIT_VERSION,
                "group_parser_version": V3_GROUP_PARSER_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for values in zip(window_ids, labels, splits, group_ids):
        digest.update("\0".join(values).encode())
        digest.update(b"\n")
    matrix = np.ascontiguousarray(raw_features, dtype="<f8")
    digest.update(memoryview(matrix).cast("B"))
    return digest.hexdigest()


def _counts(
    splits: list[str],
    labels: list[str],
    datasets: list[str],
    group_ids: list[str],
    professor_flags: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {"rows": len(labels), "splits": {}}
    values: dict[str, Any] = defaultdict(
        lambda: {"rows": 0, "labels": Counter(), "datasets": Counter(), "groups": set()}
    )
    for split, label, dataset, group_id, is_professor in zip(
        splits, labels, datasets, group_ids, professor_flags
    ):
        key = "professor_test" if bool(is_professor) else split
        values[key]["rows"] += 1
        values[key]["labels"][label] += 1
        values[key]["datasets"][dataset] += 1
        values[key]["groups"].add(group_id)
    output["splits"] = {
        split: {
            "rows": value["rows"],
            "labels": dict(sorted(value["labels"].items())),
            "datasets": dict(sorted(value["datasets"].items())),
            "groups": len(value["groups"]),
        }
        for split, value in sorted(values.items())
    }
    return output


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise V3FeatureError("V3 feature repartition requires numpy") from error
    return np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise V3FeatureError(f"refusing to overwrite V3 output(s): {', '.join(existing)}")
