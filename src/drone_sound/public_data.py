"""Build public dataset manifests from local WAV folders."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from drone_sound.constants import (
    REQUIRED_MANIFEST_FIELDS,
    WINDOW_LENGTH_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from drone_sound.dataset_registry import (
    load_dataset_registry,
    validate_dataset_registry,
)
from drone_sound.manifest import expected_window_count, validate_manifest_rows
from drone_sound.wav_audio import WavReadError, read_wav_metadata

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
PUBLIC_TEST_SPLIT = "public_test"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT, PUBLIC_TEST_SPLIT)


class PublicDatasetError(ValueError):
    """Raised when public dataset folders cannot produce a valid manifest."""


@dataclass(frozen=True)
class PublicAudioFile:
    dataset: Mapping[str, object]
    path: Path
    relative_path: str
    sha256: str
    channels: int
    sample_rate_hz: int
    sample_width_bytes: int
    frames: int
    duration_sec: float
    estimated_windows: int
    label_subtype: str
    split: str


@dataclass(frozen=True)
class SkippedPublicAudioFile:
    dataset_id: str
    relative_path: str
    reason: str


def build_public_manifest(
    registry_path: str | Path,
    raw_root: str | Path,
    manifest_out: str | Path,
    audit_out: str | Path,
    split_seed: int,
) -> dict[str, Any]:
    """Build a public manifest and audit report from local WAV datasets."""

    registry_path = Path(registry_path)
    raw_root = Path(raw_root)
    manifest_out = Path(manifest_out)
    audit_out = Path(audit_out)

    entries = load_dataset_registry(registry_path)
    validate_dataset_registry(entries)

    files_by_stratum: dict[tuple[str, str, str], list[PublicAudioFile]] = defaultdict(list)
    skipped_files: list[SkippedPublicAudioFile] = []
    for entry in entries:
        dataset_files, dataset_skipped_files = _scan_dataset(entry, raw_root)
        skipped_files.extend(dataset_skipped_files)
        for audio_file in dataset_files:
            stratum = (
                _text(entry["dataset_id"]),
                _text(entry["label_binary"]),
                audio_file.label_subtype,
            )
            files_by_stratum[stratum].append(audio_file)

    split_files = _assign_splits(files_by_stratum, split_seed)
    _validate_split_labels(split_files)

    manifest_rows = _manifest_rows(split_files)
    validate_manifest_rows(manifest_rows)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_out, manifest_rows)

    report = _audit_report(
        registry_path=registry_path,
        raw_root=raw_root,
        manifest_out=manifest_out,
        split_seed=split_seed,
        entries=entries,
        files=split_files,
        skipped_files=skipped_files,
    )
    audit_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _scan_dataset(
    entry: Mapping[str, object],
    raw_root: Path,
) -> tuple[list[PublicAudioFile], list[SkippedPublicAudioFile]]:
    dataset_id = _text(entry["dataset_id"])
    dataset_dir = raw_root / dataset_id
    if not dataset_dir.exists():
        raise PublicDatasetError(f"missing dataset directory: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise PublicDatasetError(f"dataset path is not a directory: {dataset_dir}")

    audio_files: list[PublicAudioFile] = []
    skipped_files: list[SkippedPublicAudioFile] = []
    for path in sorted(
        file_path
        for file_path in dataset_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() == ".wav"
    ):
        try:
            audited = _audit_wav(entry, raw_root, path)
        except WavReadError as error:
            skipped_files.append(
                SkippedPublicAudioFile(
                    dataset_id=dataset_id,
                    relative_path=path.relative_to(raw_root).as_posix(),
                    reason=str(error),
                )
            )
            continue
        if audited.estimated_windows > 0:
            audio_files.append(audited)

    if not audio_files:
        raise PublicDatasetError(f"dataset has no usable .wav files with complete windows: {dataset_dir}")
    return audio_files, skipped_files


def _audit_wav(entry: Mapping[str, object], raw_root: Path, path: Path) -> PublicAudioFile:
    metadata = read_wav_metadata(path)
    relative_path = path.relative_to(raw_root).as_posix()
    return PublicAudioFile(
        dataset=entry,
        path=path,
        relative_path=relative_path,
        sha256=_sha256_path(path),
        channels=metadata.channels,
        sample_rate_hz=metadata.sample_rate_hz,
        sample_width_bytes=metadata.sample_width_bytes,
        frames=metadata.frames,
        duration_sec=metadata.duration_sec,
        estimated_windows=expected_window_count(metadata.duration_sec),
        label_subtype=_infer_label_subtype(entry, relative_path),
        split="",
    )


def _infer_label_subtype(entry: Mapping[str, object], relative_path: str) -> str:
    subtypes = _as_list(entry["label_subtype"])
    if len(subtypes) == 1:
        return subtypes[0]

    dataset_id = _text(entry["dataset_id"])
    lowered = relative_path.lower()
    if dataset_id == "kielce_uav_acoustic_signals":
        if "speech" in lowered:
            return "drone_with_speech"
        return "drone_field"
    if dataset_id == "idmt_traffic":
        if "background" in lowered or _has_token(lowered, "bg"):
            return "no_drone_background"
        return "no_drone_traffic"

    return subtypes[0]


def _assign_splits(
    files_by_stratum: Mapping[tuple[str, str, str], list[PublicAudioFile]],
    split_seed: int,
) -> list[PublicAudioFile]:
    assigned: list[PublicAudioFile] = []
    for stratum, files in sorted(files_by_stratum.items()):
        shuffled = list(files)
        random.Random(_stable_seed(split_seed, *stratum)).shuffle(shuffled)
        split_names = _split_names_for_count(len(shuffled))
        for audio_file, split in zip(shuffled, split_names):
            assigned.append(
                PublicAudioFile(
                    dataset=audio_file.dataset,
                    path=audio_file.path,
                    relative_path=audio_file.relative_path,
                    sha256=audio_file.sha256,
                    channels=audio_file.channels,
                    sample_rate_hz=audio_file.sample_rate_hz,
                    sample_width_bytes=audio_file.sample_width_bytes,
                    frames=audio_file.frames,
                    duration_sec=audio_file.duration_sec,
                    estimated_windows=audio_file.estimated_windows,
                    label_subtype=audio_file.label_subtype,
                    split=split,
                )
            )
    return sorted(assigned, key=lambda item: item.relative_path)


def _split_names_for_count(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [TRAIN_SPLIT]
    if count == 2:
        return [TRAIN_SPLIT, VAL_SPLIT]

    val_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    while (val_count + test_count) >= count:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
    train_count = count - val_count - test_count
    if train_count < 1:
        train_count = 1
        if test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1

    return (
        [TRAIN_SPLIT] * train_count
        + [VAL_SPLIT] * val_count
        + [PUBLIC_TEST_SPLIT] * test_count
    )


def _validate_split_labels(files: Iterable[PublicAudioFile]) -> None:
    labels_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for audio_file in files:
        labels_by_split[audio_file.split].add(_text(audio_file.dataset["label_binary"]))

    errors = []
    for split, labels in labels_by_split.items():
        missing = {"drone", "no_drone"} - labels
        if missing:
            errors.append(f"{split} split is missing labels: {', '.join(sorted(missing))}")
    if errors:
        raise PublicDatasetError("; ".join(errors))


def _manifest_rows(files: list[PublicAudioFile]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for audio_file in files:
        rows.extend(_window_rows(audio_file))
    return rows


def _window_rows(audio_file: PublicAudioFile) -> list[dict[str, str]]:
    dataset = audio_file.dataset
    dataset_id = _text(dataset["dataset_id"])
    clip_id = f"{dataset_id}_{_slug(Path(audio_file.relative_path).stem)}_{audio_file.sha256[:8]}"
    source = _text(dataset.get("doi")) or _text(dataset["source_url"])
    notes = (
        "v1 research manifest includes all registered datasets; "
        f"license_status={_text(dataset['license_status'])}; "
        f"training_use={_text(dataset['training_use'])}"
    )

    rows = []
    for index in range(audio_file.estimated_windows):
        start = index * WINDOW_STRIDE_SECONDS
        end = start + WINDOW_LENGTH_SECONDS
        rows.append(
            {
                "clip_id": clip_id,
                "window_id": f"{clip_id}_w{index:06d}",
                "source_dataset": dataset_id,
                "source_url_or_doi": source,
                "license": _text(dataset["license"]),
                "original_file": audio_file.relative_path,
                "original_file_checksum": audio_file.sha256,
                "sample_rate_original": str(audio_file.sample_rate_hz),
                "sample_rate_processed": str(audio_file.sample_rate_hz),
                "duration_sec": f"{audio_file.duration_sec:.6f}",
                "window_start_sec": f"{start:.6f}",
                "window_end_sec": f"{end:.6f}",
                "label_binary": _text(dataset["label_binary"]),
                "label_subtype": audio_file.label_subtype,
                "drone_model": "",
                "environment": "",
                "recording_condition": "",
                "split": audio_file.split,
                "is_professor_data": "false",
                "notes": notes,
            }
        )
    return rows


def _audit_report(
    registry_path: Path,
    raw_root: Path,
    manifest_out: Path,
    split_seed: int,
    entries: list[Mapping[str, object]],
    files: list[PublicAudioFile],
    skipped_files: list[SkippedPublicAudioFile],
) -> dict[str, Any]:
    files_by_dataset: dict[str, list[PublicAudioFile]] = defaultdict(list)
    for audio_file in files:
        files_by_dataset[_text(audio_file.dataset["dataset_id"])].append(audio_file)
    skipped_by_dataset: dict[str, list[SkippedPublicAudioFile]] = defaultdict(list)
    for skipped_file in skipped_files:
        skipped_by_dataset[skipped_file.dataset_id].append(skipped_file)

    return {
        "registry": str(registry_path),
        "raw_root": str(raw_root),
        "manifest": str(manifest_out),
        "split_seed": split_seed,
        "policy": {
            "all_registered_datasets_included": True,
            "license_status_blocks_v1_training": False,
            "professor_data_included": False,
            "split_unit": "original_file",
        },
        "datasets": [
            _dataset_report(
                entry,
                files_by_dataset[_text(entry["dataset_id"])],
                skipped_by_dataset[_text(entry["dataset_id"])],
            )
            for entry in entries
        ],
        "splits": _split_report(files),
    }


def _dataset_report(
    entry: Mapping[str, object],
    files: list[PublicAudioFile],
    skipped_files: list[SkippedPublicAudioFile],
) -> dict[str, Any]:
    return {
        "dataset_id": _text(entry["dataset_id"]),
        "name": _text(entry["name"]),
        "license": _text(entry["license"]),
        "license_status": _text(entry["license_status"]),
        "training_use": _text(entry["training_use"]),
        "evaluation_use": _text(entry["evaluation_use"]),
        "label_binary": _text(entry["label_binary"]),
        "configured_label_subtypes": _as_list(entry["label_subtype"]),
        "files": len(files),
        "skipped_files": len(skipped_files),
        "windows": sum(file.estimated_windows for file in files),
        "splits": _split_report(files),
        "audio_files": [
            {
                "original_file": file.relative_path,
                "sha256": file.sha256,
                "channels": file.channels,
                "sample_rate_hz": file.sample_rate_hz,
                "sample_width_bytes": file.sample_width_bytes,
                "duration_sec": round(file.duration_sec, 6),
                "estimated_windows": file.estimated_windows,
                "label_subtype": file.label_subtype,
                "split": file.split,
            }
            for file in files
        ],
        "skipped_audio_files": [
            {
                "original_file": skipped_file.relative_path,
                "reason": skipped_file.reason,
            }
            for skipped_file in skipped_files
        ],
    }


def _split_report(files: Iterable[PublicAudioFile]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {
        split: {"files": 0, "windows": 0, "labels": {"drone": 0, "no_drone": 0}}
        for split in SPLITS
    }
    for audio_file in files:
        label = _text(audio_file.dataset["label_binary"])
        split_report = report[audio_file.split]
        split_report["files"] += 1
        split_report["windows"] += audio_file.estimated_windows
        split_report["labels"][label] += audio_file.estimated_windows
    return report


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(REQUIRED_MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big")


def _has_token(value: str, token: str) -> bool:
    return token in re.split(r"[^a-z0-9]+", value)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _as_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
