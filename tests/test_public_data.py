from __future__ import annotations

import csv
import json
import math
import wave
from collections import defaultdict
from pathlib import Path

import pytest

from drone_sound.cli import main
from drone_sound.dataset_registry import load_dataset_registry
from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.public_data import PublicDatasetError, build_public_manifest


def test_build_public_manifest_includes_all_registered_datasets(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "public"
    _write_all_registered_wavs(raw_root)
    _write_unsupported_wav(raw_root / "dronenoise_database" / "unsupported_adpcm.wav")
    manifest_out = tmp_path / "data" / "manifests" / "public_manifest.csv"
    audit_out = tmp_path / "artifacts" / "audits" / "public_data_audit.json"

    report = build_public_manifest(
        "configs/datasets.json",
        raw_root,
        manifest_out,
        audit_out,
        split_seed=20260619,
    )

    rows = load_manifest_csv(manifest_out)
    validate_manifest_rows(rows)
    audit = json.loads(audit_out.read_text(encoding="utf-8"))

    expected_ids = {entry["dataset_id"] for entry in load_dataset_registry("configs/datasets.json")}
    assert {row["source_dataset"] for row in rows} == expected_ids
    assert {dataset["dataset_id"] for dataset in audit["datasets"]} == expected_ids
    assert report["policy"]["license_status_blocks_v1_training"] is False
    assert audit["policy"]["all_registered_datasets_included"] is True
    dronenoise_report = next(
        dataset for dataset in audit["datasets"]
        if dataset["dataset_id"] == "dronenoise_database"
    )
    assert dronenoise_report["skipped_files"] == 1
    assert "unsupported" in dronenoise_report["skipped_audio_files"][0]["reason"]

    statuses = {dataset["license_status"] for dataset in audit["datasets"]}
    assert "license_unspecified" in statuses
    assert "non_commercial" in statuses

    labels_by_split = _labels_by_split(rows)
    assert labels_by_split["train"] == {"drone", "no_drone"}
    assert labels_by_split["val"] == {"drone", "no_drone"}
    assert labels_by_split["public_test"] == {"drone", "no_drone"}

    splits_by_original_file: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_original_file[row["original_file"]].add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_original_file.values())

    assert any(
        row["source_dataset"] == "kielce_uav_acoustic_signals"
        and "speech" in row["original_file"]
        and row["label_subtype"] == "drone_with_speech"
        for row in rows
    )
    assert any(
        row["source_dataset"] == "idmt_traffic"
        and "/bg/" in row["original_file"]
        and row["label_subtype"] == "no_drone_background"
        for row in rows
    )


def test_build_public_manifest_cli_can_feed_train_v1(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "public"
    _write_all_registered_wavs(raw_root)
    manifest_out = tmp_path / "data" / "manifests" / "public_manifest.csv"
    audit_out = tmp_path / "artifacts" / "audits" / "public_data_audit.json"
    model_out = tmp_path / "artifacts" / "models" / "v1_baseline_model.json"
    metrics_out = tmp_path / "artifacts" / "models" / "v1_metrics.json"

    build_exit = main(
        [
            "build-public-manifest",
            "--registry",
            "configs/datasets.json",
            "--raw-root",
            str(raw_root),
            "--manifest-out",
            str(manifest_out),
            "--audit-out",
            str(audit_out),
            "--split-seed",
            "20260619",
        ]
    )
    train_exit = main(
        [
            "train-v1",
            "--manifest",
            str(manifest_out),
            "--audio-root",
            str(raw_root),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
        ]
    )

    assert build_exit == 0
    assert train_exit == 0
    assert model_out.exists()
    assert metrics_out.exists()

    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert metrics["counts"]["public_test"]["drone"] > 0
    assert metrics["counts"]["public_test"]["no_drone"] > 0


def test_build_public_manifest_fails_for_missing_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(PublicDatasetError, match="missing dataset directory"):
        build_public_manifest(
            "configs/datasets.json",
            tmp_path / "raw" / "public",
            tmp_path / "manifest.csv",
            tmp_path / "audit.json",
            split_seed=20260619,
        )


def _write_all_registered_wavs(raw_root: Path) -> None:
    entries = load_dataset_registry("configs/datasets.json")
    for entry in entries:
        dataset_id = str(entry["dataset_id"])
        label = str(entry["label_binary"])
        amplitude = 0.8 if label == "drone" else 0.1
        frequency = 440.0 if label == "drone" else 220.0

        for relative_path in _relative_wav_paths(entry):
            path = raw_root / dataset_id / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(path, amplitude=amplitude, frequency_hz=frequency)


def _relative_wav_paths(entry: dict[str, object]) -> list[Path]:
    dataset_id = str(entry["dataset_id"])
    if dataset_id == "kielce_uav_acoustic_signals":
        return [
            *(Path("field") / f"field_{index}.wav" for index in range(3)),
            *(Path("speech") / f"speech_{index}.wav" for index in range(3)),
        ]
    if dataset_id == "idmt_traffic":
        return [
            *(Path("traffic") / f"traffic_{index}.wav" for index in range(3)),
            *(Path("bg") / f"bg_{index}.wav" for index in range(3)),
        ]
    if dataset_id == "nasa_small_uas_flyover":
        return [Path(f"sample_{index}.WAV") for index in range(3)]
    return [Path(f"sample_{index}.wav") for index in range(3)]


def _write_wav(
    path: Path,
    *,
    amplitude: float,
    frequency_hz: float,
    sample_rate_hz: int = 16_000,
    duration_sec: float = 2.0,
) -> None:
    frames = int(sample_rate_hz * duration_sec)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        samples = bytearray()
        for index in range(frames):
            value = amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate_hz)
            samples.extend(int(value * 32767).to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(samples))


def _write_unsupported_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (
        (17).to_bytes(2, byteorder="little")
        + (1).to_bytes(2, byteorder="little")
        + (16_000).to_bytes(4, byteorder="little")
        + (16_000).to_bytes(4, byteorder="little")
        + (1).to_bytes(2, byteorder="little")
        + (4).to_bytes(2, byteorder="little")
    )
    data = b"\x00" * 16
    riff_size = 4 + (8 + len(fmt)) + (8 + len(data))
    path.write_bytes(
        b"RIFF"
        + riff_size.to_bytes(4, byteorder="little")
        + b"WAVE"
        + b"fmt "
        + len(fmt).to_bytes(4, byteorder="little")
        + fmt
        + b"data"
        + len(data).to_bytes(4, byteorder="little")
        + data
    )


def _labels_by_split(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels[row["split"]].add(row["label_binary"])
    return labels
