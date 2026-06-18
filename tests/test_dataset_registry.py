from __future__ import annotations

import csv
from pathlib import Path

import pytest

from drone_sound.dataset_registry import (
    DatasetRegistryValidationError,
    export_license_manifest,
    load_dataset_registry,
    validate_dataset_registry,
)


def test_project_dataset_registry_is_valid() -> None:
    entries = load_dataset_registry("configs/datasets.json")

    validate_dataset_registry(entries)

    ids = {entry["dataset_id"] for entry in entries}
    assert {
        "dronenoise_database",
        "kielce_uav_acoustic_signals",
        "nasa_small_uas_flyover",
        "aerosonicdb_ypad_0523",
        "idmt_traffic",
        "urbansound8k",
    } == ids


def test_duplicate_dataset_id_fails() -> None:
    entry = _entry()

    with pytest.raises(DatasetRegistryValidationError, match="duplicate dataset_id"):
        validate_dataset_registry([entry, dict(entry)])


def test_restricted_license_cannot_allow_commercial_training() -> None:
    entry = _entry(
        license_status="non_commercial",
        training_use="commercial_training_allowed",
    )

    with pytest.raises(DatasetRegistryValidationError, match="restricted license"):
        validate_dataset_registry([entry])


def test_invalid_label_and_subtype_fail() -> None:
    entry = _entry(label_binary="positive", label_subtype=["unknown"])

    with pytest.raises(DatasetRegistryValidationError) as error:
        validate_dataset_registry([entry])

    message = str(error.value)
    assert "invalid label_binary" in message
    assert "invalid label_subtype" in message


def test_export_license_manifest(tmp_path: Path) -> None:
    entries = [_entry(dataset_id="one"), _entry(dataset_id="two", license_status="commercial_ok")]
    out_path = tmp_path / "license_manifest.csv"

    export_license_manifest(entries, out_path)

    rows = list(csv.DictReader(out_path.open(newline="", encoding="utf-8")))
    assert [row["dataset_id"] for row in rows] == ["one", "two"]
    assert rows[0]["license_status"] == "non_commercial"
    assert rows[1]["license_status"] == "commercial_ok"


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "dataset_id": "dataset",
        "name": "Dataset",
        "role": "negative_aircraft",
        "source_url": "https://example.com/dataset",
        "doi": "",
        "license": "CC BY-NC 4.0",
        "license_status": "non_commercial",
        "label_binary": "no_drone",
        "label_subtype": ["no_drone_aircraft"],
        "training_use": "research_training_only_non_commercial",
        "evaluation_use": "hard_negative_public_test_allowed_for_research",
        "notes": "test entry",
    }
    entry.update(overrides)
    return entry
