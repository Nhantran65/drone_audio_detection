from __future__ import annotations

import pytest

from drone_sound.constants import ALLOWED_LABELS, ALLOWED_SPLITS, ALLOWED_SUBTYPES
from drone_sound.manifest import (
    ManifestValidationError,
    expected_window_count,
    validate_manifest_rows,
)


def test_allowed_values_include_contract_defaults() -> None:
    assert {"drone", "no_drone", "ignore"} <= ALLOWED_LABELS
    assert {"train", "val", "public_test", "professor_test"} <= ALLOWED_SPLITS
    assert {
        "drone_unknown_condition",
        "drone_with_speech",
        "no_drone_aircraft",
        "no_drone_speech",
    } <= ALLOWED_SUBTYPES


def test_expected_window_count_for_professor_duration() -> None:
    assert expected_window_count(557.232) == 1113


def test_manifest_missing_required_field_fails() -> None:
    row = _valid_row()
    row.pop("window_id")

    with pytest.raises(ManifestValidationError, match="missing required fields"):
        validate_manifest_rows([row])


def test_duplicate_window_id_fails() -> None:
    first = _valid_row(window_id="duplicate")
    second = _valid_row(window_id="duplicate")

    with pytest.raises(ManifestValidationError, match="duplicate window_id"):
        validate_manifest_rows([first, second])


def test_original_file_cannot_span_multiple_splits() -> None:
    train = _valid_row(window_id="w1", split="train", original_file="same.wav")
    test = _valid_row(window_id="w2", split="public_test", original_file="same.wav")

    with pytest.raises(ManifestValidationError, match="multiple splits"):
        validate_manifest_rows([train, test])


def test_invalid_label_subtype_and_split_fail() -> None:
    row = _valid_row(
        label_binary="maybe_drone",
        label_subtype="unknown_subtype",
        split="random",
    )

    with pytest.raises(ManifestValidationError) as error:
        validate_manifest_rows([row])

    message = str(error.value)
    assert "invalid label_binary" in message
    assert "invalid label_subtype" in message
    assert "invalid split" in message


def _valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "clip_id": "clip-1",
        "window_id": "window-1",
        "source_dataset": "unit_test",
        "source_url_or_doi": "local",
        "license": "test",
        "original_file": "audio.wav",
        "original_file_checksum": "abc123",
        "sample_rate_original": "16000",
        "sample_rate_processed": "16000",
        "duration_sec": "1.000000",
        "window_start_sec": "0.000000",
        "window_end_sec": "1.000000",
        "label_binary": "drone",
        "label_subtype": "drone_unknown_condition",
        "drone_model": "",
        "environment": "",
        "recording_condition": "",
        "split": "professor_test",
        "is_professor_data": "true",
        "notes": "",
    }
    row.update(overrides)
    return row
