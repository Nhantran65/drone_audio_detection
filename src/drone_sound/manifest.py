"""Manifest validation helpers."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from drone_sound.constants import (
    ALLOWED_LABELS,
    ALLOWED_SPLITS,
    ALLOWED_SUBTYPES,
    REQUIRED_MANIFEST_FIELDS,
    WINDOW_LENGTH_SECONDS,
    WINDOW_STRIDE_SECONDS,
)


class ManifestValidationError(ValueError):
    """Raised when manifest rows violate the dataset contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def expected_window_count(
    duration_sec: float,
    window_length_sec: float = WINDOW_LENGTH_SECONDS,
    stride_sec: float = WINDOW_STRIDE_SECONDS,
) -> int:
    """Return the number of complete fixed-length windows in an audio clip."""

    if duration_sec < window_length_sec:
        return 0
    return math.floor(((duration_sec - window_length_sec) / stride_sec) + 1e-9) + 1


def load_manifest_csv(path: str | Path) -> list[dict[str, str]]:
    """Load manifest rows from a CSV file."""

    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_manifest_file(path: str | Path) -> None:
    """Validate a manifest CSV file and raise on any contract violation."""

    validate_manifest_rows(load_manifest_csv(path))


def validate_manifest_rows(rows: Iterable[Mapping[str, object]]) -> None:
    """Validate manifest rows and raise ManifestValidationError on failure."""

    row_list = list(rows)
    errors: list[str] = []
    seen_window_ids: set[str] = set()
    splits_by_original_file: dict[str, set[str]] = defaultdict(set)

    if not row_list:
        errors.append("manifest has no rows")

    for row_number, row in enumerate(row_list, start=1):
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
        if missing:
            errors.append(f"row {row_number}: missing required fields: {', '.join(missing)}")
            continue

        blank = [
            field
            for field in (
                "clip_id",
                "window_id",
                "source_dataset",
                "original_file",
                "original_file_checksum",
                "label_binary",
                "label_subtype",
                "split",
            )
            if _as_text(row[field]) == ""
        ]
        if blank:
            errors.append(f"row {row_number}: blank critical fields: {', '.join(blank)}")

        window_id = _as_text(row["window_id"])
        if window_id in seen_window_ids:
            errors.append(f"row {row_number}: duplicate window_id {window_id!r}")
        seen_window_ids.add(window_id)

        label = _as_text(row["label_binary"])
        if label not in ALLOWED_LABELS:
            errors.append(f"row {row_number}: invalid label_binary {label!r}")

        subtype = _as_text(row["label_subtype"])
        if subtype not in ALLOWED_SUBTYPES:
            errors.append(f"row {row_number}: invalid label_subtype {subtype!r}")

        split = _as_text(row["split"])
        if split not in ALLOWED_SPLITS:
            errors.append(f"row {row_number}: invalid split {split!r}")

        original_file = _as_text(row["original_file"])
        if original_file:
            splits_by_original_file[original_file].add(split)

        for numeric_field in (
            "sample_rate_original",
            "sample_rate_processed",
            "duration_sec",
            "window_start_sec",
            "window_end_sec",
        ):
            if _as_text(row[numeric_field]) == "":
                errors.append(f"row {row_number}: blank numeric field {numeric_field}")
                continue
            try:
                float(_as_text(row[numeric_field]))
            except ValueError:
                errors.append(f"row {row_number}: invalid numeric field {numeric_field}")

    for original_file, splits in sorted(splits_by_original_file.items()):
        if len(splits) > 1:
            joined = ", ".join(sorted(splits))
            errors.append(f"original_file {original_file!r} appears in multiple splits: {joined}")

    if errors:
        raise ManifestValidationError(errors)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
