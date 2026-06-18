"""Dataset registry validation and license manifest export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from drone_sound.constants import (
    ALLOWED_DATASET_ROLES,
    ALLOWED_LABELS,
    ALLOWED_LICENSE_STATUSES,
    ALLOWED_SUBTYPES,
)

REQUIRED_DATASET_FIELDS = (
    "dataset_id",
    "name",
    "role",
    "source_url",
    "license",
    "license_status",
    "label_binary",
    "label_subtype",
    "training_use",
    "evaluation_use",
    "notes",
)

LICENSE_MANIFEST_FIELDS = (
    "dataset_id",
    "name",
    "source_url",
    "license",
    "license_status",
    "training_use",
    "evaluation_use",
    "notes",
)


class DatasetRegistryValidationError(ValueError):
    """Raised when the dataset registry violates the project contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def load_dataset_registry(path: str | Path) -> list[dict[str, object]]:
    """Load dataset registry entries from JSON."""

    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), list):
        raise DatasetRegistryValidationError(["registry must contain a datasets list"])
    return data["datasets"]


def validate_dataset_registry(entries: Iterable[Mapping[str, object]]) -> None:
    """Validate dataset registry entries."""

    entry_list = list(entries)
    errors: list[str] = []
    seen_ids: set[str] = set()

    if not entry_list:
        errors.append("dataset registry has no entries")

    for index, entry in enumerate(entry_list, start=1):
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in entry]
        if missing:
            errors.append(f"entry {index}: missing required fields: {', '.join(missing)}")
            continue

        dataset_id = _as_text(entry["dataset_id"])
        if not dataset_id:
            errors.append(f"entry {index}: dataset_id is blank")
        elif dataset_id in seen_ids:
            errors.append(f"entry {index}: duplicate dataset_id {dataset_id!r}")
        seen_ids.add(dataset_id)

        role = _as_text(entry["role"])
        if role not in ALLOWED_DATASET_ROLES:
            errors.append(f"{dataset_id}: invalid role {role!r}")

        license_status = _as_text(entry["license_status"])
        if license_status not in ALLOWED_LICENSE_STATUSES:
            errors.append(f"{dataset_id}: invalid license_status {license_status!r}")

        label = _as_text(entry["label_binary"])
        if label not in ALLOWED_LABELS:
            errors.append(f"{dataset_id}: invalid label_binary {label!r}")

        subtype_values = _as_list(entry["label_subtype"])
        if not subtype_values:
            errors.append(f"{dataset_id}: label_subtype is empty")
        for subtype in subtype_values:
            if subtype not in ALLOWED_SUBTYPES:
                errors.append(f"{dataset_id}: invalid label_subtype {subtype!r}")

        if not _as_text(entry["source_url"]).startswith("https://"):
            errors.append(f"{dataset_id}: source_url must be https")

        if license_status in {"non_commercial", "license_unspecified", "permission_required"}:
            training_use = _as_text(entry["training_use"])
            if training_use == "commercial_training_allowed":
                errors.append(f"{dataset_id}: restricted license cannot allow commercial training")

    if errors:
        raise DatasetRegistryValidationError(errors)


def export_license_manifest(entries: Iterable[Mapping[str, object]], out_path: str | Path) -> None:
    """Write a compact license manifest CSV from validated registry entries."""

    entry_list = list(entries)
    validate_dataset_registry(entry_list)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LICENSE_MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entry_list:
            writer.writerow(
                {
                    field: _csv_value(entry.get(field, ""))
                    for field in LICENSE_MANIFEST_FIELDS
                }
            )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _csv_value(value: object) -> str:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return _as_text(value)
