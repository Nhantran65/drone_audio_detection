"""Project constants for dataset manifests and windowing."""

from __future__ import annotations

WINDOW_LENGTH_SECONDS = 1.0
WINDOW_STRIDE_SECONDS = 0.5
TARGET_SAMPLE_RATE_HZ = 16_000

ALLOWED_LABELS = frozenset({"drone", "no_drone", "ignore"})

ALLOWED_SPLITS = frozenset({"train", "val", "public_test", "professor_test"})

ALLOWED_DATASET_ROLES = frozenset(
    {
        "positive_drone",
        "positive_optional",
        "negative_aircraft",
        "negative_traffic",
        "negative_urban_industrial",
        "negative_speech",
        "professor_positive_test",
    }
)

ALLOWED_LICENSE_STATUSES = frozenset(
    {
        "commercial_ok",
        "research_only",
        "non_commercial",
        "license_unspecified",
        "permission_required",
    }
)

ALLOWED_SUBTYPES = frozenset(
    {
        "drone_field",
        "drone_with_speech",
        "drone_with_background",
        "drone_unknown_condition",
        "no_drone_aircraft",
        "no_drone_traffic",
        "no_drone_urban_industrial",
        "no_drone_speech",
        "no_drone_background",
        "no_drone_unknown",
    }
)

REQUIRED_MANIFEST_FIELDS = (
    "clip_id",
    "window_id",
    "source_dataset",
    "source_url_or_doi",
    "license",
    "original_file",
    "original_file_checksum",
    "sample_rate_original",
    "sample_rate_processed",
    "duration_sec",
    "window_start_sec",
    "window_end_sec",
    "label_binary",
    "label_subtype",
    "drone_model",
    "environment",
    "recording_condition",
    "split",
    "is_professor_data",
    "notes",
)

PROFESSOR_SOURCE_DATASET = "professor_real_world"
PROFESSOR_LICENSE = "private_research_only"
PROFESSOR_DEFAULT_LABEL = "drone"
PROFESSOR_DEFAULT_SUBTYPE = "drone_unknown_condition"
PROFESSOR_SPLIT = "professor_test"
