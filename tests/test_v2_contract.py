from __future__ import annotations

from pathlib import Path

import pytest

from drone_sound.v2 import (
    DEPLOYABILITY_LABELS,
    FEATURE_SETS,
    MODEL_TYPES,
    THRESHOLD_NAMES,
    V2InputError,
    validate_deployability_label,
    validate_feature_set,
    validate_model_type,
)


def test_v2_contract_names_are_documented() -> None:
    doc = Path("docs/product/v2-exploration.md").read_text(encoding="utf-8")

    for value in FEATURE_SETS + MODEL_TYPES + DEPLOYABILITY_LABELS + THRESHOLD_NAMES:
        assert value in doc


def test_v2_validation_accepts_known_contract_values() -> None:
    assert validate_feature_set("spectral_mfcc_v1") == "spectral_mfcc_v1"
    assert validate_model_type("tiny_cnn") == "tiny_cnn"
    assert validate_deployability_label("reference_only") == "reference_only"


def test_v2_validation_rejects_unknown_contract_values() -> None:
    with pytest.raises(V2InputError, match="unknown feature set"):
        validate_feature_set("spectrogram")

    with pytest.raises(V2InputError, match="unknown model type"):
        validate_model_type("xgboost")

    with pytest.raises(V2InputError, match="unknown deployability label"):
        validate_deployability_label("production_ready")
