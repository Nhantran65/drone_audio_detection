from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from drone_sound.audio_preprocessing import CANONICAL_PREPROCESSING_PROFILE
from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL
from drone_sound.v2_lightweight_trees import (
    DEFAULT_CANDIDATES,
    CandidateSpec,
    XGBoostBinaryAdapter,
    _select_group_winners,
    run_lightweight_tree_experiments,
)
from drone_sound.v2_ml import V2MLError, load_v2_model


def test_default_grid_is_bounded_to_three_requested_groups() -> None:
    counts: dict[str, int] = {}
    for candidate in DEFAULT_CANDIDATES:
        counts[candidate.group] = counts.get(candidate.group, 0) + 1

    assert counts == {
        "compressed_rf": 3,
        "hist_gradient_boosting": 2,
        "shallow_xgboost": 1,
    }


def test_group_selection_uses_validation_and_size_only() -> None:
    rows = [
        {
            "group": "compressed_rf",
            "model": "validation_winner",
            "validation_f2": 0.91,
            "validation_fpr": 0.08,
            "model_size_bytes": 200,
            "public_f2": 0.01,
            "professor_recall": 0.01,
        },
        {
            "group": "compressed_rf",
            "model": "test_winner",
            "validation_f2": 0.90,
            "validation_fpr": 0.01,
            "model_size_bytes": 10,
            "public_f2": 1.0,
            "professor_recall": 1.0,
        },
    ]

    selected = _select_group_winners(rows)

    assert selected["compressed_rf"]["model"] == "validation_winner"


def test_xgboost_label_adapter_exposes_project_label_order() -> None:
    class FakeEstimator:
        def predict_proba(self, features: object) -> object:
            return np.asarray([[0.25, 0.75]])

    adapter = XGBoostBinaryAdapter(FakeEstimator())

    assert adapter.classes_ == (NEGATIVE_LABEL, POSITIVE_LABEL)
    assert adapter.predict_proba(None).tolist() == [[0.25, 0.75]]


def test_small_rf_experiment_writes_frozen_package_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    features = _write_bundle(tmp_path / "features.npz")
    output = tmp_path / "experiment"
    candidate = CandidateSpec(
        name="compressed_rf_fixture",
        group="compressed_rf",
        family="random_forest",
        parameters={"n_estimators": 2, "max_depth": 3, "min_samples_leaf": 1},
        runtime_dependency="scikit-learn",
    )

    report = run_lightweight_tree_experiments(
        features,
        output,
        candidates=(candidate,),
    )

    assert report["winners"]["compressed_rf"]["model"] == candidate.name
    assert report["selection_excludes"] == [
        "public_test",
        "professor_test",
        "private_real_only",
    ]
    package = load_v2_model(output / "models" / f"{candidate.name}.joblib")
    assert package["threshold_name"] == "threshold_f2"
    assert package["preprocessing_profile"] == CANONICAL_PREPROCESSING_PROFILE
    assert package["source_bundle_fingerprint"] == "fixture-fingerprint"
    with pytest.raises(V2MLError, match="refusing to overwrite"):
        run_lightweight_tree_experiments(
            features,
            output,
            candidates=(candidate,),
        )


def _write_bundle(path: Path) -> Path:
    rows = [
        ("train", POSITIVE_LABEL, False, "d1", [2.0, 2.0]),
        ("train", POSITIVE_LABEL, False, "d2", [2.2, 1.8]),
        ("train", POSITIVE_LABEL, False, "d3", [1.8, 2.2]),
        ("train", NEGATIVE_LABEL, False, "n1", [-2.0, -2.0]),
        ("train", NEGATIVE_LABEL, False, "n2", [-2.2, -1.8]),
        ("train", NEGATIVE_LABEL, False, "n3", [-1.8, -2.2]),
        ("val", POSITIVE_LABEL, False, "vd1", [2.1, 2.0]),
        ("val", POSITIVE_LABEL, False, "vd2", [1.9, 2.1]),
        ("val", NEGATIVE_LABEL, False, "vn1", [-2.1, -2.0]),
        ("val", NEGATIVE_LABEL, False, "vn2", [-1.9, -2.1]),
        ("public_test", POSITIVE_LABEL, False, "td1", [2.0, 2.1]),
        ("public_test", POSITIVE_LABEL, False, "td2", [2.1, 1.9]),
        ("public_test", NEGATIVE_LABEL, False, "tn1", [-2.0, -2.1]),
        ("public_test", NEGATIVE_LABEL, False, "tn2", [-2.1, -1.9]),
        ("professor_test", POSITIVE_LABEL, True, "p1", [2.0, 2.0]),
        ("professor_test", POSITIVE_LABEL, True, "p2", [1.7, 1.8]),
        ("professor_test", POSITIVE_LABEL, True, "p3", [2.2, 2.1]),
    ]
    count = len(rows)
    np.savez_compressed(
        path,
        scaled_features=np.asarray([row[4] for row in rows], dtype=np.float64),
        labels=np.asarray([row[1] for row in rows]),
        splits=np.asarray([row[0] for row in rows]),
        window_ids=np.asarray([f"w{index}" for index in range(count)]),
        clip_ids=np.asarray([row[3] for row in rows]),
        original_files=np.asarray([f"{row[3]}.wav" for row in rows]),
        source_datasets=np.asarray(["fixture"] * count),
        group_ids=np.asarray([row[3] for row in rows]),
        label_subtypes=np.asarray(["fixture"] * count),
        environments=np.asarray(["fixture"] * count),
        window_start_sec=np.asarray([0.0] * count),
        window_end_sec=np.asarray([1.0] * count),
        is_professor_data=np.asarray([row[2] for row in rows]),
        feature_names=np.asarray(["f0", "f1"]),
        feature_set=np.asarray("spectral_mfcc_v1"),
        source_sample_rates_hz=np.asarray([16_000] * count),
        processed_sample_rates_hz=np.asarray([16_000] * count),
        preprocessing_profile=np.asarray(CANONICAL_PREPROCESSING_PROFILE),
        processed_sample_rate_hz=np.asarray(16_000),
        bundle_fingerprint=np.asarray("fixture-fingerprint"),
        bundle_schema_version=np.asarray("v2_feature_bundle_v2"),
        split_version=np.asarray("fixture_split_v1"),
        group_parser_version=np.asarray("fixture_group_v1"),
    )
    return path
