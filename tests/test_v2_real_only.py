from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
joblib = pytest.importorskip("joblib")

from drone_sound.private_real import build_private_real_manifest
from drone_sound.v2_ml import MODEL_SCHEMA_VERSION
from drone_sound.v2_real_eval import evaluate_real_v2


class FrozenEstimator:
    classes_ = np.asarray(["no_drone", "drone"])

    def fit(self, *_: object, **__: object) -> None:
        raise AssertionError("real-only evaluator must never call fit")

    def predict_proba(self, features: object) -> object:
        matrix = np.asarray(features, dtype=np.float64)
        positive = np.clip(matrix[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))


def test_private_real_manifest_has_locked_recording_and_window_counts(tmp_path: Path) -> None:
    report = build_private_real_manifest(
        Path("data/manifests/professor_manifest.csv"),
        Path("data/raw/private"),
        Path("data/raw/private/Recordings"),
        tmp_path / "private_real.csv",
        tmp_path / "private_real_audit.json",
    )
    assert report["counts"] == {
        "rows": 1_705,
        "positive_windows": 1_113,
        "negative_windows": 592,
        "positive_recordings": 1,
        "negative_recordings": 10,
    }


def test_real_only_evaluator_uses_frozen_threshold_and_reconciles_counts(tmp_path: Path) -> None:
    bundle = _write_real_bundle(tmp_path / "real.npz")
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "algorithm": "v2a_classical_ml",
            "model_type": "logistic_regression",
            "track": "v2a",
            "feature_set": "spectral_mfcc_v1",
            "feature_names": ["f0", "f1"],
            "labels": {"positive": "drone", "negative": "no_drone"},
            "threshold": 0.5,
            "threshold_name": "threshold_f2",
            "thresholds": {"threshold_f2": 0.5},
            "estimator": FrozenEstimator(),
            "deployability_label": "deployable_lightweight",
            "selection": {"uses_professor_data": False},
            "preprocessing_profile": "mono_resample_poly_16khz_v1",
            "processed_sample_rate_hz": 16_000,
            "source_bundle_fingerprint": "training-bundle",
        },
        model_path,
    )
    report = evaluate_real_v2(
        model_path,
        bundle,
        tmp_path,
        tmp_path / "cache",
        tmp_path / "metrics.json",
    )
    confusion = report["metrics"]["confusion_matrix"]
    assert sum(confusion.values()) == 1_705
    assert report["counts"]["drone"] == 1_113
    assert report["counts"]["no_drone"] == 592
    assert report["fit_called"] is False
    assert report["threshold_tuned"] is False
    assert len(report["per_recording_fpr"]) == 10


def _write_real_bundle(path: Path) -> Path:
    positive_count = 1_113
    negative_count = 592
    rows = positive_count + negative_count
    labels = np.asarray(["drone"] * positive_count + ["no_drone"] * negative_count)
    scores = np.asarray([0.9] * positive_count + [0.1] * negative_count)
    features = np.column_stack((scores, np.zeros(rows)))
    positive_files = ["professor_real_world/positive.wav"] * positive_count
    negative_files = [f"Recordings/negative_{index % 10}.wav" for index in range(negative_count)]
    environments = [""] * positive_count + ["traffic" if index % 2 else "forest" for index in range(negative_count)]
    np.savez_compressed(
        path,
        raw_features=features,
        scaled_features=features,
        labels=labels,
        splits=np.asarray(["professor_test"] * rows),
        window_ids=np.asarray([f"w{index:06d}" for index in range(rows)]),
        clip_ids=np.asarray([f"c{index:06d}" for index in range(rows)]),
        original_files=np.asarray(positive_files + negative_files),
        source_datasets=np.asarray(["private"] * rows),
        label_subtypes=np.asarray(["drone_unknown_condition"] * positive_count + ["no_drone_background"] * negative_count),
        environments=np.asarray(environments),
        window_start_sec=np.zeros(rows),
        window_end_sec=np.ones(rows),
        is_professor_data=np.ones(rows, dtype=bool),
        source_sample_rates_hz=np.asarray([16_000] * positive_count + [48_000] * negative_count),
        processed_sample_rates_hz=np.full(rows, 16_000, dtype=np.int32),
        feature_names=np.asarray(["f0", "f1"]),
        feature_set=np.asarray("spectral_mfcc_v1"),
        bundle_schema_version=np.asarray("v2_feature_bundle_v2"),
        preprocessing_profile=np.asarray("mono_resample_poly_16khz_v1"),
        processed_sample_rate_hz=np.asarray(16_000, dtype=np.int32),
        bundle_fingerprint=np.asarray("b" * 64),
    )
    return path
