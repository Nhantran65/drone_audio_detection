from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL, _metrics
from drone_sound.v2 import THRESHOLD_NAMES
from drone_sound.v2_ml import (
    V2MLError,
    evaluate_v2_model,
    load_feature_bundle,
    run_v2a_sweep,
    select_thresholds,
    train_v2_mlp,
)


def test_feature_bundle_loader_requires_eval_metadata(tmp_path: Path) -> None:
    features = tmp_path / "old_features.npz"
    np.savez_compressed(
        features,
        scaled_features=np.asarray([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float64),
        labels=np.asarray([POSITIVE_LABEL, NEGATIVE_LABEL]),
        splits=np.asarray(["train", "train"]),
        window_ids=np.asarray(["w1", "w2"]),
        original_files=np.asarray(["a.wav", "b.wav"]),
        source_datasets=np.asarray(["fixture", "fixture"]),
        is_professor_data=np.asarray([False, False]),
        feature_names=np.asarray(["f0", "f1"]),
        feature_set=np.asarray("spectral_mfcc_v1"),
    )

    with pytest.raises(V2MLError, match="regenerate features"):
        load_feature_bundle(features)


def test_threshold_selection_reports_fbeta_and_fixed_fpr_thresholds() -> None:
    labels = [
        POSITIVE_LABEL,
        POSITIVE_LABEL,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
        NEGATIVE_LABEL,
        NEGATIVE_LABEL,
    ]
    scores = [0.95, 0.80, 0.45, 0.60, 0.20, 0.10]

    thresholds = select_thresholds(labels, scores)

    assert tuple(thresholds) == THRESHOLD_NAMES
    for name, threshold in thresholds.items():
        assert isinstance(threshold, float), name
    fixed_metric = _metrics(labels, scores, thresholds["threshold_at_fpr_10"])
    assert fixed_metric["false_positive_rate"] <= 0.10


def test_v2a_sweep_trains_models_and_writes_artifacts(tmp_path: Path) -> None:
    features = _write_feature_bundle(tmp_path / "features.npz")
    out_dir = tmp_path / "sweep"

    report = run_v2a_sweep(
        features,
        ["logistic_regression", "linear_svm", "random_forest"],
        out_dir,
    )

    assert report["best_model"]["selected_model"] in {
        "logistic_regression",
        "linear_svm",
        "random_forest",
    }
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "comparison.csv").exists()
    assert (out_dir / "best_model.json").exists()
    for model_type in ["logistic_regression", "linear_svm", "random_forest"]:
        assert (out_dir / "models" / f"{model_type}.joblib").exists()
        metrics_path = out_dir / "metrics" / f"{model_type}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["training"]["counts"] == {"drone": 3, "no_drone": 3}
        assert metrics["selection"]["uses_professor_data"] is False
        assert metrics["threshold_name"] == "threshold_f2"
        assert set(metrics["thresholds"]) == set(THRESHOLD_NAMES)
        assert metrics["professor_test"]["metrics"]["rows"] == 3


def test_eval_v2_reuses_frozen_threshold_and_exports_errors(tmp_path: Path) -> None:
    features = _write_feature_bundle(tmp_path / "features.npz")
    out_dir = tmp_path / "sweep"
    run_v2a_sweep(features, ["logistic_regression"], out_dir)
    model_path = out_dir / "models" / "logistic_regression.joblib"
    metrics_out = tmp_path / "eval" / "metrics.json"
    errors_out = tmp_path / "eval" / "errors"

    report = evaluate_v2_model(model_path, features, metrics_out, errors_out)

    stored_metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert report["threshold_name"] == "threshold_f2"
    assert stored_metrics["threshold"] == report["threshold"]
    assert report["public_test"]["metrics"]["rows"] == 4
    assert report["professor_test"]["metrics"]["rows"] == 3
    assert (errors_out / "public_false_positives.csv").exists()
    assert (errors_out / "professor_missed_windows.csv").exists()


def test_train_v2_mlp_writes_frozen_model_and_metrics(tmp_path: Path) -> None:
    features = _write_feature_bundle(tmp_path / "features.npz")
    model_out = tmp_path / "models" / "tiny_mlp.joblib"
    metrics_out = tmp_path / "metrics" / "tiny_mlp.json"

    report = train_v2_mlp(
        features,
        model_out,
        metrics_out,
        epochs=2,
        batch_size=4,
        patience=2,
    )

    stored = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert report["model_type"] == "tiny_mlp"
    assert report["track"] == "v2b"
    assert stored["threshold_name"] == "threshold_f2"
    assert stored["selection"]["uses_professor_data"] is False
    assert stored["early_stopping"]["enabled"] is True
    assert stored["early_stopping"]["epochs_requested"] == 2
    assert len(stored["learning_curve"]) == stored["early_stopping"]["epochs_ran"]
    assert model_out.exists()


def test_missing_feature_bundle_reports_clear_error(tmp_path: Path) -> None:
    with pytest.raises(V2MLError, match="features does not exist"):
        run_v2a_sweep(
            tmp_path / "missing.npz",
            ["logistic_regression"],
            tmp_path / "out",
        )


def test_ml_dependency_guard_reports_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from drone_sound import v2_ml

    original_import = builtins.__import__

    def fail_sklearn_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("sklearn"):
            raise ImportError("blocked sklearn")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_sklearn_import)

    with pytest.raises(V2MLError, match="optional ml dependencies"):
        v2_ml._require_ml_dependencies()


def _write_feature_bundle(path: Path) -> Path:
    rows = [
        ("train", POSITIVE_LABEL, False, "train_d1", [2.0, 2.0]),
        ("train", POSITIVE_LABEL, False, "train_d2", [2.2, 1.8]),
        ("train", POSITIVE_LABEL, False, "train_d3", [1.8, 2.2]),
        ("train", NEGATIVE_LABEL, False, "train_n1", [-2.0, -2.0]),
        ("train", NEGATIVE_LABEL, False, "train_n2", [-2.2, -1.8]),
        ("train", NEGATIVE_LABEL, False, "train_n3", [-1.8, -2.2]),
        ("val", POSITIVE_LABEL, False, "val_d1", [2.1, 2.0]),
        ("val", POSITIVE_LABEL, False, "val_d2", [1.9, 2.1]),
        ("val", NEGATIVE_LABEL, False, "val_n1", [-2.1, -2.0]),
        ("val", NEGATIVE_LABEL, False, "val_n2", [-1.9, -2.1]),
        ("public_test", POSITIVE_LABEL, False, "pub_d1", [2.0, 1.9]),
        ("public_test", POSITIVE_LABEL, False, "pub_d2", [1.9, 2.0]),
        ("public_test", NEGATIVE_LABEL, False, "pub_n1", [-2.0, -1.9]),
        ("public_test", NEGATIVE_LABEL, False, "pub_n2", [-1.9, -2.0]),
        ("professor_test", POSITIVE_LABEL, True, "prof_1", [2.0, 2.0]),
        ("professor_test", POSITIVE_LABEL, True, "prof_2", [1.8, 1.9]),
        ("professor_test", POSITIVE_LABEL, True, "prof_3", [0.0, 0.0]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        raw_features=np.asarray([row[4] for row in rows], dtype=np.float64),
        scaled_features=np.asarray([row[4] for row in rows], dtype=np.float64),
        labels=np.asarray([row[1] for row in rows]),
        splits=np.asarray([row[0] for row in rows]),
        window_ids=np.asarray([f"w{index:03d}" for index in range(len(rows))]),
        clip_ids=np.asarray([row[3] for row in rows]),
        original_files=np.asarray([f"{row[3]}.wav" for row in rows]),
        source_datasets=np.asarray(
            ["professor_fixture" if row[2] else "public_fixture" for row in rows]
        ),
        label_subtypes=np.asarray(
            ["drone_unknown_condition" if row[2] else _subtype(row[1]) for row in rows]
        ),
        window_start_sec=np.asarray([float(index) for index in range(len(rows))]),
        window_end_sec=np.asarray([float(index + 1) for index in range(len(rows))]),
        is_professor_data=np.asarray([row[2] for row in rows], dtype=bool),
        feature_names=np.asarray(["f0", "f1"]),
        feature_set=np.asarray("spectral_mfcc_v1"),
    )
    return path


def _subtype(label: str) -> str:
    if label == POSITIVE_LABEL:
        return "drone_field"
    return "no_drone_background"
