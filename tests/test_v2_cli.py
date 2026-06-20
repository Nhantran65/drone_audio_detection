from __future__ import annotations

from pathlib import Path

import pytest

from drone_sound.cli import main
from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL


@pytest.mark.parametrize(
    ("argv_builder", "expected_error"),
    [
        (
            lambda manifest, audio_root, tmp_path: [
                "train-v2-mlp",
                "--manifest",
                str(manifest),
                "--audio-root",
                str(audio_root),
                "--feature-set",
                "spectral_mfcc_v1",
                "--model-out",
                str(tmp_path / "mlp.pt"),
                "--metrics-out",
                str(tmp_path / "mlp_metrics.json"),
            ],
            "v2 not implemented: Tiny MLP training",
        ),
        (
            lambda manifest, audio_root, tmp_path: [
                "train-v2-cnn",
                "--manifest",
                str(manifest),
                "--audio-root",
                str(audio_root),
                "--feature-set",
                "logmel_v1",
                "--model-out",
                str(tmp_path / "cnn.pt"),
                "--metrics-out",
                str(tmp_path / "cnn_metrics.json"),
            ],
            "v2 not implemented: Tiny CNN training",
        ),
        (
            lambda manifest, audio_root, tmp_path: [
                "train-v2-embedding",
                "--manifest",
                str(manifest),
                "--audio-root",
                str(audio_root),
                "--embedding-model",
                "yamnet",
                "--head",
                "logistic_regression",
                "--model-out",
                str(tmp_path / "yamnet_head.json"),
                "--metrics-out",
                str(tmp_path / "yamnet_metrics.json"),
            ],
            "v2 not implemented: pretrained embedding training",
        ),
    ],
)
def test_v2_train_skeleton_commands_validate_inputs_then_stop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv_builder: object,
    expected_error: str,
) -> None:
    manifest = tmp_path / "public_manifest.csv"
    audio_root = tmp_path / "audio"
    manifest.write_text("placeholder\n", encoding="utf-8")
    audio_root.mkdir()

    exit_code = main(argv_builder(manifest, audio_root, tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert expected_error in captured.err


def test_sweep_v2_ml_cli_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    features = _write_feature_bundle(tmp_path / "features.npz")
    out_dir = tmp_path / "v2a"

    exit_code = main(
        [
            "sweep-v2-ml",
            "--features",
            str(features),
            "--model-types",
            "logistic_regression",
            "extra_trees",
            "--out-dir",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "completed v2a sweep: best_model=" in captured.out
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "models" / "logistic_regression.joblib").exists()


def test_eval_v2_cli_writes_metrics_and_error_exports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    features = _write_feature_bundle(tmp_path / "features.npz")
    out_dir = tmp_path / "v2a"
    assert main(
        [
            "sweep-v2-ml",
            "--features",
            str(features),
            "--model-types",
            "logistic_regression",
            "--out-dir",
            str(out_dir),
        ]
    ) == 0
    capsys.readouterr()
    metrics_out = tmp_path / "eval" / "metrics.json"
    errors_out = tmp_path / "eval" / "errors"

    exit_code = main(
        [
            "eval-v2",
            "--model",
            str(out_dir / "models" / "logistic_regression.joblib"),
            "--features",
            str(features),
            "--metrics-out",
            str(metrics_out),
            "--export-errors",
            str(errors_out),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "evaluated frozen v2 model: public_test_f2=" in captured.out
    assert metrics_out.exists()
    assert (errors_out / "public_false_positives.csv").exists()
    assert (errors_out / "professor_missed_windows.csv").exists()


def test_eval_v2_cli_reports_missing_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features = _write_feature_bundle(tmp_path / "features.npz")

    exit_code = main(
        [
            "eval-v2",
            "--model",
            str(tmp_path / "missing.joblib"),
            "--features",
            str(features),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "v2 ml error: model does not exist" in captured.err


def test_v2_skeleton_reports_missing_manifest_before_not_implemented(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()

    exit_code = main(
        [
            "train-v2-mlp",
            "--manifest",
            str(tmp_path / "missing.csv"),
            "--audio-root",
            str(audio_root),
            "--feature-set",
            "spectral_mfcc_v1",
            "--model-out",
            str(tmp_path / "mlp.pt"),
            "--metrics-out",
            str(tmp_path / "mlp_metrics.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "v2 input error: manifest does not exist" in captured.err


def test_v2_cli_rejects_unknown_feature_set(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    audio_root = tmp_path / "audio"
    manifest.write_text("placeholder\n", encoding="utf-8")
    audio_root.mkdir()

    with pytest.raises(SystemExit) as error:
        main(
            [
                "train-v2-cnn",
                "--manifest",
                str(manifest),
                "--audio-root",
                str(audio_root),
                "--feature-set",
                "mel",
                "--model-out",
                str(tmp_path / "cnn.pt"),
                "--metrics-out",
                str(tmp_path / "cnn_metrics.json"),
            ]
        )

    assert error.value.code == 2


def _write_feature_bundle(path: Path) -> Path:
    np = pytest.importorskip("numpy")
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
