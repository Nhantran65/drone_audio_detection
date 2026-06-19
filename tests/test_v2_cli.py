from __future__ import annotations

from pathlib import Path

import pytest

from drone_sound.cli import main


@pytest.mark.parametrize(
    ("argv_builder", "expected_error"),
    [
        (
            lambda manifest, audio_root, tmp_path: [
                "sweep-v2-ml",
                "--manifest",
                str(manifest),
                "--audio-root",
                str(audio_root),
                "--feature-set",
                "spectral_mfcc_v1",
                "--model-types",
                "logistic_regression",
                "linear_svm",
                "--out-dir",
                str(tmp_path / "v2a"),
            ],
            "v2 not implemented: classical ML sweep",
        ),
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


def test_eval_v2_skeleton_validates_inputs_then_stops(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model.json"
    public_manifest = tmp_path / "public_manifest.csv"
    professor_manifest = tmp_path / "professor_manifest.csv"
    public_audio_root = tmp_path / "public_audio"
    professor_audio_root = tmp_path / "professor_audio"
    metrics_out = tmp_path / "metrics.json"
    errors_out = tmp_path / "errors"

    model.write_text("placeholder\n", encoding="utf-8")
    public_manifest.write_text("placeholder\n", encoding="utf-8")
    professor_manifest.write_text("placeholder\n", encoding="utf-8")
    public_audio_root.mkdir()
    professor_audio_root.mkdir()

    exit_code = main(
        [
            "eval-v2",
            "--model",
            str(model),
            "--public-manifest",
            str(public_manifest),
            "--public-audio-root",
            str(public_audio_root),
            "--professor-manifest",
            str(professor_manifest),
            "--professor-audio-root",
            str(professor_audio_root),
            "--metrics-out",
            str(metrics_out),
            "--export-errors",
            str(errors_out),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "v2 not implemented: frozen V2 evaluation" in captured.err
    assert not metrics_out.exists()
    assert not errors_out.exists()


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
