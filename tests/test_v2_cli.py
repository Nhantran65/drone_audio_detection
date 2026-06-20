from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from drone_sound import v2_embedding
from drone_sound.cli import main
from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL


def test_train_v2_mlp_cli_writes_model_and_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    features = _write_feature_bundle(tmp_path / "features.npz")
    model_out = tmp_path / "mlp.joblib"
    metrics_out = tmp_path / "mlp_metrics.json"

    exit_code = main(
        [
            "train-v2-mlp",
            "--features",
            str(features),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--patience",
            "2",
            "--min-delta",
            "0.0",
        ]
    )

    captured = capsys.readouterr()
    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "trained v2b tiny mlp" in captured.out
    assert "v2 ml progress: epoch complete model=tiny_mlp" in captured.err
    assert metrics["early_stopping"]["epochs_requested"] == 2
    assert len(metrics["learning_curve"]) == 2
    assert model_out.exists()
    assert metrics_out.exists()


def test_train_v2_embedding_cli_writes_model_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    features = _write_feature_bundle(tmp_path / "features.npz")
    public_root = tmp_path / "public"
    professor_root = tmp_path / "professor"
    public_root.mkdir()
    professor_root.mkdir()
    _write_audio_files_for_bundle(features, public_root, professor_root)
    monkeypatch.setattr(v2_embedding, "_load_yamnet_embedder", lambda: _fake_embedder(np))
    model_out = tmp_path / "head.joblib"
    metrics_out = tmp_path / "metrics.json"

    exit_code = main(
        [
            "train-v2-embedding",
            "--features",
            str(features),
            "--public-audio-root",
            str(public_root),
            "--professor-audio-root",
            str(professor_root),
            "--embedding-model",
            "yamnet",
            "--head",
            "logistic_regression",
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    captured = capsys.readouterr()
    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "trained v2d pretrained embedding" in captured.out
    assert "v2 ml progress: cache start phase=yamnet_embedding_v1" in captured.err
    assert model_out.exists()
    assert metrics["model_type"] == "yamnet_head"
    assert metrics["track"] == "v2d"
    assert metrics["feature_set"] == "pretrained_embedding_v1"
    assert metrics["efficiency"]["deployability_label"] == "reference_only"

    eval_metrics_out = tmp_path / "eval" / "yamnet_eval.json"
    exit_code = main(
        [
            "eval-v2-embedding",
            "--model",
            str(model_out),
            "--features",
            str(features),
            "--metrics-out",
            str(eval_metrics_out),
            "--batch-size",
            "4",
        ]
    )

    captured = capsys.readouterr()
    eval_metrics = json.loads(eval_metrics_out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "evaluated frozen v2d embedding model: public_test_f2=" in captured.out
    assert "v2 ml progress: eval start model=yamnet_head phase=public_test" in captured.err
    assert eval_metrics["threshold"] == metrics["threshold"]
    assert eval_metrics["combined_supplementary"]["supplementary"] is True
    assert eval_metrics["combined_supplementary"]["counts"] == {"drone": 4, "no_drone": 2}


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
    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert metrics["combined_supplementary"]["supplementary"] is True
    assert metrics["combined_supplementary"]["counts"] == {"drone": 4, "no_drone": 2}
    assert (errors_out / "public_false_positives.csv").exists()
    assert (errors_out / "professor_missed_windows.csv").exists()


def test_eval_v2_cnn_cli_writes_combined_supplementary_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    features = _write_feature_bundle(tmp_path / "features.npz")
    public_root = tmp_path / "public"
    professor_root = tmp_path / "professor"
    public_root.mkdir()
    professor_root.mkdir()
    _write_audio_files_for_bundle(features, public_root, professor_root)
    model_out = tmp_path / "models" / "tiny_cnn.pt"
    train_metrics = tmp_path / "metrics" / "tiny_cnn_train.json"
    cache_dir = tmp_path / "cache"
    assert main(
        [
            "train-v2-cnn",
            "--features",
            str(features),
            "--public-audio-root",
            str(public_root),
            "--professor-audio-root",
            str(professor_root),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(train_metrics),
            "--cache-dir",
            str(cache_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ]
    ) == 0
    capsys.readouterr()
    metrics_out = tmp_path / "eval" / "tiny_cnn_eval.json"

    exit_code = main(
        [
            "eval-v2-cnn",
            "--model",
            str(model_out),
            "--features",
            str(features),
            "--public-audio-root",
            str(public_root),
            "--professor-audio-root",
            str(professor_root),
            "--cache-dir",
            str(cache_dir),
            "--metrics-out",
            str(metrics_out),
            "--batch-size",
            "4",
        ]
    )

    captured = capsys.readouterr()
    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "evaluated frozen v2c model: public_test_f2=" in captured.out
    assert "v2 ml progress: eval start model=tiny_cnn phase=public_test" in captured.err
    assert metrics["combined_supplementary"]["supplementary"] is True
    assert metrics["combined_supplementary"]["counts"] == {"drone": 4, "no_drone": 2}
    assert metrics["threshold"] == json.loads(train_metrics.read_text(encoding="utf-8"))["threshold"]


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


def test_v2_mlp_cli_reports_missing_features(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "train-v2-mlp",
            "--features",
            str(tmp_path / "missing.npz"),
            "--model-out",
            str(tmp_path / "mlp.joblib"),
            "--metrics-out",
            str(tmp_path / "mlp_metrics.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "v2 ml error: features does not exist" in captured.err


def test_v2_cli_rejects_unknown_embedding_model(tmp_path: Path) -> None:
    features = _write_feature_bundle(tmp_path / "features.npz")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "train-v2-embedding",
                "--features",
                str(features),
                "--public-audio-root",
                str(tmp_path),
                "--professor-audio-root",
                str(tmp_path),
                "--embedding-model",
                "vggish",
                "--head",
                "logistic_regression",
                "--model-out",
                str(tmp_path / "head.joblib"),
                "--metrics-out",
                str(tmp_path / "metrics.json"),
                "--cache-dir",
                str(tmp_path / "cache"),
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
        window_start_sec=np.asarray([0.0 for _ in rows]),
        window_end_sec=np.asarray([1.0 for _ in rows]),
        is_professor_data=np.asarray([row[2] for row in rows], dtype=bool),
        feature_names=np.asarray(["f0", "f1"]),
        feature_set=np.asarray("spectral_mfcc_v1"),
    )
    return path


def _subtype(label: str) -> str:
    if label == POSITIVE_LABEL:
        return "drone_field"
    return "no_drone_background"


def _write_audio_files_for_bundle(features: Path, public_root: Path, professor_root: Path) -> None:
    np = pytest.importorskip("numpy")
    with np.load(features, allow_pickle=False) as data:
        for original_file, is_professor, label in zip(
            data["original_files"],
            data["is_professor_data"],
            data["labels"],
        ):
            root = professor_root if bool(is_professor) else public_root
            level = 0.65 if str(label) == POSITIVE_LABEL else -0.65
            _write_fixture_wav(root / str(original_file), level)


def _write_fixture_wav(path: Path, level: float) -> None:
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(16_000, level, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())


def _fake_embedder(np: object) -> object:
    def embed(window: object) -> object:
        values = np.asarray(window, dtype=np.float32)
        return np.asarray(
            [values.mean(), values.std(), abs(values.mean()), 1.0],
            dtype=np.float32,
        )

    return embed
