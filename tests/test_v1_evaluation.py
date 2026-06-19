from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import pytest

from drone_sound.cli import main
from drone_sound.training import (
    FEATURE_NAMES,
    POSITIVE_LABEL,
    TrainingExample,
    _scale,
    _score,
)
from drone_sound.v1_evaluation import (
    V1EvaluationError,
    load_frozen_v1_model,
    professor_test_metrics,
    score_v1_features,
)


def test_load_frozen_v1_model_rejects_unsupported_algorithm(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model = _model_dict()
    model["algorithm"] = "other"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(V1EvaluationError, match="unsupported V1 model algorithm"):
        load_frozen_v1_model(model_path)


def test_load_frozen_v1_model_rejects_missing_fields(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model = _model_dict()
    del model["centroids"]
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(V1EvaluationError, match="centroids"):
        load_frozen_v1_model(model_path)


def test_score_v1_features_matches_training_score_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(_model_dict()), encoding="utf-8")
    model = load_frozen_v1_model(model_path)
    features = (0.0, 0.4, 0.5, 0.7, 0.04)

    assert score_v1_features(features, model) == pytest.approx(
        _score(_scale(features, model.scaler), model.centroids)
    )


def test_professor_metrics_report_allowed_positive_only_fields() -> None:
    examples = [
        TrainingExample(
            row=_row("prof", "prof.wav", "drone", "drone_unknown_condition", "professor_test", i),
            label=POSITIVE_LABEL,
            features=(),
        )
        for i in range(5)
    ]
    scores = [0.9, 0.2, 0.1, 0.8, 0.2]

    metrics = professor_test_metrics(examples, scores, threshold=0.5)

    assert metrics["rows"] == 5
    assert metrics["recall"] == pytest.approx(0.4)
    assert metrics["missed_window_count"] == 3
    assert metrics["longest_missed_segment_sec"] == pytest.approx(1.5)
    assert metrics["mean_p_drone"] == pytest.approx(sum(scores) / len(scores))
    assert metrics["median_p_drone"] == pytest.approx(0.2)
    assert metrics["prediction_stability"]["decision_flip_count"] == 3
    assert "precision" not in metrics
    assert "false_positive_rate" not in metrics


def test_eval_v1_cli_writes_public_professor_and_combined_metrics(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    professor_root = tmp_path / "professor"
    public_root.mkdir()
    professor_root.mkdir()
    _write_wav(public_root / "public_drone.wav", amplitude=0.8, frequency_hz=440.0)
    _write_wav(public_root / "public_no_drone.wav", amplitude=0.1, frequency_hz=220.0)
    _write_wav(professor_root / "professor.wav", amplitude=0.8, frequency_hz=440.0)

    public_manifest = tmp_path / "public_manifest.csv"
    _write_manifest(
        public_manifest,
        [
            _row("public-drone", "public_drone.wav", "drone", "drone_field", "public_test", 0),
            _row(
                "public-no-drone",
                "public_no_drone.wav",
                "no_drone",
                "no_drone_urban_industrial",
                "public_test",
                0,
            ),
        ],
    )
    professor_manifest = tmp_path / "professor_manifest.csv"
    _write_manifest(
        professor_manifest,
        [
            _row(
                "professor",
                "professor.wav",
                "drone",
                "drone_unknown_condition",
                "professor_test",
                0,
                is_professor_data="true",
            ),
        ],
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(_model_dict()), encoding="utf-8")
    metrics_out = tmp_path / "v1_eval_metrics.json"

    exit_code = main(
        [
            "eval-v1",
            "--model",
            str(model_path),
            "--public-manifest",
            str(public_manifest),
            "--public-audio-root",
            str(public_root),
            "--professor-manifest",
            str(professor_manifest),
            "--professor-audio-root",
            str(professor_root),
            "--metrics-out",
            str(metrics_out),
        ]
    )

    assert exit_code == 0
    report = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert report["threshold"] == 0.5
    assert report["threshold_source"] == "val"
    assert report["public_test"]["counts"] == {"drone": 1, "no_drone": 1}
    assert report["professor_test"]["counts"] == {"drone": 1, "no_drone": 0}
    assert report["professor_test"]["metrics"]["rows"] == 1
    assert "precision" not in report["professor_test"]["metrics"]
    assert report["combined_supplementary"]["supplementary"] is True
    assert report["combined_supplementary"]["counts"] == {"drone": 2, "no_drone": 1}


def _model_dict() -> dict[str, object]:
    return {
        "algorithm": "v1_centroid_smoke_baseline",
        "feature_names": list(FEATURE_NAMES),
        "labels": {"positive": "drone", "negative": "no_drone"},
        "scaler": {
            "mean": [0.0] * len(FEATURE_NAMES),
            "std": [1.0] * len(FEATURE_NAMES),
        },
        "centroids": {
            "drone": [0.0, 0.5, 0.6, 0.8, 0.05],
            "no_drone": [0.0, 0.05, 0.07, 0.1, 0.03],
        },
        "threshold": 0.5,
        "threshold_source": "val",
        "training_manifest": "synthetic.csv",
    }


def _write_wav(
    path: Path,
    *,
    amplitude: float,
    frequency_hz: float,
    sample_rate_hz: int = 16_000,
    duration_sec: float = 1.0,
) -> None:
    frames = int(sample_rate_hz * duration_sec)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        samples = bytearray()
        for index in range(frames):
            value = amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate_hz)
            samples.extend(int(value * 32767).to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(samples))


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    prefix: str,
    original_file: str,
    label_binary: str,
    label_subtype: str,
    split: str,
    index: int,
    *,
    is_professor_data: str = "false",
) -> dict[str, str]:
    return {
        "clip_id": f"{prefix}-clip",
        "window_id": f"{prefix}-window-{index}",
        "source_dataset": "unit_test",
        "source_url_or_doi": "local",
        "license": "test",
        "original_file": original_file,
        "original_file_checksum": f"{prefix}-checksum",
        "sample_rate_original": "16000",
        "sample_rate_processed": "16000",
        "duration_sec": "3.000000",
        "window_start_sec": f"{index * 0.5:.6f}",
        "window_end_sec": f"{index * 0.5 + 1.0:.6f}",
        "label_binary": label_binary,
        "label_subtype": label_subtype,
        "drone_model": "",
        "environment": "",
        "recording_condition": "",
        "split": split,
        "is_professor_data": is_professor_data,
        "notes": "",
    }
