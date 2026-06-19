from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import pytest

from drone_sound.cli import main
from drone_sound.training import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TrainingDataError,
    _roc_auc,
    _select_threshold,
    train_v1_baseline,
)


def test_train_v1_cli_writes_model_and_metrics(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_wav(audio_root / "train_drone.wav", amplitude=0.8, frequency_hz=440.0)
    _write_wav(audio_root / "train_no_drone.wav", amplitude=0.1, frequency_hz=220.0)
    _write_wav(audio_root / "val_drone.wav", amplitude=0.7, frequency_hz=440.0)
    _write_wav(audio_root / "val_no_drone.wav", amplitude=0.08, frequency_hz=220.0)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _row("train-drone", "train_drone.wav", "drone", "drone_field", "train"),
            _row(
                "train-no-drone",
                "train_no_drone.wav",
                "no_drone",
                "no_drone_urban_industrial",
                "train",
            ),
            _row("val-drone", "val_drone.wav", "drone", "drone_field", "val"),
            _row(
                "val-no-drone",
                "val_no_drone.wav",
                "no_drone",
                "no_drone_urban_industrial",
                "val",
            ),
        ],
    )
    model_out = tmp_path / "artifacts" / "models" / "model.json"
    metrics_out = tmp_path / "artifacts" / "models" / "metrics.json"

    exit_code = main(
        [
            "train-v1",
            "--manifest",
            str(manifest),
            "--audio-root",
            str(audio_root),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
        ]
    )

    assert exit_code == 0
    assert model_out.exists()
    assert metrics_out.exists()

    model = json.loads(model_out.read_text(encoding="utf-8"))
    assert model["algorithm"] == "v1_centroid_smoke_baseline"
    assert model["threshold_source"] == "val"
    assert model["labels"] == {"negative": "no_drone", "positive": "drone"}

    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert metrics["counts"]["train"] == {"drone": 1, "no_drone": 1}
    assert metrics["counts"]["val"] == {"drone": 1, "no_drone": 1}
    assert metrics["metrics"]["val"]["confusion_matrix"]["tp"] == 1
    assert metrics["metrics"]["val"]["confusion_matrix"]["tn"] == 1
    assert metrics["coverage"]["no_drone_speech_available"] is False
    assert "no_drone_speech" in metrics["coverage"]["missing_hard_negative_subtypes"]


def test_train_v1_rejects_professor_only_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "professor_only.csv"
    _write_manifest(
        manifest,
        [
            _row(
                "professor",
                "professor.wav",
                "drone",
                "drone_unknown_condition",
                "professor_test",
                is_professor_data="true",
            )
        ],
    )

    with pytest.raises(TrainingDataError, match="requires non-professor 'train' rows"):
        train_v1_baseline(
            manifest,
            tmp_path,
            tmp_path / "model.json",
            tmp_path / "metrics.json",
        )


def test_train_v1_requires_both_train_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "one_class_train.csv"
    _write_manifest(
        manifest,
        [
            _row("train-drone", "train_drone.wav", "drone", "drone_field", "train"),
            _row("val-drone", "val_drone.wav", "drone", "drone_field", "val"),
            _row(
                "val-no-drone",
                "val_no_drone.wav",
                "no_drone",
                "no_drone_urban_industrial",
                "val",
            ),
        ],
    )

    with pytest.raises(TrainingDataError, match="'train' rows are missing required labels"):
        train_v1_baseline(
            manifest,
            tmp_path,
            tmp_path / "model.json",
            tmp_path / "metrics.json",
        )


def test_select_threshold_matches_legacy_bruteforce_with_tied_scores() -> None:
    labels = [
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
        NEGATIVE_LABEL,
        POSITIVE_LABEL,
    ]
    scores = [0.8, 0.8, 0.4, 0.3, 0.1, 0.1]

    assert _select_threshold(labels, scores) == pytest.approx(
        _legacy_select_threshold(labels, scores)
    )


def test_roc_auc_uses_average_ranks_for_tied_scores() -> None:
    labels = [POSITIVE_LABEL, NEGATIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL]
    scores = [0.5, 0.5, 1.0, 0.0]

    assert _roc_auc(labels, scores) == pytest.approx(0.875)


def _legacy_select_threshold(labels: list[str], scores: list[float]) -> float:
    unique_scores = sorted(set(scores))
    candidates = [unique_scores[0] - 1e-12, unique_scores[-1] + 1e-12]
    candidates.extend(unique_scores)
    candidates.extend(
        (left + right) / 2.0 for left, right in zip(unique_scores, unique_scores[1:])
    )

    best_threshold = candidates[0]
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for label, score in zip(labels, scores):
            predicted = POSITIVE_LABEL if score >= threshold else NEGATIVE_LABEL
            if label == POSITIVE_LABEL and predicted == POSITIVE_LABEL:
                counts["tp"] += 1
            elif label == POSITIVE_LABEL:
                counts["fn"] += 1
            elif predicted == POSITIVE_LABEL:
                counts["fp"] += 1
            else:
                counts["tn"] += 1
        key = _legacy_threshold_key(counts)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _legacy_threshold_key(counts: dict[str, int]) -> tuple[float, float, float]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn

    precision = _legacy_div(tp, tp + fp)
    recall = _legacy_div(tp, tp + fn)
    specificity = _legacy_div(tn, tn + fp)
    f2 = _legacy_fbeta(precision, recall, beta=2.0)
    balanced_accuracy = _legacy_average_defined([recall, specificity])
    accuracy = _legacy_div(tp + tn, total)
    return (
        _legacy_metric_value(f2),
        _legacy_metric_value(balanced_accuracy),
        _legacy_metric_value(accuracy),
    )


def _legacy_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _legacy_fbeta(precision: float | None, recall: float | None, beta: float) -> float | None:
    if precision is None or recall is None:
        return None
    beta_squared = beta * beta
    denominator = (beta_squared * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def _legacy_average_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _legacy_metric_value(value: float | None) -> float:
    if value is None:
        return -1.0
    return value


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
    *,
    is_professor_data: str = "false",
) -> dict[str, str]:
    return {
        "clip_id": f"{prefix}-clip",
        "window_id": f"{prefix}-window",
        "source_dataset": "unit_test",
        "source_url_or_doi": "local",
        "license": "test",
        "original_file": original_file,
        "original_file_checksum": f"{prefix}-checksum",
        "sample_rate_original": "16000",
        "sample_rate_processed": "16000",
        "duration_sec": "1.000000",
        "window_start_sec": "0.000000",
        "window_end_sec": "1.000000",
        "label_binary": label_binary,
        "label_subtype": label_subtype,
        "drone_model": "",
        "environment": "",
        "recording_condition": "",
        "split": split,
        "is_professor_data": is_professor_data,
        "notes": "",
    }
