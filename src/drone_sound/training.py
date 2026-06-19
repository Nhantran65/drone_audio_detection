"""V1 training smoke baseline."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from drone_sound.manifest import load_manifest_csv, validate_manifest_rows

POSITIVE_LABEL = "drone"
NEGATIVE_LABEL = "no_drone"
FEATURE_NAMES = (
    "mean",
    "mean_abs",
    "rms",
    "peak_abs",
    "zero_crossing_rate",
)
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
PUBLIC_TEST_SPLIT = "public_test"


class TrainingDataError(ValueError):
    """Raised when manifest or audio data cannot support v1 training."""


@dataclass(frozen=True)
class TrainingExample:
    row: Mapping[str, str]
    label: str
    features: tuple[float, ...]


def train_v1_baseline(
    manifest_path: str | Path,
    audio_root: str | Path,
    model_out: str | Path,
    metrics_out: str | Path,
) -> dict[str, Any]:
    """Train a tiny dependency-free baseline and write model/metrics JSON."""

    manifest_path = Path(manifest_path)
    audio_root = Path(audio_root)
    model_out = Path(model_out)
    metrics_out = Path(metrics_out)

    rows = load_manifest_csv(manifest_path)
    validate_manifest_rows(rows)

    train_rows = _eligible_rows(rows, TRAIN_SPLIT)
    val_rows = _eligible_rows(rows, VAL_SPLIT)
    public_test_rows = _eligible_rows(rows, PUBLIC_TEST_SPLIT)

    _require_rows(train_rows, TRAIN_SPLIT)
    _require_rows(val_rows, VAL_SPLIT)
    _require_both_labels(train_rows, TRAIN_SPLIT)
    _require_both_labels(val_rows, VAL_SPLIT)

    train_examples = _load_examples(train_rows, audio_root)
    val_examples = _load_examples(val_rows, audio_root)
    public_test_examples = _load_examples(public_test_rows, audio_root)

    scaler = _fit_scaler([example.features for example in train_examples])
    normalized_train = [
        TrainingExample(example.row, example.label, _scale(example.features, scaler))
        for example in train_examples
    ]
    normalized_val = [
        TrainingExample(example.row, example.label, _scale(example.features, scaler))
        for example in val_examples
    ]
    normalized_public_test = [
        TrainingExample(example.row, example.label, _scale(example.features, scaler))
        for example in public_test_examples
    ]

    centroids = _fit_centroids(normalized_train)
    val_scores = [_score(example.features, centroids) for example in normalized_val]
    threshold = _select_threshold(
        labels=[example.label for example in normalized_val],
        scores=val_scores,
    )

    val_metrics = _metrics(
        labels=[example.label for example in normalized_val],
        scores=val_scores,
        threshold=threshold,
    )
    public_test_metrics = None
    if normalized_public_test:
        public_test_metrics = _metrics(
            labels=[example.label for example in normalized_public_test],
            scores=[_score(example.features, centroids) for example in normalized_public_test],
            threshold=threshold,
        )

    coverage_rows = val_rows + public_test_rows
    report = {
        "algorithm": "v1_centroid_smoke_baseline",
        "manifest": str(manifest_path),
        "audio_root": str(audio_root),
        "counts": {
            "train": _label_counts(train_rows),
            "val": _label_counts(val_rows),
            "public_test": _label_counts(public_test_rows),
        },
        "threshold": threshold,
        "threshold_source": VAL_SPLIT,
        "metrics": {
            "val": val_metrics,
            "public_test": public_test_metrics,
        },
        "coverage": {
            "no_drone_speech_available": _has_subtype(coverage_rows, "no_drone_speech"),
            "missing_hard_negative_subtypes": _missing_hard_negative_subtypes(coverage_rows),
        },
        "limitations": [
            "dependency-free smoke baseline, not a product-quality model",
            "professor data is excluded from training, validation, and threshold selection",
            "speech-only false-positive coverage is unavailable when no_drone_speech is missing",
        ],
    }

    model = {
        "algorithm": report["algorithm"],
        "feature_names": FEATURE_NAMES,
        "labels": {
            "positive": POSITIVE_LABEL,
            "negative": NEGATIVE_LABEL,
        },
        "scaler": {
            "mean": scaler["mean"],
            "std": scaler["std"],
        },
        "centroids": {
            POSITIVE_LABEL: list(centroids[POSITIVE_LABEL]),
            NEGATIVE_LABEL: list(centroids[NEGATIVE_LABEL]),
        },
        "threshold": threshold,
        "threshold_source": VAL_SPLIT,
        "training_manifest": str(manifest_path),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _eligible_rows(rows: Iterable[Mapping[str, str]], split: str) -> list[Mapping[str, str]]:
    eligible = []
    for row in rows:
        if str(row.get("split", "")).strip() != split:
            continue
        if _truthy(row.get("is_professor_data", "")):
            continue
        label = str(row.get("label_binary", "")).strip()
        if label not in {POSITIVE_LABEL, NEGATIVE_LABEL}:
            continue
        eligible.append(row)
    return eligible


def _require_rows(rows: list[Mapping[str, str]], split: str) -> None:
    if not rows:
        raise TrainingDataError(
            f"v1 training requires non-professor {split!r} rows with drone/no_drone labels"
        )


def _require_both_labels(rows: list[Mapping[str, str]], split: str) -> None:
    labels = {str(row["label_binary"]).strip() for row in rows}
    missing = {POSITIVE_LABEL, NEGATIVE_LABEL} - labels
    if missing:
        joined = ", ".join(sorted(missing))
        raise TrainingDataError(f"{split!r} rows are missing required labels: {joined}")


def _load_examples(rows: Iterable[Mapping[str, str]], audio_root: Path) -> list[TrainingExample]:
    examples = []
    for row in rows:
        audio_path = _resolve_audio_path(audio_root, str(row["original_file"]).strip())
        features = _extract_features(
            audio_path=audio_path,
            start_sec=float(str(row["window_start_sec"]).strip()),
            end_sec=float(str(row["window_end_sec"]).strip()),
        )
        examples.append(
            TrainingExample(
                row=row,
                label=str(row["label_binary"]).strip(),
                features=features,
            )
        )
    return examples


def _resolve_audio_path(audio_root: Path, original_file: str) -> Path:
    path = Path(original_file)
    if not path.is_absolute():
        path = audio_root / path
    if not path.exists():
        raise TrainingDataError(f"audio file not found for manifest row: {path}")
    return path


def _extract_features(audio_path: Path, start_sec: float, end_sec: float) -> tuple[float, ...]:
    if end_sec <= start_sec:
        raise TrainingDataError(f"invalid window bounds for {audio_path}: {start_sec}..{end_sec}")

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        start_frame = int(round(start_sec * sample_rate))
        frame_count = int(round((end_sec - start_sec) * sample_rate))
        if start_frame + frame_count > wav_file.getnframes():
            raise TrainingDataError(f"manifest window exceeds audio duration: {audio_path}")
        wav_file.setpos(start_frame)
        data = wav_file.readframes(frame_count)

    samples = _decode_pcm(data, channels=channels, sample_width=sample_width)
    if not samples:
        raise TrainingDataError(f"empty audio window: {audio_path}")

    mean = sum(samples) / len(samples)
    mean_abs = sum(abs(sample) for sample in samples) / len(samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak_abs = max(abs(sample) for sample in samples)
    zero_crossing_rate = _zero_crossing_rate(samples)
    return (mean, mean_abs, rms, peak_abs, zero_crossing_rate)


def _decode_pcm(data: bytes, channels: int, sample_width: int) -> list[float]:
    if channels < 1:
        raise TrainingDataError("audio must have at least one channel")
    if sample_width not in {1, 2, 4}:
        raise TrainingDataError(f"unsupported PCM sample width: {sample_width} bytes")

    frame_width = channels * sample_width
    samples: list[float] = []
    for frame_offset in range(0, len(data), frame_width):
        channel_values = []
        for channel_index in range(channels):
            sample_offset = frame_offset + (channel_index * sample_width)
            raw = data[sample_offset : sample_offset + sample_width]
            if len(raw) != sample_width:
                continue
            if sample_width == 1:
                value = (raw[0] - 128) / 128.0
            else:
                value = int.from_bytes(raw, byteorder="little", signed=True)
                value = value / float(2 ** ((8 * sample_width) - 1))
            channel_values.append(value)
        if channel_values:
            samples.append(sum(channel_values) / len(channel_values))
    return samples


def _zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = 1 if samples[0] >= 0 else -1
    for sample in samples[1:]:
        current = 1 if sample >= 0 else -1
        if current != previous:
            crossings += 1
        previous = current
    return crossings / (len(samples) - 1)


def _fit_scaler(features: list[tuple[float, ...]]) -> dict[str, list[float]]:
    if not features:
        raise TrainingDataError("cannot fit scaler without training examples")
    means = []
    stds = []
    for index in range(len(features[0])):
        values = [feature[index] for feature in features]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)
    return {"mean": means, "std": stds}


def _scale(features: tuple[float, ...], scaler: Mapping[str, list[float]]) -> tuple[float, ...]:
    return tuple(
        (value - scaler["mean"][index]) / scaler["std"][index]
        for index, value in enumerate(features)
    )


def _fit_centroids(examples: list[TrainingExample]) -> dict[str, tuple[float, ...]]:
    grouped: dict[str, list[tuple[float, ...]]] = {
        POSITIVE_LABEL: [],
        NEGATIVE_LABEL: [],
    }
    for example in examples:
        grouped[example.label].append(example.features)

    centroids = {}
    for label, features in grouped.items():
        if not features:
            raise TrainingDataError(f"cannot fit centroid without {label!r} examples")
        centroids[label] = tuple(
            sum(feature[index] for feature in features) / len(features)
            for index in range(len(features[0]))
        )
    return centroids


def _score(features: tuple[float, ...], centroids: Mapping[str, tuple[float, ...]]) -> float:
    drone_distance = _euclidean(features, centroids[POSITIVE_LABEL])
    no_drone_distance = _euclidean(features, centroids[NEGATIVE_LABEL])
    return 1.0 / (1.0 + math.exp(drone_distance - no_drone_distance))


def _euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _select_threshold(labels: list[str], scores: list[float]) -> float:
    if not labels:
        raise TrainingDataError("cannot select threshold without validation examples")

    unique_scores = sorted(set(scores))
    candidates = [unique_scores[0] - 1e-12, unique_scores[-1] + 1e-12]
    candidates.extend(unique_scores)
    candidates.extend(
        (left + right) / 2.0 for left, right in zip(unique_scores, unique_scores[1:])
    )

    best_threshold = candidates[0]
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        metrics = _metrics(labels, scores, threshold)
        key = (
            _metric_value(metrics["f2"]),
            _metric_value(metrics["balanced_accuracy"]),
            _metric_value(metrics["accuracy"]),
        )
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _metrics(labels: list[str], scores: list[float], threshold: float) -> dict[str, Any]:
    counts = _confusion_counts(labels, scores, threshold)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _fbeta(precision, recall, beta=1.0)
    f2 = _fbeta(precision, recall, beta=2.0)

    return {
        "rows": total,
        "threshold": threshold,
        "accuracy": _safe_div(tp + tn, total),
        "balanced_accuracy": _average_defined(precision_like for precision_like in (recall, specificity)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "roc_auc": _roc_auc(labels, scores),
        "pr_auc": _pr_auc(labels, scores),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def _confusion_counts(labels: list[str], scores: list[float], threshold: float) -> dict[str, int]:
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
    return counts


def _fbeta(precision: float | None, recall: float | None, beta: float) -> float | None:
    if precision is None or recall is None:
        return None
    beta_squared = beta * beta
    denominator = (beta_squared * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def _roc_auc(labels: list[str], scores: list[float]) -> float | None:
    positive_scores = [score for label, score in zip(labels, scores) if label == POSITIVE_LABEL]
    negative_scores = [score for label, score in zip(labels, scores) if label == NEGATIVE_LABEL]
    if not positive_scores or not negative_scores:
        return None

    wins = 0.0
    for positive_score in positive_scores:
        for negative_score in negative_scores:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def _pr_auc(labels: list[str], scores: list[float]) -> float | None:
    positives = sum(1 for label in labels if label == POSITIVE_LABEL)
    if positives == 0:
        return None

    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in pairs:
        if label == POSITIVE_LABEL:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _label_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    counts = {POSITIVE_LABEL: 0, NEGATIVE_LABEL: 0}
    for row in rows:
        label = str(row.get("label_binary", "")).strip()
        if label in counts:
            counts[label] += 1
    return counts


def _has_subtype(rows: Iterable[Mapping[str, str]], subtype: str) -> bool:
    return any(str(row.get("label_subtype", "")).strip() == subtype for row in rows)


def _missing_hard_negative_subtypes(rows: Iterable[Mapping[str, str]]) -> list[str]:
    present = {
        str(row.get("label_subtype", "")).strip()
        for row in rows
        if str(row.get("label_binary", "")).strip() == NEGATIVE_LABEL
    }
    expected = {
        "no_drone_aircraft",
        "no_drone_traffic",
        "no_drone_urban_industrial",
        "no_drone_speech",
        "no_drone_background",
    }
    return sorted(expected - present)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _average_defined(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _metric_value(value: float | None) -> float:
    if value is None:
        return -1.0
    return value
