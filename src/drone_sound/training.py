"""V1 training smoke baseline."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.wav_audio import WavReadError, read_wav_mono, read_wav_mono_array

try:
    import numpy as np
except ImportError:  # pragma: no cover - base dependency path.
    np = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency.
    tqdm = None

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

    train_examples = _load_examples(train_rows, audio_root, TRAIN_SPLIT)
    val_examples = _load_examples(val_rows, audio_root, VAL_SPLIT)
    public_test_examples = _load_examples(public_test_rows, audio_root, PUBLIC_TEST_SPLIT)

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


def _load_examples(
    rows: Iterable[Mapping[str, str]],
    audio_root: Path,
    split: str,
) -> list[TrainingExample]:
    rows_by_audio_path: dict[Path, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        audio_path = _resolve_audio_path(audio_root, str(row["original_file"]).strip())
        rows_by_audio_path[audio_path].append(row)

    examples = []
    progress = _progress(
        total=sum(len(file_rows) for file_rows in rows_by_audio_path.values()),
        desc=f"extract {split}",
        unit="window",
    )
    for audio_path, file_rows in rows_by_audio_path.items():
        progress.set_postfix_str(
            f"{audio_path.name[:40]} ({len(file_rows)} windows)",
            refresh=False,
        )
        if np is not None:
            try:
                samples_array, metadata = read_wav_mono_array(audio_path)
            except WavReadError as error:
                raise TrainingDataError(str(error)) from error
            for row in file_rows:
                features = _extract_features_from_array(
                    audio_path=audio_path,
                    samples=samples_array,
                    sample_rate_hz=metadata.sample_rate_hz,
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
        else:
            try:
                samples, metadata = read_wav_mono(audio_path)
            except WavReadError as error:
                raise TrainingDataError(str(error)) from error
            for row in file_rows:
                features = _extract_features_from_samples(
                    audio_path=audio_path,
                    samples=samples,
                    sample_rate_hz=metadata.sample_rate_hz,
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
        progress.update(len(file_rows))
    progress.close()
    return examples


def _progress(total: int, desc: str, unit: str) -> Any:
    if tqdm is None:
        return _NullProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=1.0,
        miniters=1_000,
    )


class _NullProgress:
    def set_postfix_str(self, value: str, refresh: bool | None = None) -> None:
        return None

    def update(self, count: int) -> None:
        return None

    def close(self) -> None:
        return None


def _extract_features_from_array(
    audio_path: Path,
    samples: Any,
    sample_rate_hz: int,
    start_sec: float,
    end_sec: float,
) -> tuple[float, ...]:
    if np is None:
        raise TrainingDataError("numpy feature extraction requested without numpy")
    if end_sec <= start_sec:
        raise TrainingDataError(f"invalid window bounds for {audio_path}: {start_sec}..{end_sec}")

    start_index = int(round(start_sec * sample_rate_hz))
    sample_count = int(round((end_sec - start_sec) * sample_rate_hz))
    end_index = start_index + sample_count
    if start_index < 0 or end_index > len(samples):
        raise TrainingDataError(f"manifest window exceeds audio duration: {audio_path}")

    window_samples = samples[start_index:end_index]
    if len(window_samples) == 0:
        raise TrainingDataError(f"empty audio window: {audio_path}")

    abs_samples = np.abs(window_samples)
    signs = window_samples >= 0
    crossings = np.count_nonzero(signs[1:] != signs[:-1]) if len(window_samples) > 1 else 0
    zero_crossing_rate = crossings / (len(window_samples) - 1) if len(window_samples) > 1 else 0.0
    return (
        float(np.mean(window_samples)),
        float(np.mean(abs_samples)),
        float(np.sqrt(np.mean(window_samples * window_samples))),
        float(np.max(abs_samples)),
        float(zero_crossing_rate),
    )


def _resolve_audio_path(audio_root: Path, original_file: str) -> Path:
    path = Path(original_file)
    if not path.is_absolute():
        path = audio_root / path
    if not path.exists():
        raise TrainingDataError(f"audio file not found for manifest row: {path}")
    return path


def _extract_features(audio_path: Path, start_sec: float, end_sec: float) -> tuple[float, ...]:
    try:
        samples, metadata = read_wav_mono(audio_path)
    except WavReadError as error:
        raise TrainingDataError(str(error)) from error
    return _extract_features_from_samples(
        audio_path=audio_path,
        samples=samples,
        sample_rate_hz=metadata.sample_rate_hz,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def _extract_features_from_samples(
    audio_path: Path,
    samples: list[float],
    sample_rate_hz: int,
    start_sec: float,
    end_sec: float,
) -> tuple[float, ...]:
    if end_sec <= start_sec:
        raise TrainingDataError(f"invalid window bounds for {audio_path}: {start_sec}..{end_sec}")

    start_index = int(round(start_sec * sample_rate_hz))
    sample_count = int(round((end_sec - start_sec) * sample_rate_hz))
    end_index = start_index + sample_count
    if start_index < 0 or end_index > len(samples):
        raise TrainingDataError(f"manifest window exceeds audio duration: {audio_path}")

    window_samples = samples[start_index:end_index]
    if not window_samples:
        raise TrainingDataError(f"empty audio window: {audio_path}")

    mean = sum(window_samples) / len(window_samples)
    mean_abs = sum(abs(sample) for sample in window_samples) / len(window_samples)
    rms = math.sqrt(sum(sample * sample for sample in window_samples) / len(window_samples))
    peak_abs = max(abs(sample) for sample in window_samples)
    zero_crossing_rate = _zero_crossing_rate(window_samples)
    return (mean, mean_abs, rms, peak_abs, zero_crossing_rate)


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
    if len(labels) != len(scores):
        raise TrainingDataError("cannot select threshold with mismatched labels and scores")

    unique_scores = sorted(set(scores))
    positive_total = sum(1 for label in labels if label == POSITIVE_LABEL)
    negative_total = sum(1 for label in labels if label == NEGATIVE_LABEL)
    grouped_counts: dict[float, dict[str, int]] = {
        score: {POSITIVE_LABEL: 0, NEGATIVE_LABEL: 0} for score in unique_scores
    }
    for label, score in zip(labels, scores):
        if label in grouped_counts[score]:
            grouped_counts[score][label] += 1

    best_threshold = unique_scores[0] - 1e-12
    best_key = (-1.0, -1.0, -1.0)

    def consider(threshold: float, counts: dict[str, int]) -> None:
        nonlocal best_threshold, best_key
        key = _threshold_key_from_counts(counts)
        if key > best_key:
            best_key = key
            best_threshold = threshold

    consider(
        unique_scores[0] - 1e-12,
        {"tp": positive_total, "tn": 0, "fp": negative_total, "fn": 0},
    )
    consider(
        unique_scores[-1] + 1e-12,
        {"tp": 0, "tn": negative_total, "fp": 0, "fn": positive_total},
    )

    tp = 0
    fp = 0
    for threshold in reversed(unique_scores):
        tp += grouped_counts[threshold][POSITIVE_LABEL]
        fp += grouped_counts[threshold][NEGATIVE_LABEL]
        counts = {
            "tp": tp,
            "tn": negative_total - fp,
            "fp": fp,
            "fn": positive_total - tp,
        }
        consider(threshold, counts)
    return best_threshold


def _threshold_key_from_counts(counts: Mapping[str, int]) -> tuple[float, float, float]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f2 = _fbeta(precision, recall, beta=2.0)
    balanced_accuracy = _average_defined(value for value in (recall, specificity))
    accuracy = _safe_div(tp + tn, total)
    return (
        _metric_value(f2),
        _metric_value(balanced_accuracy),
        _metric_value(accuracy),
    )


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
    positive_count = sum(1 for label in labels if label == POSITIVE_LABEL)
    negative_count = sum(1 for label in labels if label == NEGATIVE_LABEL)
    if positive_count == 0 or negative_count == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank = 1
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        end = index + 1
        while end < len(pairs) and pairs[end][0] == score:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        positive_rank_sum += sum(
            average_rank for _, label in pairs[index:end] if label == POSITIVE_LABEL
        )
        rank += end - index
        index = end

    wins = positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)
    return wins / (positive_count * negative_count)


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
