"""Evaluate a frozen V1 baseline model."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.training import (
    FEATURE_NAMES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    PUBLIC_TEST_SPLIT,
    TrainingExample,
    _label_counts,
    _load_examples,
    _metrics,
    _scale,
    _score,
)

PROFESSOR_TEST_SPLIT = "professor_test"
V1_ALGORITHM = "v1_centroid_smoke_baseline"


class V1EvaluationError(ValueError):
    """Raised when a frozen V1 model or evaluation data is invalid."""


@dataclass(frozen=True)
class FrozenV1Model:
    path: Path
    algorithm: str
    feature_names: tuple[str, ...]
    scaler: dict[str, list[float]]
    centroids: dict[str, tuple[float, ...]]
    threshold: float
    threshold_source: str
    training_manifest: str


def evaluate_frozen_v1_model(
    model_path: str | Path,
    public_manifest_path: str | Path,
    public_audio_root: str | Path,
    professor_manifest_path: str | Path,
    professor_audio_root: str | Path,
    metrics_out: str | Path,
) -> dict[str, Any]:
    """Evaluate a frozen V1 model on public and professor test data."""

    model_path = Path(model_path)
    public_manifest_path = Path(public_manifest_path)
    public_audio_root = Path(public_audio_root)
    professor_manifest_path = Path(professor_manifest_path)
    professor_audio_root = Path(professor_audio_root)
    metrics_out = Path(metrics_out)

    model = load_frozen_v1_model(model_path)

    public_rows_all = load_manifest_csv(public_manifest_path)
    validate_manifest_rows(public_rows_all)
    public_rows = _public_test_rows(public_rows_all)
    _require_rows(public_rows, PUBLIC_TEST_SPLIT)
    _require_both_labels(public_rows, PUBLIC_TEST_SPLIT)

    professor_rows_all = load_manifest_csv(professor_manifest_path)
    validate_manifest_rows(professor_rows_all)
    professor_rows = _professor_test_rows(professor_rows_all)
    _require_rows(professor_rows, PROFESSOR_TEST_SPLIT)
    _require_professor_positive_only(professor_rows)

    public_examples = _load_examples(public_rows, public_audio_root, f"eval {PUBLIC_TEST_SPLIT}")
    professor_examples = _load_examples(
        professor_rows,
        professor_audio_root,
        f"eval {PROFESSOR_TEST_SPLIT}",
    )

    public_scores = score_examples(public_examples, model)
    professor_scores = score_examples(professor_examples, model)

    public_labels = [example.label for example in public_examples]
    professor_labels = [example.label for example in professor_examples]

    public_test = {
        "manifest": str(public_manifest_path),
        "audio_root": str(public_audio_root),
        "counts": _label_counts(public_rows),
        "metrics": _metrics(public_labels, public_scores, model.threshold),
        "false_positive_rate_by_subtype": _false_positive_rate_by_subtype(
            public_examples,
            public_scores,
            model.threshold,
        ),
    }
    professor_test = {
        "manifest": str(professor_manifest_path),
        "audio_root": str(professor_audio_root),
        "counts": _label_counts(professor_rows),
        "metrics": professor_test_metrics(
            professor_examples,
            professor_scores,
            model.threshold,
        ),
    }

    combined_labels = public_labels + professor_labels
    combined_scores = public_scores + professor_scores
    combined_rows = public_rows + professor_rows
    combined_supplementary = {
        "supplementary": True,
        "note": (
            "Combined metrics include positive-only or near-positive-only professor data "
            "and do not replace the public_test benchmark."
        ),
        "counts": _label_counts(combined_rows),
        "metrics": _metrics(combined_labels, combined_scores, model.threshold),
    }

    report = {
        "algorithm": model.algorithm,
        "model_path": str(model_path),
        "threshold": model.threshold,
        "threshold_source": model.threshold_source,
        "public_test": public_test,
        "professor_test": professor_test,
        "combined_supplementary": combined_supplementary,
        "limitations": [
            "Professor data is positive-only or near-positive-only evaluation data.",
            "Professor data is not used for training, validation, or threshold tuning.",
            "Do not report professor-only false positive rate, specificity, accuracy, or precision.",
            "Combined supplementary metrics do not replace the public_test binary benchmark.",
        ],
    }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_frozen_v1_model(path: str | Path) -> FrozenV1Model:
    """Load and validate a frozen V1 model JSON file."""

    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise V1EvaluationError(f"invalid model JSON: {path}") from error

    if not isinstance(data, dict):
        raise V1EvaluationError("model JSON must be an object")

    algorithm = _required_string(data, "algorithm")
    if algorithm != V1_ALGORITHM:
        raise V1EvaluationError(f"unsupported V1 model algorithm: {algorithm}")

    feature_names = tuple(_required_string_list(data, "feature_names"))
    if feature_names != FEATURE_NAMES:
        raise V1EvaluationError("model feature_names do not match the V1 feature contract")

    labels = _required_mapping(data, "labels")
    if labels.get("positive") != POSITIVE_LABEL or labels.get("negative") != NEGATIVE_LABEL:
        raise V1EvaluationError("model labels do not match the V1 label contract")

    scaler = _required_mapping(data, "scaler")
    mean = _required_number_list(scaler, "mean", len(feature_names))
    std = _required_number_list(scaler, "std", len(feature_names))
    if any(value <= 0 for value in std):
        raise V1EvaluationError("model scaler std values must be positive")

    centroids_data = _required_mapping(data, "centroids")
    centroids = {
        POSITIVE_LABEL: tuple(
            _required_number_list(centroids_data, POSITIVE_LABEL, len(feature_names))
        ),
        NEGATIVE_LABEL: tuple(
            _required_number_list(centroids_data, NEGATIVE_LABEL, len(feature_names))
        ),
    }

    threshold = _required_number(data, "threshold")
    threshold_source = _required_string(data, "threshold_source")
    training_manifest = _optional_string(data, "training_manifest")

    return FrozenV1Model(
        path=path,
        algorithm=algorithm,
        feature_names=feature_names,
        scaler={"mean": mean, "std": std},
        centroids=centroids,
        threshold=threshold,
        threshold_source=threshold_source,
        training_manifest=training_manifest,
    )


def score_examples(examples: Iterable[TrainingExample], model: FrozenV1Model) -> list[float]:
    """Score examples with the frozen model scaler and centroids."""

    return [score_v1_features(example.features, model) for example in examples]


def score_v1_features(features: tuple[float, ...], model: FrozenV1Model) -> float:
    """Score one raw V1 feature tuple with a frozen model."""

    if len(features) != len(model.feature_names):
        raise V1EvaluationError("feature vector length does not match frozen model")
    return _score(_scale(features, model.scaler), model.centroids)


def professor_test_metrics(
    examples: list[TrainingExample],
    scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    """Return professor-only metrics allowed by the evaluation contract."""

    if len(examples) != len(scores):
        raise V1EvaluationError("professor examples and scores length mismatch")
    if not examples:
        raise V1EvaluationError("cannot evaluate professor_test without rows")

    predictions = [score >= threshold for score in scores]
    missed = [not prediction for prediction in predictions]
    missed_count = sum(1 for value in missed if value)
    rows = len(examples)
    return {
        "rows": rows,
        "threshold": threshold,
        "recall": (rows - missed_count) / rows,
        "missed_window_count": missed_count,
        "longest_missed_segment_sec": _longest_missed_segment_sec(examples, missed),
        "mean_p_drone": statistics.fmean(scores),
        "median_p_drone": statistics.median(scores),
        "prediction_stability": _prediction_stability(predictions, scores),
    }


def _public_test_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    selected = []
    for row in rows:
        if str(row.get("split", "")).strip() != PUBLIC_TEST_SPLIT:
            continue
        if _truthy(row.get("is_professor_data", "")):
            continue
        label = str(row.get("label_binary", "")).strip()
        if label in {POSITIVE_LABEL, NEGATIVE_LABEL}:
            selected.append(row)
    return selected


def _professor_test_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    selected = []
    for row in rows:
        if str(row.get("split", "")).strip() != PROFESSOR_TEST_SPLIT:
            continue
        if not _truthy(row.get("is_professor_data", "")):
            continue
        selected.append(row)
    return selected


def _require_rows(rows: list[Mapping[str, str]], split: str) -> None:
    if not rows:
        raise V1EvaluationError(f"eval-v1 requires {split!r} rows")


def _require_both_labels(rows: list[Mapping[str, str]], split: str) -> None:
    labels = {str(row.get("label_binary", "")).strip() for row in rows}
    missing = {POSITIVE_LABEL, NEGATIVE_LABEL} - labels
    if missing:
        raise V1EvaluationError(
            f"{split!r} rows are missing labels: {', '.join(sorted(missing))}"
        )


def _require_professor_positive_only(rows: list[Mapping[str, str]]) -> None:
    labels = {str(row.get("label_binary", "")).strip() for row in rows}
    if labels != {POSITIVE_LABEL}:
        raise V1EvaluationError(
            "professor_test rows must be positive-only for V1 professor evaluation"
        )


def _false_positive_rate_by_subtype(
    examples: list[TrainingExample],
    scores: list[float],
    threshold: float,
) -> dict[str, dict[str, int | float | None]]:
    grouped: dict[str, list[float]] = {}
    for example, score in zip(examples, scores):
        if example.label != NEGATIVE_LABEL:
            continue
        subtype = str(example.row.get("label_subtype", "")).strip() or "no_drone_unknown"
        grouped.setdefault(subtype, []).append(score)

    report = {}
    for subtype, subtype_scores in sorted(grouped.items()):
        false_positives = sum(1 for score in subtype_scores if score >= threshold)
        true_negatives = len(subtype_scores) - false_positives
        report[subtype] = {
            "rows": len(subtype_scores),
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_positive_rate": false_positives / len(subtype_scores)
            if subtype_scores
            else None,
        }
    return report


def _longest_missed_segment_sec(examples: list[TrainingExample], missed: list[bool]) -> float:
    longest = 0.0
    active_clip = ""
    active_start: float | None = None
    active_end: float | None = None

    ordered = sorted(
        zip(examples, missed),
        key=lambda item: (
            str(item[0].row.get("clip_id", "")),
            _row_float(item[0].row, "window_start_sec"),
            _row_float(item[0].row, "window_end_sec"),
        ),
    )
    for example, is_missed in ordered:
        clip_id = str(example.row.get("clip_id", ""))
        start = _row_float(example.row, "window_start_sec")
        end = _row_float(example.row, "window_end_sec")
        if not is_missed:
            longest = _close_segment(longest, active_start, active_end)
            active_clip = ""
            active_start = None
            active_end = None
            continue

        if active_start is None or active_end is None or clip_id != active_clip:
            longest = _close_segment(longest, active_start, active_end)
            active_clip = clip_id
            active_start = start
            active_end = end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            longest = _close_segment(longest, active_start, active_end)
            active_start = start
            active_end = end

    return _close_segment(longest, active_start, active_end)


def _prediction_stability(predictions: list[bool], scores: list[float]) -> dict[str, float | int]:
    flip_count = sum(
        1
        for previous, current in zip(predictions, predictions[1:])
        if previous != current
    )
    comparable = max(0, len(predictions) - 1)
    return {
        "decision_flip_count": flip_count,
        "decision_flip_rate": flip_count / comparable if comparable else 0.0,
        "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }


def _close_segment(longest: float, start: float | None, end: float | None) -> float:
    if start is None or end is None:
        return longest
    return max(longest, end - start)


def _row_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(str(row[field]).strip())
    except (KeyError, ValueError) as error:
        raise V1EvaluationError(f"invalid {field!r} in evaluation manifest") from error
    if not math.isfinite(value):
        raise V1EvaluationError(f"non-finite {field!r} in evaluation manifest")
    return value


def _required_mapping(data: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise V1EvaluationError(f"model field {field!r} must be an object")
    return value


def _required_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise V1EvaluationError(f"model field {field!r} must be a non-empty string")
    return value


def _optional_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field, "")
    if not isinstance(value, str):
        raise V1EvaluationError(f"model field {field!r} must be a string")
    return value


def _required_string_list(data: Mapping[str, Any], field: str) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise V1EvaluationError(f"model field {field!r} must be a string list")
    return value


def _required_number_list(data: Mapping[str, Any], field: str, length: int) -> list[float]:
    value = data.get(field)
    if not isinstance(value, list) or len(value) != length:
        raise V1EvaluationError(f"model field {field!r} must contain {length} numbers")
    numbers = [_number(item, f"model field {field!r}") for item in value]
    return numbers


def _required_number(data: Mapping[str, Any], field: str) -> float:
    return _number(data.get(field), f"model field {field!r}")


def _number(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise V1EvaluationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V1EvaluationError(f"{context} must be finite")
    return result


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}
