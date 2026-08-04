"""V2A classical ML training and frozen evaluation."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from drone_sound.training import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    PUBLIC_TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    _label_counts,
    _metrics,
)
from drone_sound.audio_preprocessing import (
    CANONICAL_PREPROCESSING_PROFILE,
    LEGACY_PREPROCESSING_PROFILE,
    AudioPreprocessingError,
    require_compatible_preprocessing,
)
from drone_sound.v2 import (
    DEPLOYABILITY_LABELS,
    THRESHOLD_NAMES,
    V2A_MODEL_TYPES,
    validate_model_types,
)

PROFESSOR_TEST_SPLIT = "professor_test"
V2A_TRACK = "v2a"
V2B_TRACK = "v2b"
V2A_FEATURE_SET = "spectral_mfcc_v1"
MODEL_SCHEMA_VERSION = "v2a_model_v1"
PRIMARY_THRESHOLD_NAME = "threshold_f2"
RANDOM_SEED = 20260619
V2B_DEFAULT_EPOCHS = 8
V2B_DEFAULT_BATCH_SIZE = 4096
V2B_DEFAULT_PATIENCE = 3
V2_DEFAULT_MIN_DELTA = 0.0
FIXED_FPR_THRESHOLDS = {
    "threshold_at_fpr_10": 0.10,
    "threshold_at_fpr_5": 0.05,
    "threshold_at_fpr_2": 0.02,
    "threshold_at_fpr_1": 0.01,
}
REQUIRED_BUNDLE_ARRAYS = (
    "scaled_features",
    "labels",
    "splits",
    "window_ids",
    "clip_ids",
    "original_files",
    "source_datasets",
    "label_subtypes",
    "window_start_sec",
    "window_end_sec",
    "is_professor_data",
    "feature_names",
    "feature_set",
)
COMPARISON_COLUMNS = (
    "model",
    "track",
    "feature/input",
    "validation_f2",
    "validation_fpr",
    "public_f2",
    "public_f1",
    "public_precision",
    "public_recall",
    "public_fpr",
    "professor_recall",
    "longest_missed_segment",
    "model_size",
    "params",
    "deployability_label",
)


class V2MLError(ValueError):
    """Raised when V2 ML training or evaluation cannot complete."""


@dataclass(frozen=True)
class V2FeatureBundle:
    """Loaded V2 feature bundle with row-level metadata."""

    path: Path
    feature_set: str
    feature_names: tuple[str, ...]
    features: Any
    labels: tuple[str, ...]
    splits: tuple[str, ...]
    window_ids: tuple[str, ...]
    clip_ids: tuple[str, ...]
    original_files: tuple[str, ...]
    source_datasets: tuple[str, ...]
    label_subtypes: tuple[str, ...]
    environments: tuple[str, ...]
    window_start_sec: tuple[float, ...]
    window_end_sec: tuple[float, ...]
    is_professor_data: tuple[bool, ...]
    source_sample_rates_hz: tuple[int, ...]
    processed_sample_rates_hz: tuple[int, ...]
    preprocessing_profile: str
    processed_sample_rate_hz: int | None
    bundle_fingerprint: str
    bundle_schema_version: str

    @property
    def row_count(self) -> int:
        return len(self.labels)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def load_feature_bundle(path: str | Path) -> V2FeatureBundle:
    """Load and validate a V2 feature bundle for ML training/evaluation."""

    np = _require_numpy()
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise V2MLError(f"features does not exist: {bundle_path}")
    if not bundle_path.is_file():
        raise V2MLError(f"features is not a file: {bundle_path}")

    try:
        with np.load(bundle_path, allow_pickle=False) as data:
            missing = [name for name in REQUIRED_BUNDLE_ARRAYS if name not in data.files]
            if missing:
                joined = ", ".join(missing)
                raise V2MLError(
                    "feature bundle missing required arrays: "
                    f"{joined}; regenerate features with extract-v2-features"
                )

            features = np.asarray(data["scaled_features"], dtype=np.float64)
            labels = _string_tuple(data["labels"])
            splits = _string_tuple(data["splits"])
            window_ids = _string_tuple(data["window_ids"])
            clip_ids = _string_tuple(data["clip_ids"])
            original_files = _string_tuple(data["original_files"])
            source_datasets = _string_tuple(data["source_datasets"])
            label_subtypes = _string_tuple(data["label_subtypes"])
            window_start_sec = tuple(float(value) for value in data["window_start_sec"])
            window_end_sec = tuple(float(value) for value in data["window_end_sec"])
            is_professor_data = tuple(bool(value) for value in data["is_professor_data"])
            feature_names = _string_tuple(data["feature_names"])
            feature_set = _scalar_string(data["feature_set"])
            environments = (
                _string_tuple(data["environments"])
                if "environments" in data.files
                else tuple("" for _ in labels)
            )
            source_sample_rates_hz = (
                tuple(int(value) for value in data["source_sample_rates_hz"])
                if "source_sample_rates_hz" in data.files
                else tuple(0 for _ in labels)
            )
            processed_sample_rates_hz = (
                tuple(int(value) for value in data["processed_sample_rates_hz"])
                if "processed_sample_rates_hz" in data.files
                else tuple(0 for _ in labels)
            )
            preprocessing_profile = (
                _scalar_string(data["preprocessing_profile"])
                if "preprocessing_profile" in data.files
                else LEGACY_PREPROCESSING_PROFILE
            )
            processed_sample_rate_hz = (
                int(data["processed_sample_rate_hz"].item())
                if "processed_sample_rate_hz" in data.files
                else None
            )
            bundle_fingerprint = (
                _scalar_string(data["bundle_fingerprint"])
                if "bundle_fingerprint" in data.files
                else _sha256_file(bundle_path)
            )
            bundle_schema_version = (
                _scalar_string(data["bundle_schema_version"])
                if "bundle_schema_version" in data.files
                else "v2_feature_bundle_v1_legacy"
            )
    except V2MLError:
        raise
    except Exception as error:
        raise V2MLError(f"invalid feature bundle: {bundle_path}") from error

    if features.ndim != 2 or features.shape[0] == 0:
        raise V2MLError("feature bundle scaled_features must be a non-empty 2D matrix")
    row_count = features.shape[0]
    _require_row_length("labels", labels, row_count)
    _require_row_length("splits", splits, row_count)
    _require_row_length("window_ids", window_ids, row_count)
    _require_row_length("clip_ids", clip_ids, row_count)
    _require_row_length("original_files", original_files, row_count)
    _require_row_length("source_datasets", source_datasets, row_count)
    _require_row_length("label_subtypes", label_subtypes, row_count)
    _require_row_length("environments", environments, row_count)
    _require_row_length("window_start_sec", window_start_sec, row_count)
    _require_row_length("window_end_sec", window_end_sec, row_count)
    _require_row_length("is_professor_data", is_professor_data, row_count)
    _require_row_length("source_sample_rates_hz", source_sample_rates_hz, row_count)
    _require_row_length("processed_sample_rates_hz", processed_sample_rates_hz, row_count)
    if len(feature_names) != features.shape[1]:
        raise V2MLError("feature_names length does not match scaled_features columns")
    if any(label not in {POSITIVE_LABEL, NEGATIVE_LABEL} for label in labels):
        raise V2MLError("feature bundle labels must be drone or no_drone")
    if preprocessing_profile == CANONICAL_PREPROCESSING_PROFILE:
        if processed_sample_rate_hz != 16_000:
            raise V2MLError("canonical feature bundle must declare processed rate 16000 Hz")
        if set(processed_sample_rates_hz) != {16_000}:
            raise V2MLError("canonical feature bundle rows must all be processed at 16000 Hz")

    return V2FeatureBundle(
        path=bundle_path,
        feature_set=feature_set,
        feature_names=feature_names,
        features=features,
        labels=labels,
        splits=splits,
        window_ids=window_ids,
        clip_ids=clip_ids,
        original_files=original_files,
        source_datasets=source_datasets,
        label_subtypes=label_subtypes,
        environments=environments,
        window_start_sec=window_start_sec,
        window_end_sec=window_end_sec,
        is_professor_data=is_professor_data,
        source_sample_rates_hz=source_sample_rates_hz,
        processed_sample_rates_hz=processed_sample_rates_hz,
        preprocessing_profile=preprocessing_profile,
        processed_sample_rate_hz=processed_sample_rate_hz,
        bundle_fingerprint=bundle_fingerprint,
        bundle_schema_version=bundle_schema_version,
    )


def select_thresholds(labels: Sequence[str], scores: Sequence[float]) -> dict[str, float]:
    """Select all V2 thresholds from validation labels and scores only."""

    if len(labels) != len(scores):
        raise V2MLError("threshold selection requires matching labels and scores")
    if not labels:
        raise V2MLError("cannot select thresholds without validation rows")
    _require_both_labels(labels, VAL_SPLIT)
    sorted_pairs = sorted(
        ((float(score), str(label)) for label, score in zip(labels, scores)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    positive_total = sum(1 for label in labels if label == POSITIVE_LABEL)
    negative_total = sum(1 for label in labels if label == NEGATIVE_LABEL)
    counts = {
        "tp": 0,
        "tn": negative_total,
        "fp": 0,
        "fn": positive_total,
    }
    best_fbeta: dict[str, tuple[tuple[float, float, float], float]] = {
        "threshold_f1": ((-1.0, -1.0, -1.0), sorted_pairs[0][0] + 1e-12),
        "threshold_f2": ((-1.0, -1.0, -1.0), sorted_pairs[0][0] + 1e-12),
    }
    best_fixed: dict[str, tuple[tuple[float, float, float], float]] = {
        name: ((-1.0, -1.0, -1.0), sorted_pairs[0][0] + 1e-12)
        for name in FIXED_FPR_THRESHOLDS
    }

    def consider(threshold: float) -> None:
        metrics = _metrics_from_counts(counts)
        for name, beta in (("threshold_f1", 1.0), ("threshold_f2", 2.0)):
            fbeta = _fbeta(metrics["precision"], metrics["recall"], beta)
            key = (
                _metric_sort_value(fbeta),
                _metric_sort_value(metrics["balanced_accuracy"]),
                _metric_sort_value(metrics["accuracy"]),
            )
            if key > best_fbeta[name][0]:
                best_fbeta[name] = (key, threshold)
        for name, target_fpr in FIXED_FPR_THRESHOLDS.items():
            fpr = metrics["false_positive_rate"]
            if fpr is None or fpr > target_fpr:
                continue
            key = (
                _metric_sort_value(metrics["recall"]),
                -_metric_sort_value(fpr, missing=1.0),
                _metric_sort_value(metrics["precision"]),
            )
            if key > best_fixed[name][0]:
                best_fixed[name] = (key, threshold)

    consider(sorted_pairs[0][0] + 1e-12)
    index = 0
    while index < len(sorted_pairs):
        threshold = sorted_pairs[index][0]
        while index < len(sorted_pairs) and sorted_pairs[index][0] == threshold:
            label = sorted_pairs[index][1]
            if label == POSITIVE_LABEL:
                counts["tp"] += 1
                counts["fn"] -= 1
            else:
                counts["fp"] += 1
                counts["tn"] -= 1
            index += 1
        consider(threshold)
    thresholds = {
        "threshold_f1": float(best_fbeta["threshold_f1"][1]),
        "threshold_f2": float(best_fbeta["threshold_f2"][1]),
    }
    for name in FIXED_FPR_THRESHOLDS:
        thresholds[name] = float(best_fixed[name][1])
    return thresholds


def run_v2a_sweep(
    features_path: str | Path,
    model_types: Iterable[str],
    out_dir: str | Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train V2A candidates and write model, metrics, and comparison artifacts."""

    requested = validate_model_types(model_types)
    if not requested:
        raise V2MLError("at least one V2A model type is required")
    unsupported = [model_type for model_type in requested if model_type not in V2A_MODEL_TYPES]
    if unsupported:
        raise V2MLError(f"unsupported V2A model types: {', '.join(unsupported)}")

    bundle = load_feature_bundle(features_path)
    _require_feature_set(bundle)
    partitions = _validated_partitions(bundle)
    _, joblib, _ = _require_ml_dependencies()

    out_path = Path(out_dir)
    if out_path.exists():
        raise V2MLError(f"refusing to overwrite existing V2A output directory: {out_path}")
    model_dir = out_path / "models"
    metrics_dir = out_path / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    comparison_rows = []
    for model_type in requested:
        if progress_callback is not None:
            progress_callback({"event": "model_start", "model_type": model_type})
        estimator = _fit_estimator(model_type, bundle, partitions["train"])
        if progress_callback is not None:
            progress_callback({"event": "model_fit_complete", "model_type": model_type})
        model_path = model_dir / f"{model_type}.joblib"
        metrics_path = metrics_dir / f"{model_type}.json"
        metrics = _candidate_metrics(bundle, estimator, model_type, partitions)
        model_package = _model_package(bundle, estimator, model_type, metrics)
        joblib.dump(model_package, model_path)
        model_size = model_path.stat().st_size
        metrics["model_path"] = str(model_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics["efficiency"]["model_size_bytes"] = model_size
        metrics["efficiency"]["model_size"] = _format_bytes(model_size)
        _write_json(metrics_path, metrics)
        row = _comparison_row(metrics)
        candidates.append(metrics)
        comparison_rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "model_complete",
                    "model_type": model_type,
                    "model_path": str(model_path),
                    "metrics_path": str(metrics_path),
                }
            )

    best_row = max(
        comparison_rows,
        key=lambda row: (
            _metric_sort_value(row["validation_f2"]),
            -_metric_sort_value(row["validation_fpr"], missing=1.0),
            -int(row["model_size_bytes"]),
        ),
    )
    best_model = {
        "selection_rule": (
            "highest validation F2 at threshold_f2; ties use lower validation FPR "
            "then smaller model size"
        ),
        "selected_model": best_row["model"],
        "model_path": best_row["model_path"],
        "metrics_path": best_row["metrics_path"],
        "comparison_row": {
            key: value
            for key, value in best_row.items()
            if key in COMPARISON_COLUMNS
        },
    }
    comparison = {
        "feature_bundle": str(bundle.path),
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "source_bundle_fingerprint": bundle.bundle_fingerprint,
        "feature_set": bundle.feature_set,
        "selection_rule": best_model["selection_rule"],
        "best_model": best_model,
        "candidates": [
            {key: value for key, value in row.items() if key in COMPARISON_COLUMNS}
            for row in comparison_rows
        ],
    }

    _write_json(out_path / "comparison.json", comparison)
    _write_comparison_csv(out_path / "comparison.csv", comparison_rows)
    _write_json(out_path / "best_model.json", best_model)
    return comparison


def train_v2_mlp(
    features_path: str | Path,
    model_out: str | Path,
    metrics_out: str | Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    epochs: int = V2B_DEFAULT_EPOCHS,
    batch_size: int = V2B_DEFAULT_BATCH_SIZE,
    patience: int = V2B_DEFAULT_PATIENCE,
    min_delta: float = V2_DEFAULT_MIN_DELTA,
    hidden_layer_sizes: Sequence[int] = (64, 16),
    activation: str = "relu",
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    positive_class_weight: float = 1.0,
    balance_sources: bool = False,
    feature_noise_std: float = 0.0,
    random_seed: int = RANDOM_SEED,
    variant_name: str = "tiny_mlp",
) -> dict[str, Any]:
    """Train the V2B Tiny MLP on a V2 engineered feature bundle."""

    if epochs <= 0:
        raise V2MLError("epochs must be > 0")
    if batch_size <= 0:
        raise V2MLError("batch size must be > 0")
    if patience <= 0:
        raise V2MLError("patience must be > 0")
    if min_delta < 0:
        raise V2MLError("min_delta must be >= 0")
    hidden_layers = tuple(int(size) for size in hidden_layer_sizes)
    if not hidden_layers or any(size <= 0 for size in hidden_layers):
        raise V2MLError("hidden_layer_sizes must contain positive integers")
    if activation not in {"relu", "tanh"}:
        raise V2MLError("activation must be relu or tanh")
    if alpha < 0:
        raise V2MLError("alpha must be >= 0")
    if learning_rate_init <= 0:
        raise V2MLError("learning_rate_init must be > 0")
    if positive_class_weight <= 0:
        raise V2MLError("positive_class_weight must be > 0")
    if feature_noise_std < 0:
        raise V2MLError("feature_noise_std must be >= 0")
    if not variant_name or any(character.isspace() for character in variant_name):
        raise V2MLError("variant_name must be a non-empty token without whitespace")
    bundle = load_feature_bundle(features_path)
    _require_feature_set(bundle)
    partitions = _validated_partitions(bundle)
    np, joblib, classes = _require_ml_dependencies()
    train_indices = np.asarray(_row_indices(partitions["train"]), dtype=np.int64)
    train_labels = np.asarray(_labels_for_mask(bundle, partitions["train"]))
    train_sample_weights = _mlp_training_sample_weights(
        bundle,
        train_indices,
        positive_class_weight=positive_class_weight,
        balance_sources=balance_sources,
    )
    sample_weight_by_row = (
        {
            int(row_index): float(train_sample_weights[position])
            for position, row_index in enumerate(train_indices)
        }
        if train_sample_weights is not None
        else None
    )
    estimator = classes["MLPClassifier"](
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        random_state=random_seed,
        max_iter=1,
        warm_start=True,
        shuffle=False,
    )
    class_labels = np.asarray([NEGATIVE_LABEL, POSITIVE_LABEL])
    best_estimator: Any | None = None
    best_key: tuple[float, float] = (-1.0, -1.0)
    best_epoch: int | None = None
    no_improvement_epochs = 0
    learning_curve: list[dict[str, Any]] = []
    stopped_epoch: int | None = None
    rng = np.random.default_rng(random_seed)
    first_batch = True
    total_batches = math.ceil(len(train_indices) / batch_size)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "model_start",
                "model_type": variant_name,
                "epochs": epochs,
                "train_rows": len(train_indices),
                "batch_size": batch_size,
                "patience": patience,
                "min_delta": min_delta,
            }
        )
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_indices)
        running_loss = 0.0
        seen_batches = 0
        for batch_number, start in enumerate(range(0, len(shuffled), batch_size), start=1):
            batch_indices = shuffled[start : start + batch_size]
            batch_labels = np.asarray([bundle.labels[int(index)] for index in batch_indices])
            batch_features = bundle.features[batch_indices]
            if feature_noise_std > 0:
                batch_features = batch_features + rng.normal(
                    0.0,
                    feature_noise_std,
                    size=batch_features.shape,
                )
            kwargs = {"classes": class_labels} if first_batch else {}
            if sample_weight_by_row is not None:
                batch_weights = np.asarray(
                    [sample_weight_by_row[int(index)] for index in batch_indices],
                    dtype=np.float64,
                )
                kwargs["sample_weight"] = batch_weights
            estimator.partial_fit(batch_features, batch_labels, **kwargs)
            first_batch = False
            loss = float(getattr(estimator, "loss_", 0.0) or 0.0)
            running_loss += loss
            seen_batches += 1
            if progress_callback is not None and (
                batch_number == total_batches
                or batch_number == 1
                or batch_number % 20 == 0
            ):
                progress_callback(
                    {
                        "event": "batch_progress",
                        "model_type": variant_name,
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch": batch_number,
                        "batches": total_batches,
                        "loss": loss,
                    }
                )
        val_scores = _score_estimator(estimator, bundle.features[partitions["val"]])
        val_labels = _labels_for_mask(bundle, partitions["val"])
        thresholds = select_thresholds(val_labels, val_scores)
        threshold = thresholds[PRIMARY_THRESHOLD_NAME]
        val_metrics = _metrics(val_labels, val_scores, threshold)
        key = (
            _metric_sort_value(val_metrics["f2"]),
            -_metric_sort_value(val_metrics["false_positive_rate"], missing=1.0),
        )
        improved = _early_stopping_improved(key, best_key, min_delta)
        if improved:
            best_key = key
            best_estimator = copy.deepcopy(estimator)
            best_epoch = epoch
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        average_loss = running_loss / max(seen_batches, 1)
        learning_curve.append(
            {
                "epoch": epoch,
                "training_loss": average_loss,
                "validation_f2": val_metrics["f2"],
                "validation_fpr": val_metrics["false_positive_rate"],
                "threshold_f2": threshold,
                "is_best": improved,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_complete",
                    "model_type": variant_name,
                    "epoch": epoch,
                    "epochs": epochs,
                    "loss": average_loss,
                    "validation_f2": val_metrics["f2"],
                    "validation_fpr": val_metrics["false_positive_rate"],
                    "improved": improved,
                    "best_epoch": best_epoch,
                }
            )
        if no_improvement_epochs >= patience:
            stopped_epoch = epoch
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "early_stop",
                        "model_type": variant_name,
                        "epoch": epoch,
                        "epochs": epochs,
                        "best_epoch": best_epoch,
                        "patience": patience,
                        "min_delta": min_delta,
                    }
                )
            break
    if best_estimator is None:
        best_estimator = estimator
        best_epoch = len(learning_curve) or None

    metrics = _feature_model_metrics(
        bundle,
        best_estimator,
        "tiny_mlp",
        V2B_TRACK,
        "v2b_tiny_mlp",
        partitions,
        runtime_dependency="scikit-learn",
        deployability_label="deployable_lightweight",
    )
    metrics["learning_curve"] = learning_curve
    metrics["early_stopping"] = {
        "enabled": True,
        "patience": patience,
        "min_delta": min_delta,
        "epochs_requested": epochs,
        "epochs_ran": len(learning_curve),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_epoch is not None,
    }
    metrics["training_config"] = {
        "variant_name": variant_name,
        "hidden_layer_sizes": list(hidden_layers),
        "activation": activation,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "positive_class_weight": positive_class_weight,
        "balance_sources": balance_sources,
        "feature_noise_std": feature_noise_std,
        "random_seed": random_seed,
        "batch_size": batch_size,
    }
    output_model = Path(model_out)
    output_metrics = Path(metrics_out)
    _require_outputs_absent(output_model, output_metrics)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model_package = _model_package_for_track(
        bundle,
        best_estimator,
        "tiny_mlp",
        V2B_TRACK,
        "v2b_tiny_mlp",
        metrics,
    )
    model_package["training_config"] = dict(metrics["training_config"])
    joblib.dump(model_package, output_model)
    model_size = output_model.stat().st_size
    metrics["model_path"] = str(output_model)
    metrics["metrics_path"] = str(output_metrics)
    metrics["efficiency"]["model_size_bytes"] = model_size
    metrics["efficiency"]["model_size"] = _format_bytes(model_size)
    _write_json(output_metrics, metrics)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "model_complete",
                "model_type": variant_name,
                "model_path": str(output_model),
                "metrics_path": str(output_metrics),
            }
        )
    return metrics


def _mlp_training_sample_weights(
    bundle: V2FeatureBundle,
    train_indices: Any,
    *,
    positive_class_weight: float,
    balance_sources: bool,
) -> Any | None:
    """Build deterministic train-only weights without reading evaluation rows."""

    if positive_class_weight == 1.0 and not balance_sources:
        return None
    np = _require_numpy()
    labels = [bundle.labels[int(index)] for index in train_indices]
    sources = [bundle.source_datasets[int(index)] for index in train_indices]
    weights = np.ones(len(train_indices), dtype=np.float64)
    if balance_sources:
        group_counts: dict[tuple[str, str], int] = {}
        sources_by_label: dict[str, set[str]] = {}
        label_counts: dict[str, int] = {}
        for label, source in zip(labels, sources):
            group_counts[(label, source)] = group_counts.get((label, source), 0) + 1
            sources_by_label.setdefault(label, set()).add(source)
            label_counts[label] = label_counts.get(label, 0) + 1
        for position, (label, source) in enumerate(zip(labels, sources)):
            weights[position] *= label_counts[label] / (
                len(sources_by_label[label]) * group_counts[(label, source)]
            )
    if positive_class_weight != 1.0:
        for position, label in enumerate(labels):
            if label == POSITIVE_LABEL:
                weights[position] *= positive_class_weight
    return weights


def evaluate_v2_model(
    model_path: str | Path,
    features_path: str | Path,
    metrics_out: str | Path,
    export_errors: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen V2 model package without retuning thresholds."""

    model = load_v2_model(model_path)
    bundle = load_feature_bundle(features_path)
    _require_model_compatible_with_bundle(model, bundle)
    partitions = _validated_partitions(bundle)
    threshold = float(model["threshold"])
    estimator = model["estimator"]
    model_type = str(model["model_type"])

    public_scores, inference = _score_estimator_with_progress(
        estimator,
        bundle.features[partitions["public_test"]],
        model_type,
        "public_test",
        progress_callback,
    )
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores, _ = _score_estimator_with_progress(
        estimator,
        bundle.features[partitions["professor_test"]],
        model_type,
        "professor_test",
        progress_callback,
    )

    report = {
        "schema_version": "v2_eval_report_v1",
        "algorithm": model["algorithm"],
        "model_type": model_type,
        "track": model["track"],
        "feature_set": bundle.feature_set,
        "feature_bundle": str(bundle.path),
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "source_bundle_fingerprint": bundle.bundle_fingerprint,
        "model_path": str(Path(model_path)),
        "threshold": threshold,
        "threshold_name": model["threshold_name"],
        "thresholds": model["thresholds"],
        "public_test": {
            "counts": _counts_for_mask(bundle, partitions["public_test"]),
            "metrics": _metrics(public_labels, public_scores, threshold),
            "false_positive_rate_by_subtype": _false_positive_rate_by_subtype(
                bundle,
                partitions["public_test"],
                public_scores,
                threshold,
            ),
        },
        "professor_test": {
            "counts": _counts_for_mask(bundle, partitions["professor_test"]),
            "metrics": _professor_metrics(bundle, partitions["professor_test"], professor_scores, threshold),
        },
        "combined_supplementary": _combined_supplementary_metrics(
            bundle,
            partitions,
            public_scores,
            professor_scores,
            threshold,
        ),
        "efficiency": {
            "feature_count": bundle.feature_count,
            "model_type": model_type,
            "model_size_bytes": Path(model_path).stat().st_size,
            "model_size": _format_bytes(Path(model_path).stat().st_size),
            "parameter_count": _model_parameter_count(estimator),
            "tree_node_count": _tree_node_count(estimator),
            "average_inference_time_sec_per_window": inference,
            "runtime_dependency": "scikit-learn",
            "deployability_label": model["deployability_label"],
        },
        "limitations": [
            "Professor data is positive-only or near-positive-only evaluation data.",
            "Professor data is not used for training, validation, threshold tuning, or model selection.",
            "Combined supplementary metrics include positive-only professor rows and do not replace public_test.",
        ],
    }
    output_path = Path(metrics_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    if export_errors is not None:
        _export_error_csvs(Path(export_errors), bundle, partitions, public_scores, professor_scores, threshold)
    return report


def load_v2_model(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen V2A model package."""

    model_path = Path(path)
    if not model_path.exists():
        raise V2MLError(f"model does not exist: {model_path}")
    if not model_path.is_file():
        raise V2MLError(f"model is not a file: {model_path}")
    _, joblib, _ = _require_ml_dependencies()
    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise V2MLError(f"invalid V2 model artifact: {model_path}") from error
    if not isinstance(model, dict):
        raise V2MLError("V2 model artifact must contain a model package object")
    required = (
        "schema_version",
        "algorithm",
        "model_type",
        "track",
        "feature_set",
        "feature_names",
        "labels",
        "threshold",
        "threshold_name",
        "thresholds",
        "estimator",
        "deployability_label",
    )
    missing = [key for key in required if key not in model]
    if missing:
        raise V2MLError(f"V2 model artifact missing keys: {', '.join(missing)}")
    if model["schema_version"] != MODEL_SCHEMA_VERSION:
        raise V2MLError(f"unsupported V2 model schema: {model['schema_version']}")
    if model["threshold_name"] != PRIMARY_THRESHOLD_NAME:
        raise V2MLError("V2 model threshold_name must be threshold_f2")
    if model["deployability_label"] not in DEPLOYABILITY_LABELS:
        raise V2MLError("V2 model deployability label is invalid")
    model.setdefault("preprocessing_profile", LEGACY_PREPROCESSING_PROFILE)
    model.setdefault("processed_sample_rate_hz", None)
    model.setdefault("source_bundle_fingerprint", None)
    return model


def _candidate_metrics(
    bundle: V2FeatureBundle,
    estimator: Any,
    model_type: str,
    partitions: Mapping[str, Any],
) -> dict[str, Any]:
    val_scores = _score_estimator(estimator, bundle.features[partitions["val"]])
    val_labels = _labels_for_mask(bundle, partitions["val"])
    thresholds = select_thresholds(val_labels, val_scores)
    threshold = thresholds[PRIMARY_THRESHOLD_NAME]
    public_scores, inference = _score_with_timing(
        estimator,
        bundle.features[partitions["public_test"]],
    )
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores = _score_estimator(estimator, bundle.features[partitions["professor_test"]])

    return {
        "schema_version": "v2a_metrics_v1",
        "algorithm": "v2a_classical_ml",
        "model_type": model_type,
        "track": V2A_TRACK,
        "feature_set": bundle.feature_set,
        "feature_bundle": str(bundle.path),
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "source_bundle_fingerprint": bundle.bundle_fingerprint,
        "feature_count": bundle.feature_count,
        "threshold_name": PRIMARY_THRESHOLD_NAME,
        "threshold": threshold,
        "thresholds": thresholds,
        "training": {
            "counts": _counts_for_mask(bundle, partitions["train"]),
        },
        "validation": {
            "counts": _counts_for_mask(bundle, partitions["val"]),
            "primary": _metrics(val_labels, val_scores, threshold),
            "metrics_by_threshold": {
                name: _metrics(val_labels, val_scores, value)
                for name, value in thresholds.items()
            },
        },
        "public_test": {
            "counts": _counts_for_mask(bundle, partitions["public_test"]),
            "metrics": _metrics(public_labels, public_scores, threshold),
            "false_positive_rate_by_subtype": _false_positive_rate_by_subtype(
                bundle,
                partitions["public_test"],
                public_scores,
                threshold,
            ),
        },
        "professor_test": {
            "counts": _counts_for_mask(bundle, partitions["professor_test"]),
            "metrics": _professor_metrics(bundle, partitions["professor_test"], professor_scores, threshold),
        },
        "combined_supplementary": _combined_supplementary_metrics(
            bundle,
            partitions,
            public_scores,
            professor_scores,
            threshold,
        ),
        "efficiency": {
            "feature_count": bundle.feature_count,
            "model_type": model_type,
            "model_size_bytes": None,
            "model_size": None,
            "parameter_count": _model_parameter_count(estimator),
            "tree_node_count": _tree_node_count(estimator),
            "average_inference_time_sec_per_window": inference,
            "runtime_dependency": "scikit-learn",
            "deployability_label": _deployability_label(model_type),
        },
        "selection": {
            "uses_professor_data": False,
            "primary_metric": "validation.f2",
            "primary_threshold": PRIMARY_THRESHOLD_NAME,
        },
    }


def _feature_model_metrics(
    bundle: V2FeatureBundle,
    estimator: Any,
    model_type: str,
    track: str,
    algorithm: str,
    partitions: Mapping[str, Any],
    *,
    runtime_dependency: str,
    deployability_label: str,
) -> dict[str, Any]:
    val_scores = _score_estimator(estimator, bundle.features[partitions["val"]])
    val_labels = _labels_for_mask(bundle, partitions["val"])
    thresholds = select_thresholds(val_labels, val_scores)
    threshold = thresholds[PRIMARY_THRESHOLD_NAME]
    public_scores, inference = _score_with_timing(
        estimator,
        bundle.features[partitions["public_test"]],
    )
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores = _score_estimator(estimator, bundle.features[partitions["professor_test"]])

    return {
        "schema_version": "v2_metrics_v1",
        "algorithm": algorithm,
        "model_type": model_type,
        "track": track,
        "feature_set": bundle.feature_set,
        "feature_bundle": str(bundle.path),
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "source_bundle_fingerprint": bundle.bundle_fingerprint,
        "feature_count": bundle.feature_count,
        "threshold_name": PRIMARY_THRESHOLD_NAME,
        "threshold": threshold,
        "thresholds": thresholds,
        "training": {
            "counts": _counts_for_mask(bundle, partitions["train"]),
        },
        "validation": {
            "counts": _counts_for_mask(bundle, partitions["val"]),
            "primary": _metrics(val_labels, val_scores, threshold),
            "metrics_by_threshold": {
                name: _metrics(val_labels, val_scores, value)
                for name, value in thresholds.items()
            },
        },
        "public_test": {
            "counts": _counts_for_mask(bundle, partitions["public_test"]),
            "metrics": _metrics(public_labels, public_scores, threshold),
            "false_positive_rate_by_subtype": _false_positive_rate_by_subtype(
                bundle,
                partitions["public_test"],
                public_scores,
                threshold,
            ),
        },
        "professor_test": {
            "counts": _counts_for_mask(bundle, partitions["professor_test"]),
            "metrics": _professor_metrics(bundle, partitions["professor_test"], professor_scores, threshold),
        },
        "combined_supplementary": _combined_supplementary_metrics(
            bundle,
            partitions,
            public_scores,
            professor_scores,
            threshold,
        ),
        "efficiency": {
            "feature_count": bundle.feature_count,
            "model_type": model_type,
            "model_size_bytes": None,
            "model_size": None,
            "parameter_count": _model_parameter_count(estimator),
            "tree_node_count": _tree_node_count(estimator),
            "average_inference_time_sec_per_window": inference,
            "runtime_dependency": runtime_dependency,
            "deployability_label": deployability_label,
        },
        "selection": {
            "uses_professor_data": False,
            "primary_metric": "validation.f2",
            "primary_threshold": PRIMARY_THRESHOLD_NAME,
        },
    }


def _fit_estimator(model_type: str, bundle: V2FeatureBundle, train_mask: Any) -> Any:
    _, _, classes = _require_ml_dependencies()
    labels = _labels_for_mask(bundle, train_mask)
    features = bundle.features[train_mask]
    if model_type == "logistic_regression":
        estimator = classes["LogisticRegression"](
            class_weight="balanced",
            max_iter=300,
            random_state=RANDOM_SEED,
            solver="saga",
            tol=1e-3,
        )
    elif model_type == "linear_svm":
        min_class_count = min(labels.count(POSITIVE_LABEL), labels.count(NEGATIVE_LABEL))
        if min_class_count < 2:
            raise V2MLError("linear_svm calibration requires at least two rows per class")
        estimator = classes["CalibratedClassifierCV"](
            estimator=classes["LinearSVC"](
                class_weight="balanced",
                dual="auto",
                max_iter=2_000,
                random_state=RANDOM_SEED,
                tol=1e-3,
            ),
            cv=min(2, min_class_count),
            method="sigmoid",
        )
    elif model_type == "random_forest":
        estimator = classes["RandomForestClassifier"](
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    elif model_type == "extra_trees":
        estimator = classes["ExtraTreesClassifier"](
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    else:
        raise V2MLError(f"unsupported V2A model type: {model_type}")
    estimator.fit(features, labels)
    return estimator


def _early_stopping_improved(
    key: tuple[float, float],
    best_key: tuple[float, float],
    min_delta: float,
) -> bool:
    f2, negative_fpr = key
    best_f2, best_negative_fpr = best_key
    if f2 > best_f2 + min_delta:
        return True
    if abs(f2 - best_f2) <= min_delta and negative_fpr > best_negative_fpr:
        return True
    return False


def _model_package(
    bundle: V2FeatureBundle,
    estimator: Any,
    model_type: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "algorithm": "v2a_classical_ml",
        "model_type": model_type,
        "track": V2A_TRACK,
        "feature_set": bundle.feature_set,
        "feature_names": list(bundle.feature_names),
        "labels": {
            "positive": POSITIVE_LABEL,
            "negative": NEGATIVE_LABEL,
        },
        "threshold": metrics["threshold"],
        "threshold_name": metrics["threshold_name"],
        "thresholds": dict(metrics["thresholds"]),
        "estimator": estimator,
        "deployability_label": metrics["efficiency"]["deployability_label"],
        "selection": dict(metrics["selection"]),
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "source_bundle_fingerprint": bundle.bundle_fingerprint,
    }


def _model_package_for_track(
    bundle: V2FeatureBundle,
    estimator: Any,
    model_type: str,
    track: str,
    algorithm: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    package = _model_package(bundle, estimator, model_type, metrics)
    package["algorithm"] = algorithm
    package["track"] = track
    return package


def _validated_partitions(bundle: V2FeatureBundle) -> dict[str, Any]:
    np = _require_numpy()
    labels = np.asarray(bundle.labels)
    splits = np.asarray(bundle.splits)
    professor = np.asarray(bundle.is_professor_data, dtype=bool)
    masks = {
        "train": (splits == TRAIN_SPLIT) & (~professor),
        "val": (splits == VAL_SPLIT) & (~professor),
        "public_test": (splits == PUBLIC_TEST_SPLIT) & (~professor),
        "professor_test": (splits == PROFESSOR_TEST_SPLIT) & professor,
    }
    for name in ("train", "val", "public_test"):
        _require_non_empty_mask(masks[name], name)
        _require_both_labels(tuple(labels[masks[name]]), name)
    if masks["professor_test"].any():
        professor_labels = set(labels[masks["professor_test"]])
        if professor_labels != {POSITIVE_LABEL}:
            raise V2MLError("professor_test rows must be positive-only for V2 evaluation")
    return masks


def _score_with_timing(estimator: Any, features: Any) -> tuple[list[float], float]:
    started_at = time.perf_counter()
    scores = _score_estimator(estimator, features)
    elapsed = time.perf_counter() - started_at
    per_window = elapsed / len(scores) if scores else 0.0
    return scores, per_window


def _score_estimator_with_progress(
    estimator: Any,
    features: Any,
    model_type: str,
    phase: str,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    batch_size: int = 8192,
) -> tuple[list[float], float]:
    total = int(len(features))
    started_at = time.perf_counter()
    if progress_callback is not None:
        progress_callback(
            {
                "event": "eval_start",
                "model_type": model_type,
                "phase": phase,
                "total_rows": total,
                "batch_size": batch_size,
            }
        )
    if total == 0:
        return [], 0.0
    scores: list[float] = []
    for start in range(0, total, batch_size):
        batch = features[start : start + batch_size]
        scores.extend(_score_estimator(estimator, batch))
        processed = min(start + batch_size, total)
        if progress_callback is not None and (
            processed == total
            or processed == len(batch)
            or processed % (batch_size * 5) == 0
        ):
            elapsed = max(time.perf_counter() - started_at, 0.0)
            rate = processed / elapsed if elapsed else 0.0
            eta = (total - processed) / rate if rate > 0 else None
            progress_callback(
                {
                    "event": "eval_progress",
                    "model_type": model_type,
                    "phase": phase,
                    "processed_rows": processed,
                    "total_rows": total,
                    "rate": rate,
                    "eta_seconds": eta,
                }
            )
    elapsed = time.perf_counter() - started_at
    per_window = elapsed / total if total else 0.0
    if progress_callback is not None:
        progress_callback(
            {
                "event": "eval_complete",
                "model_type": model_type,
                "phase": phase,
                "total_rows": total,
                "rate": total / elapsed if elapsed > 0 else 0.0,
            }
        )
    return scores, per_window


def _score_estimator(estimator: Any, features: Any) -> list[float]:
    np = _require_numpy()
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classes = list(getattr(estimator, "classes_", []))
        if POSITIVE_LABEL not in classes:
            raise V2MLError("estimator classes do not include the positive label")
        positive_index = classes.index(POSITIVE_LABEL)
        return [float(value) for value in probabilities[:, positive_index]]
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(features), dtype=np.float64)
        if decision.ndim == 2:
            classes = list(getattr(estimator, "classes_", []))
            positive_index = classes.index(POSITIVE_LABEL) if POSITIVE_LABEL in classes else 1
            decision = decision[:, positive_index]
        return [float(1.0 / (1.0 + math.exp(-value))) for value in decision]
    raise V2MLError("estimator does not support predict_proba or decision_function")


def _select_threshold_for_fbeta(
    labels: Sequence[str],
    scores: Sequence[float],
    *,
    beta: float,
) -> float:
    unique_scores = _candidate_thresholds(scores)
    best_threshold = unique_scores[0]
    best_key = (-1.0, -1.0, -1.0)
    for threshold in unique_scores:
        metric = _metrics(list(labels), list(scores), threshold)
        precision = metric["precision"]
        recall = metric["recall"]
        fbeta = _fbeta(precision, recall, beta)
        balanced = metric["balanced_accuracy"]
        accuracy = metric["accuracy"]
        key = (
            _metric_sort_value(fbeta),
            _metric_sort_value(balanced),
            _metric_sort_value(accuracy),
        )
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def _select_threshold_at_fpr(
    labels: Sequence[str],
    scores: Sequence[float],
    target_fpr: float,
) -> float:
    best_threshold: float | None = None
    best_key = (-1.0, -1.0, -1.0)
    for threshold in _candidate_thresholds(scores):
        metric = _metrics(list(labels), list(scores), threshold)
        fpr = metric["false_positive_rate"]
        if fpr is None or fpr > target_fpr:
            continue
        key = (
            _metric_sort_value(metric["recall"]),
            -_metric_sort_value(fpr, missing=1.0),
            _metric_sort_value(metric["precision"]),
        )
        if key > best_key:
            best_key = key
            best_threshold = threshold
    if best_threshold is not None:
        return float(best_threshold)
    return float(max(scores) + 1e-12)


def _candidate_thresholds(scores: Sequence[float]) -> list[float]:
    unique_scores = sorted({float(score) for score in scores})
    if not unique_scores:
        raise V2MLError("cannot select threshold without scores")
    return [unique_scores[0] - 1e-12] + unique_scores + [unique_scores[-1] + 1e-12]


def _metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float | None]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "accuracy": _safe_div(tp + tn, total),
        "balanced_accuracy": _average_defined((recall, specificity)),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": _safe_div(fp, fp + tn),
    }


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _average_defined(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _false_positive_rate_by_subtype(
    bundle: V2FeatureBundle,
    mask: Any,
    scores: Sequence[float],
    threshold: float,
) -> dict[str, dict[str, int | float | None]]:
    rows = _row_indices(mask)
    grouped: dict[str, list[float]] = {}
    for row_index, score in zip(rows, scores):
        if bundle.labels[row_index] != NEGATIVE_LABEL:
            continue
        subtype = bundle.label_subtypes[row_index] or "no_drone_unknown"
        grouped.setdefault(subtype, []).append(float(score))

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


def _professor_metrics(
    bundle: V2FeatureBundle,
    mask: Any,
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any] | None:
    rows = _row_indices(mask)
    if not rows:
        return None
    predictions = [score >= threshold for score in scores]
    missed = [not prediction for prediction in predictions]
    missed_count = sum(1 for value in missed if value)
    return {
        "rows": len(rows),
        "threshold": threshold,
        "recall": (len(rows) - missed_count) / len(rows),
        "missed_window_count": missed_count,
        "longest_missed_segment_sec": _longest_missed_segment_sec(bundle, rows, missed),
        "mean_p_drone": statistics.fmean(scores),
        "median_p_drone": statistics.median(scores),
        "prediction_stability": _prediction_stability(predictions, scores),
        "score_quantiles": _score_quantiles(scores),
    }


def _combined_supplementary_metrics(
    bundle: V2FeatureBundle,
    partitions: Mapping[str, Any],
    public_scores: Sequence[float],
    professor_scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    public_rows = _row_indices(partitions["public_test"])
    professor_rows = _row_indices(partitions["professor_test"])
    combined_rows = public_rows + professor_rows
    combined_labels = [bundle.labels[index] for index in combined_rows]
    combined_scores = [float(score) for score in public_scores] + [
        float(score) for score in professor_scores
    ]
    return {
        "supplementary": True,
        "note": (
            "Combined metrics include positive-only or near-positive-only professor data "
            "and do not replace the public_test benchmark."
        ),
        "counts": _label_counts(
            [{"label_binary": bundle.labels[index]} for index in combined_rows]
        ),
        "metrics": _metrics(combined_labels, combined_scores, threshold),
    }


def _longest_missed_segment_sec(
    bundle: V2FeatureBundle,
    row_indices: Sequence[int],
    missed: Sequence[bool],
) -> float:
    longest = 0.0
    active_clip = ""
    active_start: float | None = None
    active_end: float | None = None
    ordered = sorted(
        zip(row_indices, missed),
        key=lambda item: (
            bundle.clip_ids[item[0]],
            bundle.window_start_sec[item[0]],
            bundle.window_end_sec[item[0]],
        ),
    )
    for row_index, is_missed in ordered:
        clip_id = bundle.clip_ids[row_index]
        start = bundle.window_start_sec[row_index]
        end = bundle.window_end_sec[row_index]
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


def _close_segment(longest: float, start: float | None, end: float | None) -> float:
    if start is None or end is None:
        return longest
    return max(longest, end - start)


def _prediction_stability(
    predictions: Sequence[bool],
    scores: Sequence[float],
) -> dict[str, float | int]:
    flip_count = sum(
        1
        for previous, current in zip(predictions, predictions[1:])
        if previous != current
    )
    return {
        "decision_flip_count": flip_count,
        "decision_flip_rate": flip_count / max(len(predictions) - 1, 1),
        "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }


def _score_quantiles(scores: Sequence[float]) -> dict[str, float]:
    return {
        "p05": _quantile(scores, 0.05),
        "p25": _quantile(scores, 0.25),
        "p50": _quantile(scores, 0.50),
        "p75": _quantile(scores, 0.75),
        "p95": _quantile(scores, 0.95),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise V2MLError("cannot compute quantiles without values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _export_error_csvs(
    out_dir: Path,
    bundle: V2FeatureBundle,
    partitions: Mapping[str, Any],
    public_scores: Sequence[float],
    professor_scores: Sequence[float],
    threshold: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_error_csv(
        out_dir / "public_false_positives.csv",
        bundle,
        _public_false_positive_rows(bundle, partitions["public_test"], public_scores, threshold),
    )
    _write_error_csv(
        out_dir / "professor_missed_windows.csv",
        bundle,
        _missed_professor_rows(partitions["professor_test"], professor_scores, threshold),
    )


def _public_false_positive_rows(
    bundle: V2FeatureBundle,
    mask: Any,
    scores: Sequence[float],
    threshold: float,
) -> list[tuple[int, float]]:
    rows = []
    for row_index, score in zip(_row_indices(mask), scores):
        if bundle.labels[row_index] == NEGATIVE_LABEL and score >= threshold:
            rows.append((row_index, float(score)))
    return rows


def _missed_professor_rows(
    mask: Any,
    scores: Sequence[float],
    threshold: float,
) -> list[tuple[int, float]]:
    rows = []
    for row_index, score in zip(_row_indices(mask), scores):
        if score < threshold:
            rows.append((row_index, float(score)))
    return rows


def _write_error_csv(
    path: Path,
    bundle: V2FeatureBundle,
    rows: Sequence[tuple[int, float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_id",
                "clip_id",
                "original_file",
                "source_dataset",
                "label_binary",
                "label_subtype",
                "window_start_sec",
                "window_end_sec",
                "score",
            ],
        )
        writer.writeheader()
        for row_index, score in rows:
            writer.writerow(
                {
                    "window_id": bundle.window_ids[row_index],
                    "clip_id": bundle.clip_ids[row_index],
                    "original_file": bundle.original_files[row_index],
                    "source_dataset": bundle.source_datasets[row_index],
                    "label_binary": bundle.labels[row_index],
                    "label_subtype": bundle.label_subtypes[row_index],
                    "window_start_sec": bundle.window_start_sec[row_index],
                    "window_end_sec": bundle.window_end_sec[row_index],
                    "score": score,
                }
            )


def _comparison_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    public_metrics = metrics["public_test"]["metrics"]
    professor_metrics = metrics["professor_test"]["metrics"] or {}
    efficiency = metrics["efficiency"]
    row = {
        "model": metrics["model_type"],
        "track": metrics["track"],
        "feature/input": metrics["feature_set"],
        "validation_f2": metrics["validation"]["primary"]["f2"],
        "validation_fpr": metrics["validation"]["primary"]["false_positive_rate"],
        "public_f2": public_metrics["f2"],
        "public_f1": public_metrics["f1"],
        "public_precision": public_metrics["precision"],
        "public_recall": public_metrics["recall"],
        "public_fpr": public_metrics["false_positive_rate"],
        "professor_recall": professor_metrics.get("recall"),
        "longest_missed_segment": professor_metrics.get("longest_missed_segment_sec"),
        "model_size": efficiency["model_size"],
        "model_size_bytes": efficiency["model_size_bytes"],
        "params": efficiency["parameter_count"],
        "deployability_label": efficiency["deployability_label"],
        "model_path": metrics["model_path"],
        "metrics_path": metrics["metrics_path"],
    }
    return row


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COMPARISON_COLUMNS})


def _counts_for_mask(bundle: V2FeatureBundle, mask: Any) -> dict[str, int]:
    rows = [
        {"label_binary": bundle.labels[index]}
        for index in _row_indices(mask)
    ]
    return _label_counts(rows)


def _labels_for_mask(bundle: V2FeatureBundle, mask: Any) -> list[str]:
    return [bundle.labels[index] for index in _row_indices(mask)]


def _row_indices(mask: Any) -> list[int]:
    np = _require_numpy()
    return [int(index) for index in np.flatnonzero(mask)]


def _require_feature_set(bundle: V2FeatureBundle) -> None:
    if bundle.feature_set != V2A_FEATURE_SET:
        raise V2MLError(f"V2A sweep requires feature set {V2A_FEATURE_SET}")


def _require_model_compatible_with_bundle(
    model: Mapping[str, Any],
    bundle: V2FeatureBundle,
) -> None:
    if model["feature_set"] != bundle.feature_set:
        raise V2MLError("model feature_set does not match feature bundle")
    if tuple(model["feature_names"]) != bundle.feature_names:
        raise V2MLError("model feature_names do not match feature bundle")
    labels = model["labels"]
    if labels.get("positive") != POSITIVE_LABEL or labels.get("negative") != NEGATIVE_LABEL:
        raise V2MLError("model labels do not match V2 label contract")
    model_profile = str(
        model.get("preprocessing_profile", LEGACY_PREPROCESSING_PROFILE)
    )
    try:
        require_compatible_preprocessing(
            model_profile,
            bundle.preprocessing_profile,
            left_name="model",
            right_name="feature_bundle",
        )
    except AudioPreprocessingError as error:
        raise V2MLError(str(error)) from error
    model_rate = model.get("processed_sample_rate_hz")
    if model_rate != bundle.processed_sample_rate_hz:
        raise V2MLError(
            "processed sample-rate mismatch: "
            f"model={model_rate}, feature_bundle={bundle.processed_sample_rate_hz}"
        )


def _require_non_empty_mask(mask: Any, split: str) -> None:
    if not bool(mask.any()):
        raise V2MLError(f"feature bundle requires {split!r} rows")


def _require_both_labels(labels: Sequence[str], split: str) -> None:
    present = set(labels)
    missing = {POSITIVE_LABEL, NEGATIVE_LABEL} - present
    if missing:
        raise V2MLError(f"{split!r} rows are missing labels: {', '.join(sorted(missing))}")


def _require_row_length(name: str, values: Sequence[Any], expected: int) -> None:
    if len(values) != expected:
        raise V2MLError(f"{name} length does not match scaled_features rows")


def _string_tuple(values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _scalar_string(value: Any) -> str:
    if hasattr(value, "shape") and value.shape == ():
        return str(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _fbeta(precision: float | None, recall: float | None, beta: float) -> float | None:
    if precision is None or recall is None:
        return None
    denominator = (beta * beta * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta * beta) * precision * recall / denominator


def _metric_sort_value(value: Any, *, missing: float = -1.0) -> float:
    if value is None:
        return missing
    return float(value)


def _deployability_label(model_type: str) -> str:
    if model_type in {"logistic_regression", "linear_svm"}:
        return "deployable_lightweight"
    return "maybe_deployable"


def _model_parameter_count(estimator: Any) -> int | None:
    count = _direct_parameter_count(estimator)
    if count is not None:
        return count
    total = 0
    found = False
    for child in _child_estimators(estimator):
        child_count = _direct_parameter_count(child)
        if child_count is not None:
            total += child_count
            found = True
    return total if found else None


def _direct_parameter_count(estimator: Any) -> int | None:
    total = 0
    found = False
    for attribute in ("coef_", "intercept_"):
        values = getattr(estimator, attribute, None)
        if values is not None and hasattr(values, "size"):
            total += int(values.size)
            found = True
    for attribute in ("coefs_", "intercepts_"):
        values = getattr(estimator, attribute, None)
        if values is not None:
            for value in values:
                if hasattr(value, "size"):
                    total += int(value.size)
                    found = True
    return total if found else None


def _tree_node_count(estimator: Any) -> int | None:
    estimators = getattr(estimator, "estimators_", None)
    if estimators is None:
        return None
    total = 0
    for tree_estimator in estimators:
        tree = getattr(tree_estimator, "tree_", None)
        if tree is not None:
            total += int(tree.node_count)
    return total


def _child_estimators(estimator: Any) -> list[Any]:
    children = []
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated is not None:
        for classifier in calibrated:
            for attribute in ("estimator", "base_estimator"):
                child = getattr(classifier, attribute, None)
                if child is not None:
                    children.append(child)
    return children


def _format_bytes(size: int | None) -> str | None:
    if size is None:
        return None
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise V2MLError(
            "refusing to overwrite existing V2 artifact(s): " + ", ".join(existing)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - dependency guard.
        raise V2MLError(
            "V2 ML commands require optional data dependencies; install with "
            "`python -m pip install -e '.[data,ml]'` or use "
            "`uv run --extra data --extra ml ...`"
        ) from error
    return np


def _require_ml_dependencies() -> tuple[Any, Any, dict[str, Any]]:
    np = _require_numpy()
    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import LinearSVC
    except ImportError as error:  # pragma: no cover - dependency guard.
        raise V2MLError(
            "V2 ML commands require optional ml dependencies; install with "
            "`python -m pip install -e '.[data,ml]'` or use "
            "`uv run --extra data --extra ml ...`"
        ) from error
    return np, joblib, {
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LinearSVC": LinearSVC,
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SGDClassifier": SGDClassifier,
    }
