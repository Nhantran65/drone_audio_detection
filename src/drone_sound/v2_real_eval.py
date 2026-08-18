"""Frozen-model evaluation on the private real-only external test set."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL, _metrics
from drone_sound.v2_ml import (
    V2FeatureBundle,
    V2MLError,
    _require_model_compatible_with_bundle,
    _score_estimator,
    load_feature_bundle,
    load_v2_model,
)

ProgressCallback = Callable[[dict[str, Any]], None]


def evaluate_real_v2(
    model_path: str | Path,
    features_path: str | Path,
    audio_root: str | Path,
    cache_dir: str | Path,
    metrics_out: str | Path,
    export_errors: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    *,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Auto-detect a V2 model family and run inference without fit or tuning."""

    if batch_size <= 0:
        raise V2MLError("batch size must be > 0")
    bundle = load_feature_bundle(features_path)
    _require_real_only_bundle(bundle)
    model_file = Path(model_path)
    output = Path(metrics_out)
    if output.exists():
        raise V2MLError(f"refusing to overwrite real-only metrics artifact: {output}")
    if export_errors is not None and Path(export_errors).exists():
        raise V2MLError(
            f"refusing to overwrite real-only error directory: {Path(export_errors)}"
        )
    if not model_file.is_file():
        raise V2MLError(f"model does not exist: {model_file}")
    if progress_callback is not None:
        progress_callback(
            {"event": "real_eval_start", "model": str(model_file), "rows": bundle.row_count}
        )

    if model_file.suffix.lower() in {".pt", ".pth"}:
        model_type, track, threshold_name, thresholds, scores = _score_cnn(
            model_file, bundle, audio_root, cache_dir, progress_callback, batch_size
        )
    else:
        raw = _load_joblib(model_file)
        if raw.get("track") == "v2d" or raw.get("model_type") == "yamnet_head":
            model_type, track, threshold_name, thresholds, scores = _score_embedding(
                model_file, bundle, audio_root, cache_dir, progress_callback, batch_size
            )
        else:
            model_type, track, threshold_name, thresholds, scores = _score_engineered(
                model_file, bundle, progress_callback
            )

    labels = list(bundle.labels)
    threshold = float(thresholds[threshold_name])
    metrics = _metrics(labels, scores, threshold)
    report = {
        "schema_version": "v2_real_only_evaluation_v1",
        "evaluation_only": True,
        "fit_called": False,
        "threshold_tuned": False,
        "model_path": str(model_file),
        "model_type": model_type,
        "track": track,
        "feature_bundle": str(bundle.path),
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "threshold_name": threshold_name,
        "threshold": threshold,
        "thresholds": thresholds,
        "counts": {
            POSITIVE_LABEL: labels.count(POSITIVE_LABEL),
            NEGATIVE_LABEL: labels.count(NEGATIVE_LABEL),
            "recordings": len(set(bundle.original_files)),
        },
        "metrics": metrics,
        "metrics_by_threshold": {
            name: _metrics(labels, scores, value)
            for name, value in thresholds.items()
            if name in {"threshold_f1", "threshold_f2"}
        },
        "per_environment": _per_environment(bundle, scores, threshold),
        "per_recording_fpr": _per_recording_fpr(bundle, scores, threshold),
        "limitations": [
            "The external set contains one positive and ten negative recordings.",
            "Overlapping windows are correlated and are not independent trials.",
            "The model and validation-derived threshold were frozen before this evaluation.",
            "This stress test is not a deployment-performance estimate.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if export_errors is not None:
        _write_errors(Path(export_errors), bundle, scores, threshold)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "real_eval_complete",
                "model_type": model_type,
                "rows": bundle.row_count,
                "metrics_out": str(output),
            }
        )
    return report


def _score_engineered(
    model_path: Path,
    bundle: V2FeatureBundle,
    progress_callback: ProgressCallback | None,
) -> tuple[str, str, str, dict[str, float], list[float]]:
    model = load_v2_model(model_path)
    _require_model_compatible_with_bundle(model, bundle)
    _require_frozen_threshold(model)
    if progress_callback is not None:
        progress_callback({"event": "real_eval_score", "model_type": model["model_type"]})
    scores = _score_estimator(model["estimator"], bundle.features)
    thresholds = {name: float(value) for name, value in model["thresholds"].items()}
    return (
        str(model["model_type"]),
        str(model["track"]),
        str(model["threshold_name"]),
        thresholds,
        scores,
    )


def _score_cnn(
    model_path: Path,
    bundle: V2FeatureBundle,
    audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None,
    batch_size: int,
) -> tuple[str, str, str, dict[str, float], list[float]]:
    from drone_sound.v2_torch import (
        LOGMEL_BINS,
        LOGMEL_FRAMES,
        _TinyCnn,
        _device,
        _logmel_cache,
        _require_checkpoint_compatible,
        _require_dl_dependencies,
        _torch_scores,
        _validate_torch_checkpoint,
    )

    _, torch, _, _ = _require_dl_dependencies()
    device = _device(torch)
    checkpoint = torch.load(model_path, map_location=device)
    _validate_torch_checkpoint(checkpoint)
    _require_checkpoint_compatible(checkpoint, bundle)
    _require_frozen_threshold(checkpoint)
    cache = _logmel_cache(bundle, audio_root, audio_root, cache_dir, progress_callback)
    network = _TinyCnn(LOGMEL_BINS, LOGMEL_FRAMES).to(device)
    network.load_state_dict(checkpoint["state_dict"])
    scores = _torch_scores(
        torch,
        network,
        cache,
        list(range(bundle.row_count)),
        batch_size,
        device,
        progress_callback=progress_callback,
        model_type="tiny_cnn",
        phase="real_only",
    )
    thresholds = {name: float(value) for name, value in checkpoint["thresholds"].items()}
    return "tiny_cnn", "v2c", str(checkpoint["threshold_name"]), thresholds, scores


def _score_embedding(
    model_path: Path,
    bundle: V2FeatureBundle,
    audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None,
    batch_size: int,
) -> tuple[str, str, str, dict[str, float], list[float]]:
    from drone_sound.v2_embedding import (
        _require_embedding_model_compatible,
        _yamnet_embedding_cache,
        load_v2_embedding_model,
    )

    model = load_v2_embedding_model(model_path)
    _require_embedding_model_compatible(model, bundle)
    _require_frozen_threshold(model)
    embeddings, _ = _yamnet_embedding_cache(
        bundle,
        audio_root,
        audio_root,
        cache_dir,
        progress_callback,
        progress_every=batch_size,
    )
    scores = _score_estimator(model["estimator"], embeddings)
    thresholds = {name: float(value) for name, value in model["thresholds"].items()}
    return "yamnet_head", "v2d", str(model["threshold_name"]), thresholds, scores


def _require_real_only_bundle(bundle: V2FeatureBundle) -> None:
    labels = list(bundle.labels)
    if labels.count(POSITIVE_LABEL) != 1_113 or labels.count(NEGATIVE_LABEL) != 592:
        raise V2MLError(
            "real-only bundle must contain exactly 1113 positive and 592 negative windows"
        )
    if set(bundle.splits) != {"professor_test"}:
        raise V2MLError("real-only bundle must contain professor_test rows only")
    if len(set(bundle.original_files)) != 11:
        raise V2MLError("real-only bundle must contain exactly 11 recordings")


def _require_frozen_threshold(model: Mapping[str, Any]) -> None:
    if model.get("threshold_name") not in {"threshold_f1", "threshold_f2"}:
        raise V2MLError("real-only evaluation requires a frozen F1 or F2 threshold")
    if "threshold" not in model:
        raise V2MLError("model does not contain a frozen threshold")
    if model.get("threshold_name") not in model.get("thresholds", {}):
        raise V2MLError("model threshold_name is missing from frozen thresholds")


def _per_environment(
    bundle: V2FeatureBundle, scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, environment in enumerate(bundle.environments):
        groups[environment or "unspecified"].append(index)
    return {
        environment: _metrics(
            [bundle.labels[index] for index in indices],
            [float(scores[index]) for index in indices],
            threshold,
        )
        for environment, indices in sorted(groups.items())
    }


def _per_recording_fpr(
    bundle: V2FeatureBundle, scores: Sequence[float], threshold: float
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, original_file in enumerate(bundle.original_files):
        if bundle.labels[index] == NEGATIVE_LABEL:
            groups[original_file].append(index)
    output = []
    for original_file, indices in sorted(groups.items()):
        false_positives = sum(float(scores[index]) >= threshold for index in indices)
        output.append(
            {
                "original_file": original_file,
                "environment": bundle.environments[indices[0]] or "unspecified",
                "negative_windows": len(indices),
                "false_positives": int(false_positives),
                "false_positive_rate": false_positives / len(indices),
            }
        )
    return output


def _write_errors(
    directory: Path,
    bundle: V2FeatureBundle,
    scores: Sequence[float],
    threshold: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "real_only_errors.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "window_id",
                "original_file",
                "environment",
                "label_binary",
                "score",
                "predicted_label",
            ),
        )
        writer.writeheader()
        for index, score in enumerate(scores):
            predicted = POSITIVE_LABEL if score >= threshold else NEGATIVE_LABEL
            if predicted == bundle.labels[index]:
                continue
            writer.writerow(
                {
                    "window_id": bundle.window_ids[index],
                    "original_file": bundle.original_files[index],
                    "environment": bundle.environments[index],
                    "label_binary": bundle.labels[index],
                    "score": float(score),
                    "predicted_label": predicted,
                }
            )


def _load_joblib(path: Path) -> Mapping[str, Any]:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover
        raise V2MLError("real-only joblib evaluation requires scikit-learn/joblib") from error
    value = joblib.load(path)
    if not isinstance(value, Mapping):
        raise V2MLError("V2 joblib model must contain a model package object")
    return value
