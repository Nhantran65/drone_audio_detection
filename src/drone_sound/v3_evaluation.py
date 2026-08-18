"""Frozen V3 public/professor evaluation with group-aware diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL, _metrics
from drone_sound.v2_ml import (
    V2FeatureBundle,
    V2MLError,
    _professor_metrics,
    _require_model_compatible_with_bundle,
    _score_estimator,
    load_feature_bundle,
    load_v2_model,
)
from drone_sound.v2_real_eval import _load_joblib, _require_frozen_threshold
from drone_sound.v3_group_split import V3_SPLIT_VERSION

ProgressCallback = Callable[[dict[str, Any]], None]


def evaluate_v3_frozen_model(
    model_path: str | Path,
    features_path: str | Path,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    cache_dir: str | Path,
    metrics_out: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    batch_size: int = 512,
    bootstrap_iterations: int = 1_000,
) -> dict[str, Any]:
    """Evaluate a frozen V3 model without fitting, tuning, or selecting."""

    if batch_size <= 0 or bootstrap_iterations <= 0:
        raise V2MLError("batch_size and bootstrap_iterations must be > 0")
    bundle = load_feature_bundle(features_path)
    if bundle.split_version != V3_SPLIT_VERSION:
        raise V2MLError(
            f"V3 evaluation requires split_version={V3_SPLIT_VERSION}, got {bundle.split_version}"
        )
    output = Path(metrics_out)
    if output.exists():
        raise V2MLError(f"refusing to overwrite V3 evaluation report: {output}")
    model_file = Path(model_path)
    if not model_file.is_file():
        raise V2MLError(f"model does not exist: {model_file}")
    if progress_callback is not None:
        progress_callback(
            {"event": "v3_eval_start", "model": str(model_file), "rows": bundle.row_count}
        )

    model_type, track, threshold_name, thresholds, scores = _score_all(
        model_file,
        bundle,
        public_audio_root,
        professor_audio_root,
        cache_dir,
        progress_callback,
        batch_size,
    )
    threshold = thresholds[threshold_name]
    np = _require_numpy()
    split_array = np.asarray(bundle.splits)
    professor = np.asarray(bundle.is_professor_data, dtype=bool)
    public_indices = np.flatnonzero((split_array == "public_test") & (~professor)).tolist()
    professor_indices = np.flatnonzero((split_array == "professor_test") & professor).tolist()
    public_labels = [bundle.labels[index] for index in public_indices]
    public_scores = [scores[index] for index in public_indices]
    professor_scores = [scores[index] for index in professor_indices]
    if set(public_labels) != {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise V2MLError("V3 public_test must contain both labels")

    metrics_by_threshold = {
        name: _metrics(public_labels, public_scores, value)
        for name, value in thresholds.items()
        if name in {"threshold_f1", "threshold_f2"}
    }
    report = {
        "schema_version": "v3_grouped_evaluation_v1",
        "evaluation_only": True,
        "fit_called": False,
        "threshold_tuned": False,
        "selection_performed": False,
        "model_path": str(model_file),
        "model_type": model_type,
        "track": track,
        "model_size_bytes": model_file.stat().st_size,
        "feature_bundle": str(bundle.path),
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "split_version": bundle.split_version,
        "group_parser_version": bundle.group_parser_version,
        "threshold_name": threshold_name,
        "threshold": threshold,
        "thresholds": thresholds,
        "public_test": {
            "rows": len(public_indices),
            "groups": len({bundle.group_ids[index] for index in public_indices}),
            "metrics": _metrics(public_labels, public_scores, threshold),
            "metrics_by_threshold": metrics_by_threshold,
            "per_dataset": _per_dataset(bundle, public_indices, scores, threshold),
            "per_group": _per_group(bundle, public_indices, scores, threshold),
            "cluster_bootstrap_95ci": _cluster_bootstrap(
                bundle,
                public_indices,
                scores,
                threshold,
                iterations=bootstrap_iterations,
            ),
        },
        "professor_test": {
            "rows": len(professor_indices),
            "recordings": len({bundle.original_files[index] for index in professor_indices}),
            "metrics": _professor_metrics(
                bundle,
                _mask(bundle.row_count, professor_indices),
                professor_scores,
                threshold,
            ),
            "metrics_by_threshold": {
                name: _professor_metrics(
                    bundle,
                    _mask(bundle.row_count, professor_indices),
                    professor_scores,
                    value,
                )
                for name, value in thresholds.items()
                if name in {"threshold_f1", "threshold_f2"}
            },
        },
        "limitations": [
            "The professor-positive recording is the same positive source reused in private real-only evaluation.",
            "The private benchmark has already informed prior research and is not a pristine blind test.",
            "Overlapping windows are correlated; group bootstrap is used for public confidence intervals.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "v3_eval_complete",
                "model_type": model_type,
                "rows": len(public_indices),
                "metrics_out": str(output),
            }
        )
    return report


def _score_all(
    model_file: Path,
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None,
    batch_size: int,
) -> tuple[str, str, str, dict[str, float], list[float]]:
    if model_file.suffix.lower() in {".pt", ".pth"}:
        return _score_cnn(
            model_file,
            bundle,
            public_audio_root,
            professor_audio_root,
            cache_dir,
            progress_callback,
            batch_size,
        )
    raw = _load_joblib(model_file)
    if raw.get("track") == "v2d" or raw.get("model_type") == "yamnet_head":
        return _score_embedding(
            model_file,
            bundle,
            public_audio_root,
            professor_audio_root,
            cache_dir,
            progress_callback,
            batch_size,
        )
    model = load_v2_model(model_file)
    _require_model_compatible_with_bundle(model, bundle)
    _require_frozen_threshold(model)
    scores = _score_estimator(model["estimator"], bundle.features)
    return (
        str(model["model_type"]),
        str(model["track"]),
        str(model["threshold_name"]),
        {name: float(value) for name, value in model["thresholds"].items()},
        scores,
    )


def _score_cnn(
    model_file: Path,
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
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
    checkpoint = torch.load(model_file, map_location=device)
    _validate_torch_checkpoint(checkpoint)
    _require_checkpoint_compatible(checkpoint, bundle)
    _require_frozen_threshold(checkpoint)
    cache = _logmel_cache(
        bundle,
        public_audio_root,
        professor_audio_root,
        cache_dir,
        progress_callback,
    )
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
        phase="v3_all",
    )
    return (
        "tiny_cnn",
        "v2c",
        str(checkpoint["threshold_name"]),
        {name: float(value) for name, value in checkpoint["thresholds"].items()},
        scores,
    )


def _score_embedding(
    model_file: Path,
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None,
    batch_size: int,
) -> tuple[str, str, str, dict[str, float], list[float]]:
    from drone_sound.v2_embedding import (
        _require_embedding_model_compatible,
        _yamnet_embedding_cache,
        load_v2_embedding_model,
    )

    model = load_v2_embedding_model(model_file)
    _require_embedding_model_compatible(model, bundle)
    _require_frozen_threshold(model)
    embeddings, _ = _yamnet_embedding_cache(
        bundle,
        public_audio_root,
        professor_audio_root,
        cache_dir,
        progress_callback,
        progress_every=batch_size,
    )
    scores = _score_estimator(model["estimator"], embeddings)
    return (
        "yamnet_head",
        "v2d",
        str(model["threshold_name"]),
        {name: float(value) for name, value in model["thresholds"].items()},
        scores,
    )


def _per_dataset(
    bundle: V2FeatureBundle,
    indices: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[bundle.source_datasets[index]].append(index)
    output = {}
    for dataset, rows in sorted(groups.items()):
        labels = [bundle.labels[index] for index in rows]
        values = [float(scores[index]) for index in rows]
        output[dataset] = {
            "rows": len(rows),
            "groups": len({bundle.group_ids[index] for index in rows}),
            "label": labels[0] if len(set(labels)) == 1 else "mixed",
            "metrics": _metrics(labels, values, threshold),
        }
    return output


def _per_group(
    bundle: V2FeatureBundle,
    indices: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[bundle.group_ids[index]].append(index)
    output = []
    for group_id, rows in sorted(groups.items()):
        label = bundle.labels[rows[0]]
        positives = sum(float(scores[index]) >= threshold for index in rows)
        output.append(
            {
                "group_id": group_id,
                "source_dataset": bundle.source_datasets[rows[0]],
                "label": label,
                "rows": len(rows),
                "positive_predictions": int(positives),
                "recall": positives / len(rows) if label == POSITIVE_LABEL else None,
                "false_positive_rate": positives / len(rows) if label == NEGATIVE_LABEL else None,
                "mean_score": sum(float(scores[index]) for index in rows) / len(rows),
            }
        )
    return output


def _cluster_bootstrap(
    bundle: V2FeatureBundle,
    indices: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    *,
    iterations: int,
) -> dict[str, Any]:
    np = _require_numpy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[bundle.group_ids[index]].append(index)
    group_counts = []
    for rows in groups.values():
        tp = tn = fp = fn = 0
        for index in rows:
            positive = float(scores[index]) >= threshold
            if bundle.labels[index] == POSITIVE_LABEL:
                tp += int(positive)
                fn += int(not positive)
            else:
                fp += int(positive)
                tn += int(not positive)
        group_counts.append((tp, tn, fp, fn))
    rng = np.random.default_rng(20260804)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled = rng.integers(0, len(group_counts), size=len(group_counts))
        tp = sum(group_counts[int(index)][0] for index in sampled)
        tn = sum(group_counts[int(index)][1] for index in sampled)
        fp = sum(group_counts[int(index)][2] for index in sampled)
        fn = sum(group_counts[int(index)][3] for index in sampled)
        metric = _count_metrics(tp, tn, fp, fn)
        for name, value in metric.items():
            values[name].append(value)
    return {
        name: {
            "low": float(np.quantile(samples, 0.025)),
            "high": float(np.quantile(samples, 0.975)),
        }
        for name, samples in sorted(values.items())
    }


def _count_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": 1.0 - specificity,
    }


def _mask(row_count: int, indices: Sequence[int]) -> Any:
    np = _require_numpy()
    mask = np.zeros(row_count, dtype=bool)
    mask[list(indices)] = True
    return mask


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise V2MLError("V3 evaluation requires numpy") from error
    return np
