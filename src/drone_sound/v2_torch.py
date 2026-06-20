"""PyTorch-backed V2 experiment tracks."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL, _metrics
from drone_sound.v2_ml import (
    PRIMARY_THRESHOLD_NAME,
    PROFESSOR_TEST_SPLIT,
    RANDOM_SEED,
    V2_DEFAULT_MIN_DELTA,
    V2FeatureBundle,
    V2MLError,
    _counts_for_mask,
    _combined_supplementary_metrics,
    _early_stopping_improved,
    _false_positive_rate_by_subtype,
    _format_bytes,
    _jsonable,
    _labels_for_mask,
    _metric_sort_value,
    _professor_metrics,
    _row_indices,
    _validated_partitions,
    load_feature_bundle,
    select_thresholds,
)
from drone_sound.wav_audio import WavReadError, read_wav_mono_array

V2C_TRACK = "v2c"
LOGMEL_FEATURE_SET = "logmel_v1"
TORCH_MODEL_SCHEMA_VERSION = "v2_torch_model_v1"
V2C_DEFAULT_EPOCHS = 4
V2C_DEFAULT_BATCH_SIZE = 512
V2C_DEFAULT_PATIENCE = 4
V2D_DEFAULT_BATCH_SIZE = 64
V2D_DEFAULT_EPOCHS = 5
TARGET_SAMPLE_RATE = 16_000
LOGMEL_BINS = 64
LOGMEL_FRAMES = 98
ProgressCallback = Callable[[dict[str, Any]], None]


def train_v2_cnn(
    features_path: str | Path,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    model_out: str | Path,
    metrics_out: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    epochs: int = V2C_DEFAULT_EPOCHS,
    batch_size: int = V2C_DEFAULT_BATCH_SIZE,
    patience: int = V2C_DEFAULT_PATIENCE,
    min_delta: float = V2_DEFAULT_MIN_DELTA,
) -> dict[str, Any]:
    """Train V2C Tiny CNN on cached log-mel windows."""

    if epochs <= 0:
        raise V2MLError("epochs must be > 0")
    if batch_size <= 0:
        raise V2MLError("batch size must be > 0")
    if patience <= 0:
        raise V2MLError("patience must be > 0")
    if min_delta < 0:
        raise V2MLError("min_delta must be >= 0")
    np, torch, _, _ = _require_dl_dependencies()
    bundle = load_feature_bundle(features_path)
    partitions = _validated_partitions(bundle)
    cache = _logmel_cache(
        bundle,
        public_audio_root,
        professor_audio_root,
        cache_dir,
        progress_callback,
    )
    device = _device(torch)
    model = _TinyCnn(LOGMEL_BINS, LOGMEL_FRAMES).to(device)
    labels = _label_array(np, bundle)
    train_indices = np.asarray(_row_indices(partitions["train"]), dtype=np.int64)
    train_labels = labels[train_indices]
    positive_count = int((train_labels == 1).sum())
    negative_count = int((train_labels == 0).sum())
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1)],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(RANDOM_SEED)
    total_batches = math.ceil(len(train_indices) / batch_size)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "model_start",
                "model_type": "tiny_cnn",
                "epochs": epochs,
                "train_rows": len(train_indices),
                "batch_size": batch_size,
                "device": str(device),
                "patience": patience,
                "min_delta": min_delta,
            }
        )
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] = (-1.0, -1.0)
    best_epoch: int | None = None
    no_improvement_epochs = 0
    learning_curve: list[dict[str, Any]] = []
    stopped_epoch: int | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        running_loss = 0.0
        seen_batches = 0
        for batch_number, start in enumerate(range(0, len(shuffled), batch_size), start=1):
            batch_indices = shuffled[start : start + batch_size]
            x = _torch_logmel_batch(torch, cache, batch_indices, device)
            y = torch.as_tensor(labels[batch_indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            seen_batches += 1
            if progress_callback is not None and (
                batch_number == 1
                or batch_number == total_batches
                or batch_number % 50 == 0
            ):
                progress_callback(
                    {
                        "event": "batch_progress",
                        "model_type": "tiny_cnn",
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch": batch_number,
                        "batches": total_batches,
                        "loss": loss_value,
                    }
                )
        val_scores = _torch_scores(torch, model, cache, _row_indices(partitions["val"]), batch_size, device)
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
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
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
                    "model_type": "tiny_cnn",
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
                        "model_type": "tiny_cnn",
                        "epoch": epoch,
                        "epochs": epochs,
                        "best_epoch": best_epoch,
                        "patience": patience,
                        "min_delta": min_delta,
                    }
                )
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = _torch_metrics(
        bundle,
        partitions,
        model,
        cache,
        "tiny_cnn",
        V2C_TRACK,
        "v2c_tiny_cnn",
        LOGMEL_FEATURE_SET,
        batch_size,
        device,
        runtime_dependency="torch",
        deployability_label="maybe_deployable",
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
    output_model = Path(model_out)
    output_metrics = Path(metrics_out)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": TORCH_MODEL_SCHEMA_VERSION,
            "algorithm": "v2c_tiny_cnn",
            "model_type": "tiny_cnn",
            "track": V2C_TRACK,
            "feature_set": LOGMEL_FEATURE_SET,
            "input_shape": [1, LOGMEL_BINS, LOGMEL_FRAMES],
            "threshold": metrics["threshold"],
            "threshold_name": metrics["threshold_name"],
            "thresholds": metrics["thresholds"],
            "state_dict": model.state_dict(),
            "labels": {"positive": POSITIVE_LABEL, "negative": NEGATIVE_LABEL},
        },
        output_model,
    )
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
                "model_type": "tiny_cnn",
                "model_path": str(output_model),
                "metrics_path": str(output_metrics),
            }
        )
    return metrics


def evaluate_v2_cnn(
    model_path: str | Path,
    features_path: str | Path,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    cache_dir: str | Path,
    metrics_out: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    batch_size: int = V2C_DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate a frozen V2C Tiny CNN checkpoint without retuning thresholds."""

    if batch_size <= 0:
        raise V2MLError("batch size must be > 0")
    _, torch, _, _ = _require_dl_dependencies()
    bundle = load_feature_bundle(features_path)
    partitions = _validated_partitions(bundle)
    cache = _logmel_cache(
        bundle,
        public_audio_root,
        professor_audio_root,
        cache_dir,
        progress_callback,
    )
    device = _device(torch)
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise V2MLError(f"model does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise V2MLError(f"model is not a file: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as error:
        raise V2MLError(f"invalid V2C model artifact: {checkpoint_path}") from error
    _validate_torch_checkpoint(checkpoint)
    model = _TinyCnn(LOGMEL_BINS, LOGMEL_FRAMES).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    threshold = float(checkpoint["threshold"])
    model_type = str(checkpoint["model_type"])
    track = str(checkpoint["track"])
    algorithm = str(checkpoint["algorithm"])

    public_indices = _row_indices(partitions["public_test"])
    professor_indices = _row_indices(partitions["professor_test"])
    started_at = time.perf_counter()
    public_scores = _torch_scores(
        torch,
        model,
        cache,
        public_indices,
        batch_size,
        device,
        progress_callback=progress_callback,
        model_type=model_type,
        phase="public_test",
    )
    elapsed = time.perf_counter() - started_at
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores = _torch_scores(
        torch,
        model,
        cache,
        professor_indices,
        batch_size,
        device,
        progress_callback=progress_callback,
        model_type=model_type,
        phase="professor_test",
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "schema_version": "v2_eval_report_v1",
        "algorithm": algorithm,
        "model_type": model_type,
        "track": track,
        "feature_set": LOGMEL_FEATURE_SET,
        "source_feature_bundle": str(bundle.path),
        "feature_bundle": str(bundle.path),
        "model_path": str(checkpoint_path),
        "threshold": threshold,
        "threshold_name": checkpoint["threshold_name"],
        "thresholds": dict(checkpoint["thresholds"]),
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
            "feature_count": LOGMEL_BINS * LOGMEL_FRAMES,
            "model_type": model_type,
            "model_size_bytes": checkpoint_path.stat().st_size,
            "model_size": _format_bytes(checkpoint_path.stat().st_size),
            "parameter_count": parameter_count,
            "tree_node_count": None,
            "average_inference_time_sec_per_window": elapsed / max(len(public_scores), 1),
            "runtime_dependency": "torch",
            "deployability_label": "maybe_deployable",
        },
        "limitations": [
            "Professor data is positive-only or near-positive-only evaluation data.",
            "Professor data is not used for training, validation, threshold tuning, or model selection.",
            "Combined supplementary metrics include positive-only professor rows and do not replace public_test.",
        ],
    }
    _write_json(Path(metrics_out), metrics)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "model_complete",
                "model_type": model_type,
                "model_path": str(checkpoint_path),
                "metrics_path": str(metrics_out),
            }
        )
    return metrics


def train_v2_embedding(
    features_path: str | Path,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    embedding_model: str,
    head: str,
    model_out: str | Path,
    metrics_out: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    batch_size: int = V2D_DEFAULT_BATCH_SIZE,
    epochs: int = V2D_DEFAULT_EPOCHS,
) -> dict[str, Any]:
    """Delegate V2D pretrained embeddings to the TensorFlow Hub implementation."""

    from drone_sound.v2_embedding import train_v2_embedding as train_embedding

    return train_embedding(
        features_path,
        public_audio_root,
        professor_audio_root,
        embedding_model,
        head,
        model_out,
        metrics_out,
        cache_dir,
        progress_callback=progress_callback,
        batch_size=batch_size,
        epochs=epochs,
    )


class _TinyCnn:
    def __new__(cls, mel_bins: int, frame_count: int) -> Any:
        _, torch, _, _ = _require_dl_dependencies()

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(8),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((2, 2)),
                    torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.head = torch.nn.Linear(16, 1)

            def forward(self, x: Any) -> Any:
                features = self.net(x).flatten(1)
                return self.head(features)

        return Module()


def _logmel_cache(
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None,
) -> Any:
    np, torch, torchaudio, _ = _require_dl_dependencies()
    cache_path = Path(cache_dir) / "logmel_v1.npy"
    meta_path = Path(cache_dir) / "logmel_v1_meta.json"
    expected_shape = (bundle.row_count, LOGMEL_BINS, LOGMEL_FRAMES)
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if tuple(meta.get("shape", ())) == expected_shape:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "cache_reuse",
                        "phase": LOGMEL_FEATURE_SET,
                        "rows": bundle.row_count,
                        "cache_path": str(cache_path),
                    }
                )
            return np.load(cache_path, mmap_mode="r")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=LOGMEL_BINS,
        center=False,
        power=2.0,
    )

    def extract(waveforms: Any) -> Any:
        with torch.inference_mode():
            mel = transform(waveforms)
            logmel = torch.log(mel + 1e-6)
            if logmel.shape[-1] < LOGMEL_FRAMES:
                logmel = torch.nn.functional.pad(logmel, (0, LOGMEL_FRAMES - logmel.shape[-1]))
            logmel = logmel[..., :LOGMEL_FRAMES]
            mean = logmel.mean(dim=(1, 2), keepdim=True)
            std = logmel.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            return ((logmel - mean) / std).cpu().numpy().astype(np.float16)

    _fill_window_cache(
        bundle,
        public_audio_root,
        professor_audio_root,
        matrix,
        extract,
        progress_callback,
        phase="logmel_cache",
        batch_size=512,
    )
    meta_path.write_text(
        json.dumps(
            {
                "feature_set": LOGMEL_FEATURE_SET,
                "shape": list(expected_shape),
                "dtype": "float16",
                "source_bundle": str(bundle.path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return np.load(cache_path, mmap_mode="r")


def _fill_window_cache(
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    matrix: Any,
    extract: Callable[[Any], Any],
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    batch_size: int,
) -> None:
    np, torch, torchaudio, _ = _require_dl_dependencies()
    groups = _group_rows_by_audio(bundle, public_audio_root, professor_audio_root)
    processed = 0
    next_report = 10_000
    started_at = time.monotonic()
    if progress_callback is not None:
        progress_callback(
            {
                "event": "cache_start",
                "phase": phase,
                "rows": bundle.row_count,
                "files": len(groups),
            }
        )
    for file_index, (audio_path, row_indices) in enumerate(groups.items(), start=1):
        try:
            samples, metadata = read_wav_mono_array(audio_path)
        except (ImportError, WavReadError) as error:
            raise V2MLError(str(error)) from error
        windows: list[Any] = []
        indices: list[int] = []
        for row_index in row_indices:
            start = bundle.window_start_sec[row_index]
            end = bundle.window_end_sec[row_index]
            window = _window_samples(np, samples, metadata.sample_rate_hz, start, end, audio_path)
            windows.append(window)
            indices.append(row_index)
            if len(windows) == batch_size:
                _write_window_batch(torch, torchaudio, matrix, extract, indices, windows, metadata.sample_rate_hz)
                processed += len(windows)
                if processed >= next_report or processed == bundle.row_count:
                    _cache_progress(progress_callback, phase, processed, bundle.row_count, file_index, len(groups), audio_path, started_at)
                    while next_report <= processed:
                        next_report += 10_000
                windows = []
                indices = []
        if windows:
            _write_window_batch(torch, torchaudio, matrix, extract, indices, windows, metadata.sample_rate_hz)
            processed += len(windows)
            if processed >= next_report or processed == bundle.row_count:
                _cache_progress(progress_callback, phase, processed, bundle.row_count, file_index, len(groups), audio_path, started_at)
                while next_report <= processed:
                    next_report += 10_000
    matrix.flush()


def _write_window_batch(
    torch: Any,
    torchaudio: Any,
    matrix: Any,
    extract: Callable[[Any], Any],
    indices: Sequence[int],
    windows: Sequence[Any],
    sample_rate: int,
) -> None:
    import numpy as np

    waveforms = torch.as_tensor(np.stack(windows), dtype=torch.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveforms = torchaudio.functional.resample(waveforms, sample_rate, TARGET_SAMPLE_RATE)
    if waveforms.shape[1] < TARGET_SAMPLE_RATE:
        waveforms = torch.nn.functional.pad(waveforms, (0, TARGET_SAMPLE_RATE - waveforms.shape[1]))
    waveforms = waveforms[:, :TARGET_SAMPLE_RATE]
    matrix[list(indices)] = extract(waveforms)


def _cache_progress(
    progress_callback: ProgressCallback | None,
    phase: str,
    processed: int,
    total: int,
    file_index: int,
    total_files: int,
    audio_path: Path,
    started_at: float,
) -> None:
    if progress_callback is None:
        return
    elapsed = max(time.monotonic() - started_at, 0.0)
    rate = processed / elapsed if elapsed else 0.0
    eta = (total - processed) / rate if rate > 0 else None
    progress_callback(
        {
            "event": "cache_progress",
            "phase": phase,
            "processed_rows": processed,
            "total_rows": total,
            "file_index": file_index,
            "total_files": total_files,
            "current_file": str(audio_path),
            "rate": rate,
            "eta_seconds": eta,
        }
    )


def _torch_metrics(
    bundle: V2FeatureBundle,
    partitions: Mapping[str, Any],
    model: Any,
    cache: Any,
    model_type: str,
    track: str,
    algorithm: str,
    feature_set: str,
    batch_size: int,
    device: Any,
    *,
    runtime_dependency: str,
    deployability_label: str,
) -> dict[str, Any]:
    _, torch, _, _ = _require_dl_dependencies()
    val_scores = _torch_scores(torch, model, cache, _row_indices(partitions["val"]), batch_size, device)
    val_labels = _labels_for_mask(bundle, partitions["val"])
    thresholds = select_thresholds(val_labels, val_scores)
    threshold = thresholds[PRIMARY_THRESHOLD_NAME]
    started_at = time.perf_counter()
    public_scores = _torch_scores(torch, model, cache, _row_indices(partitions["public_test"]), batch_size, device)
    elapsed = time.perf_counter() - started_at
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores = _torch_scores(torch, model, cache, _row_indices(partitions["professor_test"]), batch_size, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return _metrics_report(
        bundle,
        partitions,
        model_type,
        track,
        algorithm,
        feature_set,
        thresholds,
        threshold,
        val_labels,
        val_scores,
        public_labels,
        public_scores,
        professor_scores,
        feature_count=LOGMEL_BINS * LOGMEL_FRAMES,
        parameter_count=parameter_count,
        inference_time=elapsed / max(len(public_scores), 1),
        runtime_dependency=runtime_dependency,
        deployability_label=deployability_label,
    )


def _metrics_report(
    bundle: V2FeatureBundle,
    partitions: Mapping[str, Any],
    model_type: str,
    track: str,
    algorithm: str,
    feature_set: str,
    thresholds: Mapping[str, float],
    threshold: float,
    val_labels: Sequence[str],
    val_scores: Sequence[float],
    public_labels: Sequence[str],
    public_scores: Sequence[float],
    professor_scores: Sequence[float],
    *,
    feature_count: int,
    parameter_count: int | None,
    inference_time: float,
    runtime_dependency: str,
    deployability_label: str,
) -> dict[str, Any]:
    return {
        "schema_version": "v2_metrics_v1",
        "algorithm": algorithm,
        "model_type": model_type,
        "track": track,
        "feature_set": feature_set,
        "source_feature_bundle": str(bundle.path),
        "feature_count": feature_count,
        "threshold_name": PRIMARY_THRESHOLD_NAME,
        "threshold": threshold,
        "thresholds": dict(thresholds),
        "training": {"counts": _counts_for_mask(bundle, partitions["train"])},
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
            "feature_count": feature_count,
            "model_type": model_type,
            "model_size_bytes": None,
            "model_size": None,
            "parameter_count": parameter_count,
            "tree_node_count": None,
            "average_inference_time_sec_per_window": inference_time,
            "runtime_dependency": runtime_dependency,
            "deployability_label": deployability_label,
        },
        "selection": {
            "uses_professor_data": False,
            "primary_metric": "validation.f2",
            "primary_threshold": PRIMARY_THRESHOLD_NAME,
        },
        "limitations": [
            "Professor data is positive-only or near-positive-only evaluation data.",
            "Professor data is not used for training, validation, threshold tuning, or model selection.",
        ],
    }


def _torch_scores(
    torch: Any,
    model: Any,
    cache: Any,
    indices: Sequence[int],
    batch_size: int,
    device: Any,
    *,
    progress_callback: ProgressCallback | None = None,
    model_type: str = "tiny_cnn",
    phase: str = "eval",
) -> list[float]:
    model.eval()
    scores: list[float] = []
    total = len(indices)
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
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            x = _torch_logmel_batch(torch, cache, batch_indices, device)
            logits = model(x).reshape(-1)
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
            processed = min(start + batch_size, total)
            if progress_callback is not None and (
                processed == total
                or processed == len(batch_indices)
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
    elapsed = max(time.perf_counter() - started_at, 0.0)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "eval_complete",
                "model_type": model_type,
                "phase": phase,
                "total_rows": total,
                "rate": total / elapsed if elapsed else 0.0,
            }
        )
    return [float(value) for value in scores]


def _validate_torch_checkpoint(checkpoint: Any) -> None:
    if not isinstance(checkpoint, dict):
        raise V2MLError("V2C model artifact must contain a checkpoint object")
    required = (
        "schema_version",
        "algorithm",
        "model_type",
        "track",
        "feature_set",
        "input_shape",
        "threshold",
        "threshold_name",
        "thresholds",
        "state_dict",
        "labels",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise V2MLError(f"V2C model artifact missing keys: {', '.join(missing)}")
    if checkpoint["schema_version"] != TORCH_MODEL_SCHEMA_VERSION:
        raise V2MLError(f"unsupported V2C model schema: {checkpoint['schema_version']}")
    if checkpoint["algorithm"] != "v2c_tiny_cnn":
        raise V2MLError(f"unsupported V2C model algorithm: {checkpoint['algorithm']}")
    if checkpoint["model_type"] != "tiny_cnn":
        raise V2MLError("V2C model_type must be tiny_cnn")
    if checkpoint["track"] != V2C_TRACK:
        raise V2MLError("V2C track must be v2c")
    if checkpoint["feature_set"] != LOGMEL_FEATURE_SET:
        raise V2MLError(f"V2C feature_set must be {LOGMEL_FEATURE_SET}")
    if checkpoint["threshold_name"] != PRIMARY_THRESHOLD_NAME:
        raise V2MLError("V2C model threshold_name must be threshold_f2")


def _torch_logmel_batch(torch: Any, cache: Any, indices: Sequence[int], device: Any) -> Any:
    array = cache[list(indices)].astype("float32", copy=False)
    return torch.from_numpy(array).unsqueeze(1).to(device)


def _label_array(np: Any, bundle: V2FeatureBundle) -> Any:
    return np.asarray([1.0 if label == POSITIVE_LABEL else 0.0 for label in bundle.labels], dtype=np.float32)


def _group_rows_by_audio(
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
) -> dict[Path, list[int]]:
    groups: dict[Path, list[int]] = defaultdict(list)
    for row_index, original_file in enumerate(bundle.original_files):
        root = professor_audio_root if bundle.is_professor_data[row_index] else public_audio_root
        path = Path(original_file)
        if not path.is_absolute():
            path = Path(root) / path
        if not path.exists():
            raise V2MLError(f"audio file not found for feature bundle row: {path}")
        groups[path].append(row_index)
    return dict(groups)


def _window_samples(
    np: Any,
    samples: Any,
    sample_rate_hz: int,
    start_sec: float,
    end_sec: float,
    audio_path: Path,
) -> Any:
    if end_sec <= start_sec:
        raise V2MLError(f"invalid window bounds for {audio_path}: {start_sec}..{end_sec}")
    start_index = int(round(start_sec * sample_rate_hz))
    sample_count = int(round((end_sec - start_sec) * sample_rate_hz))
    end_index = start_index + sample_count
    if start_index < 0 or end_index > len(samples):
        raise V2MLError(f"feature bundle window exceeds audio duration: {audio_path}")
    return np.asarray(samples[start_index:end_index], dtype=np.float32)


def _device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_dl_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        import torchaudio
    except ImportError as error:  # pragma: no cover - dependency guard.
        raise V2MLError(
            "V2 DL commands require optional dl dependencies; install with "
            "`python -m pip install -e '.[data,ml,dl]'` or use "
            "`uv run --extra data --extra ml --extra dl ...`"
        ) from error
    return np, torch, torchaudio, None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
