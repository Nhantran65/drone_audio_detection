"""TensorFlow Hub-backed V2D pretrained embedding baseline."""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL, PUBLIC_TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT, _metrics
from drone_sound.v2_ml import (
    PRIMARY_THRESHOLD_NAME,
    PROFESSOR_TEST_SPLIT,
    RANDOM_SEED,
    V2FeatureBundle,
    V2MLError,
    _counts_for_mask,
    _combined_supplementary_metrics,
    _false_positive_rate_by_subtype,
    _format_bytes,
    _labels_for_mask,
    _model_parameter_count,
    _professor_metrics,
    _require_ml_dependencies,
    _require_numpy,
    _row_indices,
    _score_estimator,
    _score_estimator_with_progress,
    _score_with_timing,
    _tree_node_count,
    _validated_partitions,
    _write_json,
    load_feature_bundle,
    select_thresholds,
)
from drone_sound.wav_audio import WavReadError, read_wav_mono_array

V2D_TRACK = "v2d"
V2D_FEATURE_SET = "pretrained_embedding_v1"
V2D_MODEL_TYPE = "yamnet_head"
V2D_ALGORITHM = "v2d_yamnet_embedding"
V2D_MODEL_SCHEMA_VERSION = "v2_embedding_model_v1"
YAMNET_MODEL_NAME = "yamnet"
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_EXPECTED_EMBEDDING_DIM = 1024
YAMNET_CACHE_STEM = "yamnet_embedding_v1"
YAMNET_AGGREGATION = "mean_pool"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
ProgressCallback = Callable[[dict[str, Any]], None]
EmbeddingCallable = Callable[[Any], Any]


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
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, Any]:
    """Train the V2D YAMNet embedding head from an existing V2 feature bundle."""

    if embedding_model != YAMNET_MODEL_NAME:
        raise V2MLError(f"unsupported V2D embedding model: {embedding_model}")
    if head != "logistic_regression":
        raise V2MLError(f"unsupported V2D embedding head: {head}")
    if batch_size <= 0:
        raise V2MLError("batch size must be > 0")
    if epochs <= 0:
        raise V2MLError("epochs must be > 0")

    bundle = load_feature_bundle(features_path)
    partitions = _validated_partitions(bundle)
    embeddings, cache_report = _yamnet_embedding_cache(
        bundle,
        public_audio_root,
        professor_audio_root,
        cache_dir,
        progress_callback,
        progress_every=batch_size,
    )
    estimator = _fit_embedding_head(
        embeddings,
        bundle,
        partitions["train"],
        max_iter=max(100, epochs * 100),
        progress_callback=progress_callback,
    )
    metrics = _embedding_metrics(bundle, embeddings, estimator, partitions, cache_report)
    metrics["head_training"] = {
        "head": head,
        "max_iter": max(100, epochs * 100),
        "requested_epochs": epochs,
        "train_rows": int(partitions["train"].sum()),
    }

    output_model = Path(model_out)
    output_metrics = Path(metrics_out)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    _, joblib, _ = _require_ml_dependencies()
    joblib.dump(_model_package(embeddings, estimator, metrics, cache_report), output_model)
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
                "model_type": V2D_MODEL_TYPE,
                "model_path": str(output_model),
                "metrics_path": str(output_metrics),
            }
        )
    return metrics


def evaluate_v2_embedding(
    model_path: str | Path,
    features_path: str | Path,
    metrics_out: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    batch_size: int = 8192,
) -> dict[str, Any]:
    """Evaluate a frozen V2D embedding head without retuning thresholds."""

    if batch_size <= 0:
        raise V2MLError("batch size must be > 0")
    model = load_v2_embedding_model(model_path)
    bundle = load_feature_bundle(features_path)
    partitions = _validated_partitions(bundle)
    embeddings, cache_report = _embedding_cache_from_model(model, bundle)
    estimator = model["estimator"]
    threshold = float(model["threshold"])

    public_scores, inference = _score_estimator_with_progress(
        estimator,
        embeddings[partitions["public_test"]],
        V2D_MODEL_TYPE,
        "public_test",
        progress_callback,
        batch_size=batch_size,
    )
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores, _ = _score_estimator_with_progress(
        estimator,
        embeddings[partitions["professor_test"]],
        V2D_MODEL_TYPE,
        "professor_test",
        progress_callback,
        batch_size=batch_size,
    )
    model_size = Path(model_path).stat().st_size
    report = {
        "schema_version": "v2_eval_report_v1",
        "algorithm": model["algorithm"],
        "model_type": V2D_MODEL_TYPE,
        "track": V2D_TRACK,
        "feature_set": V2D_FEATURE_SET,
        "source_feature_bundle": str(bundle.path),
        "feature_bundle": str(bundle.path),
        "model_path": str(Path(model_path)),
        "feature_count": int(embeddings.shape[1]),
        "threshold": threshold,
        "threshold_name": model["threshold_name"],
        "thresholds": dict(model["thresholds"]),
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
            "metrics": _professor_metrics(
                bundle,
                partitions["professor_test"],
                professor_scores,
                threshold,
            ),
        },
        "combined_supplementary": _combined_supplementary_metrics(
            bundle,
            partitions,
            public_scores,
            professor_scores,
            threshold,
        ),
        "embedding": cache_report,
        "efficiency": {
            "feature_count": int(embeddings.shape[1]),
            "model_type": V2D_MODEL_TYPE,
            "model_size_bytes": model_size,
            "model_size": _format_bytes(model_size),
            "parameter_count": _model_parameter_count(estimator),
            "tree_node_count": _tree_node_count(estimator),
            "average_inference_time_sec_per_window": inference,
            "runtime_dependency": "tensorflow-hub + scikit-learn",
            "deployability_label": model["deployability_label"],
            "embedding_extractor": YAMNET_MODEL_NAME,
            "embedding_extractor_size": None,
            "classifier_head_size": None,
        },
        "selection": dict(model["selection"]),
        "limitations": [
            "V2D is reference_only unless full extractor deployment proof is added.",
            "Professor data is positive-only or near-positive-only evaluation data.",
            "Professor data is not used for training, validation, threshold tuning, or model selection.",
            "Combined supplementary metrics include positive-only professor rows and do not replace public_test.",
        ],
    }
    output_path = Path(metrics_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    return report


def load_v2_embedding_model(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen V2D embedding model package."""

    model_path = Path(path)
    if not model_path.exists():
        raise V2MLError(f"model does not exist: {model_path}")
    if not model_path.is_file():
        raise V2MLError(f"model is not a file: {model_path}")
    _, joblib, _ = _require_ml_dependencies()
    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise V2MLError(f"invalid V2D model artifact: {model_path}") from error
    if not isinstance(model, dict):
        raise V2MLError("V2D model artifact must contain a model package object")
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
        "embedding",
        "deployability_label",
        "selection",
    )
    missing = [key for key in required if key not in model]
    if missing:
        raise V2MLError(f"V2D model artifact missing keys: {', '.join(missing)}")
    if model["schema_version"] != V2D_MODEL_SCHEMA_VERSION:
        raise V2MLError(f"unsupported V2D model schema: {model['schema_version']}")
    if model["algorithm"] != V2D_ALGORITHM:
        raise V2MLError(f"unsupported V2D algorithm: {model['algorithm']}")
    if model["model_type"] != V2D_MODEL_TYPE:
        raise V2MLError(f"unsupported V2D model type: {model['model_type']}")
    if model["track"] != V2D_TRACK:
        raise V2MLError(f"unsupported V2D track: {model['track']}")
    if model["feature_set"] != V2D_FEATURE_SET:
        raise V2MLError(f"unsupported V2D feature set: {model['feature_set']}")
    if model["threshold_name"] != PRIMARY_THRESHOLD_NAME:
        raise V2MLError("V2D model threshold_name must be threshold_f2")
    if model["deployability_label"] != "reference_only":
        raise V2MLError("V2D YAMNet model must be marked reference_only")
    embedding = model["embedding"]
    if not isinstance(embedding, Mapping):
        raise V2MLError("V2D model embedding metadata must be an object")
    for key in ("path", "meta_path"):
        if key not in embedding:
            raise V2MLError(f"V2D model embedding metadata missing key: {key}")
    return model


def _yamnet_embedding_cache(
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None,
    *,
    progress_every: int,
) -> tuple[Any, dict[str, Any]]:
    np = _require_numpy()
    cache_root = Path(cache_dir)
    cache_path = cache_root / f"{YAMNET_CACHE_STEM}.npy"
    meta_path = cache_root / f"{YAMNET_CACHE_STEM}_meta.json"
    cached = _load_matching_cache(np, bundle, cache_path, meta_path)
    if cached is not None:
        matrix, report = cached
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "cache_reuse",
                    "phase": YAMNET_CACHE_STEM,
                    "rows": bundle.row_count,
                    "cache_path": str(cache_path),
                }
            )
        return matrix, report

    cache_root.mkdir(parents=True, exist_ok=True)
    groups = _group_rows_by_audio(bundle, public_audio_root, professor_audio_root)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "cache_start",
                "phase": YAMNET_CACHE_STEM,
                "rows": bundle.row_count,
                "files": len(groups),
                "model_type": V2D_MODEL_TYPE,
            }
        )
    embedder = _load_yamnet_embedder()
    matrix = None
    processed = 0
    next_report = max(progress_every, 1)
    started_at = time.monotonic()
    embedding_dim: int | None = None
    batch_indices: list[int] = []
    batch_windows: list[Any] = []

    def flush_batch(file_index: int, total_files: int, audio_path: Path) -> None:
        nonlocal matrix, embedding_dim, processed, next_report, batch_indices, batch_windows
        if not batch_windows:
            return
        vectors = _embedding_vectors(np, embedder, batch_windows)
        if matrix is None:
            embedding_dim = int(vectors.shape[1])
            matrix = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=(bundle.row_count, embedding_dim),
            )
        elif int(vectors.shape[1]) != embedding_dim:
            raise V2MLError(
                "YAMNet embedding dimension changed within one extraction run"
            )
        matrix[batch_indices] = vectors
        processed += len(batch_indices)
        if processed >= next_report or processed == bundle.row_count:
            _cache_progress(
                progress_callback,
                processed,
                bundle.row_count,
                file_index,
                total_files,
                audio_path,
                started_at,
            )
            while next_report <= processed:
                next_report += max(progress_every, 1)
        batch_indices = []
        batch_windows = []

    last_file_index = 0
    last_audio_path: Path | None = None
    for file_index, (audio_path, row_indices) in enumerate(groups.items(), start=1):
        last_file_index = file_index
        last_audio_path = audio_path
        try:
            samples, metadata = read_wav_mono_array(audio_path)
        except (ImportError, WavReadError) as error:
            raise V2MLError(str(error)) from error
        for row_index in row_indices:
            window = _window_samples(np, samples, metadata.sample_rate_hz, bundle, row_index, audio_path)
            batch_indices.append(row_index)
            batch_windows.append(window)
            if len(batch_windows) >= max(progress_every, 1):
                flush_batch(file_index, len(groups), audio_path)
    if last_audio_path is not None:
        flush_batch(last_file_index, len(groups), last_audio_path)

    if matrix is None or embedding_dim is None:
        raise V2MLError("cannot build YAMNet cache without feature rows")
    matrix.flush()
    meta = _cache_meta(bundle, embedding_dim, str(np.dtype(np.float32)))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r"), _cache_report(meta, cache_path, meta_path, reused=False)


def _load_matching_cache(
    np: Any,
    bundle: V2FeatureBundle,
    cache_path: Path,
    meta_path: Path,
) -> tuple[Any, dict[str, Any]] | None:
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not _cache_meta_matches(meta, bundle):
        return None
    matrix = np.load(cache_path, mmap_mode="r")
    if tuple(matrix.shape) != tuple(meta.get("shape", ())):
        return None
    return matrix, _cache_report(meta, cache_path, meta_path, reused=True)


def _embedding_cache_from_model(
    model: Mapping[str, Any],
    bundle: V2FeatureBundle,
) -> tuple[Any, dict[str, Any]]:
    np = _require_numpy()
    embedding = model["embedding"]
    cache_path = Path(str(embedding["path"]))
    meta_path = Path(str(embedding["meta_path"]))
    cached = _load_matching_cache(np, bundle, cache_path, meta_path)
    if cached is None:
        raise V2MLError(
            "V2D embedding cache does not match the feature bundle; rerun "
            "train-v2-embedding or provide the original feature bundle used by the model"
        )
    matrix, report = cached
    feature_names = model.get("feature_names", ())
    if int(matrix.shape[1]) != len(feature_names):
        raise V2MLError("V2D embedding cache dimension does not match model feature names")
    return matrix, report


def _cache_meta_matches(meta: Mapping[str, Any], bundle: V2FeatureBundle) -> bool:
    return (
        _same_path_value(str(meta.get("source_bundle", "")), bundle.path)
        and meta.get("feature_set") == V2D_FEATURE_SET
        and meta.get("embedding_model") == YAMNET_MODEL_NAME
        and meta.get("backend") == "tensorflow_hub"
        and meta.get("model_handle") == YAMNET_MODEL_HANDLE
        and meta.get("aggregation") == YAMNET_AGGREGATION
        and tuple(meta.get("shape", ()))[:1] == (bundle.row_count,)
    )


def _same_path_value(left: str, right: str | Path) -> bool:
    if left == str(right):
        return True
    left_path = Path(left)
    right_path = Path(right)
    try:
        return left_path.resolve() == right_path.resolve()
    except OSError:
        return False


def _cache_meta(bundle: V2FeatureBundle, embedding_dim: int, dtype: str) -> dict[str, Any]:
    return {
        "feature_set": V2D_FEATURE_SET,
        "embedding_model": YAMNET_MODEL_NAME,
        "backend": "tensorflow_hub",
        "model_handle": YAMNET_MODEL_HANDLE,
        "aggregation": YAMNET_AGGREGATION,
        "shape": [bundle.row_count, embedding_dim],
        "dtype": dtype,
        "source_bundle": str(bundle.path),
        "expected_embedding_dim": YAMNET_EXPECTED_EMBEDDING_DIM,
    }


def _cache_report(meta: Mapping[str, Any], cache_path: Path, meta_path: Path, *, reused: bool) -> dict[str, Any]:
    return {
        "path": str(cache_path),
        "meta_path": str(meta_path),
        "reused": reused,
        "feature_set": meta["feature_set"],
        "embedding_model": meta["embedding_model"],
        "backend": meta["backend"],
        "model_handle": meta["model_handle"],
        "aggregation": meta["aggregation"],
        "shape": list(meta["shape"]),
        "dtype": meta["dtype"],
        "source_bundle": meta["source_bundle"],
    }


def _group_rows_by_audio(
    bundle: V2FeatureBundle,
    public_audio_root: str | Path,
    professor_audio_root: str | Path,
) -> OrderedDict[Path, list[int]]:
    public_root = Path(public_audio_root)
    professor_root = Path(professor_audio_root)
    if not public_root.exists() or not public_root.is_dir():
        raise V2MLError(f"public audio root does not exist: {public_root}")
    if not professor_root.exists() or not professor_root.is_dir():
        raise V2MLError(f"professor audio root does not exist: {professor_root}")
    groups: OrderedDict[Path, list[int]] = OrderedDict()
    for row_index, original_file in enumerate(bundle.original_files):
        root = professor_root if bundle.is_professor_data[row_index] else public_root
        audio_path = root / original_file
        groups.setdefault(audio_path, []).append(row_index)
    return groups


def _window_samples(
    np: Any,
    samples: Any,
    sample_rate_hz: int,
    bundle: V2FeatureBundle,
    row_index: int,
    audio_path: Path,
) -> Any:
    start = bundle.window_start_sec[row_index]
    end = bundle.window_end_sec[row_index]
    if end <= start:
        raise V2MLError(f"{audio_path}: invalid window bounds {start}..{end}")
    start_frame = int(round(start * sample_rate_hz))
    frame_count = int(round((end - start) * sample_rate_hz))
    if start_frame < 0 or start_frame + frame_count > len(samples):
        raise V2MLError(f"{audio_path}: window exceeds audio duration")
    window = np.asarray(samples[start_frame : start_frame + frame_count], dtype=np.float32)
    return _standardize_window(np, window, sample_rate_hz)


def _standardize_window(np: Any, window: Any, sample_rate_hz: int) -> Any:
    if sample_rate_hz <= 0:
        raise V2MLError(f"invalid sample rate: {sample_rate_hz}")
    if sample_rate_hz != TARGET_SAMPLE_RATE:
        signal = _require_scipy_signal()
        factor = math.gcd(sample_rate_hz, TARGET_SAMPLE_RATE)
        window = signal.resample_poly(
            window,
            TARGET_SAMPLE_RATE // factor,
            sample_rate_hz // factor,
        ).astype(np.float32, copy=False)
    if window.shape[0] < TARGET_SAMPLE_RATE:
        window = np.pad(window, (0, TARGET_SAMPLE_RATE - window.shape[0]))
    return window[:TARGET_SAMPLE_RATE].astype(np.float32, copy=False)


def _embedding_vector(np: Any, embedder: EmbeddingCallable, window: Any) -> Any:
    raw = np.asarray(embedder(window), dtype=np.float32)
    if raw.ndim == 2:
        raw = raw.mean(axis=0)
    elif raw.ndim != 1:
        raw = raw.reshape(-1)
    if raw.size == 0:
        raise V2MLError("YAMNet extractor returned an empty embedding")
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _embedding_vectors(np: Any, embedder: EmbeddingCallable, windows: Sequence[Any]) -> Any:
    batch = np.stack(windows).astype(np.float32, copy=False)
    raw = np.asarray(embedder(batch), dtype=np.float32)
    if raw.ndim == 2 and raw.shape[0] == len(windows):
        values = raw
    else:
        values = np.asarray(
            [_embedding_vector(np, embedder, window) for window in windows],
            dtype=np.float32,
        )
    if values.ndim != 2 or values.shape[0] != len(windows):
        raise V2MLError("YAMNet extractor returned invalid batch embeddings")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _load_yamnet_embedder() -> EmbeddingCallable:
    np = _require_numpy()
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as error:  # pragma: no cover - dependency guard.
        raise V2MLError(
            "V2D YAMNet embeddings require optional embedding dependencies; install with "
            "`python -m pip install -e '.[data,ml,embedding]'` or use "
            "`uv run --extra data --extra ml --extra embedding ...`"
        ) from error

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    model = hub.load(YAMNET_MODEL_HANDLE)

    @tf.function(reduce_retracing=True)
    def embed_batch(waveforms: Any) -> Any:
        def embed_one(waveform: Any) -> Any:
            _, embeddings, _ = model(waveform)
            return tf.reduce_mean(embeddings, axis=0)

        return tf.map_fn(
            embed_one,
            waveforms,
            fn_output_signature=tf.TensorSpec(
                shape=(YAMNET_EXPECTED_EMBEDDING_DIM,),
                dtype=tf.float32,
            ),
        )

    def embed(window: Any) -> Any:
        waveform = tf.convert_to_tensor(window, dtype=tf.float32)
        if waveform.shape.rank == 1:
            _, embeddings, _ = model(waveform)
            values = embeddings.numpy()
            if values.ndim == 2:
                return np.mean(values, axis=0)
            return values
        return embed_batch(waveform).numpy()

    return embed


def _require_scipy_signal() -> Any:
    try:
        from scipy import signal
    except ImportError as error:  # pragma: no cover - dependency guard.
        raise V2MLError(
            "V2D YAMNet embeddings require optional data dependencies; install with "
            "`python -m pip install -e '.[data,ml,embedding]'` or use "
            "`uv run --extra data --extra ml --extra embedding ...`"
        ) from error
    return signal


def _fit_embedding_head(
    embeddings: Any,
    bundle: V2FeatureBundle,
    train_mask: Any,
    *,
    max_iter: int,
    progress_callback: ProgressCallback | None,
) -> Any:
    _, _, classes = _require_ml_dependencies()
    labels = _labels_for_mask(bundle, train_mask)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "model_fit_start",
                "model_type": V2D_MODEL_TYPE,
                "train_rows": len(labels),
                "feature_count": int(embeddings.shape[1]),
            }
        )
    estimator = classes["LogisticRegression"](
        class_weight="balanced",
        max_iter=max_iter,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    estimator.fit(embeddings[train_mask], labels)
    if progress_callback is not None:
        progress_callback({"event": "model_fit_complete", "model_type": V2D_MODEL_TYPE})
    return estimator


def _embedding_metrics(
    bundle: V2FeatureBundle,
    embeddings: Any,
    estimator: Any,
    partitions: Mapping[str, Any],
    cache_report: Mapping[str, Any],
) -> dict[str, Any]:
    val_scores = _score_estimator(estimator, embeddings[partitions["val"]])
    val_labels = _labels_for_mask(bundle, partitions["val"])
    thresholds = select_thresholds(val_labels, val_scores)
    threshold = thresholds[PRIMARY_THRESHOLD_NAME]
    public_scores, inference = _score_with_timing(
        estimator,
        embeddings[partitions["public_test"]],
    )
    public_labels = _labels_for_mask(bundle, partitions["public_test"])
    professor_scores = _score_estimator(estimator, embeddings[partitions["professor_test"]])
    return {
        "schema_version": "v2_metrics_v1",
        "algorithm": V2D_ALGORITHM,
        "model_type": V2D_MODEL_TYPE,
        "track": V2D_TRACK,
        "feature_set": V2D_FEATURE_SET,
        "source_feature_bundle": str(bundle.path),
        "feature_bundle": str(bundle.path),
        "feature_count": int(embeddings.shape[1]),
        "threshold_name": PRIMARY_THRESHOLD_NAME,
        "threshold": threshold,
        "thresholds": thresholds,
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
        "embedding": dict(cache_report),
        "efficiency": {
            "feature_count": int(embeddings.shape[1]),
            "model_type": V2D_MODEL_TYPE,
            "model_size_bytes": None,
            "model_size": None,
            "parameter_count": _model_parameter_count(estimator),
            "tree_node_count": _tree_node_count(estimator),
            "average_inference_time_sec_per_window": inference,
            "runtime_dependency": "tensorflow-hub + scikit-learn",
            "deployability_label": "reference_only",
            "embedding_extractor": YAMNET_MODEL_NAME,
            "embedding_extractor_size": None,
            "classifier_head_size": None,
        },
        "selection": {
            "uses_professor_data": False,
            "primary_metric": "validation.f2",
            "primary_threshold": PRIMARY_THRESHOLD_NAME,
        },
        "limitations": [
            "V2D is reference_only unless full extractor deployment proof is added.",
            "Professor data is positive-only or near-positive-only evaluation data.",
            "Professor data is not used for training, validation, threshold tuning, or model selection.",
        ],
    }


def _model_package(
    embeddings: Any,
    estimator: Any,
    metrics: Mapping[str, Any],
    cache_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": V2D_MODEL_SCHEMA_VERSION,
        "algorithm": V2D_ALGORITHM,
        "model_type": V2D_MODEL_TYPE,
        "track": V2D_TRACK,
        "feature_set": V2D_FEATURE_SET,
        "feature_names": [
            f"yamnet_embedding_{index:04d}"
            for index in range(int(embeddings.shape[1]))
        ],
        "labels": {"positive": POSITIVE_LABEL, "negative": NEGATIVE_LABEL},
        "threshold": metrics["threshold"],
        "threshold_name": metrics["threshold_name"],
        "thresholds": dict(metrics["thresholds"]),
        "estimator": estimator,
        "embedding": dict(cache_report),
        "deployability_label": metrics["efficiency"]["deployability_label"],
        "selection": dict(metrics["selection"]),
    }


def _cache_progress(
    progress_callback: ProgressCallback | None,
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
            "phase": YAMNET_CACHE_STEM,
            "processed_rows": processed,
            "total_rows": total,
            "file_index": file_index,
            "total_files": total_files,
            "current_file": str(audio_path),
            "rate": rate,
            "eta_seconds": eta,
            "model_type": V2D_MODEL_TYPE,
        }
    )


def comparison_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the standard V2 comparison row for a V2D metrics report."""

    public_metrics = metrics["public_test"]["metrics"]
    professor_metrics = metrics["professor_test"]["metrics"] or {}
    efficiency = metrics["efficiency"]
    return {
        "model": metrics["model_type"],
        "track": metrics["track"],
        "feature/input": metrics["feature_set"],
        "public_f2": public_metrics["f2"],
        "public_f1": public_metrics["f1"],
        "public_precision": public_metrics["precision"],
        "public_recall": public_metrics["recall"],
        "public_fpr": public_metrics["false_positive_rate"],
        "professor_recall": professor_metrics.get("recall"),
        "longest_missed_segment": professor_metrics.get("longest_missed_segment_sec"),
        "model_size": efficiency["model_size"],
        "params": efficiency["parameter_count"],
        "deployability_label": efficiency["deployability_label"],
        "metrics_path": metrics.get("metrics_path"),
    }
