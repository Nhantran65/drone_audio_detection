"""Bounded lightweight tree experiments for the canonical V2 feature bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL
from drone_sound.v2_ml import (
    PRIMARY_THRESHOLD_NAME,
    RANDOM_SEED,
    V2A_TRACK,
    V2MLError,
    _feature_model_metrics,
    _format_bytes,
    _metric_sort_value,
    _model_package_for_track,
    _require_feature_set,
    _require_ml_dependencies,
    _validated_partitions,
    _write_json,
    load_feature_bundle,
)
from drone_sound.v2_real_eval import evaluate_real_v2

EXPERIMENT_SCHEMA_VERSION = "v2_lightweight_tree_experiment_v1"
ALGORITHM = "v2a_lightweight_tree_experiment"
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CandidateSpec:
    """One deterministic candidate in the bounded experiment grid."""

    name: str
    group: str
    family: str
    parameters: Mapping[str, Any]
    runtime_dependency: str


DEFAULT_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="compressed_rf_25_d10_leaf10",
        group="compressed_rf",
        family="random_forest",
        parameters={"n_estimators": 25, "max_depth": 10, "min_samples_leaf": 10},
        runtime_dependency="scikit-learn",
    ),
    CandidateSpec(
        name="compressed_rf_50_d12_leaf8",
        group="compressed_rf",
        family="random_forest",
        parameters={"n_estimators": 50, "max_depth": 12, "min_samples_leaf": 8},
        runtime_dependency="scikit-learn",
    ),
    CandidateSpec(
        name="compressed_rf_80_d14_leaf5",
        group="compressed_rf",
        family="random_forest",
        parameters={"n_estimators": 80, "max_depth": 14, "min_samples_leaf": 5},
        runtime_dependency="scikit-learn",
    ),
    CandidateSpec(
        name="hist_gb_leaf15_iter150",
        group="hist_gradient_boosting",
        family="hist_gradient_boosting",
        parameters={
            "max_iter": 150,
            "max_leaf_nodes": 15,
            "learning_rate": 0.08,
            "l2_regularization": 1.0,
            "min_samples_leaf": 30,
        },
        runtime_dependency="scikit-learn",
    ),
    CandidateSpec(
        name="hist_gb_leaf31_iter200",
        group="hist_gradient_boosting",
        family="hist_gradient_boosting",
        parameters={
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "learning_rate": 0.05,
            "l2_regularization": 2.0,
            "min_samples_leaf": 30,
        },
        runtime_dependency="scikit-learn",
    ),
    CandidateSpec(
        name="xgb_shallow_d4_iter400",
        group="shallow_xgboost",
        family="xgboost",
        parameters={
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "early_stopping_rounds": 30,
        },
        runtime_dependency="xgboost",
    ),
)


class XGBoostBinaryAdapter:
    """Expose XGBoost's numeric labels through the project's string-label API."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.classes_ = (NEGATIVE_LABEL, POSITIVE_LABEL)

    def predict_proba(self, features: Any) -> Any:
        return self.estimator.predict_proba(features)


def run_lightweight_tree_experiments(
    features_path: str | Path,
    out_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    candidates: Sequence[CandidateSpec] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Fit candidates and select one frozen winner per group using validation only."""

    if not candidates:
        raise V2MLError("lightweight tree experiment requires at least one candidate")
    _validate_candidates(candidates)
    output = Path(out_dir)
    if output.exists():
        raise V2MLError(f"refusing to overwrite lightweight tree experiment: {output}")

    bundle = load_feature_bundle(features_path)
    _require_feature_set(bundle)
    partitions = _validated_partitions(bundle)
    np, joblib, _ = _require_ml_dependencies()
    train_features = bundle.features[partitions["train"]]
    train_labels = np.asarray(
        [bundle.labels[index] for index in np.flatnonzero(partitions["train"])]
    )
    validation_features = bundle.features[partitions["val"]]
    validation_labels = np.asarray(
        [bundle.labels[index] for index in np.flatnonzero(partitions["val"])]
    )

    model_dir = output / "models"
    metrics_dir = output / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(candidates, start=1):
        _progress(
            progress_callback,
            "candidate_start",
            candidate=spec.name,
            group=spec.group,
            candidate_index=index,
            candidate_total=len(candidates),
            elapsed_seconds=time.perf_counter() - started,
        )
        fit_started = time.perf_counter()
        estimator, fit_metadata = _fit_candidate(
            spec,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
        )
        fit_seconds = time.perf_counter() - fit_started
        _progress(
            progress_callback,
            "candidate_fit_complete",
            candidate=spec.name,
            group=spec.group,
            candidate_index=index,
            candidate_total=len(candidates),
            fit_seconds=fit_seconds,
        )

        metrics = _feature_model_metrics(
            bundle,
            estimator,
            spec.name,
            V2A_TRACK,
            ALGORITHM,
            partitions,
            runtime_dependency=spec.runtime_dependency,
            deployability_label="maybe_deployable",
            primary_threshold_name=PRIMARY_THRESHOLD_NAME,
        )
        metrics["experiment"] = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "group": spec.group,
            "family": spec.family,
            "candidate": spec.name,
            "selection_inputs": ["validation.f2", "validation.false_positive_rate", "model_size_bytes"],
            "post_freeze_metrics_excluded_from_selection": [
                "public_test",
                "professor_test",
                "private_real_only",
            ],
        }
        metrics["training_config"] = {
            "random_seed": RANDOM_SEED,
            "parameters": dict(spec.parameters),
            "fit_seconds": fit_seconds,
            **fit_metadata,
        }
        metrics["efficiency"].update(_tree_complexity(estimator, spec))
        model_path = model_dir / f"{spec.name}.joblib"
        metrics_path = metrics_dir / f"{spec.name}.json"
        package = _model_package_for_track(
            bundle,
            estimator,
            spec.name,
            V2A_TRACK,
            ALGORITHM,
            metrics,
        )
        package["experiment_group"] = spec.group
        package["runtime_dependency"] = spec.runtime_dependency
        package["training_config"] = dict(metrics["training_config"])
        joblib.dump(package, model_path)
        model_size = model_path.stat().st_size
        metrics["model_path"] = str(model_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics["efficiency"]["model_size_bytes"] = model_size
        metrics["efficiency"]["model_size"] = _format_bytes(model_size)
        _write_json(metrics_path, metrics)
        row = _result_row(spec, metrics)
        rows.append(row)
        elapsed = time.perf_counter() - started
        rate = index / elapsed if elapsed else 0.0
        _progress(
            progress_callback,
            "candidate_complete",
            candidate=spec.name,
            group=spec.group,
            candidate_index=index,
            candidate_total=len(candidates),
            elapsed_seconds=elapsed,
            eta_seconds=(len(candidates) - index) / rate if rate else None,
            validation_f2=row["validation_f2"],
            validation_fpr=row["validation_fpr"],
            model_size_bytes=model_size,
        )

    winners = _select_group_winners(rows)
    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "feature_bundle": str(bundle.path),
        "source_bundle_fingerprint": bundle.bundle_fingerprint,
        "preprocessing_profile": bundle.preprocessing_profile,
        "processed_sample_rate_hz": bundle.processed_sample_rate_hz,
        "selection_rule": (
            "within each group: highest validation F2 at frozen threshold_f2; "
            "ties use lower validation FPR then smaller serialized model size"
        ),
        "selection_excludes": ["public_test", "professor_test", "private_real_only"],
        "candidate_count": len(rows),
        "candidates": rows,
        "winners": winners,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "comparison.json", report)
    _write_rows_csv(output / "comparison.csv", rows)
    _write_json(output / "best_by_group.json", {"selection_rule": report["selection_rule"], "winners": winners})
    return report


def evaluate_group_winners(
    experiment_dir: str | Path,
    private_features_path: str | Path,
    audio_root: str | Path,
    cache_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate each validation-selected group winner once on private real-only."""

    experiment = Path(experiment_dir)
    selection_path = experiment / "best_by_group.json"
    if not selection_path.is_file():
        raise V2MLError(f"missing lightweight tree group selection: {selection_path}")
    evaluation_dir = experiment / "real_only"
    if evaluation_dir.exists():
        raise V2MLError(f"refusing to overwrite real-only winner evaluation: {evaluation_dir}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winners = selection.get("winners", {})
    if set(winners) != {"compressed_rf", "hist_gradient_boosting", "shallow_xgboost"}:
        raise V2MLError("group selection must contain all three lightweight tree groups")
    evaluation_dir.mkdir(parents=True)
    reports: dict[str, Any] = {}
    started = time.perf_counter()
    for index, group in enumerate(sorted(winners), start=1):
        winner = winners[group]
        model_path = Path(winner["model_path"])
        metrics_path = evaluation_dir / f"{group}.json"
        _progress(
            progress_callback,
            "group_real_eval_start",
            group=group,
            candidate=winner["model"],
            group_index=index,
            group_total=len(winners),
            elapsed_seconds=time.perf_counter() - started,
        )
        reports[group] = evaluate_real_v2(
            model_path,
            private_features_path,
            audio_root,
            cache_dir,
            metrics_path,
            progress_callback=progress_callback,
        )
        _progress(
            progress_callback,
            "group_real_eval_complete",
            group=group,
            candidate=winner["model"],
            group_index=index,
            group_total=len(winners),
            elapsed_seconds=time.perf_counter() - started,
        )
    report = {
        "evaluation_only": True,
        "fit_called": False,
        "threshold_tuned": False,
        "selection_source": str(selection_path),
        "selection_rule": selection["selection_rule"],
        "limitations": [
            "The private real-only benchmark was observed in earlier research iterations.",
            "It contains one positive and ten negative recordings with correlated windows.",
            "These results are descriptive and cannot select or retune a candidate.",
        ],
        "groups": reports,
    }
    _write_json(evaluation_dir / "summary.json", report)
    return report


def build_experiment_summary(
    experiment_dir: str | Path,
    baseline_metrics_path: str | Path,
    baseline_real_metrics_path: str | Path,
) -> dict[str, Any]:
    """Combine the frozen group winners with the existing canonical RF baseline."""

    experiment = Path(experiment_dir)
    output_json = experiment / "final_summary.json"
    output_csv = experiment / "final_summary.csv"
    if output_json.exists() or output_csv.exists():
        raise V2MLError("refusing to overwrite lightweight tree final summary")
    comparison = json.loads((experiment / "comparison.json").read_text(encoding="utf-8"))
    real_summary = json.loads((experiment / "real_only" / "summary.json").read_text(encoding="utf-8"))
    baseline_metrics = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    baseline_real = json.loads(Path(baseline_real_metrics_path).read_text(encoding="utf-8"))
    rows = [_summary_row("baseline", "random_forest_80_d16_leaf5", baseline_metrics, baseline_real)]
    for group, winner in sorted(comparison["winners"].items()):
        candidate_metrics = json.loads(Path(winner["metrics_path"]).read_text(encoding="utf-8"))
        real_metrics = real_summary["groups"][group]
        rows.append(_summary_row(group, winner["model"], candidate_metrics, real_metrics))
    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "selection_rule": comparison["selection_rule"],
        "private_real_only_used_for_selection": False,
        "rows": rows,
        "conclusion_guardrail": (
            "Replacement decisions require new independent positive recordings; the current "
            "private real-only benchmark is descriptive because it was previously observed."
        ),
    }
    _write_json(output_json, report)
    _write_rows_csv(output_csv, rows)
    _write_run_manifest(experiment, comparison)
    return report


def _fit_candidate(
    spec: CandidateSpec,
    train_features: Any,
    train_labels: Any,
    validation_features: Any,
    validation_labels: Any,
) -> tuple[Any, dict[str, Any]]:
    if spec.family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            **dict(spec.parameters),
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=1,
        )
        estimator.fit(train_features, train_labels)
        return estimator, {"early_stopping": False}
    if spec.family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            **dict(spec.parameters),
            class_weight="balanced",
            early_stopping=True,
            scoring="loss",
            n_iter_no_change=20,
            random_state=RANDOM_SEED,
            verbose=1,
        )
        estimator.fit(
            train_features,
            train_labels,
            X_val=validation_features,
            y_val=validation_labels,
        )
        return estimator, {
            "early_stopping": True,
            "early_stopping_source": "public_validation",
            "iterations_ran": int(estimator.n_iter_),
        }
    if spec.family == "xgboost":
        try:
            import numpy as np
            import xgboost
        except ImportError as error:
            raise V2MLError(
                "shallow XGBoost requires the optional boosting dependency; use "
                "`uv run --extra data --extra ml --extra boosting ...`"
            ) from error
        encoded_train = np.asarray(train_labels == POSITIVE_LABEL, dtype=np.int32)
        encoded_validation = np.asarray(validation_labels == POSITIVE_LABEL, dtype=np.int32)
        positives = int(encoded_train.sum())
        negatives = int(len(encoded_train) - positives)
        if positives == 0 or negatives == 0:
            raise V2MLError("XGBoost training requires both binary labels")
        estimator = xgboost.XGBClassifier(
            **dict(spec.parameters),
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / positives,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=1,
        )
        estimator.fit(
            train_features,
            encoded_train,
            eval_set=[(validation_features, encoded_validation)],
            verbose=10,
        )
        adapter = XGBoostBinaryAdapter(estimator)
        return adapter, {
            "early_stopping": True,
            "early_stopping_source": "public_validation",
            "best_iteration": int(estimator.best_iteration),
            "iterations_ran": int(estimator.best_iteration) + 1,
            "scale_pos_weight": negatives / positives,
            "xgboost_version": xgboost.__version__,
        }
    raise V2MLError(f"unsupported lightweight tree family: {spec.family}")


def _validate_candidates(candidates: Sequence[CandidateSpec]) -> None:
    names = [spec.name for spec in candidates]
    if len(names) != len(set(names)):
        raise V2MLError("lightweight tree candidate names must be unique")
    allowed_families = {"random_forest", "hist_gradient_boosting", "xgboost"}
    unknown = sorted({spec.family for spec in candidates} - allowed_families)
    if unknown:
        raise V2MLError(f"unsupported lightweight tree families: {', '.join(unknown)}")


def _select_group_winners(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group"]), []).append(row)
    winners: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(groups.items()):
        winner = max(
            group_rows,
            key=lambda row: (
                _metric_sort_value(row["validation_f2"]),
                -_metric_sort_value(row["validation_fpr"], missing=1.0),
                -int(row["model_size_bytes"]),
            ),
        )
        winners[group] = dict(winner)
    return winners


def _result_row(spec: CandidateSpec, metrics: Mapping[str, Any]) -> dict[str, Any]:
    validation = metrics["validation"]["primary"]
    public = metrics["public_test"]["metrics"]
    professor = metrics["professor_test"]["metrics"]
    efficiency = metrics["efficiency"]
    return {
        "group": spec.group,
        "family": spec.family,
        "model": spec.name,
        "validation_precision": validation["precision"],
        "validation_recall": validation["recall"],
        "validation_f1": validation["f1"],
        "validation_f2": validation["f2"],
        "validation_fpr": validation["false_positive_rate"],
        "public_precision": public["precision"],
        "public_recall": public["recall"],
        "public_f1": public["f1"],
        "public_f2": public["f2"],
        "public_fpr": public["false_positive_rate"],
        "professor_recall": professor["recall"],
        "longest_missed_segment_sec": professor["longest_missed_segment_sec"],
        "model_size_bytes": efficiency["model_size_bytes"],
        "model_size": efficiency["model_size"],
        "tree_count": efficiency.get("tree_count"),
        "tree_node_count": efficiency.get("tree_node_count"),
        "average_inference_time_sec_per_window": efficiency["average_inference_time_sec_per_window"],
        "runtime_dependency": spec.runtime_dependency,
        "model_path": metrics["model_path"],
        "metrics_path": metrics["metrics_path"],
    }


def _tree_complexity(estimator: Any, spec: CandidateSpec) -> dict[str, Any]:
    if spec.family == "random_forest":
        return {"tree_count": len(estimator.estimators_)}
    if spec.family == "hist_gradient_boosting":
        return {"tree_count": int(estimator.n_iter_)}
    if spec.family == "xgboost":
        inner = estimator.estimator
        return {"tree_count": int(inner.best_iteration) + 1}
    return {"tree_count": None}


def _summary_row(
    group: str,
    model: str,
    metrics: Mapping[str, Any],
    real_report: Mapping[str, Any],
) -> dict[str, Any]:
    validation = metrics["validation"]["primary"]
    public = metrics["public_test"]["metrics"]
    professor = metrics["professor_test"]["metrics"]
    real = real_report["metrics"]
    efficiency = metrics["efficiency"]
    return {
        "group": group,
        "model": model,
        "validation_f1": validation["f1"],
        "validation_f2": validation["f2"],
        "validation_fpr": validation["false_positive_rate"],
        "public_precision": public["precision"],
        "public_recall": public["recall"],
        "public_f1": public["f1"],
        "public_f2": public["f2"],
        "public_fpr": public["false_positive_rate"],
        "professor_recall": professor["recall"],
        "real_precision": real["precision"],
        "real_recall": real["recall"],
        "real_f1": real["f1"],
        "real_f2": real["f2"],
        "real_fpr": real["false_positive_rate"],
        "model_size_bytes": efficiency["model_size_bytes"],
        "model_size": efficiency["model_size"],
        "average_inference_time_sec_per_window": efficiency["average_inference_time_sec_per_window"],
        "runtime_dependency": efficiency["runtime_dependency"],
    }


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise V2MLError("cannot write an empty lightweight tree comparison")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _write_run_manifest(experiment: Path, comparison: Mapping[str, Any]) -> None:
    files = sorted(path for path in experiment.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        "feature_bundle": comparison["feature_bundle"],
        "source_bundle_fingerprint": comparison["source_bundle_fingerprint"],
        "preprocessing_profile": comparison["preprocessing_profile"],
        "candidate_grid": [asdict(spec) for spec in DEFAULT_CANDIDATES],
        "artifact_checksums": {
            str(path.relative_to(experiment)): _sha256(path) for path in files
        },
    }
    _write_json(experiment / "run_manifest.json", manifest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(callback: ProgressCallback | None, event: str, **values: Any) -> None:
    if callback is not None:
        callback({"event": event, **values})
