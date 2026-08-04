#!/usr/bin/env python3
"""Evaluate frozen best models on professor-positive + private-negative audio only."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from drone_sound.training import NEGATIVE_LABEL, POSITIVE_LABEL, _metrics
from drone_sound.v2_ml import _score_estimator


MODEL_PATHS = {
    "tiny_mlp_es": Path("artifacts/models/v2b_mlp/tiny_mlp_es.joblib"),
    "random_forest": Path("artifacts/models/v2a_sweep/models/random_forest.joblib"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-scores",
        type=Path,
        default=Path("artifacts/analysis/private_negative_quality/window_scores.csv"),
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        default=Path("artifacts/features/v2_spectral_mfcc_features.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/private_negative_quality/real_only_test_metrics.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    print("phase=load_private_scores", flush=True)
    rows = _read_csv(args.window_scores)
    models = _load_models(MODEL_PATHS)

    print("phase=real_only_metrics modes=2 models=2", flush=True)
    real_only = {
        mode: _evaluate_private_mode(rows, mode, models)
        for mode in ("native_48k", "resampled_16k")
    }

    print("phase=old_constructed_reference public_negative_windows=loading", flush=True)
    old_reference = _evaluate_old_constructed_reference(
        rows,
        args.feature_bundle,
        models,
    )
    report = {
        "schema_version": "real_only_test_metrics_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "positive_population": "professor real positive recording",
            "negative_population": "ten professor/private traffic and forest recordings",
            "window_seconds": 1.0,
            "stride_seconds": 0.5,
            "threshold_policy": "frozen threshold_f2 from validation; no retuning",
            "model_policy": "frozen model packages; no fitting",
            "primary_end_to_end_mode": (
                "native_48k matches the current inference implementation, which extracts at each "
                "file's native sample rate"
            ),
            "robustness_mode": (
                "resampled_16k removes the positive/negative sample-rate mismatch but is not the "
                "current inference implementation"
            ),
        },
        "real_only_test": real_only,
        "old_constructed_reference": old_reference,
        "comparison": _comparison(real_only, old_reference),
        "interpretation": {
            "valid_use": (
                "external stress-test reporting for already frozen models, provided no model or "
                "threshold choice is made from these results"
            ),
            "not_supported": (
                "a precise deployment-performance estimate or statistically independent "
                "window-level confidence claim"
            ),
            "independent_units": {
                "positive_recordings": 1,
                "negative_recordings": 10,
            },
            "caveats": [
                "All positive windows come from one recording.",
                "All negative recordings come from two dates/sessions and two environments.",
                "Overlapping windows are correlated and are not independent observations.",
                "Using the results to select a model would consume this set as validation data.",
                "Native-mode labels are perfectly confounded with 16 kHz versus 48 kHz source rate.",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"phase=complete output={args.output} elapsed={time.monotonic() - started:.2f}s",
        flush=True,
    )
    return 0


def _evaluate_private_mode(
    rows: Sequence[Mapping[str, str]],
    negative_mode: str,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    positive_rows = [row for row in rows if row["mode"] == "positive_reference_16k"]
    negative_rows = [row for row in rows if row["mode"] == negative_mode]
    if not positive_rows or not negative_rows:
        raise ValueError(f"missing positive or negative rows for mode {negative_mode}")
    labels = [POSITIVE_LABEL] * len(positive_rows) + [NEGATIVE_LABEL] * len(negative_rows)
    result = {
        "counts": {POSITIVE_LABEL: len(positive_rows), NEGATIVE_LABEL: len(negative_rows)},
        "models": {},
    }
    for name, package in models.items():
        scores = [float(row[f"{name}_score"]) for row in positive_rows]
        scores.extend(float(row[f"{name}_score"]) for row in negative_rows)
        result["models"][name] = _metrics(labels, scores, float(package["threshold"]))
    return result


def _evaluate_old_constructed_reference(
    rows: Sequence[Mapping[str, str]],
    feature_bundle: Path,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    positive_rows = [row for row in rows if row["mode"] == "positive_reference_16k"]
    with np.load(feature_bundle, allow_pickle=False) as bundle:
        features = np.asarray(bundle["scaled_features"], dtype=np.float64)
        labels = np.asarray(bundle["labels"]).astype(str)
        splits = np.asarray(bundle["splits"]).astype(str)
        professor = np.asarray(bundle["is_professor_data"], dtype=bool)
    mask = (splits == "public_test") & (labels == NEGATIVE_LABEL) & (~professor)
    public_negative = features[mask]
    combined_labels = [POSITIVE_LABEL] * len(positive_rows) + [NEGATIVE_LABEL] * len(
        public_negative
    )
    result = {
        "definition": "professor real positive + public_test negative only",
        "counts": {POSITIVE_LABEL: len(positive_rows), NEGATIVE_LABEL: int(mask.sum())},
        "models": {},
    }
    for name, package in models.items():
        scores = [float(row[f"{name}_score"]) for row in positive_rows]
        scores.extend(
            float(value)
            for value in _score_estimator(package["estimator"], public_negative)
        )
        result["models"][name] = _metrics(
            combined_labels,
            scores,
            float(package["threshold"]),
        )
    return result


def _comparison(
    real_only: Mapping[str, Any],
    old_reference: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fields = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "false_positive_rate")
    for mode, mode_report in real_only.items():
        output[mode] = {}
        for model_name, metrics in mode_report["models"].items():
            old_metrics = old_reference["models"][model_name]
            output[mode][model_name] = {
                f"delta_{field}": _delta(metrics.get(field), old_metrics.get(field))
                for field in fields
            }
    return output


def _delta(current: Any, reference: Any) -> float | None:
    if current is None or reference is None:
        return None
    return float(current) - float(reference)


def _load_models(paths: Mapping[str, Path]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for name, path in paths.items():
        package = joblib.load(path)
        if package.get("threshold_name") != "threshold_f2":
            raise ValueError(f"model threshold is not frozen threshold_f2: {path}")
        output[name] = package
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
