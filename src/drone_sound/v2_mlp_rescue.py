"""Isolated validation-selected rescue sweep for the canonical Tiny MLP."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from drone_sound.v2_ml import RANDOM_SEED, V2MLError, train_v2_mlp

ProgressCallback = Callable[[dict[str, Any]], None]


RESCUE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "variant_name": "baseline_repro",
        "hidden_layer_sizes": (64, 16),
        "alpha": 1e-4,
    },
    {
        "variant_name": "positive_weight_1p5",
        "hidden_layer_sizes": (64, 16),
        "alpha": 1e-4,
        "positive_class_weight": 1.5,
    },
    {
        "variant_name": "source_balanced",
        "hidden_layer_sizes": (64, 16),
        "alpha": 1e-4,
        "balance_sources": True,
    },
    {
        "variant_name": "source_balanced_positive_1p5",
        "hidden_layer_sizes": (64, 16),
        "alpha": 1e-4,
        "positive_class_weight": 1.5,
        "balance_sources": True,
    },
    {
        "variant_name": "feature_noise_0p02",
        "hidden_layer_sizes": (64, 16),
        "alpha": 1e-3,
        "feature_noise_std": 0.02,
    },
    {
        "variant_name": "wider_source_noise_0p02",
        "hidden_layer_sizes": (128, 32),
        "alpha": 1e-3,
        "balance_sources": True,
        "feature_noise_std": 0.02,
    },
)


def run_v2_mlp_rescue(
    features_path: str | Path,
    out_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    epochs: int = 20,
    batch_size: int = 4096,
    patience: int = 4,
    min_delta: float = 0.0,
    candidates: Sequence[Mapping[str, Any]] = RESCUE_CANDIDATES,
) -> dict[str, Any]:
    """Train a bounded candidate set and select using public validation only."""

    output = Path(out_dir)
    if output.exists():
        raise V2MLError(f"refusing to overwrite Tiny MLP rescue directory: {output}")
    if not candidates:
        raise V2MLError("Tiny MLP rescue requires at least one candidate")
    names = [str(candidate.get("variant_name", "")) for candidate in candidates]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise V2MLError("Tiny MLP rescue candidate names must be non-empty and unique")

    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(candidates, start=1):
        candidate = dict(raw_candidate)
        name = str(candidate.pop("variant_name"))
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "rescue_candidate_start",
                    "model_type": name,
                    "candidate": index,
                    "candidates": len(candidates),
                }
            )
        model_path = output / "models" / f"{name}.joblib"
        metrics_path = output / "metrics" / f"{name}.json"
        metrics = train_v2_mlp(
            features_path,
            model_path,
            metrics_path,
            progress_callback=progress_callback,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            random_seed=RANDOM_SEED,
            variant_name=name,
            **candidate,
        )
        validation = metrics["validation"]["primary"]
        public = metrics["public_test"]["metrics"]
        professor = metrics["professor_test"]["metrics"] or {}
        row = {
            "variant": name,
            "validation_f2": validation["f2"],
            "validation_fpr": validation["false_positive_rate"],
            "validation_recall": validation["recall"],
            "public_f2": public["f2"],
            "public_fpr": public["false_positive_rate"],
            "professor_recall_post_freeze": professor.get("recall"),
            "model_size_bytes": metrics["efficiency"]["model_size_bytes"],
            "model_size": metrics["efficiency"]["model_size"],
            "parameter_count": metrics["efficiency"]["parameter_count"],
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "rescue_candidate_complete",
                    "model_type": name,
                    "candidate": index,
                    "candidates": len(candidates),
                    "validation_f2": validation["f2"],
                    "validation_fpr": validation["false_positive_rate"],
                }
            )

    winner = max(
        rows,
        key=lambda row: (
            _metric(row["validation_f2"]),
            -_metric(row["validation_fpr"], missing=1.0),
            -int(row["model_size_bytes"]),
        ),
    )
    report = {
        "schema_version": "v2_mlp_rescue_sweep_v1",
        "feature_bundle": str(Path(features_path)),
        "candidate_count": len(rows),
        "selection_data": "public_validation_only",
        "selection_uses_public_test": False,
        "selection_uses_professor": False,
        "selection_uses_private_real": False,
        "selection_rule": (
            "highest validation F2 at validation-selected threshold_f2; ties use "
            "lower validation FPR then smaller serialized model size"
        ),
        "winner": dict(winner),
        "candidates": rows,
    }
    _write_json(output / "comparison.json", report)
    _write_json(output / "best_model.json", {"selection_rule": report["selection_rule"], **winner})
    _write_csv(output / "comparison.csv", rows)
    return report


def _metric(value: Any, *, missing: float = -1.0) -> float:
    return missing if value is None else float(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
