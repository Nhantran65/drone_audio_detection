#!/usr/bin/env python3
"""Build the canonical portable report artifact for legacy real-only evaluation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


REPORT_QUERY = """\
SELECT rank, model, track, model_type, size_kb, threshold,
       balanced_accuracy, precision, recall, f1, f2, roc_auc, pr_auc,
       fpr, fnr, tn, fp, fn, tp, positive_windows, negative_windows
FROM legacy_real_only_models
ORDER BY f2 DESC, fpr ASC
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/evaluation/private_real_legacy_native_all/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/private_real_legacy_native_all/artifact.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"refusing to overwrite report artifact: {args.output}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = _query_report_rows([_report_row(row) for row in summary["rows"]])
    by_name = {row["model"]: row for row in rows}
    random_forest = by_name["random_forest"]
    tiny_mlp_es = by_name["tiny_mlp_es"]
    source = {
        "id": "legacy_real_only_summary",
        "label": "Frozen legacy-native real-only evaluation summary",
        "path": "summary.json",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": REPORT_QUERY,
            "description": (
                "Frozen inference over 1,113 professor-positive and 592 private-negative "
                "overlapping windows; thresholds remain validation-derived."
            ),
            "tables_used": ["summary.json"],
            "filters": [
                "split = professor_test",
                "preprocessing_profile = legacy_native_rate_v1",
                "no fitting, model selection, or threshold tuning",
            ],
            "metric_definitions": {
                "F2": "5 * precision * recall / (4 * precision + recall)",
                "balanced_accuracy": "(recall + true_negative_rate) / 2",
                "false_positive_rate": "false positives / 592 negative windows",
            },
        },
    }
    generated_at = summary["generated_at_utc"]
    title = "Legacy-native models on the private real-only test"
    artifact: dict[str, Any] = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Frozen evaluation of all nine pre-normalization V2 artifacts.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "best_f2",
                    "description": "Highest descriptive F2 on the external stress test.",
                    "dataset": "headline",
                    "sourceId": source["id"],
                    "metrics": [
                        {"label": "Best F2 — Random Forest", "field": "best_f2", "format": "percent"},
                        {"label": "Recall", "field": "rf_recall", "format": "percent"},
                    ],
                },
                {
                    "id": "best_balanced_accuracy",
                    "description": "Highest mean of sensitivity and specificity.",
                    "dataset": "headline",
                    "sourceId": source["id"],
                    "metrics": [
                        {
                            "label": "Best balanced accuracy — tiny_mlp_es",
                            "field": "best_balanced_accuracy",
                            "format": "percent",
                        },
                        {"label": "F2", "field": "mlp_f2", "format": "percent"},
                    ],
                },
                {
                    "id": "lowest_fpr",
                    "description": "Lowest false-positive rate among the two strongest models.",
                    "dataset": "headline",
                    "sourceId": source["id"],
                    "metrics": [
                        {"label": "tiny_mlp_es FPR", "field": "lowest_fpr", "format": "percent"},
                        {"label": "False positives", "field": "mlp_fp", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "f2_ranking",
                    "title": "F2 by legacy-native model",
                    "subtitle": (
                        "Frozen validation thresholds; 1,113 positive and 592 negative windows."
                    ),
                    "type": "bar",
                    "dataset": "models",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {"field": "model", "type": "ordinal", "label": "Model"},
                        "y": {
                            "field": "f2",
                            "type": "quantitative",
                            "label": "F2",
                            "format": "percent",
                        },
                    },
                    "yAxisTitle": "F2",
                    "valueFormat": "percent",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "model_metrics",
                    "title": "Complete frozen-model metrics",
                    "description": "Exact window-level metrics, sorted by F2 descending.",
                    "dataset": "models",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "f2", "direction": "desc"},
                    "columns": [
                        {"field": "model", "label": "Model", "type": "text"},
                        {"field": "track", "label": "Track", "type": "text"},
                        {"field": "size_kb", "label": "Size (KB)", "format": "number"},
                        {"field": "balanced_accuracy", "label": "Balanced acc.", "format": "percent"},
                        {"field": "precision", "label": "Precision", "format": "percent"},
                        {"field": "recall", "label": "Recall", "format": "percent"},
                        {"field": "f1", "label": "F1", "format": "percent"},
                        {"field": "f2", "label": "F2", "format": "percent"},
                        {"field": "fpr", "label": "FPR", "format": "percent"},
                        {"field": "fnr", "label": "FNR", "format": "percent"},
                        {"field": "tn", "label": "TN", "format": "number"},
                        {"field": "fp", "label": "FP", "format": "number"},
                        {"field": "fn", "label": "FN", "format": "number"},
                        {"field": "tp", "label": "TP", "format": "number"},
                    ],
                }
            ],
            "sources": [source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## Random Forest leads on F2; tiny_mlp_es is the strongest lightweight trade-off\n\n"
                        "Random Forest achieved **F2 89.02%** and **recall 88.86%**, the strongest "
                        "positive-detection result. `tiny_mlp_es` achieved the highest balanced "
                        "accuracy (**85.86%**) with **FPR 7.43%** in a 277 KB package. These are "
                        "descriptive external-test results only; they did not drive fitting, threshold "
                        "selection, or model selection."
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": ["best_f2", "best_balanced_accuracy", "lowest_fpr"],
                },
                {
                    "id": "f2_findings",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## The two leading models express different operating trade-offs\n\n"
                        "Random Forest found **989/1,113** positive windows but produced **114/592** "
                        "false positives. `tiny_mlp_es` found **881/1,113** positives and reduced false "
                        "positives to **44/592**. The chart ranks all nine frozen artifacts by F2; the "
                        "table retains exact confusion counts for audit."
                    ),
                },
                {"id": "f2_chart", "type": "chart", "chartId": "f2_ranking", "layout": "full"},
                {"id": "metrics_table", "type": "table", "tableId": "model_metrics", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The stress test contains **1,705 overlapping one-second windows**: 1,113 "
                        "drone windows from one professor recording and 592 non-drone windows from ten "
                        "private recordings. F2 weights recall more heavily than precision; balanced "
                        "accuracy averages sensitivity and specificity; FPR uses the 592 negative "
                        "windows as denominator."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Frozen native-rate evaluation\n\n"
                        "All model packages and validation-derived `threshold_f2` values were loaded "
                        "without fitting or tuning. Engineered features used each recording's native "
                        "sample rate with the original frozen scaler. CNN and YAMNet followed their "
                        "legacy per-window input adapters. Every confusion matrix was reconciled to "
                        "1,113 positives and 592 negatives."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## The result is useful as a stress test, not a deployment estimate\n\n"
                        "Only one independent positive recording is present, overlapping windows are "
                        "correlated, and sample rate is confounded with label: the positive recording "
                        "is 16 kHz while negative recordings are 48 kHz. Ranking models from this set "
                        "would consume it as validation data."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "- Keep formal model selection based on the frozen validation metrics.\n"
                        "- Treat Random Forest as the high-recall reference and `tiny_mlp_es` as the "
                        "lightweight, lower-false-alarm reference.\n"
                        "- Collect additional independent positive recordings at multiple native "
                        "sample rates before making a deployment decision."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "Would the Random Forest advantage persist across new positive recording "
                        "sessions, microphones, distances, and environments once sample rate is no "
                        "longer confounded with label?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "best_f2": random_forest["f2"],
                        "rf_recall": random_forest["recall"],
                        "best_balanced_accuracy": tiny_mlp_es["balanced_accuracy"],
                        "mlp_f2": tiny_mlp_es["f2"],
                        "lowest_fpr": tiny_mlp_es["fpr"],
                        "mlp_fp": tiny_mlp_es["fp"],
                    }
                ],
                "models": rows,
            },
        },
        "sources": [source],
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote report artifact: {args.output}")
    return 0


def _report_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    confusion = metrics["confusion_matrix"]
    return {
        "rank": row["rank_by_f2"],
        "model": row["artifact_name"],
        "track": row["track"],
        "model_type": row["model_type"],
        "size_kb": row["model_size_bytes"] / 1024,
        "threshold": row["threshold"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "f2": metrics["f2"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "fpr": metrics["false_positive_rate"],
        "fnr": metrics["false_negative_rate"],
        "tn": confusion["tn"],
        "fp": confusion["fp"],
        "fn": confusion["fn"],
        "tp": confusion["tp"],
        "positive_windows": confusion["tp"] + confusion["fn"],
        "negative_windows": confusion["tn"] + confusion["fp"],
    }


def _query_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize the reviewed summary and run the exact report SQL."""

    columns = tuple(rows[0])
    text_columns = {"model", "track", "model_type"}
    definitions = ", ".join(
        f'"{column}" {"TEXT" if column in text_columns else "REAL"}'
        for column in columns
    )
    placeholders = ", ".join("?" for _ in columns)
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"CREATE TABLE legacy_real_only_models ({definitions})")
        connection.executemany(
            f"INSERT INTO legacy_real_only_models VALUES ({placeholders})",
            [[row[column] for column in columns] for row in rows],
        )
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(REPORT_QUERY)]
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
