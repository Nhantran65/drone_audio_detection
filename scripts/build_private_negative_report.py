#!/usr/bin/env python3
"""Build the canonical report artifact for the private negative audio audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


TITLE = "Đánh giá chất lượng private negative audio"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("artifacts/analysis/private_negative_quality"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/private_negative_quality/artifact.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_artifact(args.analysis_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"phase=report_artifact_written output={args.output}", flush=True)
    return 0


def build_artifact(analysis_dir: Path) -> dict[str, Any]:
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    profiles = _read_csv(analysis_dir / "file_profile.csv")
    file_models = _read_csv(analysis_dir / "file_model_summary.csv")
    suspects = _read_csv(analysis_dir / "suspect_windows.csv")
    drift = _read_csv(analysis_dir / "feature_drift_top.csv")
    generated_at = str(summary["generated_at_utc"])
    evaluation = summary["frozen_model_evaluation"]

    overview = [
        {
            "recordings": summary["scope"]["negative_files"],
            "duration_sec": summary["scope"]["negative_duration_sec"],
            "windows": summary["scope"]["negative_windows_per_mode"],
            "suspect_windows": summary["label_review"]["consensus_suspect_windows"],
        }
    ]
    model_rates = []
    for model in ("tiny_mlp_es", "random_forest"):
        public = evaluation["public_test_negative_reference"][model]
        native = evaluation["new_negative"]["native_48k"]["models"][model]
        sampled = evaluation["new_negative"]["resampled_16k"]["models"][model]
        model_rates.extend(
            [
                {"cohort": "Public", "model": model, "fpr": public["false_positive_rate"]},
                {"cohort": "Private native", "model": model, "fpr": native["false_positive_rate"]},
                {
                    "cohort": "Private 16 kHz",
                    "model": model,
                    "fpr": sampled["false_positive_rate"],
                },
            ]
        )
    resampled_rates = [
        {
            "mlp_fpr": evaluation["new_negative"]["resampled_16k"]["models"]["tiny_mlp_es"][
                "false_positive_rate"
            ],
            "rf_fpr": evaluation["new_negative"]["resampled_16k"]["models"]["random_forest"][
                "false_positive_rate"
            ],
            "mlp_flip": evaluation["resampling_sensitivity"]["tiny_mlp_es"][
                "decision_flip_fraction"
            ],
            "rf_flip": evaluation["resampling_sensitivity"]["random_forest"][
                "decision_flip_fraction"
            ],
        }
    ]
    sample_rates = [
        {
            "negative_sr": summary["audio_format"]["negative_sample_rates_hz"][0],
            "positive_sr": summary["audio_format"]["positive_sample_rate_hz"],
        }
    ]
    environment_rates = []
    by_environment = evaluation["new_negative"]["resampled_16k"]["by_environment"]
    for environment, model_values in by_environment.items():
        for model in ("tiny_mlp_es", "random_forest"):
            environment_rates.append(
                {
                    "environment": environment,
                    "model": model,
                    "fpr": model_values[model]["false_positive_rate"],
                }
            )
    profile_rows = [
        {
            "file": row["file"],
            "environment": row["environment"],
            "duration_sec": float(row["duration_sec"]),
            "median_dbfs": float(row["one_sec_rms_dbfs_p50"]),
            "level_spread_db": float(row["one_sec_level_spread_db"]),
            "clipped_pct": 100.0 * float(row["clipped_sample_fraction"]),
            "longest_zero_sec": float(row["longest_zero_run_sec"]),
        }
        for row in profiles
    ]
    file_model_rows = [
        {
            "mode": row["mode"],
            "file": row["file"],
            "environment": row["environment"],
            "model": row["model"],
            "windows": int(row["windows"]),
            "fpr": float(row["false_positive_rate"]),
        }
        for row in file_models
    ]
    drift_rows = [
        {
            "rank": int(row["rank"]),
            "feature": row["feature"],
            "smd": float(row["standardized_mean_difference"]),
            "absolute_smd": float(row["absolute_smd"]),
        }
        for row in drift
        if row["mode"] == "resampled_16k" and int(row["rank"]) <= 10
    ]
    suspect_rows = [
        {
            "file": row["file"],
            "environment": row["environment"],
            "start_sec": float(row["window_start_sec"]),
            "end_sec": float(row["window_end_sec"]),
            "level_dbfs": float(row["rms_dbfs"]),
            "mlp_score": float(row["tiny_mlp_es_score"]),
            "rf_score": float(row["random_forest_score"]),
        }
        for row in suspects
    ]

    datasets = {
        "overview": overview,
        "resampled_rates": resampled_rates,
        "sample_rates": sample_rates,
        "model_rates": model_rates,
        "environment_rates": environment_rates,
        "file_profiles": profile_rows,
        "file_model_rates": file_model_rows,
        "feature_drift": drift_rows,
        "suspect_windows": suspect_rows,
    }
    manifest_sources = [
        {
            "id": "analysis_summary",
            "label": "Frozen-model audit summary",
            "path": "artifacts/analysis/private_negative_quality/summary.json",
        },
        {
            "id": "file_profiles",
            "label": "WAV signal profiles",
            "path": "artifacts/analysis/private_negative_quality/file_profile.csv",
        },
        {
            "id": "file_model_rates",
            "label": "Per-recording frozen-model scores",
            "path": "artifacts/analysis/private_negative_quality/file_model_summary.csv",
        },
        {
            "id": "feature_drift",
            "label": "Feature drift audit",
            "path": "artifacts/analysis/private_negative_quality/feature_drift_top.csv",
        },
        {
            "id": "suspect_windows",
            "label": "Consensus label-review queue",
            "path": "artifacts/analysis/private_negative_quality/suspect_windows.csv",
        },
    ]
    sources = [
        {
            "id": source["id"],
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": _source_sql(source["path"]),
                "description": (
                    "Reads the bounded output generated non-destructively by "
                    "scripts/analyze_private_negative_audio.py; stored thresholds were reused "
                    "without retuning."
                ),
                "executed_at": generated_at,
                "tables_used": [source["path"]],
                "filters": [
                    "1.0 s windows",
                    "0.5 s stride",
                    "private negative WAV files only unless explicitly labeled reference",
                ],
                "metric_definitions": [
                    "FPR = negative windows scored at or above the model's frozen threshold_f2",
                    "recording macro FPR = unweighted mean of per-recording window FPR",
                    "SMD = standardized difference between private and public-negative file means",
                ],
            },
        }
        for source in manifest_sources
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Non-destructive audit and go/no-go recommendation for newly supplied negative WAV recordings.",
            "generatedAt": generated_at,
            "cards": _cards(),
            "charts": _charts(),
            "tables": _tables(),
            "sources": manifest_sources,
            "blocks": _blocks(),
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }


def _cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "scope_card",
            "description": "Quy mô dữ liệu negative mới.",
            "dataset": "overview",
            "sourceId": "analysis_summary",
            "metrics": [
                {"label": "Recordings", "field": "recordings", "format": "number"},
                {"label": "Duration (s)", "field": "duration_sec", "format": "number"},
                {"label": "Windows / mode", "field": "windows", "format": "number"},
                {"label": "Đoạn cần nghe lại", "field": "suspect_windows", "format": "number"},
            ],
        },
        {
            "id": "risk_card",
            "description": "FPR và decision flip sau resample 16 kHz.",
            "dataset": "resampled_rates",
            "sourceId": "analysis_summary",
            "metrics": [
                {"label": "MLP FPR · 16 kHz", "field": "mlp_fpr", "format": "percent"},
                {"label": "RF FPR · 16 kHz", "field": "rf_fpr", "format": "percent"},
                {"label": "MLP decision flip", "field": "mlp_flip", "format": "percent"},
                {"label": "RF decision flip", "field": "rf_flip", "format": "percent"},
            ],
        },
        {
            "id": "sample_rate_card",
            "description": "Sample-rate confound giữa hai private cohorts.",
            "dataset": "sample_rates",
            "sourceId": "analysis_summary",
            "metrics": [
                {"label": "Negative sample rate", "field": "negative_sr", "format": "number"},
                {"label": "Positive sample rate", "field": "positive_sr", "format": "number"},
            ],
        },
    ]


def _charts() -> list[dict[str, Any]]:
    return [
        {
            "id": "model_fpr_chart",
            "title": "False-positive rate thay đổi mạnh theo preprocessing",
            "subtitle": "Cùng frozen threshold_f2; không tune lại trên private data.",
            "type": "bar",
            "dataset": "model_rates",
            "sourceId": "analysis_summary",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "cohort", "type": "nominal", "label": "Cohort / mode"},
                "y": {"field": "fpr", "type": "quantitative", "label": "False-positive rate"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Model"},
                    {"field": "fpr", "type": "quantitative", "label": "FPR", "format": "percent"},
                ],
            },
        },
        {
            "id": "environment_fpr_chart",
            "title": "Forest là stress case của MLP sau resample 16 kHz",
            "subtitle": "Window-level diagnostic FPR theo environment.",
            "type": "bar",
            "dataset": "environment_rates",
            "sourceId": "analysis_summary",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "environment", "type": "nominal", "label": "Environment"},
                "y": {"field": "fpr", "type": "quantitative", "label": "False-positive rate"},
                "color": {"field": "model", "type": "nominal", "label": "Model"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Model"},
                    {"field": "fpr", "type": "quantitative", "label": "FPR", "format": "percent"},
                ],
            },
        },
    ]


def _tables() -> list[dict[str, Any]]:
    return [
        {
            "id": "profile_table",
            "title": "Signal integrity theo recording",
            "subtitle": "Level là mô tả tín hiệu; không phải ước lượng SNR.",
            "dataset": "file_profiles",
            "sourceId": "file_profiles",
            "defaultSort": {"field": "environment", "direction": "asc"},
            "columns": [
                {"field": "file", "label": "File"},
                {"field": "environment", "label": "Environment"},
                {"field": "duration_sec", "label": "Duration (s)", "format": "number"},
                {"field": "median_dbfs", "label": "Median dBFS", "format": "number"},
                {"field": "level_spread_db", "label": "P90–P10 (dB)", "format": "number"},
                {"field": "clipped_pct", "label": "Clipped (%)", "format": "number"},
                {"field": "longest_zero_sec", "label": "Max zero run (s)", "format": "number"},
            ],
        },
        {
            "id": "recording_fpr_table",
            "title": "FPR theo từng recording",
            "subtitle": "Cho thấy biến thiên giữa file; window counts không phải independent sample counts.",
            "dataset": "file_model_rates",
            "sourceId": "file_model_rates",
            "defaultSort": {"field": "fpr", "direction": "desc"},
            "columns": [
                {"field": "mode", "label": "Mode"},
                {"field": "file", "label": "File"},
                {"field": "environment", "label": "Environment"},
                {"field": "model", "label": "Model"},
                {"field": "windows", "label": "Windows", "format": "number"},
                {"field": "fpr", "label": "FPR", "format": "percent"},
            ],
        },
        {
            "id": "drift_table",
            "title": "Top feature drift sau resample 16 kHz",
            "subtitle": "Absolute standardized mean difference ở cấp recording.",
            "dataset": "feature_drift",
            "sourceId": "feature_drift",
            "defaultSort": {"field": "absolute_smd", "direction": "desc"},
            "columns": [
                {"field": "rank", "label": "Rank", "format": "number"},
                {"field": "feature", "label": "Feature"},
                {"field": "smd", "label": "SMD", "format": "number"},
                {"field": "absolute_smd", "label": "|SMD|", "format": "number"},
            ],
        },
        {
            "id": "suspect_table",
            "title": "Queue cần nghe và gán nhãn thủ công",
            "subtitle": "Chỉ gồm cửa sổ mà cả hai frozen models cùng phát hiện sau resample 16 kHz.",
            "dataset": "suspect_windows",
            "sourceId": "suspect_windows",
            "defaultSort": {"field": "mlp_score", "direction": "desc"},
            "columns": [
                {"field": "file", "label": "File"},
                {"field": "environment", "label": "Environment"},
                {"field": "start_sec", "label": "Start (s)", "format": "number"},
                {"field": "end_sec", "label": "End (s)", "format": "number"},
                {"field": "level_dbfs", "label": "Level dBFS", "format": "number"},
                {"field": "mlp_score", "label": "MLP score", "format": "number"},
                {"field": "rf_score", "label": "RF score", "format": "number"},
            ],
        },
    ]


def _blocks() -> list[dict[str, Any]]:
    return [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "analysis_summary",
            "body": (
                "## Tóm tắt kỹ thuật\n\n"
                "**Kết luận: dùng có điều kiện cho held-out private hard-negative evaluation; chưa dùng để train.** "
                "Mười WAV (302 giây) không có dấu hiệu file hỏng nghiêm trọng, nhưng negative là 48 kHz trong khi "
                "professor-positive là 16 kHz. Sau resample, MLP có FPR 45.4% còn Random Forest 1.9%; "
                "44.1% và 19.1% quyết định tương ứng bị flip so với native mode. Đây là confound preprocessing, "
                "không phải bằng chứng rằng một model tốt hơn."
            ),
        },
        {
            "id": "metrics",
            "type": "metric-strip",
            "cardIds": ["scope_card", "risk_card", "sample_rate_card"],
        },
        {
            "id": "key_finding_heading",
            "type": "markdown",
            "sourceId": "analysis_summary",
            "body": (
                "## Phát hiện chính\n\n"
                "Public-test FPR chỉ 0.32% (MLP) và 0.61% (RF), nhưng private negatives lệch mạnh khỏi baseline. "
                "Hai model còn phản ứng ngược chiều khi đổi sample rate, cho thấy feature/model pipeline chưa có "
                "sample-rate contract nhất quán."
            ),
        },
        {"id": "model_fpr", "type": "chart", "chartId": "model_fpr_chart"},
        {
            "id": "environment_note",
            "type": "markdown",
            "sourceId": "analysis_summary",
            "body": (
                "Forest audio có mức âm thấp (median gần −43 đến −49 dBFS) và là stress case rõ nhất của MLP: "
                "FPR 76.1% sau resample 16 kHz, so với 18.7% ở traffic. RF ở cả hai environment khoảng 1.8–1.9%."
            ),
        },
        {"id": "environment_fpr", "type": "chart", "chartId": "environment_fpr_chart"},
        {
            "id": "signal_quality_heading",
            "type": "markdown",
            "sourceId": "file_profiles",
            "body": (
                "## Chất lượng tín hiệu\n\n"
                "Không file nào vượt ngưỡng clipping 0.1%, DC offset 1%, zero-run 100 ms hoặc median dưới −50 dBFS. "
                "Traffic biến thiên level lớn hơn; forest rất nhỏ tiếng. Không có clean reference nên không thể báo SNR thật."
            ),
        },
        {"id": "profile", "type": "table", "tableId": "profile_table"},
        {
            "id": "recording_robustness",
            "type": "markdown",
            "sourceId": "analysis_summary",
            "body": (
                "## Robustness theo recording\n\n"
                "Với 16 kHz, recording-macro FPR của MLP là 50.9% (bootstrap 95% CI 35.0–66.4%); RF là 2.6% "
                "(0.8–4.7%). Khoảng tin cậy rộng phản ánh đúng việc chỉ có 10 recording độc lập."
            ),
        },
        {"id": "recording_fpr", "type": "table", "tableId": "recording_fpr_table"},
        {
            "id": "drift_heading",
            "type": "markdown",
            "sourceId": "feature_drift",
            "body": (
                "## Domain shift\n\n"
                "Sau resample 16 kHz, 24/100 feature có |SMD| ≥ 1 và 16/100 có |SMD| ≥ 2 so với public train negatives "
                "ở cấp file; drift lớn nhất tập trung ở độ lệch chuẩn delta-MFCC."
            ),
        },
        {"id": "drift", "type": "table", "tableId": "drift_table"},
        {
            "id": "review_heading",
            "type": "markdown",
            "sourceId": "suspect_windows",
            "body": (
                "## Các đoạn cần nghe lại\n\n"
                "Có 8 cửa sổ được cả hai model đánh dấu sau resample. Đây chỉ là queue ưu tiên để con người xác nhận "
                "có drone/âm thanh giống drone hay không; model agreement không chứng minh label sai."
            ),
        },
        {"id": "suspects", "type": "table", "tableId": "suspect_table"},
        {
            "id": "scope_method",
            "type": "markdown",
            "sourceId": "analysis_summary",
            "body": (
                "## Phạm vi, dữ liệu và định nghĩa metric\n\n"
                "Phân tích gồm 5 traffic + 5 forest recordings, chia cửa sổ 1.0 giây với stride 0.5 giây. FPR là tỷ lệ "
                "negative windows vượt frozen `threshold_f2`. Recording-macro FPR lấy trung bình không trọng số giữa WAV; "
                "bootstrap resample 20,000 lần theo recording."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Phương pháp\n\n"
                "Audio gốc được giữ nguyên. Mỗi file được profile ở native format; feature `spectral_mfcc_v1` được trích "
                "hai lần: native 48 kHz và resample polyphase trong RAM xuống 16 kHz. Scaler train cũ và hai model frozen "
                "được tái sử dụng, không tune threshold. Drift được tính trên file-mean để giảm pseudo-replication."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Hạn chế\n\n"
                "Chưa có human listening/timestamp annotation; hai environment chỉ đến từ hai ngày/session; các cửa sổ overlap "
                "có tương quan; public feature bundle hiện không enforce 16 kHz đồng nhất. Vì vậy không diễn giải window count như "
                "sample size và không dùng private set này để chọn model."
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": (
                "## Khuyến nghị\n\n"
                "1. Nghe và annotate 8 timestamp ưu tiên; sau đó spot-check ngẫu nhiên phần còn lại.\n"
                "2. Giữ raw noise; không denoise/aggressive normalization.\n"
                "3. Đăng ký các file đã duyệt thành held-out private hard-negative evaluation set; cấm threshold tuning/model selection.\n"
                "4. Chọn một sample-rate contract, rebuild toàn bộ public/private features và retrain trước khi cân nhắc training use.\n"
                "5. Thu thêm nhiều ngày, thiết bị, khoảng cách và environment độc lập."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Câu hỏi tiếp theo\n\n"
                "Negative recordings có được xác nhận tại hiện trường là hoàn toàn không có drone không? Thiết bị/mic/gain có giống "
                "professor-positive data không? Deployment sẽ nhận 48 kHz hay 16 kHz? Các câu trả lời này quyết định contract "
                "preprocessing và thiết kế vòng thu dữ liệu tiếp theo."
            ),
        },
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_sql(path: str) -> str:
    reader = "read_json_auto" if path.endswith(".json") else "read_csv_auto"
    return f"SELECT * FROM {reader}('{path}')"


if __name__ == "__main__":
    raise SystemExit(main())
