from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from drone_sound.audio_preprocessing import CANONICAL_PREPROCESSING_PROFILE
from drone_sound.constants import REQUIRED_MANIFEST_FIELDS
from drone_sound.manifest import load_manifest_csv
from drone_sound.v2_ml import load_feature_bundle
from drone_sound.v3_features import repartition_v3_features
from drone_sound.v3_group_split import (
    V3_GROUP_PARSER_VERSION,
    V3_SPLIT_VERSION,
    build_v3_grouped_manifest,
    derive_group_id,
)


def test_dataset_group_parsers_collapse_correlated_recordings() -> None:
    assert derive_group_id(
        "dronenoise_database", "dronenoise_database/Ed_run_ev1_M1.wav"
    ) == derive_group_id(
        "dronenoise_database", "dronenoise_database/Ed_run_ev1_M9.wav"
    )
    assert derive_group_id(
        "kielce_uav_acoustic_signals",
        "kielce_uav_acoustic_signals/X4_D1/REC/x4_d1_5m_230315_001.wav",
    ) == derive_group_id(
        "kielce_uav_acoustic_signals",
        "kielce_uav_acoustic_signals/X4_D1/REC/x4_d1_10m_230315_099.wav",
    )
    assert derive_group_id(
        "urbansound8k", "urbansound8k/UrbanSound8K/audio/fold4/1-2-3-4.wav"
    ).endswith(":fold4")


def test_v3_manifest_and_feature_repartition_are_group_safe(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.csv"
    source_rows = _public_rows()
    _write_manifest(source_manifest, source_rows)
    grouped_manifest = tmp_path / "v3.csv"
    group_map = tmp_path / "groups.csv"
    split_audit = tmp_path / "split.json"

    report = build_v3_grouped_manifest(
        source_manifest,
        grouped_manifest,
        group_map,
        split_audit,
        split_seed=20260804,
    )

    rows = load_manifest_csv(grouped_manifest)
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert report["group_overlap"]["valid"] is True
    assert {row["split"] for row in rows} == {"train", "val", "public_test"}
    assert [row["window_id"] for row in rows] == [row["window_id"] for row in source_rows]

    source_bundle = tmp_path / "source.npz"
    _write_source_bundle(source_bundle, source_rows)
    output = tmp_path / "v3.npz"
    scaler = tmp_path / "scaler.json"
    feature_audit = tmp_path / "features.json"
    repartition_v3_features(
        source_bundle,
        grouped_manifest,
        output,
        scaler,
        feature_audit,
    )

    bundle = load_feature_bundle(output)
    assert bundle.split_version == V3_SPLIT_VERSION
    assert bundle.group_parser_version == V3_GROUP_PARSER_VERSION
    assert len(bundle.group_ids) == bundle.row_count
    assert set(bundle.splits) == {"train", "val", "public_test", "professor_test"}
    scaler_data = json.loads(scaler.read_text(encoding="utf-8"))
    assert scaler_data["fitted_split"] == "train"
    assert scaler_data["train_rows"] == list(bundle.splits).count("train")
    audit = json.loads(feature_audit.read_text(encoding="utf-8"))
    assert audit["raw_feature_values_reused"] is True
    assert audit["scaler_refit_on_v3_train_only"] is True


def _public_rows() -> list[dict[str, str]]:
    files = {
        "dronenoise_database": [f"dronenoise_database/run{index}_M1.wav" for index in range(3)],
        "kielce_uav_acoustic_signals": [
            f"kielce_uav_acoustic_signals/X4_D{index}/REC/x4_d{index}_5m_23010{index}_001.wav"
            for index in range(1, 4)
        ],
        "nasa_small_uas_flyover": [
            f"nasa_small_uas_flyover/converted_wav/edge_flyover_00{index}.wav"
            for index in range(1, 4)
        ],
        "aerosonicdb_ypad_0523": [
            f"aerosonicdb_ypad_0523/audio/0/000000_2023-01-0{index}_0{index}-00-00_0_0.wav"
            for index in range(1, 4)
        ],
        "idmt_traffic": [
            f"idmt_traffic/audio/2019-10-2{index}-08-40_Location{index}_30Kmh_1_ME_CH12.wav"
            for index in range(1, 4)
        ],
        "urbansound8k": [
            f"urbansound8k/UrbanSound8K/audio/fold{index}/10000-3-{index}-0.wav"
            for index in range(1, 4)
        ],
    }
    rows = []
    for dataset, paths in files.items():
        label = "drone" if dataset in {
            "dronenoise_database",
            "kielce_uav_acoustic_signals",
            "nasa_small_uas_flyover",
        } else "no_drone"
        subtype = "drone_field" if label == "drone" else "no_drone_background"
        for file_index, path in enumerate(paths):
            for window_index in range(2):
                clip = f"{dataset}_{file_index}"
                rows.append(
                    {
                        "clip_id": clip,
                        "window_id": f"{clip}_w{window_index}",
                        "source_dataset": dataset,
                        "source_url_or_doi": "test",
                        "license": "test",
                        "original_file": path,
                        "original_file_checksum": f"sha-{dataset}-{file_index}",
                        "sample_rate_original": "16000",
                        "sample_rate_processed": "16000",
                        "duration_sec": "1.5",
                        "window_start_sec": str(window_index * 0.5),
                        "window_end_sec": str(window_index * 0.5 + 1.0),
                        "label_binary": label,
                        "label_subtype": subtype,
                        "drone_model": "",
                        "environment": "",
                        "recording_condition": "",
                        "split": ("train", "val", "public_test")[file_index],
                        "is_professor_data": "false",
                        "notes": "test",
                    }
                )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_source_bundle(path: Path, rows: list[dict[str, str]]) -> None:
    professor = {
        **rows[0],
        "clip_id": "professor",
        "window_id": "professor_w0",
        "original_file": "positive.wav",
        "source_dataset": "professor_real_world",
        "split": "professor_test",
        "is_professor_data": "true",
    }
    all_rows = rows + [professor]
    raw = np.arange(len(all_rows) * 3, dtype=np.float64).reshape(len(all_rows), 3)
    np.savez_compressed(
        path,
        raw_features=raw,
        scaled_features=raw,
        labels=np.asarray([row["label_binary"] for row in all_rows]),
        splits=np.asarray([row["split"] for row in all_rows]),
        window_ids=np.asarray([row["window_id"] for row in all_rows]),
        clip_ids=np.asarray([row["clip_id"] for row in all_rows]),
        original_files=np.asarray([row["original_file"] for row in all_rows]),
        source_datasets=np.asarray([row["source_dataset"] for row in all_rows]),
        label_subtypes=np.asarray([row["label_subtype"] for row in all_rows]),
        environments=np.asarray([row["environment"] for row in all_rows]),
        window_start_sec=np.asarray([float(row["window_start_sec"]) for row in all_rows]),
        window_end_sec=np.asarray([float(row["window_end_sec"]) for row in all_rows]),
        is_professor_data=np.asarray([row["is_professor_data"] == "true" for row in all_rows]),
        source_sample_rates_hz=np.full(len(all_rows), 16000),
        processed_sample_rates_hz=np.full(len(all_rows), 16000),
        feature_names=np.asarray(["a", "b", "c"]),
        feature_set=np.asarray("spectral_mfcc_v1"),
        bundle_schema_version=np.asarray("v2_feature_bundle_v2"),
        preprocessing_profile=np.asarray(CANONICAL_PREPROCESSING_PROFILE),
        processed_sample_rate_hz=np.asarray(16000),
        bundle_fingerprint=np.asarray("source-fingerprint"),
    )
