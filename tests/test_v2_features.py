from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy.fft")

from drone_sound.cli import main
from drone_sound.v2_features import (
    apply_scaler,
    extract_feature_vector,
    feature_names,
    fit_scaler,
)


def test_v1_basic_features_match_v1_formulas() -> None:
    samples = np.asarray([0.0, 1.0, -1.0, 1.0], dtype=np.float64)

    features, replacements = extract_feature_vector(samples, 16_000, "v1_basic")

    assert replacements == 0
    assert features == pytest.approx(
        (
            0.25,
            0.75,
            math.sqrt(3.0 / 4.0),
            1.0,
            2.0 / 3.0,
        )
    )


def test_spectral_features_track_sine_dominant_frequency() -> None:
    sample_rate_hz = 16_000
    duration_sec = 1.0
    frequency_hz = 440.0
    samples = _sine_samples(sample_rate_hz, duration_sec, frequency_hz)

    features, replacements = extract_feature_vector(samples, sample_rate_hz, "spectral_v1")
    values = dict(zip(feature_names("spectral_v1"), features))

    assert replacements == 0
    assert values["dominant_frequency_hz"] == pytest.approx(frequency_hz, abs=2.0)
    assert values["spectral_centroid_hz"] == pytest.approx(frequency_hz, abs=5.0)


def test_mfcc_features_are_stable_and_finite_for_sine_and_silence() -> None:
    sample_rate_hz = 16_000
    sine = _sine_samples(sample_rate_hz, 1.0, 440.0)
    silence = np.zeros(sample_rate_hz, dtype=np.float64)

    sine_features, sine_replacements = extract_feature_vector(sine, sample_rate_hz, "mfcc_v1")
    silence_features, silence_replacements = extract_feature_vector(
        silence,
        sample_rate_hz,
        "mfcc_v1",
    )

    assert len(feature_names("mfcc_v1")) == 80
    assert len(sine_features) == 80
    assert len(silence_features) == 80
    assert sine_replacements == 0
    assert silence_replacements == 0
    assert np.isfinite(np.asarray(sine_features)).all()
    assert np.isfinite(np.asarray(silence_features)).all()


def test_spectral_mfcc_feature_count_is_component_sum() -> None:
    assert len(feature_names("spectral_mfcc_v1")) == (
        len(feature_names("v1_basic"))
        + len(feature_names("spectral_v1"))
        + len(feature_names("mfcc_v1"))
    )


def test_non_finite_feature_values_are_replaced() -> None:
    features, replacements = extract_feature_vector(
        np.asarray([0.0, np.nan, np.inf, -1.0], dtype=np.float64),
        16_000,
        "v1_basic",
    )

    assert replacements > 0
    assert np.isfinite(np.asarray(features)).all()


def test_scaler_is_fit_on_train_features_only() -> None:
    train_features = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 14.0],
        ],
        dtype=np.float64,
    )
    scaler = fit_scaler(train_features)
    scaled_train = apply_scaler(train_features, scaler)
    scaled_other = apply_scaler(np.asarray([[5.0, 18.0]], dtype=np.float64), scaler)

    assert np.mean(scaled_train, axis=0) == pytest.approx([0.0, 0.0])
    assert scaled_other.shape == (1, 2)
    assert scaler["mean"] == pytest.approx([2.0, 12.0])


def test_extract_v2_features_cli_writes_feature_scaler_and_audit_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    public_root = tmp_path / "public_audio"
    professor_root = tmp_path / "professor_audio"
    public_root.mkdir()
    professor_root.mkdir()
    public_manifest = tmp_path / "public_manifest.csv"
    professor_manifest = tmp_path / "professor_manifest.csv"
    _write_public_fixture(public_root, public_manifest)
    _write_wav(professor_root / "professor.wav", amplitude=0.7, frequency_hz=440.0)
    _write_manifest(
        professor_manifest,
        [
            _row(
                "professor",
                "professor.wav",
                "drone",
                "drone_unknown_condition",
                "professor_test",
                True,
            )
        ],
    )
    features_out = tmp_path / "artifacts" / "features.npz"
    scaler_out = tmp_path / "artifacts" / "scaler.json"
    audit_out = tmp_path / "artifacts" / "audit.json"

    exit_code = main(
        [
            "extract-v2-features",
            "--public-manifest",
            str(public_manifest),
            "--public-audio-root",
            str(public_root),
            "--feature-set",
            "spectral_mfcc_v1",
            "--features-out",
            str(features_out),
            "--scaler-out",
            str(scaler_out),
            "--audit-out",
            str(audit_out),
            "--professor-manifest",
            str(professor_manifest),
            "--professor-audio-root",
            str(professor_root),
            "--progress-every",
            "2",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "v2 feature progress: start rows=7" in captured.err
    assert "v2 feature progress: 2/7 rows" in captured.err
    assert "v2 feature progress: scaling 7 rows x 100 features" in captured.err
    bundle = np.load(features_out)
    scaler = json.loads(scaler_out.read_text(encoding="utf-8"))
    audit = json.loads(audit_out.read_text(encoding="utf-8"))

    assert bundle["raw_features"].shape == (7, 100)
    assert bundle["scaled_features"].shape == (7, 100)
    assert bundle["feature_names"].shape == (100,)
    assert list(bundle["splits"]).count("train") == 2
    assert list(bundle["splits"]).count("professor_test") == 1
    assert int(np.count_nonzero(bundle["is_professor_data"])) == 1
    assert scaler["fitted_split"] == "train"
    assert scaler["train_rows"] == 2
    assert len(scaler["mean"]) == 100
    assert audit["feature_set"] == "spectral_mfcc_v1"
    assert audit["feature_count"] == 100
    assert audit["features"]["rows"] == 7
    assert audit["counts"]["splits"]["train"]["rows"] == 2
    assert audit["counts"]["splits"]["professor_test"]["rows"] == 1
    assert audit["scaler"]["train_rows"] == 2


def test_extract_v2_features_cli_reports_missing_public_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()

    exit_code = main(
        [
            "extract-v2-features",
            "--public-manifest",
            str(tmp_path / "missing.csv"),
            "--public-audio-root",
            str(audio_root),
            "--feature-set",
            "v1_basic",
            "--features-out",
            str(tmp_path / "features.npz"),
            "--scaler-out",
            str(tmp_path / "scaler.json"),
            "--audit-out",
            str(tmp_path / "audit.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "v2 input error: public manifest does not exist" in captured.err


def test_extract_v2_features_cli_rejects_invalid_feature_set(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    manifest = tmp_path / "manifest.csv"
    audio_root.mkdir()
    manifest.write_text("placeholder\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "extract-v2-features",
                "--public-manifest",
                str(manifest),
                "--public-audio-root",
                str(audio_root),
                "--feature-set",
                "logmel_v1",
                "--features-out",
                str(tmp_path / "features.npz"),
                "--scaler-out",
                str(tmp_path / "scaler.json"),
                "--audit-out",
                str(tmp_path / "audit.json"),
            ]
        )

    assert error.value.code == 2


def _sine_samples(sample_rate_hz: int, duration_sec: float, frequency_hz: float) -> object:
    times = np.arange(int(sample_rate_hz * duration_sec), dtype=np.float64) / sample_rate_hz
    return np.sin(2.0 * math.pi * frequency_hz * times)


def _write_public_fixture(audio_root: Path, manifest: Path) -> None:
    rows = []
    specs = [
        ("train-drone", "train_drone.wav", "drone", "drone_field", "train", 0.8, 440.0),
        (
            "train-no-drone",
            "train_no_drone.wav",
            "no_drone",
            "no_drone_urban_industrial",
            "train",
            0.2,
            220.0,
        ),
        ("val-drone", "val_drone.wav", "drone", "drone_field", "val", 0.7, 440.0),
        (
            "val-no-drone",
            "val_no_drone.wav",
            "no_drone",
            "no_drone_urban_industrial",
            "val",
            0.15,
            220.0,
        ),
        (
            "public-drone",
            "public_drone.wav",
            "drone",
            "drone_field",
            "public_test",
            0.75,
            440.0,
        ),
        (
            "public-no-drone",
            "public_no_drone.wav",
            "no_drone",
            "no_drone_urban_industrial",
            "public_test",
            0.18,
            220.0,
        ),
    ]
    for prefix, filename, label, subtype, split, amplitude, frequency in specs:
        _write_wav(audio_root / filename, amplitude=amplitude, frequency_hz=frequency)
        rows.append(_row(prefix, filename, label, subtype, split, False))
    _write_manifest(manifest, rows)


def _write_wav(
    path: Path,
    *,
    amplitude: float,
    frequency_hz: float,
    sample_rate_hz: int = 16_000,
    duration_sec: float = 1.0,
) -> None:
    frames = int(sample_rate_hz * duration_sec)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        samples = bytearray()
        for index in range(frames):
            value = amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate_hz)
            samples.extend(int(value * 32767).to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(samples))


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    prefix: str,
    original_file: str,
    label_binary: str,
    label_subtype: str,
    split: str,
    is_professor_data: bool,
) -> dict[str, str]:
    return {
        "clip_id": f"{prefix}-clip",
        "window_id": f"{prefix}-window-0",
        "source_dataset": "unit_test",
        "source_url_or_doi": "local",
        "license": "test",
        "original_file": original_file,
        "original_file_checksum": f"{prefix}-checksum",
        "sample_rate_original": "16000",
        "sample_rate_processed": "16000",
        "duration_sec": "1.000000",
        "window_start_sec": "0.000000",
        "window_end_sec": "1.000000",
        "label_binary": label_binary,
        "label_subtype": label_subtype,
        "drone_model": "",
        "environment": "",
        "recording_condition": "",
        "split": split,
        "is_professor_data": "true" if is_professor_data else "false",
        "notes": "",
    }
