from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
scipy_io = pytest.importorskip("scipy.io")

from drone_sound.cli import main
from drone_sound.dataset_registry import load_dataset_registry
from drone_sound.manifest import load_manifest_csv, validate_manifest_rows
from drone_sound.nasa_mat import NasaMatConversionError, convert_nasa_mat_directory


def test_convert_nasa_mat_directory_writes_wav_and_audit(tmp_path: Path) -> None:
    mat_root = tmp_path / "mat"
    wav_out = tmp_path / "wav"
    audit_out = tmp_path / "audit.json"
    _write_nasa_mat(mat_root / "flight_001.mat", amplitude=0.5, duration_sec=0.2)
    _write_nasa_mat(mat_root / "flight_002.mat", amplitude=1.0, duration_sec=0.2)

    report = convert_nasa_mat_directory(mat_root, wav_out, audit_out, 16_000)

    assert report["scale_policy"] == "batch_global_peak"
    assert report["global_peak_pascals_mono"] > 0
    assert report["scale_factor_int16_per_pascal"] > 0
    assert len(report["files"]) == 2

    wav_path = wav_out / "flight_001.wav"
    assert wav_path.exists()
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == 3_200

    audit = json.loads(audit_out.read_text(encoding="utf-8"))
    first_file = audit["files"][0]
    assert first_file["source_sample_rate_hz"] == 20_000
    assert first_file["source_channels"] == 4
    assert first_file["clipped_samples"] == 0
    assert first_file["output_wav_sha256"]


def test_convert_nasa_mat_cli_error_for_missing_incident_pascals(tmp_path: Path) -> None:
    mat_root = tmp_path / "mat"
    mat_root.mkdir()
    scipy_io.savemat(mat_root / "bad.mat", {"acoustics": {"utc_time": np.array([0.0, 0.1])}})

    exit_code = main(
        [
            "convert-nasa-mat",
            "--mat-root",
            str(mat_root),
            "--wav-out",
            str(tmp_path / "wav"),
            "--audit-out",
            str(tmp_path / "audit.json"),
            "--sample-rate",
            "16000",
        ]
    )

    assert exit_code == 1


@pytest.mark.parametrize(
    ("mat_data", "match"),
    [
        ({}, "missing acoustics"),
        (
            {"acoustics": {"incident_pascals": np.zeros((10, 4)), "utc_time": np.zeros(10)}},
            "strictly increasing",
        ),
        (
            {"acoustics": {"incident_pascals": np.zeros((10, 4)), "utc_time": np.arange(9)}},
            "shape does not match",
        ),
    ],
)
def test_convert_nasa_mat_rejects_invalid_inputs(
    tmp_path: Path,
    mat_data: dict[str, object],
    match: str,
) -> None:
    mat_root = tmp_path / "mat"
    mat_root.mkdir()
    scipy_io.savemat(mat_root / "bad.mat", mat_data)

    with pytest.raises(NasaMatConversionError, match=match):
        convert_nasa_mat_directory(
            mat_root,
            tmp_path / "wav",
            tmp_path / "audit.json",
            16_000,
        )


def test_convert_nasa_mat_rejects_all_zero_audio(tmp_path: Path) -> None:
    mat_root = tmp_path / "mat"
    _write_nasa_mat(mat_root / "zero.mat", amplitude=0.0, duration_sec=0.1)

    with pytest.raises(NasaMatConversionError, match="no non-zero finite mono samples"):
        convert_nasa_mat_directory(
            mat_root,
            tmp_path / "wav",
            tmp_path / "audit.json",
            16_000,
        )


def test_converted_nasa_wavs_feed_public_manifest_and_train_v1(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "public"
    _write_registered_public_wavs_except_nasa(raw_root)
    mat_root = raw_root / "nasa_small_uas_flyover" / "data"
    for index in range(3):
        _write_nasa_mat(
            mat_root / f"nasa_{index}.mat",
            amplitude=0.6,
            frequency_hz=330.0 + index,
            duration_sec=2.0,
        )

    convert_nasa_mat_directory(
        mat_root,
        raw_root / "nasa_small_uas_flyover" / "converted_wav",
        tmp_path / "artifacts" / "audits" / "nasa_mat_conversion_audit.json",
        16_000,
    )

    manifest_out = tmp_path / "data" / "manifests" / "public_manifest.csv"
    audit_out = tmp_path / "artifacts" / "audits" / "public_data_audit.json"
    model_out = tmp_path / "artifacts" / "models" / "v1_baseline_model.json"
    metrics_out = tmp_path / "artifacts" / "models" / "v1_metrics.json"

    build_exit = main(
        [
            "build-public-manifest",
            "--registry",
            "configs/datasets.json",
            "--raw-root",
            str(raw_root),
            "--manifest-out",
            str(manifest_out),
            "--audit-out",
            str(audit_out),
            "--split-seed",
            "20260619",
        ]
    )
    train_exit = main(
        [
            "train-v1",
            "--manifest",
            str(manifest_out),
            "--audio-root",
            str(raw_root),
            "--model-out",
            str(model_out),
            "--metrics-out",
            str(metrics_out),
        ]
    )

    assert build_exit == 0
    assert train_exit == 0
    rows = load_manifest_csv(manifest_out)
    validate_manifest_rows(rows)
    assert "nasa_small_uas_flyover" in {row["source_dataset"] for row in rows}
    assert model_out.exists()
    assert metrics_out.exists()


def _write_nasa_mat(
    path: Path,
    *,
    amplitude: float,
    frequency_hz: float = 440.0,
    source_sample_rate_hz: int = 20_000,
    duration_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(source_sample_rate_hz * duration_sec)
    utc_time = np.arange(sample_count, dtype=np.float64) / source_sample_rate_hz
    channels = []
    for channel_index in range(4):
        phase = channel_index * 0.1
        channels.append(
            amplitude
            * np.sin((2.0 * math.pi * frequency_hz * utc_time) + phase)
        )
    incident_pascals = np.stack(channels, axis=1)
    scipy_io.savemat(
        path,
        {
            "acoustics": {
                "incident_pascals": incident_pascals,
                "utc_time": utc_time,
                "mic_ned_meters": np.zeros((4, 3), dtype=np.float64),
                "origin": "synthetic",
            }
        },
    )


def _write_registered_public_wavs_except_nasa(raw_root: Path) -> None:
    for entry in load_dataset_registry("configs/datasets.json"):
        dataset_id = str(entry["dataset_id"])
        if dataset_id == "nasa_small_uas_flyover":
            continue
        label = str(entry["label_binary"])
        amplitude = 0.8 if label == "drone" else 0.1
        frequency = 440.0 if label == "drone" else 220.0

        for relative_path in _relative_wav_paths(entry):
            path = raw_root / dataset_id / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(path, amplitude=amplitude, frequency_hz=frequency)


def _relative_wav_paths(entry: dict[str, object]) -> list[Path]:
    dataset_id = str(entry["dataset_id"])
    if dataset_id == "kielce_uav_acoustic_signals":
        return [
            *(Path("field") / f"field_{index}.wav" for index in range(3)),
            *(Path("speech") / f"speech_{index}.wav" for index in range(3)),
        ]
    if dataset_id == "idmt_traffic":
        return [
            *(Path("traffic") / f"traffic_{index}.wav" for index in range(3)),
            *(Path("bg") / f"bg_{index}.wav" for index in range(3)),
        ]
    return [Path(f"sample_{index}.wav") for index in range(3)]


def _write_wav(
    path: Path,
    *,
    amplitude: float,
    frequency_hz: float,
    sample_rate_hz: int = 16_000,
    duration_sec: float = 2.0,
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
