from __future__ import annotations

import csv
import io
import json
import wave
import zipfile
from pathlib import Path

from drone_sound.cli import main


def test_professor_audit_cli_writes_redacted_report_and_manifest(tmp_path: Path) -> None:
    zip_path = tmp_path / "professor.zip"
    manifest_out = tmp_path / "data" / "manifests" / "professor_manifest.csv"
    audit_out = tmp_path / "artifacts" / "audits" / "professor_data_audit.json"
    _write_synthetic_professor_zip(zip_path)

    exit_code = main(
        [
            "audit-professor-data",
            "--zip",
            str(zip_path),
            "--manifest-out",
            str(manifest_out),
            "--audit-out",
            str(audit_out),
        ]
    )

    assert exit_code == 0
    assert manifest_out.exists()
    assert audit_out.exists()
    assert not (tmp_path / "data" / "raw").exists()
    assert not (tmp_path / "data" / "processed").exists()

    report_text = audit_out.read_text(encoding="utf-8")
    assert "63.83380" not in report_text
    assert "23.16243" not in report_text

    report = json.loads(report_text)
    assert report["summary_files"][0]["location_present"] is True
    assert report["audio_files"][0]["channels"] == 1
    assert report["audio_files"][0]["sample_rate_hz"] == 16000
    assert report["audio_files"][0]["sample_width_bytes"] == 2
    assert report["audio_files"][0]["estimated_windows"] == 3
    assert report["manifest"]["rows"] == 3
    assert report["manifest"]["split"] == "professor_test"

    rows = list(csv.DictReader(manifest_out.open(newline="", encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["window_id"].endswith("_w000000")
    assert rows[1]["window_start_sec"] == "0.500000"
    assert rows[0]["label_binary"] == "drone"
    assert rows[0]["label_subtype"] == "drone_unknown_condition"
    assert rows[0]["split"] == "professor_test"
    assert rows[0]["is_professor_data"] == "true"


def _write_synthetic_professor_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("synthetic.wav", _wav_bytes(duration_sec=2.0))
        archive.writestr(
            "synthetic_summary.txt",
            (
                "DATE,TIME,LAT,NS,LON,EW,POWER(V),TEMP(C),#FSFILES,#ZCFILES,#SCRUBBED\n"
                "2026-Jun-08,13:53:00,63.83380,N,23.16243,E,7.1,28.50,0,0,0\n"
            ),
        )


def _wav_bytes(duration_sec: float, sample_rate_hz: int = 16000) -> bytes:
    frames = int(duration_sec * sample_rate_hz)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()
