"""Audit professor-provided drone audio zip files."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_sound.constants import (
    PROFESSOR_DEFAULT_LABEL,
    PROFESSOR_DEFAULT_SUBTYPE,
    PROFESSOR_LICENSE,
    PROFESSOR_SOURCE_DATASET,
    PROFESSOR_SPLIT,
    TARGET_SAMPLE_RATE_HZ,
    WINDOW_LENGTH_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from drone_sound.manifest import expected_window_count, validate_manifest_rows


@dataclass(frozen=True)
class AudioAudit:
    filename: str
    sha256: str
    channels: int
    sample_rate_hz: int
    sample_width_bytes: int
    frames: int
    duration_sec: float
    estimated_windows: int


def audit_professor_zip(
    zip_path: str | Path,
    manifest_out: str | Path,
    audit_out: str | Path,
) -> dict[str, Any]:
    """Audit a professor zip and write manifest plus redacted audit JSON."""

    zip_path = Path(zip_path)
    manifest_out = Path(manifest_out)
    audit_out = Path(audit_out)

    audio_audits: list[AudioAudit] = []
    summary_reports: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, str]] = []

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            data = archive.read(info.filename)
            if suffix == ".wav":
                audio_audit = _audit_wav(info.filename, data)
                audio_audits.append(audio_audit)
                manifest_rows.extend(_window_rows_for_audio(audio_audit, zip_path.name))
            elif suffix == ".txt":
                summary_reports.append(_audit_summary_text(info.filename, data))

    validate_manifest_rows(manifest_rows)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_out, manifest_rows)

    report = {
        "source_zip": zip_path.name,
        "source_zip_sha256": _sha256_path(zip_path),
        "redaction": {
            "gps_coordinates": "redacted",
            "policy": "exact professor-data location is omitted by default",
        },
        "audio_files": [_audio_audit_to_json(item) for item in audio_audits],
        "summary_files": summary_reports,
        "manifest": {
            "path": str(manifest_out),
            "rows": len(manifest_rows),
            "split": PROFESSOR_SPLIT,
            "label_binary": PROFESSOR_DEFAULT_LABEL,
            "label_subtype": PROFESSOR_DEFAULT_SUBTYPE,
        },
    }

    audit_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _audit_wav(filename: str, data: bytes) -> AudioAudit:
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate_hz = wav_file.getframerate()
        duration_sec = frames / sample_rate_hz
        return AudioAudit(
            filename=filename,
            sha256=hashlib.sha256(data).hexdigest(),
            channels=wav_file.getnchannels(),
            sample_rate_hz=sample_rate_hz,
            sample_width_bytes=wav_file.getsampwidth(),
            frames=frames,
            duration_sec=duration_sec,
            estimated_windows=expected_window_count(duration_sec),
        )


def _audit_summary_text(filename: str, data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    location_present = any(
        row.get("LAT", "").strip() and row.get("LON", "").strip() for row in rows
    )
    return {
        "filename": filename,
        "rows": len(rows),
        "columns": reader.fieldnames or [],
        "location_present": location_present,
    }


def _window_rows_for_audio(audio: AudioAudit, source_zip: str) -> list[dict[str, str]]:
    clip_id = f"professor_{_slug(Path(audio.filename).stem)}_{audio.sha256[:8]}"
    rows = []
    for index in range(audio.estimated_windows):
        start = index * WINDOW_STRIDE_SECONDS
        end = start + WINDOW_LENGTH_SECONDS
        rows.append(
            {
                "clip_id": clip_id,
                "window_id": f"{clip_id}_w{index:06d}",
                "source_dataset": PROFESSOR_SOURCE_DATASET,
                "source_url_or_doi": source_zip,
                "license": PROFESSOR_LICENSE,
                "original_file": audio.filename,
                "original_file_checksum": audio.sha256,
                "sample_rate_original": str(audio.sample_rate_hz),
                "sample_rate_processed": str(TARGET_SAMPLE_RATE_HZ),
                "duration_sec": f"{audio.duration_sec:.6f}",
                "window_start_sec": f"{start:.6f}",
                "window_end_sec": f"{end:.6f}",
                "label_binary": PROFESSOR_DEFAULT_LABEL,
                "label_subtype": PROFESSOR_DEFAULT_SUBTYPE,
                "drone_model": "",
                "environment": "",
                "recording_condition": "",
                "split": PROFESSOR_SPLIT,
                "is_professor_data": "true",
                "notes": "professor positive-only or near-positive-only data",
            }
        )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty professor manifest")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _audio_audit_to_json(audio: AudioAudit) -> dict[str, Any]:
    return {
        "filename": audio.filename,
        "sha256": audio.sha256,
        "channels": audio.channels,
        "sample_rate_hz": audio.sample_rate_hz,
        "sample_width_bytes": audio.sample_width_bytes,
        "frames": audio.frames,
        "duration_sec": round(audio.duration_sec, 6),
        "estimated_windows": audio.estimated_windows,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
