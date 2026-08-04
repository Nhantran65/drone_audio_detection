"""Private real-only evaluation manifest construction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from drone_sound.constants import REQUIRED_MANIFEST_FIELDS, TARGET_SAMPLE_RATE_HZ
from drone_sound.manifest import expected_window_count, load_manifest_csv, validate_manifest_rows
from drone_sound.wav_audio import read_wav_metadata


class PrivateRealError(ValueError):
    """Raised when the private real-only manifest cannot be built safely."""


def build_private_real_manifest(
    professor_manifest_path: str | Path,
    private_audio_root: str | Path,
    negatives_dir: str | Path,
    manifest_out: str | Path,
    audit_out: str | Path,
) -> dict[str, Any]:
    """Combine professor positives and asserted private negatives for testing only."""

    professor_manifest = Path(professor_manifest_path)
    audio_root = Path(private_audio_root)
    negative_root = Path(negatives_dir)
    output = Path(manifest_out)
    audit_path = Path(audit_out)
    if output.exists() or audit_path.exists():
        raise PrivateRealError("private real-only outputs already exist; refusing overwrite")
    if not professor_manifest.is_file():
        raise PrivateRealError(f"professor manifest does not exist: {professor_manifest}")
    if not audio_root.is_dir() or not negative_root.is_dir():
        raise PrivateRealError("private audio root and negative directory must exist")

    professor_rows = load_manifest_csv(professor_manifest)
    validate_manifest_rows(professor_rows)
    positives: list[dict[str, str]] = []
    for source in professor_rows:
        if source["split"] != "professor_test" or source["label_binary"] != "drone":
            continue
        row = dict(source)
        row["original_file"] = str(Path("professor_real_world") / source["original_file"])
        row["sample_rate_processed"] = str(TARGET_SAMPLE_RATE_HZ)
        row["is_professor_data"] = "true"
        row["notes"] = "real-only external stress test; test-only positive"
        positives.append(row)

    negatives: list[dict[str, str]] = []
    files = sorted(negative_root.glob("*.wav"))
    if len(files) != 10:
        raise PrivateRealError(f"expected 10 private negative WAV files, found {len(files)}")
    for wav_path in files:
        metadata = read_wav_metadata(wav_path)
        checksum = _sha256_file(wav_path)
        environment = "traffic" if "traffic" in wav_path.stem.lower() else "forest"
        subtype = "no_drone_traffic" if environment == "traffic" else "no_drone_background"
        clip_id = f"private_negative_{wav_path.stem}_{checksum[:8]}"
        count = expected_window_count(metadata.duration_sec)
        relative = str(Path("Recordings") / wav_path.name)
        for index in range(count):
            start = index * 0.5
            end = start + 1.0
            negatives.append(
                {
                    "clip_id": clip_id,
                    "window_id": f"{clip_id}_w{index:06d}",
                    "source_dataset": "professor_private_negative",
                    "source_url_or_doi": "professor supplied private recordings",
                    "license": "private_research_only",
                    "original_file": relative,
                    "original_file_checksum": checksum,
                    "sample_rate_original": str(metadata.sample_rate_hz),
                    "sample_rate_processed": str(TARGET_SAMPLE_RATE_HZ),
                    "duration_sec": f"{metadata.duration_sec:.6f}",
                    "window_start_sec": f"{start:.6f}",
                    "window_end_sec": f"{end:.6f}",
                    "label_binary": "no_drone",
                    "label_subtype": subtype,
                    "drone_model": "",
                    "environment": environment,
                    "recording_condition": "real_world_noisy",
                    "split": "professor_test",
                    "is_professor_data": "true",
                    "notes": "asserted negative; external test only; never fit or tune",
                }
            )

    rows = positives + negatives
    validate_manifest_rows(rows)
    if len(positives) != 1_113 or len(negatives) != 592:
        raise PrivateRealError(
            "real-only manifest count mismatch: "
            f"positive={len(positives)}, negative={len(negatives)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "private_real_manifest_audit_v1",
        "evaluation_only": True,
        "fit_or_tune_allowed": False,
        "manifest": str(output),
        "audio_root": str(audio_root),
        "preprocessing_profile": "mono_resample_poly_16khz_v1",
        "processed_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "counts": {
            "rows": len(rows),
            "positive_windows": len(positives),
            "negative_windows": len(negatives),
            "positive_recordings": len({row["original_file"] for row in positives}),
            "negative_recordings": len(files),
        },
        "caveats": [
            "Only one independent positive and ten independent negative recordings.",
            "Eight questionable negative timestamps remain in the review queue and are not used for tuning.",
            "This is an external stress test, not a deployment-performance estimate.",
        ],
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
