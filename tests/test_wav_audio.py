from __future__ import annotations

import struct
from pathlib import Path

from drone_sound.wav_audio import read_wav_metadata, read_wav_window_mono


def test_read_ieee_float_wav_metadata_and_window(tmp_path: Path) -> None:
    wav_path = tmp_path / "float.wav"
    _write_float_wav(wav_path, samples=[0.25, -0.25, 0.5, -0.5], channels=2, sample_rate=2)

    metadata = read_wav_metadata(wav_path)
    samples = read_wav_window_mono(wav_path, 0.0, 1.0)

    assert metadata.decoded_format_tag == 3
    assert metadata.channels == 2
    assert metadata.sample_rate_hz == 2
    assert metadata.frames == 2
    assert samples == [0.0, 0.0]


def _write_float_wav(path: Path, samples: list[float], channels: int, sample_rate: int) -> None:
    data = b"".join(struct.pack("<f", sample) for sample in samples)
    fmt = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate,
        sample_rate * channels * 4,
        channels * 4,
        32,
    )
    riff_size = 4 + (8 + len(fmt)) + (8 + len(data))
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
