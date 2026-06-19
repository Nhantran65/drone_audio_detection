"""Small WAV reader for PCM and IEEE float research audio."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 65534


class WavReadError(ValueError):
    """Raised when a WAV file cannot be read by the v1 pipeline."""


@dataclass(frozen=True)
class WavMetadata:
    path: Path
    format_tag: int
    decoded_format_tag: int
    channels: int
    sample_rate_hz: int
    sample_width_bytes: int
    bits_per_sample: int
    block_align: int
    frames: int
    duration_sec: float
    data_offset: int
    data_size_bytes: int


def read_wav_metadata(path: str | Path) -> WavMetadata:
    """Read WAV metadata, including IEEE float and extensible PCM/float."""

    path = Path(path)
    with path.open("rb") as file:
        if file.read(4) != b"RIFF":
            raise WavReadError(f"{path}: not a RIFF file")
        file.seek(8)
        if file.read(4) != b"WAVE":
            raise WavReadError(f"{path}: not a WAVE file")

        fmt_payload: bytes | None = None
        data_offset: int | None = None
        data_size: int | None = None

        while True:
            header = file.read(8)
            if not header:
                break
            if len(header) != 8:
                raise WavReadError(f"{path}: truncated chunk header")
            chunk_id, chunk_size = struct.unpack("<4sI", header)
            chunk_data_offset = file.tell()
            if chunk_id == b"fmt ":
                fmt_payload = file.read(chunk_size)
            elif chunk_id == b"data":
                data_offset = chunk_data_offset
                data_size = chunk_size
                file.seek(chunk_size, 1)
            else:
                file.seek(chunk_size, 1)
            if chunk_size % 2:
                file.seek(1, 1)

    if fmt_payload is None:
        raise WavReadError(f"{path}: missing fmt chunk")
    if data_offset is None or data_size is None:
        raise WavReadError(f"{path}: missing data chunk")
    if len(fmt_payload) < 16:
        raise WavReadError(f"{path}: invalid fmt chunk")

    format_tag, channels, sample_rate_hz, _, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH",
        fmt_payload[:16],
    )
    decoded_format_tag = _decoded_format_tag(format_tag, fmt_payload, path)
    if decoded_format_tag not in {WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT}:
        raise WavReadError(f"{path}: unsupported WAV format tag {format_tag}")
    if channels < 1:
        raise WavReadError(f"{path}: WAV must have at least one channel")
    if sample_rate_hz <= 0:
        raise WavReadError(f"{path}: invalid sample rate")
    if bits_per_sample % 8 != 0:
        raise WavReadError(f"{path}: unsupported bit depth {bits_per_sample}")
    sample_width_bytes = bits_per_sample // 8
    if block_align <= 0:
        raise WavReadError(f"{path}: invalid block alignment")
    frames = data_size // block_align
    return WavMetadata(
        path=path,
        format_tag=format_tag,
        decoded_format_tag=decoded_format_tag,
        channels=channels,
        sample_rate_hz=sample_rate_hz,
        sample_width_bytes=sample_width_bytes,
        bits_per_sample=bits_per_sample,
        block_align=block_align,
        frames=frames,
        duration_sec=frames / sample_rate_hz,
        data_offset=data_offset,
        data_size_bytes=data_size,
    )


def read_wav_window_mono(path: str | Path, start_sec: float, end_sec: float) -> list[float]:
    """Read one WAV window and return mono float samples in approximately [-1, 1]."""

    if end_sec <= start_sec:
        raise WavReadError(f"{path}: invalid window bounds {start_sec}..{end_sec}")
    metadata = read_wav_metadata(path)
    start_frame = int(round(start_sec * metadata.sample_rate_hz))
    frame_count = int(round((end_sec - start_sec) * metadata.sample_rate_hz))
    if start_frame < 0:
        raise WavReadError(f"{path}: negative window start")
    if start_frame + frame_count > metadata.frames:
        raise WavReadError(f"{path}: window exceeds audio duration")
    byte_count = frame_count * metadata.block_align
    byte_offset = metadata.data_offset + (start_frame * metadata.block_align)
    with metadata.path.open("rb") as file:
        file.seek(byte_offset)
        data = file.read(byte_count)
    if len(data) != byte_count:
        raise WavReadError(f"{path}: truncated audio window")
    return _decode_frames(data, metadata)


def read_wav_mono(path: str | Path) -> tuple[list[float], WavMetadata]:
    """Read a whole WAV file and return mono float samples plus metadata."""

    metadata = read_wav_metadata(path)
    with metadata.path.open("rb") as file:
        file.seek(metadata.data_offset)
        data = file.read(metadata.data_size_bytes)
    if len(data) != metadata.data_size_bytes:
        raise WavReadError(f"{path}: truncated audio data")
    return _decode_frames(data, metadata), metadata


def read_wav_mono_array(path: str | Path) -> tuple[object, WavMetadata]:
    """Read a whole WAV file as a mono numpy array when numpy is available."""

    import numpy as np

    metadata = read_wav_metadata(path)
    with metadata.path.open("rb") as file:
        file.seek(metadata.data_offset)
        data = file.read(metadata.data_size_bytes)
    if len(data) != metadata.data_size_bytes:
        raise WavReadError(f"{path}: truncated audio data")

    if metadata.decoded_format_tag == WAVE_FORMAT_IEEE_FLOAT:
        if metadata.sample_width_bytes == 4:
            array = np.frombuffer(data, dtype="<f4")
        elif metadata.sample_width_bytes == 8:
            array = np.frombuffer(data, dtype="<f8")
        else:
            raise WavReadError(
                f"{metadata.path}: unsupported IEEE float width {metadata.sample_width_bytes}"
            )
        return _mono_array(array, metadata.channels), metadata

    if metadata.sample_width_bytes == 1:
        array = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif metadata.sample_width_bytes == 2:
        array = np.frombuffer(data, dtype="<i2").astype(np.float64) / float(2**15)
    elif metadata.sample_width_bytes == 3:
        return np.asarray(_decode_frames(data, metadata), dtype=np.float64), metadata
    elif metadata.sample_width_bytes == 4:
        array = np.frombuffer(data, dtype="<i4").astype(np.float64) / float(2**31)
    else:
        raise WavReadError(f"{metadata.path}: unsupported PCM width {metadata.sample_width_bytes}")
    return _mono_array(array, metadata.channels), metadata


def _mono_array(array: object, channels: int) -> object:
    import numpy as np

    if channels == 1:
        return array.astype(np.float64, copy=False)
    return array.reshape(-1, channels).mean(axis=1)


def _decoded_format_tag(format_tag: int, fmt_payload: bytes, path: Path) -> int:
    if format_tag != WAVE_FORMAT_EXTENSIBLE:
        return format_tag
    if len(fmt_payload) < 40:
        raise WavReadError(f"{path}: invalid extensible fmt chunk")
    subformat = fmt_payload[24:40]
    if len(subformat) != 16:
        raise WavReadError(f"{path}: invalid extensible subformat")
    return int.from_bytes(subformat[:2], byteorder="little")


def _decode_frames(data: bytes, metadata: WavMetadata) -> list[float]:
    samples: list[float] = []
    for frame_offset in range(0, len(data), metadata.block_align):
        channel_values = []
        for channel_index in range(metadata.channels):
            sample_offset = frame_offset + (channel_index * metadata.sample_width_bytes)
            raw = data[sample_offset : sample_offset + metadata.sample_width_bytes]
            if len(raw) != metadata.sample_width_bytes:
                continue
            channel_values.append(_decode_sample(raw, metadata))
        if channel_values:
            samples.append(sum(channel_values) / len(channel_values))
    return samples


def _decode_sample(raw: bytes, metadata: WavMetadata) -> float:
    if metadata.decoded_format_tag == WAVE_FORMAT_IEEE_FLOAT:
        if metadata.sample_width_bytes == 4:
            return float(struct.unpack("<f", raw)[0])
        if metadata.sample_width_bytes == 8:
            return float(struct.unpack("<d", raw)[0])
        raise WavReadError(
            f"{metadata.path}: unsupported IEEE float width {metadata.sample_width_bytes}"
        )

    if metadata.sample_width_bytes == 1:
        return (raw[0] - 128) / 128.0
    if metadata.sample_width_bytes == 2:
        return int.from_bytes(raw, byteorder="little", signed=True) / float(2**15)
    if metadata.sample_width_bytes == 3:
        sign_byte = b"\xff" if raw[2] & 0x80 else b"\x00"
        value = int.from_bytes(raw + sign_byte, byteorder="little", signed=True)
        return value / float(2**23)
    if metadata.sample_width_bytes == 4:
        return int.from_bytes(raw, byteorder="little", signed=True) / float(2**31)
    raise WavReadError(f"{metadata.path}: unsupported PCM width {metadata.sample_width_bytes}")
