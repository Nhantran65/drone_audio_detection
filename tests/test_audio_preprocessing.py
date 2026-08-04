from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy.signal")

from drone_sound.audio_preprocessing import canonicalize_audio


def test_identity_at_16khz_is_sample_exact() -> None:
    samples = np.linspace(-1.0, 1.0, 16_000, dtype=np.float64)
    processed = canonicalize_audio(samples, 16_000)
    assert np.array_equal(processed, samples)
    assert processed is not samples


@pytest.mark.parametrize("source_rate", [48_000, 44_100, 22_050])
def test_common_rates_become_exactly_16000_samples_per_second(source_rate: int) -> None:
    samples = np.zeros(source_rate, dtype=np.float64)
    processed = canonicalize_audio(samples, source_rate)
    assert processed.shape == (16_000,)


def test_polyphase_filter_suppresses_content_above_8khz() -> None:
    source_rate = 48_000
    times = np.arange(source_rate, dtype=np.float64) / source_rate
    below = np.sin(2.0 * math.pi * 2_000.0 * times)
    above = np.sin(2.0 * math.pi * 12_000.0 * times)
    below_processed = canonicalize_audio(below, source_rate)
    above_processed = canonicalize_audio(above, source_rate)
    below_rms = float(np.sqrt(np.mean(below_processed**2)))
    above_rms = float(np.sqrt(np.mean(above_processed**2)))
    assert above_rms < below_rms * 0.02


def test_resampling_is_deterministic() -> None:
    rng = np.random.default_rng(20260804)
    samples = rng.normal(size=44_100)
    first = canonicalize_audio(samples, 44_100)
    second = canonicalize_audio(samples, 44_100)
    assert np.array_equal(first, second)
