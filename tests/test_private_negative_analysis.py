from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_private_negative_audio import (  # noqa: E402
    _bootstrap_mean_ci,
    _longest_true_run,
    _recording_level_summary,
    _resample,
    _window_count,
)
from evaluate_real_only_test import _evaluate_private_mode  # noqa: E402


def test_window_count_matches_one_second_half_stride_contract() -> None:
    assert _window_count(48_000, 48_000) == 1
    assert _window_count(72_000, 48_000) == 2
    assert _window_count(47_999, 48_000) == 0


def test_resample_48k_to_16k_preserves_duration_and_is_finite() -> None:
    time = np.arange(48_000, dtype=np.float64) / 48_000.0
    samples = 0.25 * np.sin(2.0 * np.pi * 1_000.0 * time)

    output = _resample(samples, 48_000)

    assert output.shape == (16_000,)
    assert np.isfinite(output).all()
    assert np.sqrt(np.mean(output**2)) == pytest.approx(
        np.sqrt(np.mean(samples**2)),
        rel=0.01,
    )


def test_recording_bootstrap_uses_file_rates_instead_of_window_micro_average() -> None:
    rows = [
        {
            "file": "long.wav",
            "environment": "traffic",
            "mode": "resampled_16k",
            "model_detected": 0,
        }
        for _ in range(9)
    ]
    rows.append(
        {
            "file": "short.wav",
            "environment": "forest",
            "mode": "resampled_16k",
            "model_detected": 1,
        }
    )

    summary = _recording_level_summary(rows, {"model": {"threshold": 0.5}})

    assert summary["model"]["macro_false_positive_rate"] == 0.5
    assert summary["model"]["recordings"] == 2


def test_bootstrap_and_longest_run_are_deterministic() -> None:
    first = _bootstrap_mean_ci(np.asarray([0.0, 0.5, 1.0]))
    second = _bootstrap_mean_ci(np.asarray([0.0, 0.5, 1.0]))

    assert first == second
    assert _longest_true_run(np.asarray([False, True, True, False, True])) == 2


def test_real_only_metrics_keep_positive_and_negative_denominators_separate() -> None:
    rows = [
        {"mode": "positive_reference_16k", "model_score": "0.9"},
        {"mode": "positive_reference_16k", "model_score": "0.2"},
        {"mode": "resampled_16k", "model_score": "0.8"},
        {"mode": "resampled_16k", "model_score": "0.1"},
    ]

    report = _evaluate_private_mode(
        rows,
        "resampled_16k",
        {"model": {"threshold": 0.5}},
    )

    assert report["counts"] == {"drone": 2, "no_drone": 2}
    assert report["models"]["model"]["confusion_matrix"] == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
    }
