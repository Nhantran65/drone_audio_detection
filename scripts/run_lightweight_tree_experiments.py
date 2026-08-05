#!/usr/bin/env python3
"""Run the bounded canonical-16-kHz lightweight tree experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from drone_sound.v2_lightweight_trees import (
    build_experiment_summary,
    evaluate_group_winners,
    run_lightweight_tree_experiments,
)
from drone_sound.v2_ml import V2MLError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/v2_16khz/features/public_professor_features.npz"),
    )
    parser.add_argument(
        "--private-features",
        type=Path,
        default=Path("artifacts/v2_16khz/features/private_real_features.npz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/v2_16khz/models/lightweight_tree_experiments"),
    )
    parser.add_argument("--private-audio-root", type=Path, default=Path("data/raw/private"))
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("artifacts/v2_16khz/models/v2a/metrics/random_forest.json"),
    )
    parser.add_argument(
        "--baseline-real-metrics",
        type=Path,
        default=Path("artifacts/v2_16khz/evaluation/real_only/random_forest.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()

    def progress(event: dict[str, object]) -> None:
        elapsed = time.perf_counter() - started
        print(
            f"[lightweight-trees elapsed={elapsed:.1f}s] "
            + json.dumps(event, sort_keys=True),
            flush=True,
        )

    try:
        run_lightweight_tree_experiments(args.features, args.out_dir, progress)
        evaluate_group_winners(
            args.out_dir,
            args.private_features,
            args.private_audio_root,
            args.out_dir / "cache",
            progress,
        )
        summary = build_experiment_summary(
            args.out_dir,
            args.baseline_metrics,
            args.baseline_real_metrics,
        )
    except (V2MLError, OSError, ValueError) as error:
        print(f"lightweight tree experiment failed: {error}", flush=True)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
