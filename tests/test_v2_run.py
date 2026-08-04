from __future__ import annotations

from pathlib import Path

import pytest

from drone_sound.v2_run import V2RunError, initialize_v2_run


def test_run_initialization_hashes_legacy_and_refuses_overwrite(tmp_path: Path) -> None:
    legacy = tmp_path / "artifacts"
    legacy.mkdir()
    (legacy / "old.bin").write_bytes(b"legacy")
    run_root = legacy / "v2_16khz"

    report = initialize_v2_run(run_root, legacy)

    assert report["status"] == "initialized"
    assert report["legacy_artifacts"]["before"]["file_count"] == 1
    assert (run_root / "run_manifest.json").exists()
    for name in ("features", "cache", "models", "evaluation", "comparison"):
        assert (run_root / name).is_dir()

    with pytest.raises(V2RunError, match="already exists"):
        initialize_v2_run(run_root, legacy)
