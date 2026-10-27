from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from drone_sound.v3_cache import V3CacheError, rebind_v3_cache
from drone_sound.v3_group_split import V3_SPLIT_VERSION


def _bundle(path, fingerprint: str, *, v3: bool, window_ids=("a", "b")):
    values = {
        "window_ids": np.asarray(window_ids),
        "labels": np.asarray(["drone", "no_drone"]),
        "original_files": np.asarray(["a.wav", "b.wav"]),
        "window_start_sec": np.asarray([0.0, 0.0]),
        "window_end_sec": np.asarray([1.0, 1.0]),
        "processed_sample_rates_hz": np.asarray([16000, 16000]),
        "preprocessing_profile": np.asarray("mono_resample_poly_16khz_v1"),
        "bundle_fingerprint": np.asarray(fingerprint),
    }
    if v3:
        values["split_version"] = np.asarray(V3_SPLIT_VERSION)
    np.savez(path, **values)


def test_rebind_v3_cache_requires_exact_row_identity(tmp_path):
    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    source_cache = tmp_path / "source-cache"
    output_cache = tmp_path / "v3-cache"
    source_cache.mkdir()
    _bundle(old, "old-fingerprint", v3=False)
    _bundle(new, "new-fingerprint", v3=True)
    data = np.zeros((2, 3), dtype=np.float32)
    np.save(source_cache / "example.npy", data)
    (source_cache / "example_meta.json").write_text(
        json.dumps(
            {
                "bundle_fingerprint": "old-fingerprint",
                "shape": [2, 3],
                "preprocessing_profile": "mono_resample_poly_16khz_v1",
            }
        ),
        encoding="utf-8",
    )

    report = rebind_v3_cache(old, new, source_cache, output_cache)

    assert report["row_identity_verified"] is True
    assert (output_cache / "example.npy").stat().st_ino == (
        source_cache / "example.npy"
    ).stat().st_ino
    metadata = json.loads((output_cache / "example_meta.json").read_text())
    assert metadata["bundle_fingerprint"] == "new-fingerprint"
    assert metadata["immutable_hard_link"] is True


def test_rebind_v3_cache_rejects_changed_window_order(tmp_path):
    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    _bundle(old, "old-fingerprint", v3=False)
    _bundle(new, "new-fingerprint", v3=True, window_ids=("b", "a"))
    np.save(source_cache / "example.npy", np.zeros((2, 3), dtype=np.float32))
    (source_cache / "example_meta.json").write_text(
        json.dumps({"bundle_fingerprint": "old-fingerprint", "shape": [2, 3]}),
        encoding="utf-8",
    )

    with pytest.raises(V3CacheError, match="window_ids"):
        rebind_v3_cache(old, new, source_cache, tmp_path / "out")
