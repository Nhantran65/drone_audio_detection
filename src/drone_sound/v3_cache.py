"""Safe zero-copy cache rebinding for the V3 split-only experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from drone_sound.audio_preprocessing import CANONICAL_PREPROCESSING_PROFILE
from drone_sound.v3_group_split import V3_SPLIT_VERSION


class V3CacheError(ValueError):
    """Raised when a canonical cache cannot be safely rebound to V3."""


def rebind_v3_cache(
    source_bundle: str | Path,
    v3_bundle: str | Path,
    source_cache_dir: str | Path,
    cache_out_dir: str | Path,
    *,
    evaluation_only: bool = False,
) -> dict[str, Any]:
    """Hard-link immutable cache data after proving row/preprocessing identity."""

    np = _require_numpy()
    old_bundle = Path(source_bundle)
    new_bundle = Path(v3_bundle)
    old_cache = Path(source_cache_dir)
    new_cache = Path(cache_out_dir)
    if new_cache.exists():
        raise V3CacheError(f"refusing to overwrite V3 cache directory: {new_cache}")
    if not old_bundle.is_file() or not new_bundle.is_file():
        raise V3CacheError("source and V3 feature bundles are required")
    meta_files = sorted(old_cache.glob("*_meta.json"))
    data_files = sorted(old_cache.glob("*.npy"))
    if len(meta_files) != 1 or len(data_files) != 1:
        raise V3CacheError("source cache must contain exactly one .npy and one metadata JSON")

    with np.load(old_bundle, allow_pickle=False) as old, np.load(
        new_bundle, allow_pickle=False
    ) as new:
        for field in (
            "window_ids",
            "labels",
            "original_files",
            "window_start_sec",
            "window_end_sec",
            "processed_sample_rates_hz",
        ):
            if field not in old.files or field not in new.files:
                raise V3CacheError(f"bundle identity field is missing: {field}")
            if not np.array_equal(old[field], new[field]):
                raise V3CacheError(f"cache row identity mismatch: {field}")
        old_profile = str(old["preprocessing_profile"].item())
        new_profile = str(new["preprocessing_profile"].item())
        new_split = (
            str(new["split_version"].item())
            if "split_version" in new.files
            else None
        )
        old_fingerprint = str(old["bundle_fingerprint"].item())
        new_fingerprint = str(new["bundle_fingerprint"].item())
    if old_profile != CANONICAL_PREPROCESSING_PROFILE or new_profile != old_profile:
        raise V3CacheError("cache rebinding requires identical canonical preprocessing")
    if not evaluation_only and new_split != V3_SPLIT_VERSION:
        raise V3CacheError(f"unexpected V3 split version: {new_split}")
    if evaluation_only and old_fingerprint != new_fingerprint:
        raise V3CacheError(
            "evaluation cache rebinding requires identical raw bundle fingerprints"
        )

    metadata = json.loads(meta_files[0].read_text(encoding="utf-8"))
    if metadata.get("bundle_fingerprint") != old_fingerprint:
        raise V3CacheError("source cache fingerprint does not match its source bundle")
    shape = tuple(int(value) for value in metadata.get("shape", ()))
    array = np.load(data_files[0], mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != shape or shape[0] != len(array):
        raise V3CacheError("source cache data shape does not match metadata")

    new_cache.mkdir(parents=True)
    output_data = new_cache / data_files[0].name
    output_meta = new_cache / meta_files[0].name
    os.link(data_files[0], output_data)
    metadata.update(
        {
            "bundle_fingerprint": new_fingerprint,
            "source_bundle": str(new_bundle),
            "split_version": new_split or "evaluation_only_raw_identity_v1",
            "cache_data_reused_by_verified_row_identity": True,
            "source_cache": str(old_cache),
            "source_cache_bundle_fingerprint": old_fingerprint,
            "immutable_hard_link": True,
        }
    )
    output_meta.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "schema_version": "v3_cache_rebind_v1",
        "cache_dir": str(new_cache),
        "data": str(output_data),
        "metadata": str(output_meta),
        "shape": list(shape),
        "row_identity_verified": True,
        "preprocessing_identity_verified": True,
        "evaluation_only": evaluation_only,
        "source_bundle_fingerprint": old_fingerprint,
        "bundle_fingerprint": new_fingerprint,
    }


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise V3CacheError("V3 cache rebinding requires numpy") from error
    return np
