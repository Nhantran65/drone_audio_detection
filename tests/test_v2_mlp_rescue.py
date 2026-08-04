from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from drone_sound import v2_mlp_rescue
from drone_sound.v2_ml import V2MLError


def test_rescue_selection_uses_validation_not_professor_or_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        {"variant_name": "better_validation"},
        {"variant_name": "better_external"},
    )

    def fake_train(
        _features: object,
        model_path: Path,
        metrics_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        name = str(kwargs["variant_name"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"model")
        metrics_path.write_text("{}\n", encoding="utf-8")
        better_validation = name == "better_validation"
        return {
            "validation": {
                "primary": {
                    "f2": 0.95 if better_validation else 0.94,
                    "false_positive_rate": 0.02 if better_validation else 0.01,
                    "recall": 0.96,
                }
            },
            "public_test": {
                "metrics": {
                    "f2": 0.10 if better_validation else 0.99,
                    "false_positive_rate": 0.50 if better_validation else 0.0,
                }
            },
            "professor_test": {
                "metrics": {"recall": 0.10 if better_validation else 1.0}
            },
            "efficiency": {
                "model_size_bytes": 5,
                "model_size": "5 B",
                "parameter_count": 1,
            },
        }

    monkeypatch.setattr(v2_mlp_rescue, "train_v2_mlp", fake_train)
    output = tmp_path / "rescue"
    report = v2_mlp_rescue.run_v2_mlp_rescue(
        tmp_path / "features.npz",
        output,
        candidates=candidates,
        epochs=1,
    )

    assert report["winner"]["variant"] == "better_validation"
    assert report["selection_uses_public_test"] is False
    assert report["selection_uses_professor"] is False
    assert report["selection_uses_private_real"] is False
    assert (output / "comparison.csv").exists()
    assert json.loads((output / "best_model.json").read_text())["variant"] == "better_validation"


def test_rescue_refuses_overwrite_and_duplicate_names(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(V2MLError, match="refusing to overwrite"):
        v2_mlp_rescue.run_v2_mlp_rescue(tmp_path / "features.npz", existing)

    with pytest.raises(V2MLError, match="unique"):
        v2_mlp_rescue.run_v2_mlp_rescue(
            tmp_path / "features.npz",
            tmp_path / "new",
            candidates=(
                {"variant_name": "same"},
                {"variant_name": "same"},
            ),
        )
