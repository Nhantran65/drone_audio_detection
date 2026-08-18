"""Group-aware public manifest construction for the V3 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from drone_sound.constants import REQUIRED_MANIFEST_FIELDS
from drone_sound.manifest import load_manifest_csv, validate_manifest_rows

V3_SPLIT_VERSION = "group_aware_70_15_15_v1"
V3_GROUP_PARSER_VERSION = "dataset_session_group_v1"
V3_SPLIT_SEED = 20260804
V3_EXTRA_MANIFEST_FIELDS = (
    "group_id",
    "group_parser_version",
    "split_version",
    "split_seed",
)
V3_MANIFEST_FIELDS = REQUIRED_MANIFEST_FIELDS + V3_EXTRA_MANIFEST_FIELDS
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "public_test": 0.15}
SPLITS = tuple(SPLIT_FRACTIONS)


class V3SplitError(ValueError):
    """Raised when a leakage-safe V3 split cannot be constructed."""


@dataclass(frozen=True)
class _Recording:
    original_file: str
    source_dataset: str
    label: str
    label_subtype: str
    group_id: str
    windows: int


@dataclass(frozen=True)
class _Group:
    group_id: str
    source_dataset: str
    label: str
    recordings: tuple[_Recording, ...]
    windows: int
    subtype_windows: Mapping[str, int]


def build_v3_grouped_manifest(
    source_manifest_path: str | Path,
    manifest_out: str | Path,
    group_map_out: str | Path,
    audit_out: str | Path,
    *,
    split_seed: int = V3_SPLIT_SEED,
) -> dict[str, Any]:
    """Reassign a public manifest at recording-session group granularity."""

    source = Path(source_manifest_path)
    manifest_path = Path(manifest_out)
    group_map_path = Path(group_map_out)
    audit_path = Path(audit_out)
    _require_outputs_absent(manifest_path, group_map_path, audit_path)
    if not source.is_file():
        raise V3SplitError(f"source manifest does not exist: {source}")

    rows = load_manifest_csv(source)
    validate_manifest_rows(rows)
    public_rows = [
        dict(row)
        for row in rows
        if row.get("split") in SPLITS
        and str(row.get("is_professor_data", "")).strip().lower() != "true"
    ]
    if len(public_rows) != len(rows):
        raise V3SplitError("V3 grouped manifest accepts public train/val/test rows only")

    recordings = _recordings(public_rows)
    groups = _groups(recordings)
    assignments: dict[str, str] = {}
    for dataset, dataset_groups in sorted(_by_dataset(groups).items()):
        if len(dataset_groups) < len(SPLITS):
            raise V3SplitError(
                f"dataset {dataset} has {len(dataset_groups)} groups; at least 3 are required"
            )
        assignments.update(_assign_dataset_groups(dataset_groups, split_seed))

    rewritten = []
    for row in public_rows:
        group_id = derive_group_id(row["source_dataset"], row["original_file"])
        updated = dict(row)
        updated["split"] = assignments[group_id]
        updated["group_id"] = group_id
        updated["group_parser_version"] = V3_GROUP_PARSER_VERSION
        updated["split_version"] = V3_SPLIT_VERSION
        updated["split_seed"] = str(split_seed)
        note = str(updated.get("notes", "")).strip()
        updated["notes"] = (
            f"{note}; V3 group-aware split; group_id={group_id}"
            if note
            else f"V3 group-aware split; group_id={group_id}"
        )
        rewritten.append(updated)

    validate_manifest_rows(rewritten)
    _validate_group_isolation(rewritten)
    _validate_split_labels(rewritten)
    _validate_identity(rows, rewritten)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    group_map_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=V3_MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rewritten)
    _write_group_map(group_map_path, recordings, assignments)

    audit = {
        "schema_version": "v3_grouped_manifest_audit_v1",
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256_file(source),
        "manifest": str(manifest_path),
        "group_map": str(group_map_path),
        "split_seed": split_seed,
        "split_version": V3_SPLIT_VERSION,
        "group_parser_version": V3_GROUP_PARSER_VERSION,
        "split_fractions": SPLIT_FRACTIONS,
        "policy": {
            "split_unit": "recording_session_group",
            "professor_data_included": False,
            "private_data_included": False,
            "window_ids_preserved": True,
            "group_overlap_allowed": False,
        },
        "parser_rules": _parser_rules(),
        "counts": _audit_counts(rewritten),
        "group_overlap": _group_overlap(rewritten),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def derive_group_id(source_dataset: str, original_file: str) -> str:
    """Derive a conservative dataset-specific recording-session identifier."""

    dataset = str(source_dataset).strip()
    path = Path(str(original_file).strip())
    stem = path.stem.lower()

    if dataset == "kielce_uav_acoustic_signals":
        match = re.search(r"_(\d{6})_+", stem)
        if not match or len(path.parts) < 2:
            raise V3SplitError(f"cannot derive Kielce campaign group: {original_file}")
        return _group_token(dataset, path.parts[1], match.group(1))

    if dataset == "dronenoise_database":
        event = re.sub(r"_(?:m\d+|ch\d+)$", "", stem)
        if not event:
            raise V3SplitError(f"cannot derive DroneNoise event group: {original_file}")
        return _group_token(dataset, event)

    if dataset == "nasa_small_uas_flyover":
        return _group_token(dataset, stem)

    if dataset == "aerosonicdb_ypad_0523":
        match = re.search(r"_(\d{4})[-_](\d{2})[-_](\d{2})_(\d{2})-", stem)
        if match:
            parent = path.parent.name.lower()
            date_hour = "-".join(match.groups())
            return _group_token(dataset, parent, date_hour)
        if "env_audio" in {part.lower() for part in path.parts}:
            return _group_token(dataset, "environment", stem)
        raise V3SplitError(f"cannot derive AeroSonic session group: {original_file}")

    if dataset == "idmt_traffic":
        match = re.match(
            r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}_[^_]+_[^_]+)",
            stem,
        )
        if not match:
            raise V3SplitError(f"cannot derive IDMT session group: {original_file}")
        return _group_token(dataset, match.group(1))

    if dataset == "urbansound8k":
        fold = next(
            (part.lower() for part in path.parts if re.fullmatch(r"fold\d+", part.lower())),
            None,
        )
        if fold is None:
            raise V3SplitError(f"cannot derive UrbanSound official fold: {original_file}")
        return _group_token(dataset, fold)

    raise V3SplitError(f"unsupported V3 dataset group parser: {dataset}")


def _recordings(rows: Iterable[Mapping[str, str]]) -> list[_Recording]:
    by_file: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_file[str(row["original_file"])].append(row)
    output = []
    for original_file, file_rows in sorted(by_file.items()):
        datasets = {str(row["source_dataset"]) for row in file_rows}
        labels = {str(row["label_binary"]) for row in file_rows}
        subtypes = {str(row["label_subtype"]) for row in file_rows}
        if len(datasets) != 1 or len(labels) != 1 or len(subtypes) != 1:
            raise V3SplitError(f"recording metadata is inconsistent: {original_file}")
        dataset = next(iter(datasets))
        output.append(
            _Recording(
                original_file=original_file,
                source_dataset=dataset,
                label=next(iter(labels)),
                label_subtype=next(iter(subtypes)),
                group_id=derive_group_id(dataset, original_file),
                windows=len(file_rows),
            )
        )
    return output


def _groups(recordings: Iterable[_Recording]) -> list[_Group]:
    by_group: dict[str, list[_Recording]] = defaultdict(list)
    for recording in recordings:
        by_group[recording.group_id].append(recording)
    output = []
    for group_id, values in sorted(by_group.items()):
        datasets = {value.source_dataset for value in values}
        labels = {value.label for value in values}
        if len(datasets) != 1 or len(labels) != 1:
            raise V3SplitError(f"group spans datasets or labels: {group_id}")
        subtype_windows: Counter[str] = Counter()
        for value in values:
            subtype_windows[value.label_subtype] += value.windows
        output.append(
            _Group(
                group_id=group_id,
                source_dataset=next(iter(datasets)),
                label=next(iter(labels)),
                recordings=tuple(values),
                windows=sum(value.windows for value in values),
                subtype_windows=dict(subtype_windows),
            )
        )
    return output


def _by_dataset(groups: Iterable[_Group]) -> dict[str, list[_Group]]:
    output: dict[str, list[_Group]] = defaultdict(list)
    for group in groups:
        output[group.source_dataset].append(group)
    return output


def _assign_dataset_groups(groups: list[_Group], seed: int) -> dict[str, str]:
    totals = {
        "windows": sum(group.windows for group in groups),
        "groups": len(groups),
        "subtypes": Counter(),
    }
    for group in groups:
        totals["subtypes"].update(group.subtype_windows)
    ordered = sorted(
        groups,
        key=lambda group: (
            -group.windows,
            _stable_hash(seed, group.source_dataset, group.group_id),
        ),
    )
    assigned: dict[str, str] = {}
    state = {
        split: {"windows": 0, "groups": 0, "subtypes": Counter()}
        for split in SPLITS
    }
    for index, group in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [split for split in SPLITS if state[split]["groups"] == 0]
        candidates = empty if remaining == len(empty) and empty else list(SPLITS)
        split = min(
            candidates,
            key=lambda candidate: (
                _assignment_cost(state, totals, group, candidate),
                _stable_hash(seed, group.group_id, candidate),
            ),
        )
        assigned[group.group_id] = split
        state[split]["windows"] += group.windows
        state[split]["groups"] += 1
        state[split]["subtypes"].update(group.subtype_windows)
    if set(assigned.values()) != set(SPLITS):
        raise V3SplitError(f"group allocator produced an empty split for {groups[0].source_dataset}")
    return assigned


def _assignment_cost(
    state: Mapping[str, Mapping[str, Any]],
    totals: Mapping[str, Any],
    group: _Group,
    candidate: str,
) -> float:
    cost = 0.0
    subtype_names = sorted(totals["subtypes"])
    for split, fraction in SPLIT_FRACTIONS.items():
        projected_windows = state[split]["windows"] + (group.windows if split == candidate else 0)
        projected_groups = state[split]["groups"] + (1 if split == candidate else 0)
        target_windows = max(float(totals["windows"]) * fraction, 1.0)
        target_groups = max(float(totals["groups"]) * fraction, 1.0)
        cost += ((projected_windows - target_windows) / target_windows) ** 2
        cost += 0.20 * ((projected_groups - target_groups) / target_groups) ** 2
        for subtype in subtype_names:
            projected = state[split]["subtypes"][subtype]
            if split == candidate:
                projected += group.subtype_windows.get(subtype, 0)
            target = max(float(totals["subtypes"][subtype]) * fraction, 1.0)
            cost += 0.35 * ((projected - target) / target) ** 2
    return cost


def _write_group_map(
    path: Path,
    recordings: Iterable[_Recording],
    assignments: Mapping[str, str],
) -> None:
    fields = (
        "source_dataset",
        "label_binary",
        "label_subtype",
        "original_file",
        "group_id",
        "split",
        "windows",
        "group_parser_version",
        "split_version",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for recording in recordings:
            writer.writerow(
                {
                    "source_dataset": recording.source_dataset,
                    "label_binary": recording.label,
                    "label_subtype": recording.label_subtype,
                    "original_file": recording.original_file,
                    "group_id": recording.group_id,
                    "split": assignments[recording.group_id],
                    "windows": recording.windows,
                    "group_parser_version": V3_GROUP_PARSER_VERSION,
                    "split_version": V3_SPLIT_VERSION,
                }
            )


def _audit_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    output: dict[str, Any] = {"rows": 0, "datasets": {}, "splits": {}}
    datasets: dict[str, Any] = defaultdict(
        lambda: {"rows": 0, "recordings": set(), "groups": set(), "splits": defaultdict(int)}
    )
    splits: dict[str, Any] = defaultdict(
        lambda: {"rows": 0, "recordings": set(), "groups": set(), "labels": Counter(), "datasets": Counter()}
    )
    for row in rows:
        dataset = str(row["source_dataset"])
        split = str(row["split"])
        label = str(row["label_binary"])
        group_id = str(row["group_id"])
        original_file = str(row["original_file"])
        output["rows"] += 1
        datasets[dataset]["rows"] += 1
        datasets[dataset]["recordings"].add(original_file)
        datasets[dataset]["groups"].add(group_id)
        datasets[dataset]["splits"][split] += 1
        splits[split]["rows"] += 1
        splits[split]["recordings"].add(original_file)
        splits[split]["groups"].add(group_id)
        splits[split]["labels"][label] += 1
        splits[split]["datasets"][dataset] += 1
    output["datasets"] = {
        dataset: {
            "rows": value["rows"],
            "recordings": len(value["recordings"]),
            "groups": len(value["groups"]),
            "split_rows": dict(sorted(value["splits"].items())),
        }
        for dataset, value in sorted(datasets.items())
    }
    output["splits"] = {
        split: {
            "rows": value["rows"],
            "recordings": len(value["recordings"]),
            "groups": len(value["groups"]),
            "labels": dict(sorted(value["labels"].items())),
            "datasets": dict(sorted(value["datasets"].items())),
        }
        for split, value in sorted(splits.items())
    }
    return output


def _group_overlap(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["split"])].add(str(row["group_id"]))
    pairs = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            pairs[f"{left}__{right}"] = len(groups[left] & groups[right])
    return {"pairs": pairs, "valid": all(value == 0 for value in pairs.values())}


def _validate_group_isolation(rows: Iterable[Mapping[str, str]]) -> None:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_group[str(row["group_id"])].add(str(row["split"]))
    leaking = {group: splits for group, splits in splits_by_group.items() if len(splits) > 1}
    if leaking:
        first = next(iter(sorted(leaking)))
        raise V3SplitError(f"group appears in multiple splits: {first}")


def _validate_split_labels(rows: Iterable[Mapping[str, str]]) -> None:
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels[str(row["split"])].add(str(row["label_binary"]))
    for split in SPLITS:
        if labels[split] != {"drone", "no_drone"}:
            raise V3SplitError(f"split {split} does not contain both binary labels")


def _validate_identity(source: Iterable[Mapping[str, str]], rewritten: Iterable[Mapping[str, str]]) -> None:
    source_rows = list(source)
    new_rows = list(rewritten)
    if len(source_rows) != len(new_rows):
        raise V3SplitError("V3 split changed the manifest row count")
    source_identity = [(row["window_id"], row["label_binary"]) for row in source_rows]
    new_identity = [(row["window_id"], row["label_binary"]) for row in new_rows]
    if source_identity != new_identity:
        raise V3SplitError("V3 split changed window order, IDs, or labels")


def _parser_rules() -> dict[str, str]:
    return {
        "kielce_uav_acoustic_signals": "drone/archive folder + recording campaign date",
        "dronenoise_database": "event name with microphone/channel suffix removed",
        "nasa_small_uas_flyover": "one converted flyover/hover run",
        "aerosonicdb_ypad_0523": "class directory + recording date/hour; environment files isolated",
        "idmt_traffic": "recording timestamp + location + speed session",
        "urbansound8k": "official UrbanSound8K fold",
    }


def _group_token(*parts: str) -> str:
    value = ":".join(str(part).strip().lower() for part in parts)
    return re.sub(r"[^a-z0-9:_-]+", "_", value).strip("_")


def _stable_hash(seed: int, *parts: str) -> str:
    digest = hashlib.sha256(str(seed).encode())
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise V3SplitError(f"refusing to overwrite V3 output(s): {', '.join(existing)}")
