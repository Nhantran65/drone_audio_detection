# Dataset Contract

## Dataset Roles

- Public positive datasets train and evaluate drone acoustics.
- Public negative and hard-negative datasets train and evaluate false alarm
  resistance.
- Professor data is reserved for `professor_test`.

The initial public dataset source registry is maintained in
`docs/product/dataset-sources.md` and `configs/datasets.json`.

## Professor Data Rule

`fwdronedata.zip` is private professor-provided real-world data. It is treated
as positive-only or near-positive-only evaluation data until stronger labels
say otherwise.

Rules:

- `split = professor_test`
- `label_binary = drone`
- `label_subtype = drone_unknown_condition` by default
- `is_professor_data = true`
- Do not use for training.
- Do not use for validation.
- Do not use for threshold tuning.
- Do not report accuracy, precision, specificity, or false positive rate from
  professor data alone.

## Manifest Fields

All manifest rows must carry these fields:

```csv
clip_id,
window_id,
source_dataset,
source_url_or_doi,
license,
original_file,
original_file_checksum,
sample_rate_original,
sample_rate_processed,
duration_sec,
window_start_sec,
window_end_sec,
label_binary,
label_subtype,
drone_model,
environment,
recording_condition,
split,
is_professor_data,
notes
```

## Allowed Values

Labels:

- `drone`
- `no_drone`
- `ignore`

Splits:

- `train`
- `val`
- `public_test`
- `professor_test`

License status:

- `commercial_ok`
- `research_only`
- `non_commercial`
- `license_unspecified`
- `permission_required`

Subtypes:

- `drone_field`
- `drone_with_speech`
- `drone_with_background`
- `drone_unknown_condition`
- `no_drone_aircraft`
- `no_drone_traffic`
- `no_drone_urban_industrial`
- `no_drone_speech`
- `no_drone_background`
- `no_drone_unknown`

## Leakage Rule

No `original_file` may appear in more than one split. Window-level random split
is not allowed.

## Location Privacy

Professor data audit reports must redact exact GPS coordinates by default. The
report may state that a location is present with `location_present = true`.

## License Boundary

Datasets with `non_commercial`, `license_unspecified`, or
`permission_required` status are not cleared for commercial product training.
They may be used only in the documented research lane until permission or a
compatible license is confirmed.
