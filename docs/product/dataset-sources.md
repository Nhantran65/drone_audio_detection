# Public Dataset Sources

This document records the initial public dataset registry for the drone audio
POC. The executable source of truth is `configs/datasets.json`.

## License Status Vocabulary

- `commercial_ok`: license appears compatible with commercial reuse when
  attribution and license terms are followed.
- `research_only`: use only for research until permission is clarified.
- `non_commercial`: license restricts use to non-commercial purposes.
- `license_unspecified`: no usable license was found on the source page.
- `permission_required`: use requires explicit permission.

This is project risk tracking, not legal advice.

## Positive Datasets

| Dataset | Role | License status | Label mapping | Notes |
| --- | --- | --- | --- | --- |
| DroneNoise Database | Positive drone | `commercial_ok` | `drone` / `drone_field` | Salford/Figshare source, CC BY 4.0. |
| Kielce UAV acoustic signals | Positive drone | `commercial_ok` | `drone` / `drone_field`, `drone_with_speech` | Zenodo source, CC BY 4.0; speech with UAV remains positive. |
| NASA Small UAS Flyover Acoustics Data | Optional positive | `license_unspecified` | `drone` / `drone_field` | Exclude from training until license is clarified. |

## Negative And Hard-Negative Datasets

| Dataset | Role | License status | Label mapping | Notes |
| --- | --- | --- | --- | --- |
| AeroSonicDB | Aircraft hard negative | `non_commercial` | `no_drone` / `no_drone_aircraft` | Treat conservatively as CC BY-NC 4.0 because the page has conflicting license signals. |
| IDMT-TRAFFIC | Traffic hard negative | `non_commercial` | `no_drone` / `no_drone_traffic`, `no_drone_background` | CC BY-NC-ND 4.0; review before redistributing transformed data. |
| UrbanSound8K | Urban/industrial negative | `non_commercial` | `no_drone` / `no_drone_urban_industrial` | Use selected mechanical/urban classes as hard negatives. |

## Open Items

- Select a speech-only negative dataset.
- Clarify NASA training license before inclusion in train or validation splits.
- Decide whether IDMT-TRAFFIC may be used for transformed training features or
  only for research/evaluation artifacts.
- Resolve AeroSonicDB license conflict before productization.
