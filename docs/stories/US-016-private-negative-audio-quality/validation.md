# US-016 Validation

## Proof Strategy

Prove that all raw files remain unchanged, every derived row is attributable to
one source recording, the analysis is deterministic, frozen thresholds are
reused, and native/resampled results are both reported.

## Test Plan

| Layer | Cases |
| --- | --- |
| Unit | Signal metrics, window counts, resampling, and score aggregation |
| Integration | Run the full analysis on all ten negatives and one positive |
| E2E | Execute the companion notebook top-to-bottom and build the HTML report |
| Platform | Not applicable; no deployed surface |
| Performance | Progress output present for window extraction and scoring |
| Logs/Audit | Raw SHA-256 hashes, derived paths, limitations, and no GPS output |

## Fixtures

- Ten private negative WAV files in `data/raw/private/Recordings/`.
- Existing private professor positive WAV.
- Frozen `tiny_mlp_es` and `random_forest` artifacts plus train scaler.

## Commands

```text
.venv/bin/python scripts/analyze_private_negative_audio.py
.venv/bin/python scripts/evaluate_real_only_test.py
uv run --extra data --extra ml --with nbclient --with nbformat --with ipykernel --python 3.13 python -c "execute notebooks/professor_negative_data_quality.ipynb with NotebookClient"
uv run --offline --extra data --extra ml --with pytest --python 3.13 pytest -q
node .../build_portable_artifact.mjs --input artifacts/analysis/private_negative_quality/artifact.json --output artifacts/analysis/private_negative_quality/report.html
python scripts/fix_portable_report_overflow.py artifacts/analysis/private_negative_quality/report.html
node .../verify_portable_artifact.mjs --html artifacts/analysis/private_negative_quality/report.html --artifact artifacts/analysis/private_negative_quality/artifact.json
```

## Acceptance Evidence

- Full non-destructive run completed in 16 seconds with visible phase, count,
  rate, elapsed-time, and ETA logs; 592 negative windows per processing mode
  and 1,113 professor-positive reference windows were extracted.
- All ten negatives are PCM16 mono 48 kHz; no file crossed the clipping, DC,
  zero-dropout, or very-low-median hard-failure thresholds. Exact duplicates
  within the private set and hash matches with the public manifest were zero.
- The sample-rate sensitivity gate failed: MLP window FPR changed from 7.43%
  native to 45.44% at 16 kHz, while Random Forest changed from 19.26% to 1.86%.
  Decision-flip rates were 44.09% and 19.09%, respectively.
- Recording-level 16 kHz macro FPR (20,000 recording bootstraps) was 50.92%
  [34.96%, 66.36%] for MLP and 2.58% [0.83%, 4.67%] for Random Forest.
- Eight consensus timestamps were emitted for mandatory manual listening.
- Notebook executed top-to-bottom: four code cells with execution counts and
  outputs, no execution error.
- Portable HTML verifier passed at 1,440 px and 390 px with 22 blocks, two
  charts, three metric cards, four tables, and a working source dialog.
- The real-only evaluation contains 1,113 positive and 592 negative windows.
  Native-mode balanced accuracy was 85.86% for MLP and 84.80% for RF;
  standardized-16-kHz balanced accuracy was 66.86% and 93.50%, respectively.
- Full regression suite: 67 passed, four existing scikit-learn batch-size
  warnings, 5.76 seconds.

## Decision

Usable as a real-only external stress test after manual timestamp review:
professor real positive plus private real negative data, with frozen models and
thresholds. It is not a precise deployment-performance estimate because only
one positive recording and ten negative recordings are independent. Raw
environmental noise should remain intact. Report native end-to-end behavior and
the standardized-16-kHz sensitivity slice separately.
