# Story Backlog

This backlog tracks candidate follow-up product stories. Do not create every
possible story packet up front. Create story packets when the work is selected
or when a product decision needs a durable place to land.

## Candidate Epics

| Epic | Description | Status |
| --- | --- | --- |
| Data prep | Add NASA Small UAS MAT conversion or reader support so the real local registry can build a complete public manifest | implemented as US-005 |
| V1 run | Generate the real public manifest and train the v1 smoke baseline from local public data | candidate after US-005 |
| V1 eval | Evaluate the frozen V1 model on public test plus professor positive-only data | implemented as US-006 |
| Preprocessing | Add 16 kHz mono preprocessing and log-mel or MFCC feature extraction | candidate |
| Modeling | Add the first non-smoke baseline after feature extraction, likely classical MFCC or a tiny PyTorch model | candidate |
| V2-M0 setup | Keep V1 commands unchanged while adding V2 feature/model/metrics schema and CLI skeletons | implemented as US-007 |
| V2-M1 engineered features | Implement `spectral_v1`, `mfcc_v1`, `spectral_mfcc_v1`, train-fitted scaler, and feature audit JSON | implemented as US-008 |
| V2-M2 classical ML sweep | Train at least three V2A classical ML candidates on `spectral_mfcc_v1` and generate comparison artifacts | implemented as US-009 |
| V2-M3 tiny MLP | Train and evaluate V2B Tiny MLP on engineered features, including parameter count and model size | candidate |
| V2-M4 tiny CNN | Implement `logmel_v1`, train V2C Tiny CNN, and evaluate frozen public/professor metrics | candidate |
| V2-M5 pretrained embedding | Evaluate at least one V2D pretrained embedding baseline, recommended first YAMNet, as reference-only unless deployable | candidate |
| V2-M6 unified comparison | Generate V1 vs V2 comparison table and select the leading candidate with deployability label | candidate |
| V2-M7 error analysis | Export missed professor windows, public false positives, score quantiles, and error analysis JSON | candidate |
| V2-M8 report candidate | Freeze final V2 candidate, threshold, metrics, limitations, and professor-report summary | candidate |

## V2.1 Stop Criteria

Stop and report if a deployable lightweight V2A/V2B/V2C candidate reaches:

```text
public_test_f2 >= 0.88
professor_recall >= 0.85
longest_professor_missed_segment_sec <= 8.0
public_fpr <= 0.168729
model remains lightweight
```

If no candidate reaches every target, report if one candidate improves at least
two of professor recall, longest missed professor segment, public F2, public
FPR, or model complexity.
