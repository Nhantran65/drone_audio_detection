# SPEC v1.0 — Lightweight Drone Audio Classification POC

## 1. Project Overview

### 1.1 Project Name

Lightweight Drone Audio Classification POC

### 1.2 Objective

Build a lightweight audio classification model that detects whether a short audio window contains drone sound.

The initial POC task is binary classification:

* `drone`
* `no_drone`

The target model should be as lightweight as possible while maintaining strong detection performance, especially under realistic noisy conditions.

### 1.3 Primary Goal

Train and evaluate a small edge-friendly model for window-level drone audio detection.

The model should answer:

> Given a 1-second audio window, does this window contain drone sound?

### 1.4 Secondary Goal

Evaluate whether the model trained on public datasets can generalize to the professor-provided real-world drone recording dataset.

### 1.5 Out of Scope

This POC does not include:

* drone image/video detection
* RF-based drone detection
* radar-based drone detection
* drone localization
* drone tracking
* drone model identification
* multi-class sound event classification
* real-time product UI
* cloud deployment
* long-term monitoring dashboard

These can be future extensions after the binary audio classifier is validated.

---

## 2. Problem Definition

### 2.1 Task Type

Binary audio classification.

### 2.2 Input

A short audio segment/window.

Initial setting:

* window length: 1 second
* stride: 0.5 second
* sample rate: 16 kHz
* channel: mono

### 2.3 Output

A probability score:

```text
P(drone) ∈ [0, 1]
```

A binary decision:

```text
drone if P(drone) >= threshold
no_drone otherwise
```

### 2.4 Model Granularity

Initial POC uses window-level classification.

Event-level detection can be implemented later by smoothing multiple window predictions.

---

## 3. Key Design Decisions

### 3.1 Binary Class Only

The model will be trained with only two labels:

```text
drone
no_drone
```

No multi-class classification will be used in the first POC.

### 3.2 Keep Subtype as Metadata Only

Although the model is binary, each sample should still keep a metadata field called `label_subtype`.

Examples:

```text
drone_field
drone_with_speech
drone_with_background
no_drone_aircraft
no_drone_traffic
no_drone_urban_industrial
no_drone_speech
no_drone_background
```

This subtype is not used as the training label. It is used for:

* dataset audit
* source balancing
* hard-negative sampling
* false positive analysis
* reporting model weaknesses
* avoiding data leakage

### 3.3 Merge Public Datasets into One Unified Dataset

All approved public datasets will be merged into one unified public dataset.

However, each sample must preserve:

* source dataset
* original file name
* license
* binary label
* subtype
* split
* checksum

### 3.4 Professor Data Usage

The professor-provided real-world dataset appears to contain mostly drone sound plus human speech.

Therefore, it should be treated as:

```text
real-world positive-only or near-positive-only test data
```

It can be used to evaluate:

* drone recall
* missed drone windows
* robustness to human speech interference
* confidence stability

It should not be used alone to evaluate:

* accuracy
* precision
* specificity
* false positive rate
* false alarms per hour

because it likely does not contain enough real-world no-drone examples.

---

## 4. Dataset Plan

## 4.1 Positive Drone Datasets

### Dataset P1 — DroneNoise Database

Role:

* positive drone dataset
* field-recorded drone audio
* useful for realistic drone acoustic signatures

Usage:

```text
label_binary = drone
label_subtype = drone_field
```

### Dataset P2 — Zenodo UAV Acoustic Signals — Kielce University of Technology

Role:

* positive drone dataset
* multiple UAV acoustic signals
* includes UAV sound and speech recorded in presence of UAV

Usage:

```text
label_binary = drone
label_subtype = drone_field or drone_with_speech
```

Important handling:

* samples with human speech but UAV present should remain `drone`
* subtype should mark `drone_with_speech`
* later evaluation should separately report performance on `drone_with_speech`

### Dataset P3 — NASA Small UAS Flyover Acoustics Data

Role:

* optional positive drone dataset
* optional hard-negative aircraft source
* use carefully because license is not clearly specified

Usage option A:

```text
multicopter SUAS samples → drone
fixed-wing aircraft samples → no_drone_aircraft or exclude
```

Usage option B:

```text
Use NASA dataset only for research-only evaluation until license is clarified.
```

Decision required:

* confirm whether NASA data can be used for training in this POC
* confirm whether fixed-wing aircraft should be treated as no-drone hard negative

---

## 4.2 Negative and Hard-Negative Datasets

### Dataset N1 — AeroSonicDB / AeroSonicDB 3K

Role:

* hard negative
* aircraft / low-flying aircraft sounds

Usage:

```text
label_binary = no_drone
label_subtype = no_drone_aircraft
```

Purpose:

* reduce false positives from aircraft, airplanes, helicopters, and low-flying aerial vehicle sounds

License handling:

* treat as research/POC-only unless commercial usage permission is clarified

### Dataset N2 — IDMT-TRAFFIC

Role:

* hard negative
* traffic and engine sounds

Usage:

```text
label_binary = no_drone
label_subtype = no_drone_traffic
```

Useful classes:

* motorcycle
* car
* truck
* bus
* road noise
* wet road noise
* engine pass-by

Purpose:

* reduce false positives in outdoor, industrial, roadside, and urban deployments

License handling:

* treat as research/POC-only unless commercial usage permission is clarified

### Dataset N3 — UrbanSound8K

Role:

* urban / industrial negative dataset

Usage:

```text
label_binary = no_drone
label_subtype = no_drone_urban_industrial
```

Useful classes:

* air_conditioner
* drilling
* engine_idling
* jackhammer
* siren
* car_horn
* street_music

Purpose:

* reduce false positives from mechanical, urban, and industrial noise

### Dataset N4 — Speech-Only Negative Dataset

Candidate sources:

* LibriSpeech
* Mozilla Common Voice
* other permitted speech-only dataset

Role:

* speech-only negative

Usage:

```text
label_binary = no_drone
label_subtype = no_drone_speech
```

Purpose:

* prevent the model from associating human speech with the drone class
* important because the professor dataset contains drone plus human conversation

Recommended amount:

```text
1–3 hours of speech-only negative data for POC
```

---

## 5. Dataset Structure

### 5.1 Unified Dataset Layout

Recommended folder structure:

```text
dataset_root/
  raw/
    positive/
      DroneNoise/
      Kielce_UAV/
      NASA_SUAS/
    negative/
      AeroSonicDB/
      IDMT_TRAFFIC/
      UrbanSound8K/
      Speech/
    professor_real_world/
  processed/
    wav_16k_mono/
    windows_1s/
    features/
  manifests/
    dataset_manifest.csv
    split_manifest.csv
    license_manifest.csv
```

### 5.2 Manifest Fields

The unified manifest must include:

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

### 5.3 Binary Labels

Allowed values:

```text
drone
no_drone
```

### 5.4 Subtype Values

Recommended values:

```text
drone_field
drone_with_speech
drone_with_background
drone_unknown_condition
no_drone_aircraft
no_drone_traffic
no_drone_urban_industrial
no_drone_speech
no_drone_background
no_drone_unknown
```

---

## 6. Data Processing Pipeline

## 6.1 Audio Standardization

All audio files should be converted to:

```text
sample rate: 16 kHz
channel: mono
format: WAV or float32 tensor
```

Processing steps:

1. load original audio
2. convert to mono
3. resample to 16 kHz
4. normalize amplitude safely
5. save processed copy or generate features directly
6. record all processing results in manifest

### 6.2 Windowing

Initial windowing strategy:

```text
window length = 1.0 second
stride = 0.5 second
```

Rules:

* clips shorter than 1 second may be padded or ignored
* uncertain windows should be marked as `ignore`
* boundary windows should be excluded if drone presence is unclear
* windows from the same original file must not be split across train/test

### 6.3 Feature Extraction

Primary feature:

```text
log-mel spectrogram
```

Recommended parameters:

```text
sample rate: 16 kHz
FFT window: 25 ms
hop length: 10 ms
mel bins: 64
frequency range: 50 Hz – 8 kHz
normalization: per-dataset or global train-set normalization
```

Secondary feature for ablation:

```text
MFCC, 20–40 coefficients
```

### 6.4 Augmentation

Use moderate augmentation:

```text
random gain
random time shift
background noise mixing
band-pass / microphone EQ simulation
SpecAugment
light pitch shift
```

Recommended background mixing SNR:

```text
clean
20 dB
10 dB
5 dB
0 dB
-5 dB for stress test only
```

Avoid excessive augmentation that destroys drone acoustic signatures.

---

## 7. Data Splitting Strategy

### 7.1 Do Not Random Split by Window

Random window split is not allowed because it can cause leakage.

Example of leakage:

```text
same original recording → nearby 1-second windows → both train and test
```

This would make test performance unrealistically high.

### 7.2 Split Level

Split should be done by:

```text
source_dataset
original_file
recording_session
recording_location
drone_model if available
```

### 7.3 Proposed Splits

#### Train Set

Used for model training.

Contains:

* public positive datasets
* public negative datasets
* public hard-negative datasets

Does not contain:

* professor real-world data

#### Validation Set

Used for:

* early stopping
* model selection
* threshold tuning
* hyperparameter tuning

#### Public Test Set

Used for final public benchmark.

Contains held-out public samples only.

#### Professor Real-World Test Set

Used only after model and threshold are frozen.

Contains professor-provided real-world recordings.

#### Combined Test Set

Optional final supplementary evaluation.

Contains:

* public test samples
* professor positive samples

Must be clearly reported as supplementary, not as the only final metric.

---

## 8. Model Plan

## 8.1 Baseline 0 — Classical Lightweight Baseline

Feature:

```text
MFCC statistics
```

Models:

```text
Logistic Regression
Linear SVM
small Random Forest
```

Purpose:

* sanity check
* fast first result
* explainable baseline
* benchmark against Tiny CNN

Expected use:

* not final deploy model unless performance is surprisingly strong

---

## 8.2 Baseline 1 — Main Tiny CNN

Feature:

```text
log-mel spectrogram
```

Model:

```text
Tiny Depthwise-Separable CNN
```

Example architecture:

```text
Input: 64 x 100 log-mel spectrogram

Conv2D 8 filters
BatchNorm
ReLU

DepthwiseSeparableConv2D 16 filters
BatchNorm
ReLU

DepthwiseSeparableConv2D 32 filters
BatchNorm
ReLU

DepthwiseSeparableConv2D 64 filters
BatchNorm
ReLU

GlobalAveragePooling

Dense 1
Sigmoid
```

Target:

```text
< 100k parameters preferred
< 500 KB INT8 model preferred
CPU-friendly inference
near real-time on edge device
```

---

## 8.3 Baseline 2 — Teacher / Reference Model

Candidate models:

```text
YAMNet
PANNs
other pretrained audio classifiers
```

Usage:

* teacher model
* benchmark model
* optional embedding extractor

Not preferred as primary deployment model unless hardware allows.

---

## 9. Training Strategy

### 9.1 Loss

Initial options:

```text
Binary Cross Entropy
Focal Loss
```

Use Focal Loss if false positives or class imbalance become severe.

### 9.2 Sampling

Use balanced sampling:

```text
balanced drone / no_drone ratio
source-balanced sampling
hard-negative oversampling
speech-negative inclusion
```

Recommended initial training batch composition:

```text
50% drone
50% no_drone
```

Within `no_drone`, include:

```text
aircraft
traffic
urban_industrial
speech
background
```

### 9.3 Training Controls

Use:

```text
early stopping
learning rate scheduler
validation F1/F2 monitoring
validation FPR monitoring
checkpoint best model
```

### 9.4 Threshold Selection

Do not assume threshold = 0.5.

Choose threshold on validation set.

Two threshold strategies:

#### Strategy A — Balanced POC

Optimize:

```text
maximum F1
```

#### Strategy B — Product-Oriented

Set a target false positive rate, then maximize recall.

Example:

```text
choose threshold where FPR <= 1%
maximize recall under that constraint
```

Preferred strategy for product direction:

```text
Recall at fixed FPR
```

---

## 10. Evaluation Plan

## 10.1 Evaluation A — Public Merged Test

Purpose:

* main binary benchmark

Data:

```text
held-out public positive samples
held-out public negative samples
```

Metrics:

```text
Accuracy
Balanced Accuracy
Precision
Recall
F1-score
F2-score
ROC-AUC
PR-AUC
False Positive Rate
False Negative Rate
Confusion Matrix
```

### 10.2 Evaluation B — Hard-Negative Analysis

Purpose:

* understand model failure modes

Report FPR by subtype:

```text
no_drone_aircraft
no_drone_traffic
no_drone_urban_industrial
no_drone_speech
no_drone_background
```

Metrics:

```text
FPR by subtype
false positives per hour by subtype
top false-positive examples
```

### 10.3 Evaluation C — Professor Real-World Positive Test

Purpose:

* test generalization to real-world field data

Data:

```text
professor-provided drone + speech recordings
```

Metrics:

```text
Recall
missed drone windows
longest missed segment
mean P(drone)
median P(drone)
prediction stability over time
speech-interference robustness
```

Do not use this dataset alone to report:

```text
accuracy
precision
specificity
false positive rate
false alarms per hour
```

unless real-world no-drone data is added.

### 10.4 Evaluation D — Combined Test

Purpose:

* supplementary evaluation

Data:

```text
public test set
+ professor positive samples
```

Metrics:

```text
F1
F2
Recall
Precision
PR-AUC
Balanced Accuracy
```

Important note:

The combined test must clearly state that professor data is positive-only or near-positive-only.

### 10.5 Evaluation E — Edge Performance

Metrics:

```text
number of parameters
FP32 model size
INT8 model size
latency per 1-second window
RAM usage
CPU usage
real-time factor
```

Target:

```text
real-time factor < 1.0
```

Meaning:

```text
processing 1 second of audio takes less than 1 second
```

---

## 11. Primary Metrics

### 11.1 Model Quality Metrics

Primary metric:

```text
F1-score or F2-score on public merged test
```

If recall is more important than precision:

```text
Use F2-score
```

Operational metric:

```text
Recall / TPR at fixed FPR
```

Example:

```text
Recall when FPR <= 1%
```

Real-world metric:

```text
Recall on professor positive-only test
```

### 11.2 Product-Oriented Metrics

Important operational metrics:

```text
false positives per hour
false positives per negative subtype
missed drone windows
longest missed drone segment
latency
model size
CPU/RAM usage
```

### 11.3 Why Accuracy Is Not Enough

Accuracy can be misleading when:

* dataset is imbalanced
* negative data is easy
* real-world false alarms matter
* professor dataset is positive-only

Therefore, accuracy should be reported but not used as the main success criterion.

---

## 12. Phase Plan with User Stories and Acceptance Criteria

# Phase 0 — Requirement Clarification

## Goal

Clarify deployment assumptions and evaluation expectations.

## User Story 0.1 — Hardware Target

As a research engineer,
I want to know the expected edge hardware,
so that I can choose the correct model size and inference runtime.

### Acceptance Criteria

* Target hardware is documented, or default assumption is recorded.
* CPU/GPU/NPU availability is documented.
* RAM limit is documented if available.
* Preferred runtime is documented if available: ONNX Runtime, TensorFlow Lite, PyTorch, or other.
* If hardware is unknown, the POC will target generic CPU edge deployment.

## User Story 0.2 — Detection Granularity

As a product stakeholder,
I want the first POC to classify short audio windows,
so that we can validate the model before implementing event-level detection.

### Acceptance Criteria

* Window-level classification is confirmed as POC scope.
* Initial window size is set to 1 second.
* Initial stride is set to 0.5 second.
* Event-level detection is explicitly marked as future work.

## User Story 0.3 — Professor Data Role

As a researcher,
I want to define how the professor-provided dataset will be used,
so that evaluation results are not misleading.

### Acceptance Criteria

* Professor dataset is marked as real-world positive-only or near-positive-only.
* It is excluded from training during the first POC.
* It is used only after model and threshold are frozen.
* Report clearly states that professor data cannot estimate false positive rate unless no-drone samples are added.

---

# Phase 1 — Dataset Acquisition and License Audit

## Goal

Collect all selected public datasets and verify whether they can be used in the POC.

## User Story 1.1 — Positive Dataset Collection

As a research engineer,
I want to collect trusted public positive drone audio datasets,
so that the model can learn drone acoustic signatures from traceable sources.

### Acceptance Criteria

* DroneNoise Database is downloaded or access path is documented.
* Zenodo Kielce UAV dataset is downloaded or access path is documented.
* NASA Small UAS dataset is downloaded or marked pending license decision.
* Each dataset has a source URL or DOI recorded.
* Each dataset has license status recorded.
* Any dataset with unclear license is marked as `restricted_or_pending`.

## User Story 1.2 — Negative Dataset Collection

As a research engineer,
I want to collect hard-negative and background datasets,
so that the model learns not to confuse drone sounds with other real-world sounds.

### Acceptance Criteria

* AeroSonicDB or AeroSonicDB 3K is downloaded or access request is documented.
* IDMT-TRAFFIC is downloaded or access path is documented.
* UrbanSound8K is downloaded or access path is documented.
* A speech-only negative source is selected.
* Each negative dataset has a subtype mapping.
* License restrictions are documented.

## User Story 1.3 — License Risk Tracking

As a project owner,
I want all dataset license risks documented,
so that the POC can avoid legal or productization issues.

### Acceptance Criteria

* License manifest is created.
* Each dataset is classified as one of:

  * `commercial_ok`
  * `research_only`
  * `non_commercial`
  * `license_unspecified`
  * `permission_required`
* NASA is marked as `license_unspecified` unless clarified.
* AeroSonicDB and IDMT-TRAFFIC are treated conservatively unless permission is confirmed.
* Final POC report includes a license risk section.

---

# Phase 2 — Dataset Manifest and Merge

## Goal

Merge selected datasets into one unified public dataset while preserving traceability.

## User Story 2.1 — Unified Manifest

As a research engineer,
I want a unified manifest for all samples,
so that training, evaluation, and auditing are reproducible.

### Acceptance Criteria

* `dataset_manifest.csv` is created.
* Each row has a unique `clip_id`.
* Each window has a unique `window_id` after windowing.
* Each row records source dataset, original file, label, subtype, split, and license.
* Missing critical fields are flagged.
* Corrupted or unreadable files are excluded and logged.

## User Story 2.2 — Binary Label Mapping

As a model developer,
I want all datasets mapped into `drone` and `no_drone`,
so that the first model remains simple and lightweight.

### Acceptance Criteria

* All positive drone samples are mapped to `drone`.
* All hard-negative and background samples are mapped to `no_drone`.
* `label_binary` contains only `drone`, `no_drone`, or `ignore`.
* Subtype remains metadata only.
* Multi-class training is not introduced in Phase 2.

## User Story 2.3 — Dataset Statistics

As a project reviewer,
I want dataset statistics before training,
so that I can understand whether the data is sufficient and balanced.

### Acceptance Criteria

* Total duration by dataset is reported.
* Total duration by binary label is reported.
* Total duration by subtype is reported.
* Number of original files by dataset is reported.
* Number of windows by label and split is reported.
* Any severe imbalance is documented.

---

# Phase 3 — Preprocessing and Feature Extraction

## Goal

Convert all selected audio into a consistent format and extract model-ready features.

## User Story 3.1 — Audio Standardization

As a model developer,
I want all audio standardized to the same format,
so that the model receives consistent input.

### Acceptance Criteria

* All audio is converted to mono.
* All audio is resampled to 16 kHz.
* Output format is consistent.
* Failed conversions are logged.
* Processed files can be reproduced from the raw data.

## User Story 3.2 — Window Generation

As a model developer,
I want audio split into fixed-length windows,
so that each model input has the same duration.

### Acceptance Criteria

* Window length is 1 second.
* Stride is 0.5 second.
* Window start and end time are stored.
* Short or ambiguous windows are handled consistently.
* Windows from the same original file inherit source and label metadata.

## User Story 3.3 — Feature Extraction

As a model developer,
I want log-mel features generated for every window,
so that the Tiny CNN can be trained efficiently.

### Acceptance Criteria

* Log-mel spectrograms are generated for all valid windows.
* Feature shape is consistent across samples.
* Feature normalization parameters are fitted only on the train split.
* Validation and test features use train-fitted normalization only.
* MFCC features are optionally generated for classical baselines.

---

# Phase 4 — Data Splitting and Leakage Control

## Goal

Create train, validation, and test splits without leakage.

## User Story 4.1 — File-Level Split

As a researcher,
I want to split data by original file/session,
so that adjacent windows from the same recording do not appear in both training and test.

### Acceptance Criteria

* No original file appears in more than one split.
* No recording session appears in more than one split where metadata allows.
* Split manifest is created.
* Leakage check script passes.
* The split strategy is documented.

## User Story 4.2 — Balanced Public Test Set

As a project reviewer,
I want a public test set with both drone and no-drone samples,
so that binary classification metrics are meaningful.

### Acceptance Criteria

* Public test contains drone samples.
* Public test contains no-drone samples.
* Public test includes at least one hard-negative subtype.
* Public test metrics can include precision, recall, F1, FPR, and confusion matrix.
* Professor data is not mixed into public test before final model freezing.

---

# Phase 5 — Baseline Modeling

## Goal

Establish simple baselines before training the final lightweight CNN.

## User Story 5.1 — Classical Baseline

As a researcher,
I want a classical ML baseline using MFCC features,
so that I can verify that the dataset and labels are learnable.

### Acceptance Criteria

* MFCC baseline is trained.
* At least one of Logistic Regression, Linear SVM, or Random Forest is evaluated.
* Public validation metrics are reported.
* Confusion matrix is reported.
* Baseline results are stored for comparison.

## User Story 5.2 — Teacher/Reference Baseline

As a researcher,
I want to compare against a pretrained audio model,
so that I can estimate the performance ceiling.

### Acceptance Criteria

* YAMNet, PANNs, or a similar pretrained audio model is evaluated.
* The teacher/reference model is not treated as the default deploy model.
* Metrics are reported on the same validation/test split.
* Runtime and model size are recorded if available.

---

# Phase 6 — Lightweight Model Training

## Goal

Train the main lightweight binary classifier.

## User Story 6.1 — Tiny CNN Training

As a model developer,
I want to train a Tiny CNN on log-mel spectrograms,
so that the model can detect drone audio with low computational cost.

### Acceptance Criteria

* Tiny CNN model is implemented.
* Model parameter count is reported.
* Model trains successfully on the unified public train set.
* Validation metrics are logged per epoch.
* Best checkpoint is saved.
* Training configuration is versioned.

## User Story 6.2 — Hard-Negative Training

As a product stakeholder,
I want the model trained with hard negatives,
so that it does not frequently misclassify aircraft, traffic, or industrial sounds as drones.

### Acceptance Criteria

* Hard-negative samples are included during training.
* Hard-negative subtype distribution is reported.
* Model performance is evaluated per negative subtype.
* If one subtype has high FPR, it is flagged for follow-up.

## User Story 6.3 — Threshold Tuning

As a product stakeholder,
I want the decision threshold tuned on validation data,
so that detection behavior matches the expected false alarm tolerance.

### Acceptance Criteria

* Threshold is selected using validation data only.
* Threshold selection rule is documented.
* F1-optimal threshold is reported.
* F2-optimal threshold is reported.
* Product-oriented threshold under fixed FPR is reported if possible.
* No threshold tuning is performed on the professor dataset.

---

# Phase 7 — Evaluation

## Goal

Evaluate model performance on public, hard-negative, professor, and combined test sets.

## User Story 7.1 — Public Test Evaluation

As a project reviewer,
I want final metrics on a held-out public test set,
so that I can judge whether the model works on balanced binary data.

### Acceptance Criteria

* Accuracy is reported.
* Balanced Accuracy is reported.
* Precision is reported.
* Recall is reported.
* F1-score is reported.
* F2-score is reported.
* ROC-AUC is reported.
* PR-AUC is reported.
* FPR and FNR are reported.
* Confusion matrix is reported.

## User Story 7.2 — Hard-Negative Evaluation

As a product stakeholder,
I want false positives broken down by negative subtype,
so that we understand real-world false alarm risks.

### Acceptance Criteria

* FPR by subtype is reported.
* Aircraft false positives are reported.
* Traffic false positives are reported.
* Urban/industrial false positives are reported.
* Speech-only false positives are reported.
* Top false-positive examples are saved for listening review.

## User Story 7.3 — Professor Real-World Positive Evaluation

As a researcher,
I want to test the frozen model on professor-provided recordings,
so that I can measure real-world drone recall.

### Acceptance Criteria

* Professor data is evaluated only after model and threshold are frozen.
* Recall on professor data is reported.
* Missed window count is reported.
* Longest missed segment is reported.
* Mean and median drone probability are reported.
* Prediction timeline is generated.
* Results are clearly described as positive-only or near-positive-only.

## User Story 7.4 — Combined Evaluation

As a project reviewer,
I want a combined evaluation that includes public test data and professor data,
so that I can see how the model behaves when real-world positives are added.

### Acceptance Criteria

* Combined test includes public test and professor positive data.
* Report explicitly states professor data is positive-only or near-positive-only.
* Combined metrics are reported separately from public metrics.
* Combined results are not used to hide public hard-negative performance.

---

# Phase 8 — Model Compression and Edge Benchmark

## Goal

Optimize the model for edge deployment.

## User Story 8.1 — Quantization

As an edge deployment engineer,
I want the model quantized to INT8,
so that it can run faster and use less memory.

### Acceptance Criteria

* FP32 model is exported.
* INT8 model is exported if runtime supports it.
* FP32 vs INT8 metrics are compared.
* Accuracy/F1 degradation after quantization is reported.
* Quantized model size is reported.

## User Story 8.2 — Edge Runtime Benchmark

As a product stakeholder,
I want inference latency measured on target or proxy hardware,
so that I know whether the model can run in real time.

### Acceptance Criteria

* Inference latency per 1-second window is reported.
* RAM usage is reported if measurable.
* CPU usage is reported if measurable.
* Real-time factor is reported.
* Model can process audio faster than real time on selected hardware or proxy hardware.
* If target hardware is unavailable, benchmark is marked as proxy-only.

---

# Phase 9 — POC Report

## Goal

Create a final research report for professor/leader review.

## User Story 9.1 — Dataset Report

As a reviewer,
I want a clear dataset summary,
so that I can verify the credibility of the training data.

### Acceptance Criteria

* All datasets are listed.
* Dataset source URLs/DOIs are listed.
* License status is listed.
* Total duration by dataset is reported.
* Train/validation/test split statistics are reported.
* Dataset limitations are documented.

## User Story 9.2 — Model Report

As a reviewer,
I want a clear model summary,
so that I can understand the accuracy-vs-lightweight tradeoff.

### Acceptance Criteria

* Model architecture is summarized.
* Parameter count is reported.
* Model size is reported.
* Feature extraction settings are reported.
* Training configuration is reported.
* Threshold strategy is reported.

## User Story 9.3 — Evaluation Report

As a reviewer,
I want transparent evaluation results,
so that I can judge whether the POC is product-relevant.

### Acceptance Criteria

* Public test metrics are reported.
* Hard-negative metrics are reported.
* Professor real-world positive test metrics are reported.
* Combined test metrics are reported separately.
* False-positive examples are summarized.
* Known limitations are clearly stated.

---

## 13. Success Criteria

### 13.1 Minimum Success Criteria

The POC is considered successful if:

* the model trains on the unified public dataset
* public test F1/F2 is meaningfully above classical baseline
* professor positive-only recall is acceptable
* model size is small enough for edge direction
* false positives on hard negatives are measured and understood
* all dataset sources and licenses are documented

### 13.2 Preferred Success Criteria

The POC is considered strong if:

* Tiny CNN achieves strong F1/F2 on public test
* recall remains high on professor real-world positive data
* aircraft/traffic/urban false positives are reasonably low
* INT8 quantization causes minimal performance drop
* inference is faster than real time on CPU or target edge hardware
* final report clearly explains dataset provenance, risks, and next steps

### 13.3 Suggested Numerical Targets for POC

Initial suggested targets:

```text
Public Test F1: >= 0.85
Public Test F2: >= 0.85
Professor Positive Recall: >= 0.85
INT8 model size: < 500 KB preferred
Parameter count: < 100k preferred
Real-time factor: < 1.0
```

These are initial POC targets and should be adjusted after dataset audit and hardware clarification.

---

## 14. Risks and Mitigations

### Risk 1 — License Ambiguity

Some datasets may not be clearly usable for commercial product training.

Mitigation:

* maintain license manifest
* treat unclear datasets as research-only
* request permission if productization is planned

### Risk 2 — Professor Dataset Is Positive-Only

The professor dataset cannot measure false alarms.

Mitigation:

* report it as positive-only recall test
* request or record real-world no-drone samples
* use public hard negatives for false-positive analysis

### Risk 3 — Domain Shift

Public datasets may not match the real recorder, microphone, distance, or environment.

Mitigation:

* use multiple public sources
* use hard negatives
* use augmentation
* evaluate professor data separately
* collect same-device no-drone audio if possible

### Risk 4 — Speech Bias

Professor data contains human conversation.

Mitigation:

* add speech-only negative data
* tag `drone_with_speech`
* evaluate speech-only false positive rate

### Risk 5 — False Positives from Aircraft and Traffic

Aircraft, helicopters, motorcycles, and engines may sound similar to drones.

Mitigation:

* include AeroSonicDB
* include IDMT-TRAFFIC
* report FPR by subtype
* oversample hard negatives if needed

### Risk 6 — Model Too Heavy for Edge

A high-performing model may be too large.

Mitigation:

* prioritize Tiny CNN
* use depthwise separable convolutions
* compare FP32 and INT8
* benchmark latency early

---

## 15. Open Questions

The following questions should be clarified with the professor or project owner:

1. What exact edge hardware is expected?
2. What is the target runtime: TFLite, ONNX Runtime, PyTorch, or custom embedded runtime?
3. What is the acceptable false alarm rate?
4. What drone distance should the model target?
5. What microphone or recorder was used for the professor dataset?
6. What is the sample rate and format of the professor recording?
7. Can we record 10–30 minutes of no-drone audio using the same device?
8. Is this POC strictly research/internal, or will it be used for commercial product training?
9. Should fixed-wing aircraft be treated as `no_drone` hard negative or excluded?
10. Is high recall more important than low false alarm, or should the model optimize a balanced F1 score?

---

## 16. Deliverables

### D1 — Dataset Audit Package

Includes:

* dataset manifest
* license manifest
* duration statistics
* source mapping
* issue log

### D2 — Preprocessing Pipeline

Includes:

* audio conversion script
* windowing script
* feature extraction script
* reproducibility instructions

### D3 — Baseline Results

Includes:

* MFCC + classical ML baseline
* optional teacher/reference model baseline

### D4 — Tiny CNN Model

Includes:

* trained FP32 model
* optional INT8 model
* architecture summary
* training configuration

### D5 — Evaluation Report

Includes:

* public test metrics
* hard-negative metrics
* professor positive-only metrics
* combined test metrics
* false-positive analysis
* model efficiency metrics

### D6 — Final POC Recommendation

Includes:

* whether the POC is technically feasible
* whether more data is needed
* whether model is edge-feasible
* recommended next steps for productization

---

## 17. Recommended Next Step

Start with Phase 1 and Phase 2:

1. Download selected datasets.
2. Verify license and access constraints.
3. Build `dataset_manifest.csv`.
4. Compute exact duration by label/subtype.
5. Confirm professor data format and sample rate.
6. Record or request real-world no-drone samples if possible.

After that, proceed to preprocessing and baseline training.

---

# Appendix A - SPEC Addendum V2.1: Lightweight ML/DL Exploration

## A.1 Objective

V2.1 improves the V1 smoke baseline by exploring richer audio features,
lightweight classical ML models, lightweight self-designed DL models, and
pretrained audio embedding baselines.

The task remains binary classification:

```text
drone / no_drone
```

Professor real-world data remains evaluation-only. It must not be used for
training, validation, threshold tuning, or model selection.

The goal is to identify a reportable lightweight candidate that improves over
V1 while also checking whether pretrained audio representations provide a useful
reference or upper-bound.

## A.2 V1 Reference Baseline

Current V1 result:

```text
public_test_f2: 0.871481
public_fpr: 0.168729
professor_recall: 0.752920
longest_professor_missed_segment: 16.0s
```

V1 is a valid end-to-end smoke baseline, but not the final model. Its main
weakness is that it uses simple time-domain features and may miss frequency or
time-frequency patterns that matter for drone audio.

## A.3 V2.1 Tracks

V2.1 contains four experiment tracks:

- V2A Classical ML on engineered `spectral_mfcc_v1` features.
- V2B Tiny MLP on engineered `spectral_mfcc_v1` features.
- V2C Tiny CNN on `logmel_v1` spectrograms.
- V2D pretrained audio embeddings, reported separately as a reference baseline
  unless the full inference path satisfies lightweight deployment constraints.

V2A candidate models:

```text
Logistic Regression
Linear SVM / SGD Classifier
Random Forest Small
ExtraTrees Small
Gradient Boosting Small, optional
```

V2B suggested architecture:

```text
Input: spectral_mfcc_v1 feature vector
Dense 64
ReLU
Dropout optional
Dense 16
ReLU
Dense 1
Sigmoid
```

V2C suggested input and architecture:

```text
Input: 64 x approximately 100 log-mel spectrogram
Conv2D 8 -> BatchNorm -> ReLU
DepthwiseSeparableConv2D 16 -> BatchNorm -> ReLU
DepthwiseSeparableConv2D 32 -> BatchNorm -> ReLU
GlobalAveragePooling
Dense 1
Sigmoid
```

V2D candidate pretrained models:

```text
YAMNet
EfficientAT
VGGish
PANNs, optional
```

## A.4 Feature Sets

```text
F0 v1_basic:
  mean, mean_abs, RMS, peak_abs, zero-crossing rate

F1 spectral_v1:
  spectral_centroid, spectral_bandwidth, spectral_rolloff,
  spectral_flatness, spectral_contrast, dominant_frequency,
  low_band_energy, mid_band_energy, high_band_energy,
  band_energy_ratios

F2 mfcc_v1:
  MFCC 13 or 20 coefficients, MFCC mean, MFCC std,
  optional delta MFCC mean/std

F3 spectral_mfcc_v1:
  v1_basic + spectral_v1 + mfcc_v1

F4 logmel_v1:
  64 mel bins, 1-second audio window, approximately 100 frames

F5 pretrained_embedding_v1:
  YAMNet / EfficientAT / VGGish / PANNs embedding
```

## A.5 Model Selection And Threshold Strategy

The main selection metric remains validation F2 because the POC prioritizes
recall over precision. Guardrail metrics:

```text
validation precision
validation F1
validation FPR
public_test FPR
professor recall
longest missed professor segment
model size
inference latency
```

A candidate should not be selected if it improves F2 only by significantly
increasing false positives.

Do not assume threshold `0.5`. For each model, tune thresholds using validation
data only and report:

```text
threshold_f1
threshold_f2
threshold_at_fpr_10
threshold_at_fpr_5
threshold_at_fpr_2
threshold_at_fpr_1
```

The primary threshold for the reportable POC is `threshold_f2`. The secondary
product-oriented view is recall at fixed FPR levels.

## A.6 Evaluation Plan

Public test remains the main binary benchmark because it contains both `drone`
and `no_drone`. Report:

```text
accuracy
balanced_accuracy
precision
recall
f1
f2
roc_auc
pr_auc
false_positive_rate
false_negative_rate
confusion_matrix
```

Professor positive-only evaluation reports only:

```text
professor_recall
missed_window_count
longest_missed_segment_sec
mean_p_drone
median_p_drone
score_std
decision_flip_count
decision_flip_rate
score_quantiles
```

Do not claim professor-only accuracy, precision, specificity, false positive
rate, or false alarms per hour.

Combined public-test plus professor-positive evaluation is supplementary only
and must not replace public test metrics.

For every candidate, also report:

```text
feature_count
model_type
model_file_size
parameter_count if applicable
average_inference_time_per_window
runtime dependency
deployability note
```

## A.7 Milestones

```text
V2-M0 Setup and compatibility:
  keep train-v1 and eval-v1 unchanged; define V2 command, metric, feature,
  and model schemas.

V2-M1 Engineered feature extraction:
  implement spectral_v1, mfcc_v1, spectral_mfcc_v1, train-fitted scaler,
  feature audit, and stable feature dimension report.

V2-M2 V2A classical ML sweep:
  train at least three lightweight classical models and generate validation
  comparison artifacts.

V2-M3 V2B Tiny MLP:
  train a small MLP on spectral_mfcc_v1 and report params/model size.

V2-M4 V2C Tiny CNN:
  implement logmel_v1 plus Tiny CNN and evaluate frozen public/professor
  metrics.

V2-M5 V2D pretrained embedding baseline:
  evaluate at least one pretrained embedding model, recommended first YAMNet,
  and document deployability limitations.

V2-M6 Unified evaluation and comparison:
  generate V1 vs V2 comparison table and best candidate summary.

V2-M7 Error analysis:
  export missed professor windows, public false positives, score quantiles,
  and traceable error analysis JSON.

V2-M8 Reportable V2 candidate:
  freeze one candidate, threshold, metrics, limitations, and final summary.
```

## A.8 Stop Criteria For Professor Report

Stop and report if any deployable lightweight candidate from V2A, V2B, or V2C
achieves:

```text
public_test_f2 >= 0.88
professor_recall >= 0.85
longest_professor_missed_segment_sec <= 8.0
public_fpr <= V1 public_fpr
model remains lightweight
```

V1 reference:

```text
public_test_f2 = 0.871481
public_fpr = 0.168729
professor_recall = 0.752920
longest_professor_missed_segment = 16.0s
```

If no candidate reaches all targets, still report if one candidate clearly
improves at least two of:

```text
professor recall
longest missed segment
public F2
public FPR
model complexity
```

If V2D pretrained embedding wins but is not deployable lightweight, report it
as `reference_only` and select the best V2A/V2B/V2C model as the main
deployable lightweight candidate.

## A.9 Suggested CLI

```bash
uv run --extra data --python 3.13 python -m drone_sound.cli sweep-v2-ml \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --feature-set spectral_mfcc_v1 \
  --model-types logistic_regression linear_svm random_forest extra_trees \
  --out-dir artifacts/models/v2a_ml_sweep
```

```bash
uv run --extra data --python 3.13 python -m drone_sound.cli train-v2-mlp \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --feature-set spectral_mfcc_v1 \
  --model-out artifacts/models/v2b_tiny_mlp.pt \
  --metrics-out artifacts/models/v2b_tiny_mlp_metrics.json
```

```bash
uv run --extra data --python 3.13 python -m drone_sound.cli train-v2-cnn \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --feature-set logmel_v1 \
  --model-out artifacts/models/v2c_tiny_cnn.pt \
  --metrics-out artifacts/models/v2c_tiny_cnn_metrics.json
```

```bash
uv run --extra data --python 3.13 python -m drone_sound.cli train-v2-embedding \
  --manifest data/manifests/public_manifest.csv \
  --audio-root data/raw/public \
  --embedding-model yamnet \
  --head logistic_regression \
  --model-out artifacts/models/v2d_yamnet_head.json \
  --metrics-out artifacts/models/v2d_yamnet_metrics.json
```

```bash
uv run --extra data --python 3.13 python -m drone_sound.cli eval-v2 \
  --model artifacts/models/v2_report_model.* \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --metrics-out artifacts/models/v2_report_eval_metrics.json \
  --export-errors artifacts/errors/v2_report
```

## A.10 Final V2 Report Structure

The final V2 report should contain:

```text
1. Objective
2. Dataset and split summary
3. V1 baseline recap
4. V2 track definitions
5. Metrics and threshold strategy
6. Public test comparison
7. Professor real-world positive-only comparison
8. Model size / lightweight comparison
9. Selected reportable candidate
10. Limitations
11. Next steps
```

Required comparison table columns:

```text
model
track
feature/input
public_f2
public_f1
public_precision
public_recall
public_fpr
professor_recall
longest_missed_segment
model_size
params
deployability_label
```

Deployability labels:

```text
deployable_lightweight
maybe_deployable
reference_only
```

Preferred final candidate:

```text
Best V2A/V2B/V2C model that improves professor recall and public F2 without
increasing public FPR.
```
