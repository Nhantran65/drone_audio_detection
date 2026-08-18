# 0016: Use recording-session groups for V3 splits

Status: accepted

Window-level and file-level splits are insufficient when many files originate from the
same recording campaign or acoustic session. They can place near-duplicate conditions
in train and validation, making validation materially over-optimistic.

V3 therefore assigns complete dataset-specific session groups to exactly one of train,
validation, or public test. The 16 kHz preprocessing and feature definitions stay fixed,
so the principal experimental change is the split contract. Thresholds and model choice
use validation F1 only. Professor-positive and private-real data are never used for fit,
early stopping, threshold selection, or model selection.

The consequence is fewer independent split units and wider uncertainty than window
counts imply. Reports must include per-group/per-dataset results and cluster-bootstrap
confidence intervals where applicable.
