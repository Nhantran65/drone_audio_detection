# Drone Audio Classification - Short Project Summary

## Goal

Build a lightweight binary classifier for 1-second audio windows:

```text
drone / no_drone
```

Professor real-world audio is evaluation-only. It is not used for training,
validation, threshold tuning, or model selection.

## Data And Features

| Item | Summary |
| --- | --- |
| Windowing | 1.0s window, 0.5s stride |
| Public dataset rows | 401,270 public rows: 274,728 train, 59,298 val, 67,244 public test |
| Professor eval rows | 1,113 positive-only real-world rows |
| V1 features | 5 simple time-domain features: mean, mean abs, RMS, peak abs, zero-crossing rate |
| V2A/V2B features | `spectral_mfcc_v1`: 100 features = V1 basic + spectral + MFCC stats |
| V2C features | `logmel_v1`: 64-bin log-mel windows |
| V2D features | `pretrained_embedding_v1`: YAMNet mean-pooled embeddings |
| Scaling | V2 scaler fit on public train only, then reused for val/test/professor |

## Model Comparison

| Track | Model | Train style | Feature/input | Params | Size | Public F2 | Public FPR | Professor recall | Longest missed | Deployability |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V1 | `v1_baseline` | dependency-free centroid smoke baseline | `v1_basic` | n/a | n/a | 0.8715 | 0.1687 | 0.7529 | 16.0s | baseline only |
| V2A | `logistic_regression` | scikit-learn classical ML | `spectral_mfcc_v1` | 101 | 4.0 KB | 0.9756 | 0.0159 | 0.7260 | 26.5s | deployable lightweight |
| V2A | `linear_svm` | scikit-learn linear SVM/SGD | `spectral_mfcc_v1` | 202 | 5.8 KB | 0.9733 | 0.0327 | 0.7781 | 10.5s | deployable lightweight |
| V2A | `random_forest` | scikit-learn tree ensemble | `spectral_mfcc_v1` | n/a | 12.1 MB | 0.9963 | 0.0061 | 0.8886 | 7.5s | maybe deployable |
| V2A | `extra_trees` | scikit-learn tree ensemble | `spectral_mfcc_v1` | n/a | 27.8 MB | 0.9898 | 0.0173 | 0.5337 | 35.0s | maybe deployable |
| V2B | `tiny_mlp` | small neural net | `spectral_mfcc_v1` | 7,521 | 259.0 KB | 0.9947 | 0.0048 | 0.7233 | 9.5s | deployable lightweight |
| V2B | `tiny_mlp_es` | tiny MLP + early-stopping checkpoint | `spectral_mfcc_v1` | 7,521 | 277.3 KB | 0.9975 | 0.0032 | 0.7916 | 6.5s | deployable lightweight |
| V2C | `tiny_cnn` | small PyTorch CNN | `logmel_v1` | 1,313 | 11.5 KB | 0.9416 | 0.1027 | 0.4465 | 21.0s | maybe deployable |
| V2C | `tiny_cnn_es` | tiny CNN + early-stopping checkpoint | `logmel_v1` | 1,313 | 11.6 KB | 0.9665 | 0.0398 | 0.5544 | 19.0s | maybe deployable |
| V2D | `yamnet_head` | TensorFlow Hub YAMNet + logistic head | `pretrained_embedding_v1` | 1,025 | 30.0 KB | 0.9744 | 0.0419 | 0.6199 | 12.0s | reference only |

## Public Test Metrics

These metrics are computed on the public held-out test set only, not on the
professor positive-only set and not on a combined set.
Confusion matrix format is `TN/FP/FN/TP`.

| Track | Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FPR | FNR | CM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V1 | `v1_baseline` | 0.8592 | 0.8720 | 0.7381 | 0.9127 | 0.8162 | 0.8715 | 0.9368 | 0.8441 | 0.1687 | 0.0873 | 36753/7460/2010/21021 |
| V2A | `logistic_regression` | 0.9817 | 0.9806 | 0.9698 | 0.9771 | 0.9734 | 0.9756 | 0.9971 | 0.9958 | 0.0159 | 0.0229 | 43512/701/528/22503 |
| V2A | `linear_svm` | 0.9724 | 0.9747 | 0.9399 | 0.9820 | 0.9605 | 0.9733 | 0.9970 | 0.9934 | 0.0327 | 0.0180 | 42768/1445/414/22617 |
| V2A | `random_forest` | 0.9954 | 0.9961 | 0.9884 | 0.9983 | 0.9933 | 0.9963 | 0.9999 | 0.9998 | 0.0061 | 0.0017 | 43943/270/39/22992 |
| V2A | `extra_trees` | 0.9871 | 0.9891 | 0.9677 | 0.9955 | 0.9814 | 0.9898 | 0.9995 | 0.9991 | 0.0173 | 0.0045 | 43448/765/104/22927 |
| V2B | `tiny_mlp` | 0.9954 | 0.9955 | 0.9909 | 0.9957 | 0.9933 | 0.9947 | 0.9997 | 0.9996 | 0.0048 | 0.0043 | 44002/211/99/22932 |
| V2B | `tiny_mlp_es` | 0.9973 | 0.9976 | 0.9938 | 0.9984 | 0.9961 | 0.9975 | 0.9999 | 0.9999 | 0.0032 | 0.0016 | 44070/143/37/22994 |
| V2C | `tiny_cnn` | 0.9235 | 0.9355 | 0.8316 | 0.9738 | 0.8971 | 0.9416 | 0.9878 | 0.9833 | 0.1027 | 0.0262 | 39673/4540/604/22427 |
| V2C | `tiny_cnn_es` | 0.9659 | 0.9685 | 0.9274 | 0.9768 | 0.9515 | 0.9665 | 0.9942 | 0.9923 | 0.0398 | 0.0232 | 42452/1761/534/22497 |
| V2D | `yamnet_head` | 0.9682 | 0.9729 | 0.9247 | 0.9877 | 0.9552 | 0.9744 | 0.9976 | 0.9960 | 0.0419 | 0.0123 | 42361/1852/283/22748 |

## Combined Supplementary Metrics

These metrics combine public held-out test rows with professor positive-only
rows. Use them as supplementary context only; model selection still depends on
public test metrics plus professor recall/longest-miss reporting. Confusion
matrix format is `TN/FP/FN/TP`.

| Track | Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FPR | FNR | CM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V1 | `v1_baseline` | 0.8574 | 0.8683 | 0.7456 | 0.9054 | 0.8177 | 0.8681 | 0.9340 | 0.8435 | 0.1687 | 0.0946 | 36753/7460/2285/21859 |
| V2A | `logistic_regression` | 0.9776 | 0.9748 | 0.9708 | 0.9655 | 0.9681 | 0.9666 | 0.9959 | 0.9941 | 0.0159 | 0.0345 | 43512/701/833/23311 |
| V2A | `linear_svm` | 0.9692 | 0.9700 | 0.9420 | 0.9726 | 0.9571 | 0.9663 | 0.9955 | 0.9912 | 0.0327 | 0.0274 | 42768/1445/661/23483 |
| V2A | `random_forest` | 0.9937 | 0.9936 | 0.9889 | 0.9932 | 0.9911 | 0.9924 | 0.9997 | 0.9995 | 0.0061 | 0.0068 | 43943/270/163/23981 |
| V2A | `extra_trees` | 0.9797 | 0.9784 | 0.9685 | 0.9742 | 0.9713 | 0.9731 | 0.9974 | 0.9961 | 0.0173 | 0.0258 | 43448/765/623/23521 |
| V2B | `tiny_mlp` | 0.9910 | 0.9892 | 0.9912 | 0.9831 | 0.9871 | 0.9847 | 0.9986 | 0.9982 | 0.0048 | 0.0169 | 44002/211/407/23737 |
| V2B | `tiny_mlp_es` | 0.9940 | 0.9928 | 0.9940 | 0.9889 | 0.9914 | 0.9899 | 0.9992 | 0.9990 | 0.0032 | 0.0111 | 44070/143/269/23875 |
| V2C | `tiny_cnn` | 0.9157 | 0.9234 | 0.8347 | 0.9495 | 0.8884 | 0.9241 | 0.9813 | 0.9751 | 0.1027 | 0.0505 | 39673/4540/1220/22924 |
| V2C | `tiny_cnn_es` | 0.9592 | 0.9588 | 0.9292 | 0.9573 | 0.9431 | 0.9516 | 0.9913 | 0.9883 | 0.0398 | 0.0427 | 42452/1761/1030/23114 |
| V2D | `yamnet_head` | 0.9626 | 0.9644 | 0.9268 | 0.9708 | 0.9483 | 0.9616 | 0.9940 | 0.9916 | 0.0419 | 0.0292 | 42361/1852/706/23438 |

## Best Candidates

- Best public-test F2: `tiny_mlp_es` at 0.9975, with very low public FPR
  0.0032 and small size 277.3 KB.
- Best professor real-world recall: `random_forest` at 0.8886, with longest
  missed segment 7.5s and public FPR 0.0061.
- Best lightweight report candidate: `tiny_mlp_es`, because it is small and has
  the strongest public-test result.
- Best quality candidate if 12.1 MB is acceptable: `random_forest`, because it
  is the only current model that clears professor recall >= 0.85 and longest
  missed segment <= 8s.

## Current Status

- V1 proves the full training/evaluation loop, but is not competitive enough.
- V2A/V2B/V2C are implemented and compared.
- V2B early-stopping MLP is the best lightweight candidate so far.
- V2D YAMNet is implemented as a `reference_only` baseline and now has the same
  combined supplementary reporting as the other implemented V2 models.
