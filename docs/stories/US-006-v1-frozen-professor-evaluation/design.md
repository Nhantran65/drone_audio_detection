# Design

## Interface

Add `eval-v1`:

```text
python -m drone_sound.cli eval-v1 \
  --model artifacts/models/v1_baseline_model.json \
  --public-manifest data/manifests/public_manifest.csv \
  --public-audio-root data/raw/public \
  --professor-manifest data/manifests/professor_manifest.csv \
  --professor-audio-root data/raw/private/professor_real_world \
  --metrics-out artifacts/models/v1_eval_metrics.json
```

Professor audio is extracted locally before running eval:

```text
mkdir -p data/raw/private/professor_real_world
unzip -o fwdronedata.zip -d data/raw/private/professor_real_world
```

## Behavior

- Load and validate the frozen V1 model schema.
- Reuse V1 feature extraction, scaler, centroid score, and stored threshold.
- Compute full binary metrics for public `public_test`.
- Compute recall-oriented metrics for positive-only professor data.
- Compute combined supplementary metrics with an explicit caveat.

## Output

`v1_eval_metrics.json` records model path, threshold source, public metrics,
professor metrics, combined supplementary metrics, and limitations.
