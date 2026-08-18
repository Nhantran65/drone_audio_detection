# Execution plan

1. Inventory legacy artifacts and initialize `artifacts/v3_group_split_16khz/`.
2. Build and audit the grouped public manifest.
3. Repartition canonical features and refit the scaler on train only.
4. Compare natural and group-balanced weighting without consulting test data.
5. Retrain classical ML, Tiny MLP, Tiny CNN, and YAMNet-head tracks.
6. Freeze validation thresholds, then evaluate public, professor-positive, and private-real.
7. Build the comparison report and verify the legacy SHA-256 inventory is unchanged.
