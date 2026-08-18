# Design

V3 derives conservative dataset-specific `group_id` values: Kielce campaigns,
DroneNoise events, NASA runs, AeroSonic sessions, IDMT sessions, and official
UrbanSound folds. A deterministic seeded allocator assigns whole groups to 70/15/15
splits while balancing rows and label subtypes per dataset.

Canonical raw 100-dimensional features are unchanged. Only split metadata changes,
and the scaler is refit on V3 train rows. Hierarchical sample weighting is optional
per model and equalizes class, dataset, group, then row contribution. Every model
freezes both F1/F2 thresholds from validation; F1 is the V3 primary threshold.
