# M16 Implementation Plan

## Scope

Train the first valid post-first-kill baselines on the M15 repaired-key artifact.
Keep the feature profile, split, metrics, and untuned XGBoost parameters fixed.

## Slices

1. Freeze the canonical non-redundant feature profile and metric targets.
2. Add failing tests for feature selection, split preparation, and model contracts.
3. Implement Dummy, logistic-regression, and untuned-XGBoost training.
4. Add a matched-row pre-round XGBoost control and external comparison.
5. Run the full artifact, verify all gates, document, commit, and push.

## Review Gates

- Gate A: the M16 specification and targets exist before model code.
- Gate B: focused tests fail before the M16 module exists, then pass after implementation.
- Gate C: all models use identical keys and the exact M15 split/data fingerprint.
- Gate D: the report retains misses against targets and non-comparable external gaps.

## Outcome

M16 passed all four review gates and all eight blocking acceptance checks. The formal
XGBoost test AUC is 0.808896, while logistic regression is 0.809059. The matched-row
first-kill feature profile adds 0.088015 validation AUC over the pre-round control.
M17 will tune XGBoost using train/validation only.
