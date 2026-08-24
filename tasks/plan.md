# M22 Implementation Plan

## Scope

Close the XGBoost research line with a consolidated final report, then start the
pre-round LightGBM line as a controlled algorithm comparison. Keep the accepted M14
data, grouped split, feature encoding, XGBoost test probabilities, and metrics fixed.

## Slices

1. Freeze the M22 data, feature, training, comparison, metric, and blocker contracts.
2. Add failing tests for LightGBM parameters, train-only encoding, validation-only early
   stopping, frozen-XGBoost replay, metric directions, and acceptance decisions.
3. Implement the reusable LightGBM trainer and M22 acceptance primitives.
4. Run the real 41,074-row experiment and generate model, probability, comparison,
   external benchmark, manifest, and beginner-facing reports.
5. Run focused and complete tests, compile checks, inspect artifacts, commit, and push.

## Risks

- Unfair comparison: align both models to the same 43 columns and exact 4,172 test rows.
- Test-set tuning: use only validation Log Loss for early stopping and save that policy.
- XGBoost drift: load the accepted bundle and compare replayed probabilities to M9.
- Category leakage: derive dummy columns from train only and align validation/test afterward.
- Misleading claims: report LightGBM losses as honestly as wins and label external results.
- Circular testing: unit tests call small functions; the formal runner invokes the suite once.

## Review Gates

- Gate A: the M22 specification exists before tests or implementation.
- Gate B: focused M22 tests fail because the module does not yet exist.
- Gate C: focused tests pass before the formal real-artifact run.
- Gate D: the formal run passes every blocker with no XGBoost fit call.
- Gate E: the complete suite, compile check, artifact parsing, and Git checks pass.

## Outcome

In progress. M22 will record the first fair LightGBM baseline and the exact metric
differences from the frozen pre-round XGBoost before any LightGBM tuning begins.
