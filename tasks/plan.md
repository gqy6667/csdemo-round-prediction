# M26 Implementation Plan

## Scope

Package the accepted M23/M24/M25 pre-round LightGBM as a strict one-row JSON/CSV
prediction interface. Bind inference to the frozen model, identity calibrator,
feature order, map vocabulary, deployment tree count, and artifact hashes.

## Slices

1. Freeze M26 artifacts, input/output contracts, invalid cases, outputs, and blockers.
2. Add failing tests for model/calibrator drift, input validation, real prediction,
   JSON/CSV equality, CLI behavior, and acceptance artifacts.
3. Implement the strict LightGBM predictor and make focused tests green.
4. Implement prerequisite checks, invalid-case audit, frozen metrics, external
   comparison, report, manifest, and one-click runner.
5. Run the formal interface acceptance, full suite and compile checks; document,
   commit, and push the complete stage.

## Risks

- Interface drift: require exact 36 raw and 43 encoded columns in saved order.
- Artifact mismatch: bind calibrator model/data hashes to the loaded model.
- Silent category fallback: reject maps absent from the training vocabulary.
- Invalid snapshots: collect type, range, round, inventory, future, and identity errors.
- Metric misuse: example probability never becomes a performance metric.
- Training leakage: no fit path exists in the predictor or stage acceptance.

## Review Gates

- Gate A: the M26 specification exists before tests or implementation.
- Gate B: focused M26 tests fail because the modules do not yet exist.
- Gate C: focused tests pass before the formal real-artifact run.
- Gate D: JSON/CSV and CLI paths pass against frozen artifacts, including all invalid cases.
- Gate E: the complete suite, compile check, artifact parsing, and Git checks pass.

## Outcome

Pending.
