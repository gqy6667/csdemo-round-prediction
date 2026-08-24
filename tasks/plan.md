# M25 Implementation Plan

## Scope

Explain the accepted M23/M24 pre-round LightGBM without training or tuning. Add
native TreeSHAP, encoded/source/macro permutation importance, a complete leakage
contract, and a same-feature explanation comparison with M12 XGBoost.

## Slices

1. Freeze M25 inputs, explanation methods, feature mapping, comparison policy,
   outputs, and blockers.
2. Add failing tests for mapping, leakage, grouped permutation, native TreeSHAP,
   XGBoost rank comparison, and acceptance behavior.
3. Implement exact M24 replay and the three LightGBM explanation methods.
4. Add cases, M12 explanation comparison, plots, report, manifest, and one-click runner.
5. Run the formal 20-repeat experiment, full suite and compile checks; document,
   commit, and push the complete stage.

## Risks

- Test-driven feature selection: explanation results never trigger retraining or feature removal.
- Attribution mismatch: algorithm ranking agreement is reported, never required for acceptance.
- Identity drift: replay joins use series_id, game_id, and round_id together.
- Post-outcome leakage: only the 36 M14 purchase-end features may map into the model.
- Artifact drift: verify M24 acceptance, data/model hashes, feature columns, and probabilities.
- SHAP misuse: report model contributions, not causal effects.

## Review Gates

- Gate A: the M25 specification exists before tests or implementation.
- Gate B: focused M25 tests fail because the module does not yet exist.
- Gate C: focused tests pass before the formal real-artifact run.
- Gate D: the formal run completes Gain, permutation, TreeSHAP, leakage, comparison,
  and case outputs without changing the model.
- Gate E: the complete suite, compile check, artifact parsing, and Git checks pass.

## Outcome

Pending.
