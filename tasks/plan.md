# M24 Implementation Plan

## Scope

Evaluate the accepted M23 pre-round LightGBM without training or tuning. Add
series-level uncertainty, paired XGBoost differences, fixed robustness slices,
validation-only calibration, high-confidence error review, and reproducible evidence.

## Slices

1. Freeze M24 inputs, interval targets, paired comparison policy, group definitions,
   calibration protocol, error threshold, outputs, and blockers.
2. Add failing tests for exact replay, complete-key joins, paired series bootstrap,
   interval/group assessment, calibration isolation, and acceptance behavior.
3. Implement frozen-model replay and statistical evaluation, then make focused tests green.
4. Add calibration, error review, external comparison, report, manifest, and one-click runner.
5. Run the formal 2,000-bootstrap experiment, full suite and compile checks; document,
   commit, and push the complete stage.

## Risks

- Test leakage: calibration selection accepts validation predictions only; test is evaluated later.
- False superiority: paired bootstrap reports intervals and never requires LightGBM to win.
- Correlated rounds: all uncertainty resamples complete series, never individual rows.
- Identity drift: every join uses series_id, game_id, and round_id together.
- Post-outcome leakage: first-kill fields appear only in saved error diagnostics.
- Artifact drift: verify M23 data/model hashes, feature columns, exact keys, and probabilities.

## Review Gates

- Gate A: the M24 specification exists before tests or implementation.
- Gate B: focused M24 tests fail because the module does not yet exist.
- Gate C: focused tests pass before the formal real-artifact run.
- Gate D: the formal run completes all interval, robustness, calibration, and error outputs.
- Gate E: the complete suite, compile check, artifact parsing, and Git checks pass.

## Outcome

M24 completed. The frozen M23 model was replayed with zero fit calls; all global,
paired, grouped, calibration, error-review, reporting, and reproducibility outputs
were generated with 2,000 series bootstraps. All 16 blockers and 176 tests passed.
All five LightGBM point metrics beat XGBoost, but every paired 95% interval included
zero. M25 will explain the frozen model without changing it.
