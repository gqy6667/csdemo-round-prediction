# M17 Implementation Plan

## Scope

Tune the accepted M16 post-first-kill XGBoost using train/validation only. Keep the
M15 artifact, M16 feature contract, grouped split, and external comparison policy fixed.

## Slices

1. Freeze candidate grids, selection threshold, targets, and test-use boundary.
2. Add failing tests for candidate isolation and validation-only selection.
3. Implement sequential search and verify each phase before adding final evaluation.
4. Add seed stability, frozen test evaluation, internal/external comparisons, and reports.
5. Run the full artifact, document the result, commit, and push.

## Risks

- Validation overfitting: retain the incumbent unless Log Loss improves by at least 0.0001.
- Accidental test leakage: tuning functions accept only train/validation prepared frames.
- Misleading external ranking: retain comparability labels and report numeric gaps only.
- Runtime growth: keep the frozen 39-candidate greedy grid and CPU execution.

## Review Gates

- Gate A: the M17 specification exists before model implementation or experiments.
- Gate B: protocol tests fail before the M17 module exists, then pass after each slice.
- Gate C: all 39 candidate rows are validation-only and pass control-variable audit.
- Gate D: test is evaluated only after the eight phase selections and seed audit are frozen.
- Gate E: reports retain target misses, test regressions, and external comparability limits.

## Outcome

M17 passed all five review gates and all 12 blocking checks. Validation Log Loss
improved by 0.002038; the tuned test AUC/Log Loss/Brier are 0.809837/0.523146/0.175656.
Accuracy decreased by 0.001199 and ECE10 worsened by 0.004541, both retained in the
report. M18 will evaluate the frozen model without further parameter selection.
