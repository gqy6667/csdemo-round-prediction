# M19 Implementation Plan

## Scope

Explain the accepted M17/M18 post-first-kill XGBoost without retraining it. Reuse the
M12 native TreeSHAP implementation, add first-kill-specific feature timing audit, raw
feature grouped permutation, formal target distance, and external metric gaps.

## Slices

1. Freeze explanation methods, leakage whitelist, target-gap formulas, and no-training boundary.
2. Add failing tests for encoded-to-source mapping, grouped permutation, leakage, and gaps.
3. Implement the deterministic contracts and verify focused tests.
4. Add Gain, encoded/grouped permutation, TreeSHAP, cases, plots, and reports.
5. Run the full artifact, document target/external gaps, commit, and push.

## Risks

- Misusing test explanations for selection: M19 contains no model fit or feature selection.
- Incorrect categorical aggregation: require each encoded column to map exactly once.
- Explaining unused trees: limit Gain and TreeSHAP to `best_iteration + 1` trees.
- Misleading external ranking: retain comparability labels and report numeric gaps only.
- Runtime growth: use 20 deterministic repeats and XGBoost native TreeSHAP on 4,170 rows.

## Review Gates

- Gate A: the M19 specification exists before explanation code or experiments.
- Gate B: mapping, leakage, grouped permutation, and target-gap tests fail before implementation.
- Gate C: all 82 encoded columns map to exactly 40 accepted source features.
- Gate D: TreeSHAP reconstructs deployed probabilities and no XGBoost training occurs.
- Gate E: reports retain target margins, internal non-superiority, and external limitations.

## Outcome

Completed on 2026-08-20. M19 passed all 9 blocking checks, all 10 formal target
gates, and 118 automated tests. The frozen M17/M18 model was explained without
training; all 82 encoded columns map to 40 permitted raw features with zero leakage
failures. M20 is the next stage.
