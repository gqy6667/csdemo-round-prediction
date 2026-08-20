# M21 Implementation Plan

## Scope

Close the post-first-kill XGBoost track with a no-training final acceptance runner,
a three-mode PowerShell reproduction entrypoint, and a data-backed M6-to-M21 progress
report. Preserve all frozen M17-M20 probabilities, metrics, hashes, and target margins.

## Slices

1. Freeze the M21 evidence, blocker, replay, reproduction, and progress-report contracts.
2. Add failing tests for stage continuity, split isolation, replay tolerance, script coverage,
   acceptance decisions, and progress comparability labels.
3. Implement the acceptance primitives and prove the focused tests green.
4. Implement the real-artifact runner, experiment manifest, final report, and one-click script.
5. Commit executable M21 code, run the formal acceptance from that clean commit, then commit
   generated evidence and documentation.

## Risks

- Artifact drift: compare bytes and SHA-256 across M15-M20 and the loaded bundles.
- Hidden split leakage: audit series, game, and round overlap directly from the Parquet data.
- Metric drift: replay all 4,170 test probabilities and compare at `1e-12` tolerance.
- Circular testing: unit tests call small functions; the formal runner invokes the full suite once.
- Misleading progress: label same-task comparisons separately from prediction-time changes.
- Dirty manifest: commit code before the formal run so the manifest records a stable revision.

## Review Gates

- Gate A: the M21 specification exists before tests or implementation.
- Gate B: focused M21 tests fail because the acceptance module does not yet exist.
- Gate C: focused tests pass before the formal real-artifact run.
- Gate D: the formal run passes 17/17 blockers without calling `fit()`.
- Gate E: the complete suite, compile check, JSON parsing, and Git diff checks pass.

## Outcome

M21 completed. All 17 blockers passed, all 4,170 frozen test probabilities replayed
within `1.11e-16`, all five metrics matched M18 exactly, and all ten formal targets
retained zero remaining gap. The full suite reached 145 passing tests with zero fit
calls. The post-first-kill XGBoost track is closed and ready for the LightGBM
controlled comparison.
