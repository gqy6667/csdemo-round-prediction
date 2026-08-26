# M27-M28 Implementation Plan

## Scope

First close the frozen pre-round LightGBM line with M27 final acceptance. Then create
independent teacher-review reports for the accepted pre-round XGBoost, pre-round
LightGBM, and post-first-kill XGBoost lines. Only after those reports are verified,
start M28 by replacing the accepted first-kill M21 XGBoost algorithm with a fixed
LightGBM baseline while retaining data, grouped split, prediction point, features,
and metrics. The fourth report must use completed LightGBM evidence, never placeholders.

## Slices

1. Freeze M27 inputs, blockers, reproduction modes, outputs, and no-fit policy.
2. Add failing M27 contract tests, implement replay and acceptance, then run the
   formal real-artifact gate.
3. Commit and push M27 before opening the first-kill LightGBM stage.
4. Build and verify three independent frozen-result reports with one consistent
   review structure; do not compare different prediction times as algorithm effects.
5. Freeze M28 baseline parameters, metrics, paired-series uncertainty, and
   acceptance thresholds before training.
6. Add failing M28 tests, implement the controlled baseline, train using train with
   validation-only early stopping, and evaluate test exactly once.
7. Complete the post-first-kill LightGBM evaluation and acceptance, then write its
   independent report and the teacher review index.
8. Run focused/full tests and compile checks, verify manifests and links, document,
   commit, and push the report deliverables and LightGBM stages.

## Risks

- Test leakage: no test metric can appear in M27 replay selection or M28 training.
- Contract drift: require exact M21 first-kill rows, keys, split, and feature order.
- False superiority: paired confidence intervals govern claims, not point metrics.
- Artifact drift: hash model, calibrator, data, summaries, and outputs.
- Scope mixing: M27 must pass and be committed before M28 implementation begins.
- Report drift: every number and hash must trace to a frozen machine-readable artifact.
- Timing confusion: pre-round and post-first-kill metrics are not fair algorithm comparisons.

## Review Gates

- Gate A: M27 specification exists before M27 tests and code.
- Gate B: M27 focused tests fail for the missing module, then pass after implementation.
- Gate C: M27 formal artifacts, hashes, full suite, compile, commit, and push pass.
- Gate D: the first three teacher reports agree with their accepted source artifacts.
- Gate E: M28 specification exists before training code or model fitting.
- Gate F: the fourth report is created only after completed M28+ evidence exists.
- Gate G: all four reports and the index pass number, hash, link, and test checks.

## Outcome

M27, the first three teacher reports, and M28-M30 are complete. M31 frozen-model
post-first-kill LightGBM explanation is next; the fourth report remains blocked on
explanation, interface, and final acceptance evidence.
