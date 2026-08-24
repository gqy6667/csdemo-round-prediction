# M23 Implementation Plan

## Scope

Tune the accepted M22 pre-round LightGBM baseline with a validation-only greedy
sequential search. Keep data, grouped split, feature encoding, test rows, metrics,
and M22/XGBoost references fixed.

## Slices

1. Freeze the nine tuning phases, 36 candidates, selection threshold, seed limits,
   stage goals, test-use policy, and blockers.
2. Add failing tests for grid integrity, validation-only signatures and tables,
   deterministic selection, frozen parameters, seed stability, and exact test keys.
3. Implement sequential search and prove focused logic tests green.
4. Run the real search, freeze seed 42, evaluate test once, and generate all evidence.
5. Run the complete suite and compile checks, update learning docs, commit, and push.

## Risks

- Test leakage: search and seed functions accept only train/validation and output no test columns.
- Greedy-order dependence: freeze phase order and preserve every rejected candidate.
- Tiny noise gains: require at least 0.0001 validation Log Loss improvement per phase.
- Seed instability: enforce five-seed Log Loss and AUC range gates before test evaluation.
- Test over-interpretation: report M23 versus M22/XGBoost even when tuning loses.
- Artifact drift: verify M22 data hash, encoded columns, test keys, and baseline metrics.

## Review Gates

- Gate A: the M23 specification exists before tests or implementation.
- Gate B: focused M23 tests fail because the module does not yet exist.
- Gate C: focused tests pass before the formal real-artifact run.
- Gate D: all tuning tables remain validation-only and the formal run passes 14 blockers.
- Gate E: the complete suite, compile check, artifact parsing, and Git checks pass.

## Outcome

In progress. M23 will preserve every candidate result and select only by validation
Log Loss before a single frozen-model test evaluation.
