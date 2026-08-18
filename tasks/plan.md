# M15 Implementation Plan

## Scope

Repair and accept the post-first-kill dataset built from the repaired three-column round identity.
Do not train or tune a model in this stage.

## Slices

1. Specify the prediction point, valid event rule, key contract, split contract, and gates.
2. Add regression tests for tick ordering and post-kill alive state.
3. Fix first-kill feature construction.
4. Add an auditable M15 command and tests for its contracts.
5. Rebuild the full artifact, run all tests, write reports, and update beginner-facing docs.

## Review Gates

- Gate A: the specification records all assumptions before production code changes.
- Gate B: regression tests fail against the old event selection.
- Gate C: the rebuilt 41,027-row dataset passes every blocking check.
- Gate D: no model metric is reported as an M15 result because M15 does not train a model.
