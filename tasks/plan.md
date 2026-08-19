# M20 Implementation Plan

## Scope

Wrap the accepted M17/M18 post-first-kill XGBoost and identity calibrator in a strict
one-snapshot JSON/CSV inference interface. Reuse the M13 purchase-state validation,
add first-kill event and artifact contracts, and keep M19 metrics unchanged.

## Slices

1. Freeze the input, output, model, calibrator, error, and no-training contracts.
2. Add failing tests for event validation, bundle association, prediction, CLI, and artifacts.
3. Implement one-row validation and prediction as the first vertical slice.
4. Add M20 prerequisite checks, invalid cases, external comparison, and Chinese report.
5. Run the formal artifact, document usage and remaining work, commit, and push.

## Risks

- Silent all-zero categories: reject maps and weapons absent from saved encoded columns.
- Model/calibrator drift: verify SHA-256 and task/data association before predicting.
- Redundant or future fields: allow only the documented 31 base and 9 optional difference fields.
- CSV type drift: normalize numeric and boolean event values through the same validator as JSON.
- Misleading example probability: retain fixed test metrics and external comparability labels.

## Review Gates

- Gate A: the M20 specification exists before interface code.
- Gate B: first-kill validation and bundle tests fail before implementation.
- Gate C: valid JSON/CSV produce identical finite complementary probabilities.
- Gate D: every frozen invalid example is rejected with a specific message.
- Gate E: model metrics, ten target margins, and external limitations remain unchanged.

## Outcome

Pending M20 implementation.
