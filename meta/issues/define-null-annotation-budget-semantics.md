# Decide explicit null semantics for annotation token settings

## Summary

The settings decoder seeds existing integer budgets before JSON unmarshalling, preserving omitted fields for older clients. Go also treats explicit `null` into an integer as a no-op, so `{"output_tokens":{"image":null}}` currently preserves the saved value instead of rejecting it.

## Context and risks

- Found in independent review of configurable annotation budgets (`internal/settings/http.go`, `decodeUpdate`).
- No invalid budget is persisted; Atelier submits numbers and uses required numeric controls.
- This is API strictness, not a user-facing blocker. Do not change omission compatibility while deciding whether explicit null should be invalid.

## Acceptance Criteria

- Document whether explicit null means omission or is rejected for the output-token object and its individual keys.
- Add focused API tests for the chosen semantics, including unchanged settings version on rejection.

## Notes

- Follow-up only; current omission/null behavior is retained in the configurability change.
