# Clarify token-draft validation for remote annotation probes

## Summary

Settings Save uses native form validation for all six output-token limits, but the Test connection and Fetch models buttons bypass form submission and send the current draft, including output-token fields. A blank budget is omitted while an out-of-range budget is rejected by the API, even though neither probe uses annotation generation budgets.

## Requirements

- Decide whether remote probes should validate the entire draft or omit unrelated output-token fields.
- Keep saving strict and preserve existing API-key handling.

## Acceptance Criteria

- Consistent, understandable behavior for blank and out-of-range budget drafts when running either remote probe.
- UI/API regression tests cover the chosen behavior; no invalid budget can be persisted.

## Notes

- Nonblocking UI review follow-up from configurable token budgets. Save validation and persistence are already covered; no production configuration changed.
- UX05 review also found stale connection-probe feedback: unlike model fetching, an in-flight test response can reappear after the URL/model is edited or discarded. Guard probe responses by current draft/mount identity and add a deferred-response test when addressing this issue.
- The UX05 isolated smoke fixture simplifies stored-key preservation (`api_key_set` from the current request only). Correct that fixture before expanding credential-save tests; current retention cases do not exercise a subsequent blank-key save.
