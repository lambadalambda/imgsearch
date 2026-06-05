# 058: Align Atelier Tag All-Mode with Backend Parameters

## Priority

P1

## Status

Open.

## Summary

The Atelier frontend sends `tag_mode=all`, but the backend `/api/search/tags` handler reads `mode=all`. The UI all-tags mode therefore behaves like any-tags mode against the real backend.

## Context

- `frontend/src/lib/api.ts` sets `tag_mode` for tag searches.
- `internal/search/http.go` reads `mode` to decide whether all tags must match.
- Existing backend tests exercise `mode=all`.
- The Atelier smoke stub reads `tag_mode`, which masks the real API contract mismatch.

## Risks

- Users asking for all selected tags get broader any-tag results.
- Smoke coverage can pass while production behavior is wrong.
- Future clients may copy the wrong parameter name.

## Acceptance Criteria

- [ ] Add a regression test that proves Atelier/typed API all-mode reaches backend all-mode.
- [ ] Either make the backend accept `tag_mode` as an alias or change the frontend to send `mode`.
- [ ] Update smoke stubs to match the real backend parameter contract.
- [ ] Preserve existing `mode=all` backend compatibility.

## Related Files

- `frontend/src/lib/api.ts`
- `internal/search/http.go`
- `internal/search/http_test.go`
- `scripts/ui_smoke_test_atelier.mjs`
