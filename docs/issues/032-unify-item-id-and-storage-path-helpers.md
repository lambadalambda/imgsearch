# 032: Unify Item ID Parsing and Stored-Path Removal Helpers

## Priority

P2

## Status

Open.

## Summary

`parseItemID` and `removeStoredPath` are duplicated across images and videos handlers. This is a good consolidation point and the right place to harden path safety checks.

## Context

Duplicated helper implementations currently live in:

- `internal/images/http.go`
- `internal/videos/http.go`

## Risks

- Duplicate logic can diverge over time.
- Path cleanup helpers currently do not enforce an explicit `dataDir` containment check.

## Acceptance Criteria

- [ ] Introduce shared helper(s) for item ID parsing and stored-path removal.
- [ ] Enforce safe path resolution before file deletion.
- [ ] Replace duplicate helper implementations in both handlers.
- [ ] Add tests for invalid ID paths and deletion path containment.

## Related Files

- `internal/images/http.go`
- `internal/videos/http.go`
- `internal/httputil/`
