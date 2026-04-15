# 008 Shared Helper Dedup

## Priority

P2

## Status

Completed for the highest-value shared helpers.

- Shared JSON response helpers now live in `internal/httputil`.
- Shared float32/blob codec helpers now live in `internal/vectorindex`.
- The obvious duplicate helper implementations in handler packages and vector index packages have been removed.
- Remaining low-value duplication such as local tag decoding and pagination parsing can stay local unless a future change gives them a clearer shared home.

## Summary

There are several small but repeated helpers across packages that are now worth centralizing. This is not the highest-priority work, but it will make future changes less noisy and less error-prone.

## Repeated Areas

- float/blob conversion logic in worker and vector index code
- `writeJSON` and `writeJSONError` helpers across handler packages
- tag JSON decoding in more than one handler
- limit/offset parsing overlap
- annotation-missing logic expressed in multiple ways
- repeated native env/task wiring in `mise.toml`

## Why This Matters

- Upcoming performance and pipeline work will touch many of these packages.
- Repeated helpers create drift risk.
- Mechanical refactors are easier before the next feature wave.

## Desired Outcome

- One clear owner per shared helper.
- Less copy-paste across handlers and vector backends.
- Easier review diffs for future behavior changes.

## Suggested Approach

- Extract a small shared HTTP utility package for JSON responses.
- Extract shared vector blob codec helpers.
- Define one canonical annotation-missing rule.
- Consider a small script or shared task wrapper to reduce repeated native env wiring in `mise.toml`.

## Acceptance Criteria

- The most obvious helper duplication is removed.
- Existing tests still pass unchanged or with only mechanical updates.
- Shared logic has a natural home and is not over-abstracted.

All acceptance criteria are satisfied by the current implementation.

## Related Files

- `internal/worker/queue.go`
- `internal/vectorindex/sqlitevector/index.go`
- `internal/vectorindex/bruteforce/index.go`
- `internal/images/http.go`
- `internal/search/http.go`
- `internal/stats/http.go`
- `internal/live/http.go`
- `internal/jobs/http.go`
- `mise.toml`
