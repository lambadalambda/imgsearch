# 024: Require API Authentication and Route Guarding

## Priority

P0

## Status

Completed.

## Summary

All `/api/*` routes are currently exposed without authentication. This includes upload, retry, and delete endpoints with direct data and compute impact.

## Context

- `cmd/imgsearch/main.go` registers API handlers directly on `http.ServeMux` with no auth middleware.
- Mutating/expensive routes include:
  - `POST /api/upload`
  - `POST /api/jobs/retry-failed`
  - `DELETE /api/images/{id}`
  - `DELETE /api/videos/{id}`

## Risks

- Any reachable client can upload, delete, or requeue work.
- If bound to non-loopback addresses, this becomes remote unauthorized access.
- Localhost-only setups are still susceptible to browser-driven local request abuse when no token is required.

## Acceptance Criteria

- [x] Introduce auth middleware for `/api/*` routes (token or equivalent).
- [x] Unauthorized requests return `401` and perform no side effects.
- [x] `healthz` and static web UI routes remain accessible without auth by default.
- [x] Configuration and usage are documented for local and networked deployments.

## Related Files

- `cmd/imgsearch/main.go`
- `internal/upload/http.go`
- `internal/jobs/http.go`
- `internal/images/http.go`
- `internal/videos/http.go`
- `README.md`
