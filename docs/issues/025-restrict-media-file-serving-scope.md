# 025: Restrict `/media/` File Serving Scope

## Priority

P0

## Status

Completed.

## Summary

The web UI currently mounts `/media/` to the full `dataDir`, which can expose non-media files such as SQLite databases and temporary artifacts.

## Context

- `internal/webui/http.go` uses:
  - `http.FileServer(http.Dir(mediaRoot))`
  - mounted under `/media/`
- `mediaRoot` is the full data directory.

## Risks

- Accidental or unauthorized read access to `imgsearch.sqlite` and other internal files.
- Increased blast radius when the service is reachable over a network.

## Acceptance Criteria

- [x] `/media/` serves only explicit media subdirectories (at minimum `images/` and `videos/`).
- [x] Requests for non-media paths (DB, WAL, tmp, etc.) return `404`.
- [x] Existing UI media rendering continues to work.
- [x] Add regression tests for allowed and denied paths.

## Related Files

- `internal/webui/http.go`
- `internal/webui/http_test.go`
- `internal/webui/static/app.js`
