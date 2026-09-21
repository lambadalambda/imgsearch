# Extend CI beyond `go test`

## Summary

`ci.yml` runs exactly `go test ./...`. It does not run `go vet`, `gofmt -l` (currently failing on `internal/stats/http.go`), `-race`, `svelte-check`, a frontend build, the nine shell tests under `scripts/`, or the Playwright smoke suites. A broken frontend build is only caught in the release job. AGENTS.md says to run formatting and all tests before every commit, but nothing enforces it.

## Requirements

- Add `gofmt -l` (fail on output), `go vet ./...`, and `go test -race` for the non-cgo packages to CI.
- Install Node, run `mise run check:frontend` and `mise run build:frontend`.
- Run `mise run test:scripts`.
- Run the Atelier Playwright smoke against the built dist.
- Fix the current gofmt failure in `internal/stats/http.go`.

## Acceptance Criteria

- CI fails on an unformatted Go file, a vet warning, a svelte-check error, or a smoke regression.
- CI passes on `master` after the gofmt fix.

## Notes

- Depends on issue 103 for tolerable run times.
- Found during the 2026-09-21 review.
