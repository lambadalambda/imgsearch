# 027: Add HTTP Server Timeouts and Header Limits

## Priority

P1

## Status

Completed.

## Summary

The server currently uses `http.ListenAndServe` directly without explicit timeout and header-size controls.

## Context

- `cmd/imgsearch/main.go` starts HTTP with `http.ListenAndServe(*addr, mux)`.
- No configured `ReadHeaderTimeout`, `ReadTimeout`, `WriteTimeout`, `IdleTimeout`, or `MaxHeaderBytes`.

## Risks

- Increased susceptibility to slow-connection and connection-exhaustion attacks.
- Reduced predictability under adverse network behavior.

## Acceptance Criteria

- [x] Replace `http.ListenAndServe` with explicit `http.Server` configuration.
- [x] Set sane defaults for read/write/header/idle timeout values.
- [x] Set `MaxHeaderBytes` to a bounded value.
- [x] Add startup logging of active HTTP limits.

## Related Files

- `cmd/imgsearch/main.go`
