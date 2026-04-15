# 009 Default Model Download Hardening

## Priority

P2

## Summary

Default model downloads currently use a bare `http.Client` and do not verify integrity. That makes first-run startup behavior more fragile than it should be.

## Why This Matters

- Network hangs can stall startup indefinitely.
- Partial or corrupt downloads are not strongly validated.
- This is painful to debug on first-run setups and packaged environments.

## Current Behavior

- `cmd/imgsearch/default_model_assets.go`
- `downloadFile()` creates `http.Client{}` with no timeout when none is supplied.
- Downloads are streamed to a temp file and renamed into place, which is good.
- There is no checksum validation or stronger integrity verification for default assets.
- `cmd/imgsearch/main.go`
- Startup model resolution uses `context.Background()`.

## Desired Outcome

- Startup downloads fail predictably.
- Users get clearer failure behavior on network problems.
- Default asset downloads have a stronger integrity story.

## Suggested Approach

- Add an HTTP timeout or request deadline.
- Consider checksums for known default assets.
- Preserve the current temp-file-and-rename behavior.

## Acceptance Criteria

- Network stalls no longer hang startup indefinitely.
- Failed downloads leave no partial final files behind.
- Default asset integrity expectations are documented in code.

## Related Files

- `cmd/imgsearch/default_model_assets.go`
- `cmd/imgsearch/default_model_assets_test.go`
- `cmd/imgsearch/main.go`
