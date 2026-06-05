# 054: Document Trusted UI API Cookie Model

## Priority

P1

## Status

Open.

## Summary

The API authentication cookie is intentionally minted for anonymous non-API page loads and is accepted by `/api/*`. The chosen model is: anyone who can reach the UI is trusted to use the API. This needs explicit documentation, tests, and exposure warnings so it is not mistaken for a separate API-auth boundary.

## Context

- `cmd/imgsearch/main.go` wraps the handler with `NewAPIAuthMiddleware` and then `NewAPIKeyCookieMiddleware`.
- `internal/httputil/auth.go` issues `imgsearch_api_key` on any non-API path when an API key is configured.
- The API middleware accepts that cookie as equivalent to the configured API key.
- Decision: remote UI access is trusted. Anyone who can reach the UI can use the API.
- On a non-loopback deployment, a visitor can request `/`, receive the cookie, and then call protected API routes by design.

## Risks

- Operators may incorrectly assume a configured API key protects browser visitors from using the UI API.
- The UI and API security model can look like a bug unless it is documented as an intentional trust boundary.
- Future API routes may assume route guarding means remote UI visitors are untrusted.

## Acceptance Criteria

- [ ] Add tests proving the intended behavior: non-API UI requests mint the cookie and subsequent same-origin API calls succeed.
- [ ] Add docs/warnings that exposing the UI exposes API capability to anyone who can reach it.
- [ ] Keep non-loopback startup requiring an explicit API key so accidental default-key exposure remains blocked.
- [ ] Ensure API-header auth for scripts/imports still works independently of browser cookies.

## Related Files

- `cmd/imgsearch/main.go`
- `internal/httputil/auth.go`
- `internal/webui/http_test.go`
