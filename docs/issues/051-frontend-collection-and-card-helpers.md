# 051: Extract Shared Frontend Collection and Card Helpers

## Priority

P2

## Status

Open.

## Summary

The static frontend duplicates image/video collection loading, pagination, rendering, and action markup. Extract shared helpers after browser smoke tests are in place.

## Context

- `internal/webui/static/app.js` is a large monolithic file.
- Image and video pagination/loading flows are similar but separately implemented.
- Card markup mixes data formatting, permissions, action buttons, accessibility labels, and HTML construction in one template.
- `docs/issues/046-ui-smoke-tests.md` should land first to provide a browser-level safety net.

## Risks

- Image and video UI behavior can drift as one path receives fixes first.
- Large template literals increase XSS and accessibility regression risk when new fields are added.
- Frontend changes are difficult to review because unrelated UI concerns live in one file.

## Acceptance Criteria

- [ ] Extract shared collection pagination/loading helpers for images and videos.
- [ ] Split card markup into smaller helpers or safer DOM/template utilities.
- [ ] Keep current image, video, tag, and result UI behavior covered by browser tests.
- [ ] Preserve existing no-build static asset serving unless a separate build-tool decision is made.
- [ ] Keep the refactor incremental enough to review without visual churn.

## Related Files

- `internal/webui/static/app.js`
- `internal/webui/static/index.html`
- `internal/webui/static/styles.css`
- `internal/webui/http_test.go`
- `package.json`
