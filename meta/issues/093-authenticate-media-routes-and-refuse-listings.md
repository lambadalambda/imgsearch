# Authenticate `/media/` routes and refuse directory listings

## Summary

API key auth only guards `/api/*` (`internal/httputil/auth.go:114-116`). The media file server mounted in `internal/webui/http.go:72-76` is unauthenticated and `http.FileServer` serves directory indexes, so `GET /media/images/` returns an HTML list of every stored file. On a non-loopback bind with a strong API key, anyone who can reach the port can still enumerate and download the whole library.

## Requirements

- Put `/media/` behind the same auth middleware as `/api/*`. UI visitors already receive the `imgsearch_api_key` cookie, so the Atelier and legacy UIs keep working without changes.
- Refuse directory paths (trailing slash or any path resolving to a directory) with `404`, for both the image and video subtrees.
- Update the README "Security Model" section to state that media routes are covered by the trust boundary.

## Acceptance Criteria

- `GET /media/images/` and `GET /media/videos/` return `404` with no listing.
- `GET /media/images/<sha>` without a cookie or API key header returns `401`; with either, it returns the file.
- Existing UI smoke tests still pass (media loads in the grid and lightbox).

## Notes

- Found during the 2026-09-21 review; highest-impact finding.
