# Raise upload limits and timeouts to fit video uploads

## Summary

`internal/upload/http.go:13-15` caps the whole multipart request at 64 MiB shared by up to 32 files, so most videos exceed it. `ReadTimeout: 30s` (`cmd/imgsearch/main.go:674`) applies to the entire body, so uploads slower than roughly 17 Mbit/s fail with an opaque error. ffmpeg frame sampling runs synchronously in the handler under a 60s `WriteTimeout`.

## Requirements

- Replace the shared request cap with a per-file limit and a larger, configurable cap for video MIME types.
- Replace `ReadTimeout` with `ReadHeaderTimeout` plus per-handler deadlines so large uploads on slow links are not cut off.
- Return a clear `413` with the applicable limit in the JSON error body.

## Acceptance Criteria

- A 200 MB MP4 uploads successfully with default settings.
- Upload handler tests cover the per-file and per-type limits and the 413 body.

## Notes

- Found during the 2026-09-21 review.
