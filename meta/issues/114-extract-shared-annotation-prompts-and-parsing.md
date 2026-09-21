# Extract shared annotation prompts and response parsing

## Summary

The system/user prompts (`internal/embedder/llamacppnative/gemma_runtime_native.go:34-45, 95-260`) and the JSON response decoding with `extractJSONObject` fallback (lines 630-700, 905) are private to the native package. A remote annotator must send the same prompts and parse the same response shape, so this logic needs a backend-neutral home.

## Requirements

- New package `internal/annotation` (or similar) exporting: prompt builders for image, video frame, and video (including retry prompts), the response structs, and `Parse*` functions that accept raw model text and return `embedder.ImageAnnotation` / `embedder.VideoAnnotation` with the existing normalisation (lowercase unique tags, NSFW tag injection, title/summary derivation).
- Native runtime calls the shared package; no behaviour change.
- Pure refactor: existing native tests keep passing unchanged or with import-only edits.

## Acceptance Criteria

- Unit tests for the parsers cover clean JSON, fenced JSON, prose-wrapped JSON, and invalid output.
- `go test ./...` passes; native annotation integration tests (build-tagged) are untouched.

## Notes

- Subissue of [113](113-annotation-backend-settings-page.md). Do this first; 116 depends on it.
