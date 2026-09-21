# Cache llama.cpp and libvips builds in CI

## Summary

`.github/workflows/ci.yml` and `rolling-release.yml` download and meson-build libvips 8.18 and cmake-build llama.cpp from source on every run with no `actions/cache`. Recent runs: CI 21 to 36 minutes, rolling release 30 to 71 minutes. The two workflows also duplicate about 40 lines of identical dependency-install steps.

## Requirements

- Cache `deps/llama.cpp/build` keyed on the submodule SHA plus `IMGSEARCH_LLAMA_CMAKE_ARGS` and the runner OS.
- Cache the built libvips install keyed on the libvips version and runner OS.
- Move the shared dependency steps into a composite action used by both workflows.
- Add `shallow = true` to `.gitmodules` for `deps/llama.cpp` so CI does not fetch full upstream history.

## Acceptance Criteria

- A CI run with a warm cache completes in under 10 minutes.
- Both workflows still produce working artifacts after a cold cache.

## Notes

- Found during the 2026-09-21 review; largest developer-experience win available.
