# Ship or clearly disable video transcription in releases

## Summary

Video transcription (Parakeet) is silently absent from every release and from the CUDA container. `scripts/package_release.sh` and `Containerfile.cuda` never bundle an onnxruntime shared library, and `run_imgsearch_cuda_container.sh` never passes `-parakeet-onnxruntime-lib`. `cmd/imgsearch/main.go:216-235` only constructs the transcriber when that flag is set, with no startup warning. The dev resolver `scripts/resolve_onnxruntime_lib.sh` points at a test fixture inside the `onnxruntime_go` module and only exists for arm64, so `mise run serve` fails on Linux x86_64.

## Requirements

- Decide: bundle a pinned onnxruntime for each release platform, or download it on first run like the models, or document that transcription is a source-build feature.
- Log a clear startup line when transcription is disabled because no runtime library was provided.
- Fix the dev resolver so Linux x86_64 works or fails with an actionable message.
- Document the outcome in README and the release `README.txt`.

## Acceptance Criteria

- A fresh release download either transcribes videos or prints one obvious line explaining why it does not.
- `mise run serve` works on Linux x86_64 without manual onnxruntime setup, or the failure message says exactly what to install.

## Notes

- Found during the 2026-09-21 review.
