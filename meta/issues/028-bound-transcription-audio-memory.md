# 028: Bound Transcription Audio Memory and Feature Allocation

## Priority

P1

## Status

Completed.

## Summary

Video transcription currently buffers ffmpeg PCM output in memory and then allocates feature tensors proportional to full audio length.

## Context

- `internal/transcribe/parakeetonnx/audio.go` uses `cmd.Output()` for ffmpeg PCM extraction.
- `internal/transcribe/parakeetonnx/features.go` allocates feature arrays based on frame count with no hard upper bound.

## Risks

- Memory exhaustion on long videos.
- Unbounded CPU/memory workload from large transcription jobs.

## Acceptance Criteria

- [x] Add explicit max-duration or max-sample guardrails before feature extraction.
- [x] Avoid full-output buffering when extracting PCM (stream or bounded read strategy).
- [x] Return a deterministic, user-visible error when limits are exceeded.
- [x] Add tests covering over-limit inputs.

## Related Files

- `internal/transcribe/parakeetonnx/audio.go`
- `internal/transcribe/parakeetonnx/audio_test.go`
- `internal/transcribe/parakeetonnx/features.go`
- `internal/transcribe/transcribe.go`
