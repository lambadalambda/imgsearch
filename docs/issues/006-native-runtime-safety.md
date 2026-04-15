# 006 Native Runtime Safety

## Priority

P2

## Summary

The native runtime layer has a few safety gaps that are not always visible today because the app currently uses a single worker goroutine. These include annotator handle synchronization and incomplete use of `context.Context`.

## Why This Matters

- Future batching or multiple workers can expose data races.
- Long-running native calls currently ignore cancellation.
- Performance work will likely increase pressure on these code paths.

## Current Behavior

- `internal/embedder/llamacppnative/gemma_runtime_native.go`
- `nativeGemmaRuntime` does not currently mirror the mutex protection used by the main native embedder.
- Some helper paths use `context.Background()` instead of the passed context.
- `internal/embedder/llamacppnative/embedder_native.go`
- Embed operations largely ignore cancellation.

## Desired Outcome

- Native handles are protected consistently.
- Cancellation behavior is explicit and as strong as the native bridge allows.
- The code does not silently discard caller context.

## Suggested Approach

- Add synchronization to the separate Gemma annotator runtime.
- Remove avoidable `context.Background()` substitutions.
- If true mid-inference cancellation is not possible, document that clearly and still honor cancellation at the Go boundary where possible.

## Acceptance Criteria

- The annotator runtime has consistent handle-safety semantics.
- No obvious `context.Background()` overrides remain in request/worker-driven paths.
- Tests or comments make the remaining cancellation limitations explicit.

## Related Files

- `internal/embedder/llamacppnative/embedder_native.go`
- `internal/embedder/llamacppnative/gemma_runtime_native.go`
- `internal/embedder/llamacppnative/embedder_stub.go`
