# 040: Narrow Native Runtime Locks Around Preprocessing

## Priority

P2

## Status

Open.

## Summary

Native embedder and annotator paths hold runtime mutexes while preprocessing images. The lock should protect native llama calls and handle lifecycle, not CPU/libvips preprocessing.

## Context

- Single-image embedding preprocesses outside the mutex before `executePreparedImage`.
- Batch embedding in `internal/embedder/llamacppnative/embedder_native.go:318` locks before `runEmbedChunk`, and `runEmbedChunk` performs preprocessing through `prepareImageForEmbedding`.
- Annotation calls in `internal/embedder/llamacppnative/gemma_runtime_native.go:430` and `467` hold the runtime mutex before calling `describeAndTagImageWithHandle`.
- `generateImageJSONForHandle` preprocesses at `internal/embedder/llamacppnative/gemma_runtime_native.go:653`.

## Risks

- Slow preprocessing blocks unrelated text/image operations on the same runtime.
- Shutdown and model switching can wait behind CPU preprocessing work.
- Batch embedding gains are limited by serialized preprocessing.

## Acceptance Criteria

- [ ] Preprocess batch images outside the embedder mutex.
- [ ] Preprocess annotation images outside the Gemma runtime mutex while preserving safe handle checks.
- [ ] Keep native C calls serialized where required by llama.cpp runtime constraints.
- [ ] Add tests or targeted integration coverage that cancellation/close behavior remains safe.
- [ ] Benchmark or log before/after impact for batch preprocessing overlap if practical.

## Related Files

- `internal/embedder/llamacppnative/embedder_native.go`
- `internal/embedder/llamacppnative/gemma_runtime_native.go`
- `internal/embedder/llamacppnative/embedder_native_context_test.go`
- `internal/embedder/llamacppnative/embedder_pipeline_test.go`
