# 015: Pipeline CPU Preprocessing with GPU Inference

## Status: Completed

## What Landed

- `EmbedImages` now pipelines preprocessing and embedding within each claimed chunk so preprocessing for item N+1 can run while item N is embedding.
- Added cancellation-safe producer/consumer cleanup in the native embedder pipeline.
- Added serial fallback for chunk processing when pipeline preprocessing fails, so prefetch-path failures can degrade gracefully instead of aborting immediately.
- Split `EmbedImage` into explicit prepare and execute phases (`prepareImageForEmbedding` + `executePreparedImage`/`executePreparedImageLocked`) so CPU preprocessing is isolated from mutex-guarded native embedding execution.
- Added focused pipeline tests for ordering, overlap behavior, cleanup on embed failure, preprocess-error propagation, and context cancellation.

## Context

The current pipeline is strictly serial within each job:
1. CPU: libvips thumbnail + JPEG encode → temp file (preprocessing)
2. GPU: llama.cpp forward pass (inference)
3. CPU: SQLite write (DB)

While the GPU processes image N, the CPU is idle. While the CPU preprocesses image N, the GPU is idle. These could overlap.

However, research suggests the expected speedup is modest (~5-15%) because:
- libvips preprocessing at 384px or 1024px is likely 5-15ms per image
- GPU inference for the 8B embedding model is likely 100-400ms per image
- The preprocessing is ~10x faster than inference, so the pipeline would be GPU-bound regardless
- The CPU would spend most of its time waiting for the GPU

After issue 012 eliminates the double preprocessing, the CPU-bound portion shrinks further.

## Proposed Changes

### A. Prefetch goroutine
Add a goroutine that preprocesses the next image while the current image is being processed by the GPU:

1. Worker claims a batch of jobs (from issue 013)
2. Prefetch goroutine preprocesses image N+1 (libvips → raw pixels)
3. When GPU finishes image N, immediately start image N+1 (already preprocessed)
4. Prefetch goroutine starts on image N+2

### B. Restructure EmbedImage/AnnotateImage
Split the current methods into "prepare" and "execute" phases:
- `PrepareImage(path) → PreparedImage` (CPU only, no mutex needed)
- `EmbedImage(prepared) → []float32` (GPU only, requires mutex)

This separation is also needed for issue 014 (batched inference).

### C. Double-buffering
Maintain two slots: one being processed by GPU, one being filled by CPU prefetch. This is a simple producer-consumer pattern with a channel.

## Risks

- **Low expected return**: If preprocessing is 5-15ms and inference is 100-400ms, the pipeline eliminates only the preprocessing time from the total, giving a 3-10% wall-clock improvement.
- **Complexity**: Adds concurrency to a currently simple serial loop. Must handle errors in the prefetch goroutine, cancellation, and shutdown gracefully.
- **Mutex interaction**: The embedder/annotator use `sync.Mutex`. The prefetch goroutine must not hold the mutex during preprocessing.

## Dependencies

- Issue 012 (eliminate double preprocessing) should be done first to simplify the preprocessing step.
- Issue 013 (batch claiming) should be done first to provide multiple jobs to pipeline.
- Issue 014 (batched inference) may subsume this: if we batch 4 images in one GPU call, pipelining becomes less relevant because the GPU utilization is already higher.

## Acceptance Criteria

- [x] `EmbedImage` split into prepare + execute phases
- [x] Prefetch goroutine overlaps CPU preprocessing with GPU inference
- [x] Measure wall-clock improvement on a batch of 50 images
- [x] Error handling for prefetch failures (fall back to serial)
- [x] Clean shutdown of prefetch goroutine on context cancellation

## Benchmark Results (50-image workload)

Dataset and setup:

- Dataset path: `/tmp/imgsearch-bench-data-50-issue015` (50 symlinked images from `~/old`).
- Model: `Qwen3-VL-Embedding-8B-Q4_K_M.gguf` + `mmproj-Qwen3-VL-Embedding-8B-f16.gguf`.
- Runtime config: `LLAMA_NATIVE_CONTEXT_SIZE=512`, `LLAMA_NATIVE_BATCH_SIZE=512`, `LLAMA_NATIVE_IMAGE_MAX_SIDE=384`.
- Comparison points:
  - Baseline (before overlap): commit `cd5e024`
  - Current (with overlap): commit `33d84e2`

Benchmark commands:

```bash
RUN_LLAMACPP_NATIVE_INTEGRATION=1 \
LLAMA_NATIVE_MODEL_PATH="/Users/lainsoykaf/repos/imgsearch/models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf" \
LLAMA_NATIVE_MMPROJ_PATH="/Users/lainsoykaf/repos/imgsearch/models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf" \
LLAMA_NATIVE_DIMS=4096 \
LLAMA_NATIVE_CONTEXT_SIZE=512 \
LLAMA_NATIVE_BATCH_SIZE=512 \
LLAMA_NATIVE_IMAGE_MAX_SIDE=384 \
LLAMA_NATIVE_BENCH_IMAGE_DIR="/tmp/imgsearch-bench-data-50-issue015" \
LLAMA_NATIVE_BENCH_IMAGE_LIMIT=50 \
go test ./internal/embedder/llamacppnative -run '^$' -bench 'BenchmarkNativeEmbedImage/batch-4$' -benchtime=13x -count=3 -benchmem

RUN_LLAMACPP_NATIVE_INTEGRATION=1 \
LLAMA_NATIVE_MODEL_PATH="/Users/lainsoykaf/repos/imgsearch/models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf" \
LLAMA_NATIVE_MMPROJ_PATH="/Users/lainsoykaf/repos/imgsearch/models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf" \
LLAMA_NATIVE_DIMS=4096 \
LLAMA_NATIVE_CONTEXT_SIZE=512 \
LLAMA_NATIVE_BATCH_SIZE=512 \
LLAMA_NATIVE_IMAGE_MAX_SIDE=384 \
LLAMA_NATIVE_BENCH_IMAGE_DIR="/tmp/imgsearch-bench-data-50-issue015" \
LLAMA_NATIVE_BENCH_IMAGE_LIMIT=50 \
go test ./internal/embedder/llamacppnative -run '^$' -bench 'BenchmarkNativeEmbedImage/batch-2$' -benchtime=25x -count=3 -benchmem
```

Observed results (mean across 3 runs):

- `batch-4`: `3,264,976,235 ns/op` -> `2,922,153,066 ns/op` (**10.5% faster**)
- `batch-2`: `1,471,853,436 ns/op` -> `1,407,293,460 ns/op` (**4.4% faster**)

Estimated 50-image wall-clock from benchmark means:

- `batch-4` equivalent: ~`40.8s` -> ~`36.5s` (about `4.3s` faster)
- `batch-2` equivalent: ~`36.8s` -> ~`35.2s` (about `1.6s` faster)

Conclusion:

- The overlap change shows a measurable throughput gain on this 50-image workload.
- All acceptance criteria are now satisfied.

## Priority

Low. The effort-to-reward ratio is poor given that preprocessing is likely a small fraction of total time. Revisit if profiling shows preprocessing is more significant than expected, or if issue 014's batched inference makes pipelining the next bottleneck.

## Benchmarking Protocol

### Baseline (before changes)
1. Use the same 100-image dataset from prior benchmarks.
2. Record total wall-clock time for all embedding jobs.
3. From worker logs, extract the `embed` stage timing and `load`/`stat` stage timing to quantify the preprocessing fraction.

### After changes
1. Same 100-image dataset.
2. Record total wall-clock time and per-stage timings.
3. Compare: was preprocessing actually a significant fraction? If preprocessing was < 5% of total time, the pipeline improvement will be negligible and this confirms the low-priority assessment.

### Success criteria
- Measurable wall-clock improvement on 100-image batch (even 3-5% would validate the approach)
- If improvement is < 3%, document findings and consider cancelling further work on this issue

## Estimated Effort

Medium. Requires restructuring the embedder interface, adding a prefetch goroutine, and handling the concurrency. ~100-150 lines changed. Best implemented after issues 012-014 provide the infrastructure.
