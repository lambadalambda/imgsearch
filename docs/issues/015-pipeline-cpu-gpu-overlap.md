# 015: Pipeline CPU Preprocessing with GPU Inference

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

- [ ] `EmbedImage` split into prepare + execute phases
- [ ] Prefetch goroutine overlaps CPU preprocessing with GPU inference
- [ ] Measure wall-clock improvement on a batch of 50 images
- [ ] Error handling for prefetch failures (fall back to serial)
- [ ] Clean shutdown of prefetch goroutine on context cancellation

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
