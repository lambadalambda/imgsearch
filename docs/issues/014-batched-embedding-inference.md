# 014: Batched Embedding Inference via Multi-Sequence llama.cpp

## Context

Currently each image embedding is processed as a separate forward pass through the model. The llama.cpp context is initialized with `n_seq_max = 1` (`bridge.cc:606`), and the bridge calls `llama_memory_clear()` before each inference.

llama.cpp supports multi-sequence batched inference: set `n_seq_max > 1`, load N images' tokens into the batch with different sequence IDs, and process them in one `llama_decode()` call. Each sequence gets its own embedding via `llama_get_embeddings_seq(lctx, seq_id)`.

For the Qwen3-VL-Embedding-8B at Q4_K_M (~5GB model), processing 2-4 images simultaneously should be feasible on a 24GB GPU, especially after issue 011 frees VRAM from oversized context allocations.

Research consensus: embedding batching could yield 30-70% throughput improvement because the prompt processing (vision encoder + transformer layers) parallelizes well across sequences on GPU. The GPU is compute-bound during the forward pass, and batching increases utilization.

Annotation batching is NOT recommended: autoregressive generation with variable-length output + JSON grammar constraints makes batching extremely complex for marginal (~10-25%) gain.

## Proposed Changes

### A. Increase n_seq_max
Set `cparams.n_seq_max` to a configurable value (default 4 for embedder, keep 1 for annotator).

### B. Multi-sequence embedding in bridge.cc
New C function `imgsearch_llama_embed_images_batch()` that:
1. Accepts an array of image paths (or raw pixel buffers after issue 012)
2. Tokenizes each image into separate chunks with distinct sequence IDs (0, 1, 2, ...)
3. Processes all chunks in one or few `llama_decode()` calls
4. Extracts per-sequence embeddings via `llama_get_embeddings_seq(lctx, seq_id)`
5. L2-normalizes each embedding independently

### C. Batch-aware Go wrapper
New Go method `EmbedImages(ctx, paths) ([][]float32, error)` on the embedder interface. Falls back to sequential if batch size is 1.

### D. Wire to batch claiming
After issue 013's `claimBatch` returns N embed_image jobs, pass all N image paths to the batched embedding method.

### E. Context sizing
`n_ctx` must accommodate all sequences: `n_ctx >= n_seq_max * tokens_per_image`. If each image uses ~500 tokens and n_seq_max=4, n_ctx should be ~2048-4096.

## Risks

- **VRAM pressure**: Each sequence's KV cache requires VRAM. At 4096 dims, n_ctx=2048, n_seq_max=4 with Q8_0 quantization, the KV cache alone could be several hundred MB. Must profile on target hardware.
- **mtmd API limitations**: The current mtmd tokenize API is designed for multi-image single-sequence prompts, not independent multi-sequence batched inference. May need to tokenize each image separately and manually construct the batch with sequence IDs. The mtmd header warns "BREAKING CHANGES are expected."
- **Correctness**: Must verify that batched embeddings are identical (within float precision) to sequential embeddings. Any difference would invalidate existing search results and require re-indexing.
- **Complexity**: This is the highest-complexity optimization. The C++ bridge changes are substantial, and error handling (one image in the batch fails) must be designed carefully.

## Dependencies

- Issue 011 (Flash Attention + context tuning) should be done first to free VRAM and establish the parameter tuning infrastructure.
- Issue 013 (batch claiming) should be done first to provide the batch of jobs.
- Issue 012 (eliminate double preprocessing) is helpful but not blocking.

## Acceptance Criteria

- [ ] `n_seq_max` configurable via CLI flag for the embedder
- [ ] New C function processes 2-4 images in one forward pass
- [ ] New Go method `EmbedImages` on the embedder interface
- [ ] Worker uses batched embedding when multiple embed_image jobs are claimed
- [ ] Batched embeddings match sequential embeddings within float precision tolerance
- [ ] `BenchmarkNativeEmbedImage` extended to compare sequential vs batched throughput
- [ ] VRAM usage profiled at different batch sizes (1, 2, 4, 8)
- [ ] Graceful fallback to sequential if batch size is 1 or if VRAM is insufficient

## Priority

Medium. High potential reward but high complexity. Should be attempted after issues 011-013 are complete and their impact is measured.

## Benchmarking Protocol

### Baseline (before changes)
1. Use the same 100-image dataset from prior benchmarks.
2. Record per-image embedding time (the `embed` stage in worker logs).
3. Record VRAM usage at steady state during sequential embedding.

### After changes
1. Same 100-image dataset.
2. Benchmark at batch sizes 1, 2, 4, and 8 (if VRAM allows).
3. For each batch size, record:
   - Throughput (images/sec)
   - Per-image latency (total batch time / batch size)
   - Peak VRAM usage
4. **Correctness check**: Embed 20 images sequentially, then embed the same 20 images batched. Compare cosine similarity of each pair — must be >= 0.999.

### Success criteria
- Batch size 2-4 shows >= 30% throughput improvement over sequential
- Batched embeddings match sequential within cosine similarity >= 0.999
- VRAM usage fits within 24GB with both models loaded
- Graceful degradation when VRAM is insufficient (falls back to smaller batch or sequential)

## Estimated Effort

Large. Significant changes to bridge.cc (new batch function, multi-sequence management), the Go embedder interface, and the worker queue. 200-400 lines of new/changed code. Requires careful correctness testing.
