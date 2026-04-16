# 014: Batched Embedding Inference via Multi-Sequence llama.cpp

## Status: Partially Landed, Issue Still Open

## What Landed

- Added `embedder.BatchImageEmbedder` with `EmbedImages(ctx, paths)` so the worker can hand multiple image paths to an embedder in one call.
- Updated the worker to use `EmbedImages` when multiple `embed_image` jobs are claimed.
- Added `-llama-native-max-sequences` as a Go-side chunking control for `EmbedImages`.
- Added a native correctness test that verifies `EmbedImages` returns embeddings matching `EmbedImage` within cosine similarity `>= 0.999`.
- Extended `BenchmarkNativeEmbedImage` with `sequential`, `batch-2`, and `batch-4` sub-benchmarks.

## What Did Not Land

- A stable native multi-sequence llama.cpp bridge did not ship.
- An experimental C++ bridge implementation was attempted, but it was not stable enough to keep enabled on this branch.
- The shipped native path remains single-sequence under the hood, with `EmbedImages` executing safely via the existing single-image bridge.

## Benchmark Results

Benchmark command used:

```bash
RUN_LLAMACPP_NATIVE_INTEGRATION=1 \
LLAMA_NATIVE_MODEL_PATH="/Users/lainsoykaf/repos/imgsearch/models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf" \
LLAMA_NATIVE_MMPROJ_PATH="/Users/lainsoykaf/repos/imgsearch/models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf" \
LLAMA_NATIVE_DIMS=4096 \
LLAMA_NATIVE_CONTEXT_SIZE=512 \
LLAMA_NATIVE_BATCH_SIZE=512 \
LLAMA_NATIVE_IMAGE_MAX_SIDE=384 \
LLAMA_NATIVE_BENCH_IMAGE_DIR="/tmp/imgsearch-bench-data" \
LLAMA_NATIVE_BENCH_IMAGE_LIMIT=100 \
go test ./internal/embedder/llamacppnative -run '^$' -bench BenchmarkNativeEmbedImage -benchtime=10x -count=3 -benchmem
```

Dataset setup:

```bash
mkdir -p /tmp/imgsearch-bench-data && shuf -zn 100 -e ~/old/* | xargs -0 -I{} cp "{}" /tmp/imgsearch-bench-data/
```

Results on the M2 Max:

- `sequential`
  - `758722204 ns/op`
  - `785352488 ns/op`
  - `792620000 ns/op`
  - average: about `779 ms/image`
- `batch-2`
  - `1566811846 ns/op`
  - `1561692875 ns/op`
  - `1585275862 ns/op`
  - average: about `786 ms/image`
- `batch-4`
  - `3324163746 ns/op`
  - `3533542592 ns/op`
  - `3552332442 ns/op`
  - average: about `868 ms/image`

Because `batch-2` and `batch-4` still route through the safe single-image native bridge internally, there is no throughput win yet.

## Correctness Result

- `TestNativeBatchEmbeddingsMatchSequential` passes with cosine similarity `>= 0.999` between `EmbedImage` and `EmbedImages` outputs.

## Conclusion

- The Go API and worker plumbing needed for future batched embedding work are now in place.
- The actual native multi-sequence optimization remains unresolved and this issue should stay open.
- The current shipped implementation is intentionally conservative: correct and safe, but not faster.

## Acceptance Criteria

- [ ] `n_seq_max` configurable via CLI flag for the native embedder runtime itself
- [ ] New C function processes 2-4 images in one forward pass
- [x] New Go method `EmbedImages` on the embedder interface
- [x] Worker uses batched embedding entrypoint when multiple `embed_image` jobs are claimed
- [x] `EmbedImages` results match `EmbedImage` within cosine similarity tolerance
- [x] `BenchmarkNativeEmbedImage` compares sequential vs grouped image embedding calls
- [ ] VRAM usage profiled at different native batch sizes (1, 2, 4, 8)
- [x] Graceful fallback to the stable single-image path when grouped embedding is used
