# Development

## Running Locally

`llama-cpp-native` is the default and recommended embedder.

1. Install sqlite-vector for your platform:
   - `mise run sqlite-vector-setup`
2. Add and initialize llama.cpp submodule:
   - `git submodule update --init --recursive deps/llama.cpp`
3. Build llama.cpp runtime libraries:
   - `mise run llama-cpp-native-build`
4. Start the app with native embedding + sqlite-vector:
   - `go run ./cmd/imgsearch -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`
5. On first run, if `./models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf` or `./models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf` are missing, imgsearch downloads them automatically from `lainsoykaf/Qwen3-VL-Embedding-8B-GGUF` on Hugging Face.
6. Open the UI:
   - `http://127.0.0.1:8080/`

One-command startup:
- `mise run serve`
- `mise run "serve:8b"`

Optional native tuning env vars for `mise run serve`:
- `LLAMA_NATIVE_IMAGE_MAX_SIDE` default `512`
- `LLAMA_NATIVE_IMAGE_MAX_TOKENS` default `0`

Reset local database files:
- `mise run reset-db`

Note: imgsearch now depends on the native llama.cpp path. In source checkouts, the llama.cpp runtime libraries still need to be built first, and non-CGO builds fall back to the stub implementation.

Build artifact note:
- `mise run llama-cpp-native-build` uses `scripts/ensure_llama_cpp_native_build.sh`, which treats `./deps/llama.cpp/build` as the host-native build directory and deletes foreign shared libraries before rebuilding.
- Do not copy Linux Docker build outputs into `./deps/llama.cpp/build` on macOS.
- Keep cross-built artifacts in an explicit separate directory such as `./build-artifacts/llama.cpp/linux-cuda13/`.
- When packaging from explicit cross-built Linux libs, set `IMGSEARCH_LLAMA_LIB_DIR=/absolute/path/to/build-artifacts/llama.cpp/linux-cuda13/bin` before running `scripts/package_release.sh`.

## llama.cpp Native Runtime

`imgsearch` now uses the in-process `llama-cpp-native` path only.

Run explicitly:

- `go run ./cmd/imgsearch -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`

Convenience tasks:

- `mise run llama-cpp-native-build`
- `mise run serve`
- `mise run serve-llama-cpp-native`

Notes:

- Native defaults target `lainsoykaf/Qwen3-VL-Embedding-8B-GGUF` at `4096` dimensions.
- Native path defaults to `-llama-native-image-max-side 512` to cap indexing latency on very large images.
- Native image embedding preprocesses every image through libvips (via `github.com/cshum/vipsgen`) and writes a temporary JPEG before mtmd, which avoids WEBP/AVIF decode failures in llama.cpp input handling.
- Native prompting uses Qwen chat-template style framing (`system` + `user` + assistant generation prompt) for text and image embeddings.
- Optional: set `-llama-native-image-max-tokens` to override mtmd image token cap (`0` keeps model defaults).
- Optional: use `-llama-native-query-instruction` and `-llama-native-passage-instruction` to tune retrieval framing.
- Native embedding model metadata includes model/mmproj filenames, image cap settings, and retrieval instruction settings; changing them creates a new model version and queues missing index jobs for re-embedding.

## Bulk Import

With the app running on `http://127.0.0.1:8080`, import a directory recursively:

- `mise run import-images -- ./fixtures/images`

Optional arguments and behavior:

- `mise run import-images -- ./photos http://127.0.0.1:8080`
- `IMGSEARCH_IMPORT_CONVERT=auto` tries direct upload first, then auto-converts WEBP/AVIF with `vips` on failure.
- `IMGSEARCH_IMPORT_CONVERT=vips` always converts WEBP/AVIF via `vips` before upload.
- `IMGSEARCH_IMPORT_CONVERT=never` never converts and uploads files as-is.

## Observability

- App health endpoint: `GET /healthz`
- API status endpoint: `GET /api/stats`
- API action endpoint: `POST /api/jobs/retry-failed`

From the UI, use the **Retry Failed / Queue Missing** button in the Indexing Status panel.

`/api/stats` is the main queue snapshot for before/after comparisons when changing indexing behavior.

Useful fields:
- `queue.runnable`: number of `embed_image` jobs ready to run now
- `queue.oldest_runnable_age_seconds`: age of the oldest runnable `embed_image` job
- `job_kinds`: per-job-kind counts, intended to stay useful if indexing is later split into `embed_image` and `annotate_image`

Successful worker runs also log a timing summary line with per-stage durations:
- `total`
- `load`
- `stat`
- `embed`
- `annotate`
- `db`
- `index`

Example workflow before a pipeline change:
1. Run a representative import.
2. Capture `/api/stats` during the run and when the queue drains.
3. Save the worker timing logs.
4. Run the relevant benchmark commands below.
5. Repeat after the change and compare the same data.

## Profiling And Benchmarking

Embedding benchmark:
- `mise run llama-cpp-native-bench-qwen8b`

Annotation benchmark:
- `mise run llama-cpp-native-bench-gemma`

Useful benchmark knobs:
- `BENCH_TIME=5x`
- `BENCH_COUNT=5`

Useful annotation overrides:
- `LLAMA_NATIVE_GEMMA_MODEL_PATH=/path/to/model.gguf`
- `LLAMA_NATIVE_GEMMA_MMPROJ_PATH=/path/to/mmproj.gguf`
- `LLAMA_NATIVE_GEMMA_IMAGE_MAX_SIDE=768`
- `LLAMA_NATIVE_GEMMA_IMAGE_MAX_TOKENS=0`

If you want to benchmark the 26B annotator instead of the default smaller Gemma model, override the Gemma model and mmproj paths when running `mise run llama-cpp-native-bench-gemma`.

## Integration Tests

Run integration suites:

- `mise run llama-cpp-native-test`
- `mise run llama-cpp-native-test-fixtures`
- `mise run sqlite-vector-test`
- `mise run llama-cpp-native-bench-qwen8b`

The semantic checks verify expected relative similarity trends, such as cat images ranking closer to each other than cat-vs-dog, and woman portraits clustering together.

## UI Summary

The UI includes:

- upload form
- indexing status panel with queue totals, progress, and recent failures
- gallery view with indexing states
- live status/gallery updates over WebSocket with polling fallback
- text search with optional negative prompt via `neg`
- similar-image search buttons on cards
