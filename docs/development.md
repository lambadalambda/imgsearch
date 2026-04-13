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

Note: native embedding is the default build path when CGO is enabled. In source checkouts, the llama.cpp runtime libraries still need to be built first.

## llama.cpp Paths

This repo includes two llama.cpp paths:

- `llama-cpp-native`: direct in-process llama.cpp/mtmd calls via cgo (default)
- `llama-cpp`: HTTP calls to `llama-server` (legacy)

Run native explicitly:

- `go run ./cmd/imgsearch -embedder llama-cpp-native -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`

Convenience tasks:

- `mise run llama-cpp-native-build`
- `mise run serve-llama-cpp-native`

Legacy `llama-server` HTTP path:

1. Start server:
   - `./deps/llama.cpp/build/bin/llama-server --host 127.0.0.1 --port 8081 --model ./models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf --mmproj ./models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf --embeddings --pooling last --ctx-size 8192 --gpu-layers 99`
2. Run app:
   - `go run ./cmd/imgsearch -embedder llama-cpp -llama-cpp-url http://127.0.0.1:8081 -llama-cpp-dimensions 4096 -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`

Notes:

- Use `-llama-cpp-dimensions 4096` for Qwen3-VL-Embedding-8B models.
- You can pass `-llama-cpp-model` to set the optional `model` field in `/v1/embeddings` requests.
- Native defaults target `lainsoykaf/Qwen3-VL-Embedding-8B-GGUF` at `4096` dimensions.
- Native path defaults to `-llama-native-image-max-side 512` to cap indexing latency on very large images.
- Native image embedding preprocesses every image through libvips (via `github.com/cshum/vipsgen`) and writes a temporary JPEG before mtmd, which avoids WEBP/AVIF decode failures in llama.cpp input handling.
- Native llama.cpp prompting uses Qwen chat-template style framing (`system` + `user` + assistant generation prompt) for text and image embeddings.
- Optional: set `-llama-native-image-max-tokens` to override mtmd image token cap (`0` keeps model defaults).
- Native embedding model metadata includes model/mmproj filenames plus image cap settings; changing them creates a new model version and queues missing index jobs for re-embedding.

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

## Integration Tests

Run integration suites:

- `mise run llama-cpp-native-test`
- `mise run llama-cpp-native-test-fixtures`
- `mise run llama-cpp-test`
- `mise run llama-cpp-test-fixtures`
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
