# imgsearch

`imgsearch` is a local-first image organization and similarity search app.

It is designed as a simple Go application that:
- indexes images from disk uploads through a background queue,
- stores metadata and vectors in SQLite,
- serves a small web UI for browsing indexed images,
- supports text-to-image and image-to-image search.

## Goals
- Keep setup simple (single binary + single SQLite database).
- Keep data local and private.
- Make indexing reliable and resumable.
- Deliver useful search quality quickly with an MVP.

## Planned Stack
- Backend: Go
- Frontend: HTML/CSS/JavaScript
- Persistence: SQLite (metadata + vector storage)
- Search (MVP): `sqlite-vector` via a `VectorIndex` interface
- Embeddings: `llama-cpp-native` direct in-process multimodal embedding (primary path)

## MVP Constraints
- Scope target: personal collections up to about 10k indexed images.
- Similar image search starts from an already indexed image in the gallery.
- Supported formats (MVP): JPEG, PNG, WEBP, and AVIF.
- Server binds to localhost by default (`127.0.0.1`).
- Model runtime defaults to `llama-cpp-native`.
- Vector backend is swappable; default implementation uses `sqlite-vector`.

## Initial Scope
- Upload images from UI
- Queue-based background indexing
- Image gallery with pagination
- Search by text
- Search by similar image

## Running Locally (Default: llama.cpp Native)

`llama-cpp-native` is the default and recommended embedder.

1. Install sqlite-vector for your platform:
   - `mise run sqlite-vector-setup`
2. Add and initialize llama.cpp submodule:
   - `git submodule update --init --recursive deps/llama.cpp`
3. Build llama.cpp runtime libraries:
   - `mise run llama-cpp-native-build`
4. Download a model pair (example: Qwen3-VL-Embedding-2B):
   - `python3 -m pip install --user "huggingface_hub[cli]"`
   - `mkdir -p ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF`
   - `huggingface-cli download VesNFF/Qwen3-VL-Embedding-2B-GGUF Qwen3-VL-Embedding-2B-Q6_K.gguf --local-dir ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF`
   - `huggingface-cli download VesNFF/Qwen3-VL-Embedding-2B-GGUF mmproj-Qwen3-VL-Embedding-2B-f16.gguf --local-dir ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF`
5. Start the app with native embedding + sqlite-vector:
   - `go run -tags llamacpp_native ./cmd/imgsearch -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector -llama-native-model-path ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B-Q6_K.gguf -llama-native-mmproj-path ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/mmproj-Qwen3-VL-Embedding-2B-f16.gguf -llama-native-dimensions 2048`
6. Open the UI:
   - `http://127.0.0.1:8080/`

One-command startup (native default path):
- `mise run serve`
- `mise run "serve:8b"` for the 8B model defaults
- Optional native tuning env vars for `mise run serve`: `LLAMA_NATIVE_IMAGE_MAX_SIDE` (default `512`), `LLAMA_NATIVE_IMAGE_MAX_TOKENS` (default `0` = model default).

Reset local database files:
- `mise run reset-db`

Note: native embedding requires build tag `llamacpp_native`; use `go run -tags llamacpp_native ...` or `mise run serve`.

### llama.cpp Embedder

This repo includes two llama.cpp paths:

- `llama-cpp-native`: direct in-process llama.cpp/mtmd calls via cgo (default)
- `llama-cpp`: HTTP calls to `llama-server` (legacy)

1. Add and initialize the submodule:
   - `git submodule update --init --recursive deps/llama.cpp`
2. Build llama.cpp runtime libraries:
   - `cmake -S ./deps/llama.cpp -B ./deps/llama.cpp/build`
   - `cmake --build ./deps/llama.cpp/build --target llama-server -j`
3. Start imgsearch with direct llama.cpp native embedding (example with Qwen3-VL-Embedding-2B):
   - `go run -tags llamacpp_native ./cmd/imgsearch -embedder llama-cpp-native -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector -llama-native-model-path ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B-Q6_K.gguf -llama-native-mmproj-path ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/mmproj-Qwen3-VL-Embedding-2B-f16.gguf -llama-native-dimensions 2048`
4. Convenience tasks:
   - `mise run llama-cpp-native-build`
   - `mise run serve-llama-cpp-native`
5. (Legacy) Run via `llama-server` HTTP API:
   - Start server: `./deps/llama.cpp/build/bin/llama-server --host 127.0.0.1 --port 8081 --model ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B-Q6_K.gguf --mmproj ../sqlite-ai/tests/models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/mmproj-Qwen3-VL-Embedding-2B-f16.gguf --embeddings --pooling last --ctx-size 8192 --gpu-layers 99`
   - Run app: `go run ./cmd/imgsearch -embedder llama-cpp -llama-cpp-url http://127.0.0.1:8081 -llama-cpp-dimensions 2048 -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`

Notes:
- Use `-llama-cpp-dimensions 4096` for Qwen3-VL-Embedding-8B models.
- You can pass `-llama-cpp-model` to set the optional `model` field in `/v1/embeddings` requests.
- Use `-llama-native-dimensions 4096` for Qwen3-VL-Embedding-8B models when using `llama-cpp-native`.
- Native path defaults to `-llama-native-image-max-side 512` to cap indexing latency on very large images.
- Native image embedding preprocesses every image through libvips (via `github.com/cshum/vipsgen`) and writes a temporary JPEG before mtmd, which avoids WEBP/AVIF decode failures in llama.cpp input handling.
- Native llama.cpp prompting uses Qwen chat-template style framing (`system` + `user` + assistant generation prompt) for text and image embeddings.
- Optional: set `-llama-native-image-max-tokens` to override mtmd image token cap (`0` keeps model defaults).
- Native embedding model metadata includes model/mmproj filenames plus image cap settings; changing them creates a new model version and queues missing index jobs for re-embedding.
- Sanity checks for embedding quality:
  - `mise run llama-cpp-test`
  - `mise run llama-cpp-test-fixtures`
  - `mise run llama-cpp-native-test`
  - `mise run llama-cpp-native-test-fixtures`
- Benchmark checks:
  - `mise run llama-cpp-native-bench`
  - `mise run llama-cpp-native-bench-qwen8b`

The UI includes:
- upload form,
- indexing status panel (queue totals, progress, recent failures),
- gallery view with indexing states,
- live status/gallery updates over WebSocket (`/api/live`, ~2s snapshots) with automatic polling fallback,
- connection badge showing live vs reconnecting vs polling fallback mode,
- text search (with optional negative prompt via `neg` query param),
- similar-image search buttons on cards.

Supported upload formats: JPEG, PNG, WEBP, and AVIF.

## Bulk Import

With app running on `http://127.0.0.1:8080`, import a directory recursively:

- `mise run import-images -- ./fixtures/images`

Optional arguments and behavior:

- `mise run import-images -- ./photos http://127.0.0.1:8080`
- `IMGSEARCH_IMPORT_CONVERT=auto` (default): try direct upload first, then auto-convert WEBP/AVIF with `vips` on failure.
- `IMGSEARCH_IMPORT_CONVERT=vips`: always convert WEBP/AVIF via `vips` before upload.
- `IMGSEARCH_IMPORT_CONVERT=never`: never convert; upload files as-is.

## Observability

- App health endpoint: `GET /healthz`
- API status endpoint: `GET /api/stats`
- API action endpoint: `POST /api/jobs/retry-failed` (requeue failed jobs and enqueue missing jobs for the active model)

From the UI, use the **Retry Failed / Queue Missing** button in the Indexing Status panel.

## Integration Tests

Run integration suites:

- `mise run llama-cpp-native-test` for semantic sanity checks against direct llama.cpp native embedding.
- `mise run llama-cpp-native-test-fixtures` to validate fixture retrieval behavior against direct llama.cpp native embedding.
- `mise run llama-cpp-test` for semantic sanity checks against a running `llama-server`.
- `mise run llama-cpp-test-fixtures` to validate fixture retrieval behavior against a running `llama-server`.
- `mise run sqlite-vector-test` for sqlite-vector index integration checks.
- `mise run llama-cpp-native-bench-qwen8b` for 8B image-embedding benchmarks.

The semantic checks verify expected relative similarity trends, such as cat images ranking closer to each other than cat-vs-dog, and woman portraits clustering together.

See `docs/architecture.md`, `docs/mvp-plan.md`, and `docs/decisions.md` for implementation details.
