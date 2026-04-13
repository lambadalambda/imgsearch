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
- Embeddings: `sqlite-ai` in-DB multimodal embedding (primary path)

## MVP Constraints
- Scope target: personal collections up to about 10k indexed images.
- Similar image search starts from an already indexed image in the gallery.
- Supported formats (MVP): JPEG, PNG, WEBP, and AVIF.
- Server binds to localhost by default (`127.0.0.1`).
- Model runtime is `sqlite-ai` loaded as a SQLite extension.
- Vector backend is swappable; default implementation uses `sqlite-vector`.

## Initial Scope
- Upload images from UI
- Queue-based background indexing
- Image gallery with pagination
- Search by text
- Search by similar image

## Running Locally (SQLite-AI)

`sqlite-ai` is the primary embedder for this project.
`jina-mlx`, `jina-torch`, and `qwen3-vl-embedding-8b` are deprecated and kept only for backward compatibility.

1. Install sqlite-vector for your platform:
   - `mise run sqlite-vector-setup`
2. Ensure the sibling `sqlite-ai` repo exists and build the extension:
   - `git clone https://github.com/sqliteai/sqlite-ai ../sqlite-ai` (if you do not already have it)
   - `make -C ../sqlite-ai extension`
3. Install `vips` (required for sqlite-ai image preprocessing):
   - macOS: `brew install vips`
   - Ubuntu/Debian: `sudo apt-get install -y libvips-tools`
4. Download a model pair (example: Qwen3-VL-Embedding-2B):
   - `python3 -m pip install --user "huggingface_hub[cli]"`
   - `mkdir -p ../sqlite-ai/tests/models/Qwen3-2B`
   - `huggingface-cli download VesNFF/Qwen3-VL-Embedding-2B-GGUF Qwen3-VL-Embedding-2B-Q6_K.gguf --local-dir ../sqlite-ai/tests/models/Qwen3-2B`
   - `huggingface-cli download VesNFF/Qwen3-VL-Embedding-2B-GGUF mmproj-Qwen3-VL-Embedding-2B-f16.gguf --local-dir ../sqlite-ai/tests/models/Qwen3-2B`
5. Start the app with sqlite-ai + sqlite-vector:
   - `go run ./cmd/imgsearch -embedder sqlite-ai -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector -sqlite-ai-path ../sqlite-ai/dist/ai -sqlite-ai-model-path ../sqlite-ai/tests/models/Qwen3-2B/Qwen3-VL-Embedding-2B-Q6_K.gguf -sqlite-ai-vision-model-path ../sqlite-ai/tests/models/Qwen3-2B/mmproj-Qwen3-VL-Embedding-2B-f16.gguf -sqlite-ai-dimensions 2048`
6. Open the UI:
   - `http://127.0.0.1:8080/`

One-command startup (sqlite-ai path):
- `mise run serve-sqlite-ai`

Reset local database files:
- `mise run reset-db`

Note: the binary default embedder is still `jina-mlx` for backward compatibility, so use `-embedder sqlite-ai` (or `mise run serve-sqlite-ai`).

### Linux + CUDA Setup (sqlite-ai)

Use this setup on Linux with NVIDIA GPUs.

1. Install system dependencies:
   - `sudo apt-get update`
   - `sudo apt-get install -y build-essential cmake pkg-config libsqlite3-dev libvips-tools curl git python3-pip`
2. Install NVIDIA driver + CUDA toolkit, then verify:
   - `nvidia-smi`
   - `nvcc --version`
3. Build `sqlite-ai` with CUDA-enabled llama.cpp:
   - `CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | sed -n '1p' | tr -d '.')"`
   - `make -C ../sqlite-ai clean extension LLAMA="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"`
4. If CUDA shared libraries are not found at runtime, set:
   - `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}`
5. Run imgsearch using sqlite-ai GPU options:
   - `go run ./cmd/imgsearch -embedder sqlite-ai -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector -sqlite-ai-path ../sqlite-ai/dist/ai.so -sqlite-ai-model-path ../sqlite-ai/tests/models/Qwen3-2B/Qwen3-VL-Embedding-2B-Q6_K.gguf -sqlite-ai-vision-model-path ../sqlite-ai/tests/models/Qwen3-2B/mmproj-Qwen3-VL-Embedding-2B-f16.gguf -sqlite-ai-dimensions 2048 -sqlite-ai-model-options "gpu_layers=99" -sqlite-ai-vision-options "use_gpu=1"`

SQLite-AI options:
- `SQLITE_AI_PATH=../sqlite-ai/dist/ai` (or `../sqlite-ai/dist/ai.so` on Linux)
- `-sqlite-ai-model-path` path to embedding GGUF model
- `-sqlite-ai-vision-model-path` path to vision projector GGUF model
- `-sqlite-ai-dimensions` embedding dimensions (`2048` for Qwen3-VL-Embedding-2B, `4096` for Qwen3-VL-Embedding-8B)
- `-sqlite-ai-image-max-side` max side length for pre-embedding resize (default: `512`)
- `-sqlite-ai-vips-path` optional path to `vips` binary (default: PATH lookup)
- `-sqlite-ai-query-instruction` default: `Retrieve images or text relevant to the user's query.`
- `-sqlite-ai-passage-instruction` default: `Represent this image or text for retrieval.`
- Optional: `-sqlite-ai-model-options`, `-sqlite-ai-vision-options`, `-sqlite-ai-context-options`

Run sqlite-ai integration checks (local extension + GGUF files):
- `mise run sqlite-ai-test` (embedder-level semantic sanity)
- `mise run sqlite-ai-test-api` (API end-to-end: upload -> index -> text/similar search)
- `mise run sqlite-ai-test-all` (both sqlite-ai integration suites)
- `mise run sqlite-ai-test-fixtures` (checks `fixtures/images/expected.txt` retrieval expectations)

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

- `mise run sqlite-ai-test` for embedder-level semantic similarity checks against fixture images.
- `mise run sqlite-ai-test-api` for API end-to-end flow (`/api/upload` -> queue processing -> `/api/search/text` and `/api/search/similar`).
- `mise run sqlite-ai-test-all` to run both sqlite-ai integration suites together.
- `mise run sqlite-ai-test-fixtures` to validate retrieval behavior against `fixtures/images/expected.txt`.
- `mise run sqlite-vector-test` for sqlite-vector index integration checks.

The semantic checks verify expected relative similarity trends, such as cat images ranking closer to each other than cat-vs-dog, and woman portraits clustering together.

See `docs/architecture.md`, `docs/mvp-plan.md`, and `docs/decisions.md` for implementation details.
