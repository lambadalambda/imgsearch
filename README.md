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
- Embeddings: local multimodal embedding model through a pluggable adapter

## MVP Constraints
- Scope target: personal collections up to about 10k indexed images.
- Similar image search starts from an already indexed image in the gallery.
- Supported formats (MVP): JPEG, PNG, WEBP, and AVIF.
- Server binds to localhost by default (`127.0.0.1`).
- Model runtime may be a local sidecar process in MVP; single-binary model packaging is post-MVP.
- Vector backend is swappable; default implementation uses `sqlite-vector`.

## Initial Scope
- Upload images from UI
- Queue-based background indexing
- Image gallery with pagination
- Search by text
- Search by similar image

## Running Locally

1. Install sqlite-vector extension for your platform:
   - `mise run sqlite-vector-setup`
2. Install Python deps for embedding sidecars:
   - `mise run jina-setup`
3. Start one embedding sidecar:
   - Torch (recommended for quality): `mise run jina-torch-serve`
   - MLX (faster, experimental quality): `mise run jina-serve`
   - Qwen3-VL-Embedding-8B (remote recommended): `scripts/qwen3_vl_server.py` on a GPU host
   - SQLite-AI in-DB embedder (no sidecar): `mise run sqlite-ai-build` (requires `vips` CLI for image preprocessing)
4. Start the app with sqlite-vector backend:
   - Torch sidecar: `go run ./cmd/imgsearch -embedder jina-torch -embed-image-mode auto -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`
   - MLX sidecar: `go run ./cmd/imgsearch -embedder jina-mlx -embed-image-mode auto -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`
   - Qwen3 sidecar: `go run ./cmd/imgsearch -embedder qwen3-vl-embedding-8b -jina-mlx-url http://127.0.0.1:9010 -embed-image-mode auto -vector-backend sqlite-vector -sqlite-vector-path ./tools/sqlite-vector/vector`
   - SQLite-AI embedder: `go run ./cmd/imgsearch -embedder sqlite-ai -vector-backend sqlite-vector -sqlite-ai-path ../sqlite-ai/dist/ai -sqlite-ai-model-path ../sqlite-ai/tests/models/Qwen/Qwen3-VL-Embedding-8B-Q4_K_M.gguf -sqlite-ai-vision-model-path ../sqlite-ai/tests/models/Qwen/mmproj-Qwen3-VL-Embedding-8B-f16.gguf -sqlite-ai-dimensions 4096`
5. Open the UI:
   - `http://127.0.0.1:8080/`

One-command app startup (auto-installs sqlite-vector if missing):
- `mise run serve`
- `mise run serve-torch` (recommended when using `jina-torch` sidecar)
- `mise run serve-qwen-remote` (run app against remote Qwen3 sidecar URL)
- `mise run serve-sqlite-ai` (run app with sqlite-ai in-DB embedder)

Reset local database files:
- `mise run reset-db`

The app defaults to `-embedder jina-mlx` with `-jina-mlx-url http://127.0.0.1:9009`.
Use `-embedder jina-torch` with `mise run jina-torch-serve` for higher retrieval quality.
Use `-embedder qwen3-vl-embedding-8b` for Qwen3-VL-Embedding-8B sidecar (4096-dim embeddings).
Use `-embedder sqlite-ai` to run multimodal embedding inside SQLite (requires `sqlite-ai` extension, GGUF model files, and `vips`).
When using `-embedder sqlite-ai`, embedding runs on a dedicated in-memory SQLite runtime so app DB reads/writes are not blocked by long embedding calls.
For sqlite-ai image indexing, images are preprocessed with `vips`: converted to JPEG and resized to max side `512` by default before embedding.
For fallback local testing without model runtime, run with `-embedder deterministic`.
Use `-embed-image-mode path|bytes|auto` for sidecar image transport:
- `path`: send file path to sidecar (fastest, requires shared filesystem access)
- `bytes`: send base64 image payload to sidecar (works with remote sidecar)
- `auto` (default): try `path`, then automatically fall back to `bytes` on path access errors
The app defaults to `-vector-backend auto`, which uses `sqlite-vector` when available and falls back to `bruteforce` when it is not.
Use `-vector-backend sqlite-vector` to require the extension, or `-vector-backend bruteforce` for compatibility-only mode.
You can set `SQLITE_VECTOR_PATH` once instead of passing `-sqlite-vector-path` every run.
If you change `-data-dir`, start the sidecar with matching allowed image roots, e.g.
`python3 scripts/jina_mlx_server.py --allow-dir /path/to/data/images`.

Torch sidecar tuning options:
- `JINA_TORCH_DEVICE=auto|cuda|cuda:N|mps|cpu` (default: `auto`, which prefers CUDA then MPS then CPU)
- `JINA_TORCH_MAX_IMAGE_PIXELS=602112` (default used by HF model)

Qwen3 sidecar tuning options:
- `QWEN3_VL_URL=http://127.0.0.1:9010` (app/test URL)
- `QWEN3_VL_REPO_PATH=~/repos/Qwen3-VL-Embedding` (optional override for `mise run qwen3-serve`; autodetects common `~/repos` paths)
- `QWEN3_VL_MODEL_ID=Qwen/Qwen3-VL-Embedding-8B` (optional; defaults to repo path when local model files exist)
- `QWEN3_VL_MAX_IMAGE_PIXELS=1843200`
- `QWEN3_VL_ATTN_IMPL=sdpa`
- `QWEN3_VL_TORCH_DTYPE=auto|bfloat16|float16|float32`

SQLite-AI embedder options:
- `SQLITE_AI_PATH=../sqlite-ai/dist/ai`
- `-sqlite-ai-model-path` path to embedding GGUF model
- `-sqlite-ai-vision-model-path` path to vision projector GGUF model
- `-sqlite-ai-dimensions` embedding dimensions (Qwen3-VL-Embedding-8B is 4096)
- `-sqlite-ai-image-max-side` max side length for pre-embedding resize (default: `512`)
- `-sqlite-ai-vips-path` optional path to `vips` binary (default: PATH lookup)
- `-sqlite-ai-query-instruction` default: `Retrieve images or text relevant to the user's query.`
- `-sqlite-ai-passage-instruction` default: `Represent this image or text for retrieval.`
- Optional: `-sqlite-ai-model-options`, `-sqlite-ai-vision-options`, `-sqlite-ai-context-options`

## Qwen3-VL-Embedding-8B Remote Sidecar

Run Qwen3 sidecar on a GPU machine and tunnel it locally:

1. On remote host, clone official repo and install deps in a dedicated venv:
   - `git clone https://github.com/QwenLM/Qwen3-VL-Embedding ~/Qwen3-VL-Embedding`
   - `~/.local/bin/virtualenv ~/imgsearch-qwen/.venv` (or `python3 -m venv` if `python3-venv` is installed)
   - `~/imgsearch-qwen/.venv/bin/pip install --upgrade pip`
   - NVIDIA/CUDA host: `~/imgsearch-qwen/.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 'torch>=2.9,<3' torchvision`
   - macOS/CPU-only host: `~/imgsearch-qwen/.venv/bin/pip install 'torch>=2.9,<3' torchvision`
   - `~/imgsearch-qwen/.venv/bin/pip install accelerate qwen-vl-utils pillow requests 'transformers>=4.57.3,<5'`
2. Copy this project script to remote host and start server:
   - `scp scripts/qwen3_vl_server.py lain@aiko-1:~/imgsearch-sidecar/qwen3_vl_server.py`
   - `~/imgsearch-qwen/.venv/bin/python ~/imgsearch-sidecar/qwen3_vl_server.py --host 127.0.0.1 --port 9010 --repo-path ~/Qwen3-VL-Embedding --allow-dir ~/imgsearch-sidecar/data/images`
3. Tunnel local port to remote:
   - `ssh -L 9010:127.0.0.1:9010 lain@aiko-1`
4. Run app locally against remote Qwen sidecar:
   - `mise run serve-qwen-remote`

Run fixture sanity checks against Qwen3 sidecar:
- `mise run qwen3-test`
- `mise run qwen3-test-api`
- `mise run qwen3-test-all`
- `mise run qwen3-eval-fixtures` (checks `fixtures/images/expected.txt` retrieval expectations)

Run sqlite-ai integration checks (local extension + GGUF files):
- `mise run sqlite-ai-test` (embedder-level semantic sanity)
- `mise run sqlite-ai-test-api` (API end-to-end: upload -> index -> text/similar search)
- `mise run sqlite-ai-test-all` (both sqlite-ai integration suites)
- `mise run sqlite-ai-test-fixtures` (checks `fixtures/images/expected.txt` retrieval expectations)

## Remote Sidecar (No Shared Filesystem)

If the sidecar runs on another machine (for example via SSH tunnel), run app with byte transport:

0. Start remote sidecar and wait for readiness (first startup may take several minutes to download model files):
   - `~/imgsearch-sidecar/.venv/bin/python ~/imgsearch-sidecar/jina_torch_server.py --host 127.0.0.1 --port 9009 --allow-dir ~/imgsearch-sidecar/data/images --device auto`
   - `curl -fsS http://127.0.0.1:9009/readyz`
1. Forward local port to remote sidecar:
   - `ssh -L 9009:127.0.0.1:9009 lain@aiko-1`
2. Start app locally with remote sidecar URL and byte mode:
   - `go run ./cmd/imgsearch -embedder jina-torch -jina-mlx-url http://127.0.0.1:9009 -embed-image-mode bytes`

The UI includes:
- upload form,
- indexing status panel (queue totals, progress, recent failures),
- gallery view with indexing states,
- live status/gallery updates over WebSocket (`/api/live`, ~2s snapshots) with automatic polling fallback,
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

- API status endpoint: `GET /api/stats`
- API action endpoint: `POST /api/jobs/retry-failed` (requeue failed jobs and enqueue missing jobs for the active model)
- Sidecar status endpoints: `GET /healthz`, `GET /readyz`, `GET /stats`
- Sidecar prints one log line per embed request with request id, status, and duration.
- In sidecar `GET /stats`, `embed_image_*` counters include both `/embed/image` and `/embed/image-bytes` requests.
- Sidecar `GET /stats` also reports runtime config (model id, device for torch, and max image pixels).

From the UI, use the **Retry Failed / Queue Missing** button in the Indexing Status panel.

## Optional Integration Test (Requires Sidecar)

Run integration suites (all are skipped by default unless `RUN_JINA_MLX_INTEGRATION=1` is set by the task):

- `mise run jina-test` for embedder-level semantic similarity checks against fixture images.
- `mise run jina-test-api` for API end-to-end flow (`/api/upload` -> queue processing -> `/api/search/text` and `/api/search/similar`).
- `mise run jina-test-all` to run both integration suites together.
- `mise run sqlite-vector-test` for sqlite-vector index integration checks.

The semantic checks verify expected relative similarity trends, such as cat images ranking closer to each other than cat-vs-dog, and woman portraits clustering together.

See `docs/architecture.md`, `docs/mvp-plan.md`, and `docs/decisions.md` for implementation details.
