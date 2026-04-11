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
- Supported formats (MVP): JPEG and PNG.
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

1. Install Python deps for the Jina MLX sidecar:
   - `mise run jina-setup`
2. Start the local embedding sidecar:
   - `mise run jina-serve`
3. Start the app:
   - `go run ./cmd/imgsearch`

The app defaults to `-embedder jina-mlx` with `-jina-mlx-url http://127.0.0.1:9009`.
For fallback local testing without model runtime, run with `-embedder deterministic`.
If you change `-data-dir`, start the sidecar with matching allowed image roots, e.g.
`python3 scripts/jina_mlx_server.py --allow-dir /path/to/data/images`.

## Optional Integration Test (Requires Sidecar)

Run integration suites (all are skipped by default unless `RUN_JINA_MLX_INTEGRATION=1` is set by the task):

- `mise run jina-test` for embedder-level semantic similarity checks against fixture images.
- `mise run jina-test-api` for API end-to-end flow (`/api/upload` -> queue processing -> `/api/search/text` and `/api/search/similar`).
- `mise run jina-test-all` to run both integration suites together.

The semantic checks verify expected relative similarity trends, such as cat images ranking closer to each other than cat-vs-dog, and woman portraits clustering together.

See `docs/architecture.md`, `docs/mvp-plan.md`, and `docs/decisions.md` for implementation details.
