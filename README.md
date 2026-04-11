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

See `docs/architecture.md`, `docs/mvp-plan.md`, and `docs/decisions.md` for implementation details.
