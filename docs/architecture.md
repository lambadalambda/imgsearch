# Planned Architecture

## Overview
The application is a single Go process that exposes an HTTP server and runs a background indexing worker.

Core flow:
1. User uploads image(s) from the web UI.
2. Backend persists file and creates a queued indexing job in SQLite.
3. Worker dequeues jobs, computes embeddings, and stores vector + metadata.
4. Search endpoint performs nearest-neighbor lookup and returns ranked image results.

MVP design priorities:
- keep runtime simple and local,
- preserve data integrity across restarts,
- choose reliable defaults over maximum scale.

## Components

### 1) Web Server (Go)
- Serves HTML/CSS/JS assets.
- Handles upload endpoints and search endpoints.
- Serves image files or thumbnails.

### 2) Queue + Worker (Go)
- Job states: `pending`, `leased`, `done`, `failed`.
- Retry policy with capped attempts.
- Idempotent indexing by content hash to avoid duplicate work.
- Lease-based claiming with expiry (`leased_until`) so crash-recovery can requeue stale jobs.
- Single worker goroutine in MVP to minimize SQLite write contention.

### 3) Storage (SQLite)
- Stores image metadata, queue jobs, and vector representations.
- Maintains transactional integrity for database records.
- Enables app restart recovery from persisted queue state.
- Uses schema migrations with forward-only versions.

### 4) Embedding Adapter
- Pluggable interface for embedding provider.
- Methods:
  - `EmbedText(query string) ([]float32, error)`
  - `EmbedImage(path string) ([]float32, error)`
- Keeps model-specific logic isolated from app logic.
- MVP default is an adapter to a local model runtime endpoint.

### 5) Search Layer
- Text search: embed query text then cosine-similarity against indexed image vectors.
- Similar-image search: start from an indexed image ID and compare against other indexed vectors.
- Optional future rerank step using metadata/tags.

### 6) Vector Search Strategy (MVP)
- Store vectors as float32 blobs in SQLite.
- Keep an in-memory vector cache in the app process.
- Use brute-force cosine similarity in Go for exact nearest neighbors.
- Rebuild/refresh cache when new embeddings are written.
- Target envelope: up to ~10k images in MVP.

### 7) File Storage
- Configurable data directory (default: `./data`).
- Layout:
  - `./data/images/<sha256>` for original images
  - `./data/thumbs/<sha256>.jpg` for thumbnails
  - `./data/tmp/<uuid>` for upload staging
- Upload flow:
  1. write upload to temp file,
  2. validate image and hash content,
  3. commit DB rows,
  4. atomically move temp file to final location.

## Data Model (Initial)

### `schema_migrations`
- `version` (PK)
- `applied_at`

### `images`
- `id` (PK)
- `sha256` (unique)
- `original_name`
- `storage_path`
- `thumbnail_path` (nullable)
- `mime_type`
- `width`
- `height`
- `created_at`

### `embedding_models`
- `id` (PK)
- `name`
- `version`
- `dimensions`
- `metric` (e.g. `cosine`)
- `normalized` (boolean)
- `created_at`

### `image_embeddings`
- `image_id` (FK)
- `model_id` (FK)
- `dim`
- `vector_blob`
- `created_at`
- `updated_at`

Primary key: (`image_id`, `model_id`)

### `index_jobs`
- `id` (PK)
- `kind` (e.g. `embed_image`)
- `image_id` (FK)
- `model_id` (FK)
- `state` (`pending`, `leased`, `done`, `failed`)
- `run_after`
- `leased_until`
- `lease_owner`
- `attempts`
- `max_attempts`
- `last_error`
- `created_at`
- `updated_at`

Unique key: (`kind`, `image_id`, `model_id`)

## Operational Notes
- Use WAL mode for better concurrency.
- Keep uploads under a configured max size.
- Generate thumbnails asynchronously after base indexing.
- Add health endpoint for worker queue depth and failure count.
- On startup, recover expired leases so no job stays stuck in `leased` indefinitely.
- Restrict network binding to localhost by default.
