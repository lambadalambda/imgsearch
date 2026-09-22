# Architecture

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
- Serves the embedded Atelier SPA at `/` (and the legacy shell at `/legacy`).
- Handles upload, list, search, stats, settings, and live WebSocket endpoints under `/api/`.
- Serves stored media under `/media/images/` and `/media/videos/`, behind the same auth as `/api/`.

### 2) Queue + Worker (Go)
- Job kinds: `embed_image`, `annotate_image`, `annotate_video`, `transcribe_video`; states: `pending`, `leased`, `done`, `failed`.
- Retry policy with capped attempts and `run_after` backoff.
- Idempotent indexing by content hash to avoid duplicate work.
- Lease-based claiming with expiry (`leased_until`) and periodic renewal for long jobs, so crash-recovery can requeue stale jobs.
- Single dispatcher per process, with optionally batched embeds and bounded parallel remote annotation jobs. Native annotation, embedding batches, and transcription do not overlap annotation jobs; `-mode=api` / `-mode=worker` split the server and worker into separate processes.
- Settings → remote backend → **Parallel annotations** controls actual remote job admission (`1` serial, `2` up to two overlapping image/video annotation jobs), separately from `worker-batch-size`. Limits are per worker process. The HTTP client also retains its request cap.
- The annotator can be swapped at runtime from the settings page (native Gemma `e4b`/`26b` or an OpenAI-compatible remote server). Settings checks use a five-second polling throttle; changes apply after the next check and any required drain. Running embedding/transcription work can delay the check. Changed concurrency limits/backends pause admission and let active requests finish before applying; no restart is required.
- Settings → **Output token limits** stores first-attempt/compact-retry budgets for images (`1024/512`), video frames (`320/256`) and video summaries (`1200/640`). Each accepts an integer from `64` to `32768`. The API field is `annotation.output_tokens`, with keys `image`, `image_retry`, `video_frame`, `video_frame_retry`, `video`, and `video_retry`. Existing saved settings get the original defaults; omitted fields in updates preserve saved values.
- Output-budget-only changes publish an immutable resolver snapshot without rebuilding either backend or reloading native models. Each request receives its primary/retry pair by value, so in-flight retries keep their original budget. Changes apply after settings refresh; they do not requeue old failed jobs. These limits are distinct from input/context sizes and `-llama-native-image-max-tokens`; higher output limits can increase latency and remain subject to provider/model limits. Native generation also currently has a fixed 32 KiB output buffer, so very large outputs can still fail before reaching the token limit (tracked separately).
- See `docs/indexing-annotation-pipeline-notes.md` for pipeline notes.

### 3) Storage (SQLite)
- Stores image metadata, queue jobs, and vector representations.
- Maintains transactional integrity for database records.
- Enables app restart recovery from persisted queue state.
- Uses schema migrations with forward-only versions.

### 4) Native Embedding Runtime
- Embedding uses the in-process `llama-cpp-native` runtime.
- Query-time text embedding stays in the serving process so search remains available while background indexing runs.
- A separate optional annotator produces titles, summaries, descriptions, and tags: native Gemma (`e4b` default, `26b` optional) in-process, or an OpenAI-compatible remote server chosen in the settings page.
- Video transcription (optional, ONNX Runtime + Parakeet) embeds transcript text alongside sampled frames.
- Runtime configuration still lives behind Go interfaces so handlers and worker code stay decoupled from model details.

### 5) Search Layer
- Text search: embed query text then cosine-similarity against indexed image vectors.
- Similar-image search: start from an indexed image ID and compare against other indexed vectors.
- Optional future rerank step using metadata/tags.

### 6) Vector Index Abstraction
- Use a `VectorIndex` interface so search backend can be replaced without changing handlers.
- MVP implementation: `SQLiteVectorIndex` backed by `sqlite-vector`.
- Secondary implementation (test/fallback): `BruteForceVectorIndex` in Go.

Proposed interface:
- `Upsert(imageID int64, modelID int64, vec []float32) error`
- `Delete(imageID int64, modelID int64) error`
- `Search(modelID int64, query []float32, limit int) ([]SearchHit, error)`
- `SearchByImageID(modelID int64, imageID int64, limit int) ([]SearchHit, error)`

### 7) Vector Search Strategy (MVP)
- Store vectors as float32 blobs in SQLite (`image_embeddings`) as source-of-truth.
- `VectorIndex.Upsert` owns persistence of the active embedding row.
- `sqlite-vector` reads from `image_embeddings` for nearest-neighbor queries.
- Keep search backend behind `VectorIndex` so migration to another ANN library stays low-risk.

### 8) File Storage
- Configurable data directory (default: `./data`).
- Layout:
  - `./data/images/<sha256>` for original images and sampled video frames
  - `./data/videos/<sha256>` for original videos
  - `./data/tmp/` for upload staging
  - `./data/imgsearch.sqlite` for the database
- There are no thumbnail derivatives yet; the grid serves originals (see issue 079).
- Upload flow:
  1. write upload to temp file,
  2. sniff the media type, validate, and hash content,
  3. for videos, sample frames with ffmpeg,
  4. commit DB rows (adopting an existing row on a hash conflict),
  5. atomically move temp files to their final locations.

## Data Model

Migrations live in `internal/db/migrations.go`; this is the shape after the current version.

### `schema_migrations`
- `version` (PK), `applied_at`

### `images`
- `id` (PK), `sha256` (unique), `original_name`, `storage_path`, `thumbnail_path` (nullable, unused), `mime_type`, `width`, `height`, `created_at`
- annotation text: `title`, `summary`, `description`, `tags_json`, `annotation_updated_at`, `reannotate_requested`
- Sampled video frames are rows here too, linked through `video_frames`.

### `videos`
- `id` (PK), `sha256` (unique), `original_name`, `storage_path`, `mime_type`, `duration_ms`, `width`, `height`, `frame_count`, `created_at`
- annotation text: `title`, `summary`, `description`, `tags_json`, `annotation_updated_at`, `reannotate_requested`

### `video_frames`
- `video_id` (FK), `image_id` (FK), `frame_index`, `timestamp_ms`

### `video_transcript_embeddings`
- `video_id` (FK), `model_id` (FK), transcript text and its embedding, `created_at`, `updated_at`

### `embedding_models`
- `id` (PK), `name`, `version`, `dimensions`, `metric` (e.g. `cosine`), `normalized`, `created_at`

### `image_embeddings`
- `image_id` (FK), `model_id` (FK), `dim`, `vector_blob`, `created_at`, `updated_at`
- Primary key: (`image_id`, `model_id`)
- `image_embeddings_generation` is a one-row counter bumped by triggers on every write, used by the vector index to decide when to requantize.

### `index_jobs`
- `id` (PK), `kind`, `image_id` (FK, nullable), `video_id` (FK, nullable), `model_id` (FK), `state`, `run_after`, `leased_until`, `lease_owner`, `attempts`, `max_attempts`, `last_error`, `created_at`, `updated_at`
- Exactly one of `image_id` / `video_id` is set; unique per (`kind`, target, `model_id`).
- Lookup indexes on (`image_id`, `model_id`, `kind`), (`video_id`, `model_id`, `kind`), and (`state`, `kind`, `run_after`, `created_at`).

### `collections` / `collection_items`
- Named instance-wide lists reference existing images/videos, with exactly one media foreign key per membership and uniqueness per collection/target.
- Foreign-key cascades remove memberships when a collection or target is deleted; deleting collections never deletes media. Membership IDs provide newest-saved-first ordering.
- See [saved collections](collections.md) for API and semantics.

### `settings` / `settings_version`
- Key/value JSON settings (currently the annotation backend) plus a version counter the worker polls to hot-swap the annotator.

## Operational Notes
- Use WAL mode for better concurrency.
- Keep uploads under a configured max size.
- Add health endpoint for worker queue depth and failure count.
- On startup, recover expired leases so no job stays stuck in `leased` indefinitely.
- Restrict network binding to localhost by default.
- Initialize and validate `sqlite-vector` on startup when selected; in `auto` mode, log a warning and fall back to brute-force search if extension cannot load.
