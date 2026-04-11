# MVP Plan

## MVP Outcome
Ship a local app where users can upload images, browse them, and search by text or by similar image, with indexing handled asynchronously in a persistent queue.

## Milestones

### Milestone 0: Feasibility Spike
- Validate one local multimodal embedding runtime for text and image embeddings.
- Confirm output dimension and cosine similarity behavior.
- Measure rough indexing throughput on a small local sample set.

Acceptance:
- A simple script can embed one text query and one image with consistent dimensions.
- Measured baseline is captured in notes for future comparison.

### Milestone 1: Project Bootstrap
- Initialize Go module and directory layout.
- Add SQLite connection and migration runner.
- Add baseline test harness.

Acceptance:
- App starts, runs migrations idempotently, and passes initial tests.
- Migration version is tracked in `schema_migrations`.

### Milestone 2: Upload + Persistence
- Implement upload endpoint.
- Persist files to local storage directory via temp-file staging.
- Create `images` and `index_jobs` records transactionally.
- Enforce MIME/format checks (JPEG/PNG in MVP).

Acceptance:
- Uploaded file appears in database and pending queue.
- Duplicate file uploads are idempotent by content hash.

### Milestone 3: Indexing Worker
- Implement queue polling and lease-based job claiming.
- Add embedding adapter interface and stub implementation for tests.
- Store vectors in `image_embeddings`.

Acceptance:
- Pending jobs move to done/failed state correctly.
- Expired leases are recovered after restart.

### Milestone 4: Search Endpoints
- Add text-to-image search endpoint.
- Add image-to-image search endpoint from indexed image ID.
- Return top-k results with metadata.

Acceptance:
- Test fixtures return deterministic top results.
- Cosine similarity ranking is deterministic and excludes self-match for similar-image search.

### Milestone 5: Web UI
- Build minimal UI with:
  - upload form,
  - gallery grid,
  - text search box,
  - similar-image action from gallery cards.

Acceptance:
- End-to-end manual flow works on localhost.

## Testing Strategy (TDD)
- Start each backend behavior with failing unit/integration tests.
- Use table-driven tests for handlers and queue transitions.
- Add integration tests against temporary SQLite files.
- Use a fake embedder in early slices to avoid model-runtime coupling.

## First 3 TDD Slices
1. SQLite migrations + queue leasing primitives + recovery tests.
2. Upload endpoint + staged file writes + enqueue tests.
3. Worker `ProcessOne` + fake embedder + deterministic ranking tests.

## Out of Scope for MVP
- User accounts/auth.
- Distributed workers.
- Cloud storage.
- Advanced reranking and relevance feedback.
- ANN indexes and SQLite vector extensions.
- Full model packaging into a single binary.

## Post-MVP Priorities
- Incremental reindexing and model version migration.
- Batch import from folders.
- Duplicate detection and clustering.
- Better ranking with reranker/metadata signals.
