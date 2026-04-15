# Architecture Decisions

## ADR-001: Vector Search Strategy for MVP
- Context: The app needs nearest-neighbor search over image vectors with better scaling than linear scans.
- Decision: Use `sqlite-vector` as the default vector backend behind a `VectorIndex` interface.
- Consequences: Better retrieval performance in SQLite while keeping backend swappable for future changes.

## ADR-002: Embedding Integration for MVP
- Context: Different local model runtimes have different packaging constraints.
- Decision: Standardize on the in-process `llama-cpp-native` runtime for text and image embedding, while keeping Go-side interfaces for search and worker code.
- Consequences: Packaging and local development are simpler, query-time search keeps direct access to the embedder, and future worker/process splits can happen above the runtime boundary instead of through a sidecar protocol.

## ADR-003: Queue Reliability Model
- Context: Background indexing must survive process crashes and restarts.
- Decision: Use lease-based jobs (`pending`, `leased`, `done`, `failed`) with expiry and recovery.
- Consequences: No permanently stuck in-flight jobs and predictable retry behavior.

## ADR-004: Content-Addressed File Storage
- Context: Duplicate uploads and idempotency should be easy to reason about.
- Decision: Store images by SHA-256 content hash and deduplicate by hash in the database.
- Consequences: Stable identifiers, easier reprocessing, and simplified duplicate handling.
