# Architecture Decisions

## ADR-001: Vector Search Strategy for MVP
- Context: The app needs nearest-neighbor search over image vectors with better scaling than linear scans.
- Decision: Use `sqlite-vector` as the default vector backend behind a `VectorIndex` interface.
- Consequences: Better retrieval performance in SQLite while keeping backend swappable for future changes.

## ADR-002: Embedding Integration for MVP
- Context: Different local model runtimes have different packaging constraints.
- Decision: Use a pluggable embedding adapter and start with a local runtime endpoint for text/image embedding.
- Consequences: Faster development and easier model iteration; true single-binary model packaging is deferred.

## ADR-003: Queue Reliability Model
- Context: Background indexing must survive process crashes and restarts.
- Decision: Use lease-based jobs (`pending`, `leased`, `done`, `failed`) with expiry and recovery.
- Consequences: No permanently stuck in-flight jobs and predictable retry behavior.

## ADR-004: Content-Addressed File Storage
- Context: Duplicate uploads and idempotency should be easy to reason about.
- Decision: Store images by SHA-256 content hash and deduplicate by hash in the database.
- Consequences: Stable identifiers, easier reprocessing, and simplified duplicate handling.
