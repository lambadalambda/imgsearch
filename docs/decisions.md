# Architecture Decisions

## ADR-001: Vector Search Strategy for MVP
- Context: The app needs nearest-neighbor search over image vectors while staying simple.
- Decision: Use brute-force cosine similarity in Go over vectors stored in SQLite and cached in memory.
- Consequences: Exact results and simple implementation, but limited practical scale for MVP-sized libraries.

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
