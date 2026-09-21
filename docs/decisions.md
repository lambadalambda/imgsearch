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

## ADR-005: Svelte SPA Embedded in the Go Binary
- Context: The single-file plain-JS shell grew past what is maintainable, and the library UI needed real components (masonry, lightbox, Feed, settings).
- Decision: Build the Atelier frontend with Svelte 5 + Tailwind 4 + Vite under `frontend/`, embed the built `dist` into the Go binary, and keep the legacy shell at `/legacy`. No SSR or Node at runtime; routing is query-string based.
- Consequences: One binary still deploys everything; the frontend must be built before `go build` for `/` to work (`mise run build:frontend`); details in `docs/frontend.md`.

## ADR-006: Video Support Through Sampled Frames
- Context: Users upload short videos alongside images and want them searchable with the same text and similarity queries.
- Decision: Sample a small number of frames per video with ffmpeg at upload time, store them as `images` rows linked by `video_frames`, embed them with the image model, and let the annotator summarize the video from its frame annotations. Optional ASR (Parakeet via ONNX Runtime) adds transcript embeddings.
- Consequences: No separate video model; search results carry a matching timestamp; frame count is tunable with `-video-frame-count`; frames share the image pipeline and its quotas.

## ADR-007: Annotation Model Choice
- Context: Titles, summaries, descriptions, and tags need a vision-language model; the search embedder cannot generate text.
- Decision: Default to the native in-process Gemma `e4b` GGUF annotator via llama.cpp, with `26b` as an opt-in for more capable hardware, and allow an OpenAI-compatible remote server as an alternative backend selectable at runtime from the settings page. Prompts and response parsing are shared so all backends produce the same annotation shape.
- Consequences: Zero-config annotations on a laptop, a path to better quality on a GPU host, and no vendor lock-in; annotation quality and speed depend on the chosen backend.
