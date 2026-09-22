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

## ADR-008: Lazy Persistent Image Thumbnails
- Context: Grid cards were transferring and decoding full originals, including for existing libraries. Resizing synchronously during list queries or backfilling every image at startup would delay browsing.
- Decision: Image list/get/duplicate and image-search responses expose `thumbnail_path`; when the stored column is empty, it points to `/media/images/thumbnails/{id}.webp` (without the `/media/` prefix in the field). The authenticated media handler generates a max-side 640px, quality-80 WebP on first request using the same libvips resize module as inference. It auto-orients, retains transparency, strips metadata, and does not upscale. Animated sources use their first frame. Originals remain unchanged.
- Persistence: Generate into a same-directory temporary file, rename the complete derivative, then record the existing `images.thumbnail_path` column. Existing image/video-frame deletion already removes recorded derivatives. Missing cache files regenerate; unrecorded cache files are not trusted, preventing stale ID-based thumbnails after a database reset.
- Trade-offs: The first view pays resizing cost; cache misses are serialized per handler to bound native work, without holding a DB connection during resize. Subsequent requests use the disk cache and private browser caching with revalidation (ID-only URLs can be reused after database resets). No new dependency, schema migration, background job kind, or eager backfill is required. Native resizing requires the existing cgo/libvips build; Atelier falls back to the original on thumbnail failure.
- UI: Atelier library/search/duplicate grids use derivatives; lightbox and Open original use full media. Video previews and the legacy UI are unchanged.

## ADR-009: Saved Collections Reference Existing Media
- Context: Users want named lists of images/videos they discover, without duplicating files or changing the existing single-library authentication model.
- Decision: Store collections and mixed-media memberships in SQLite (migration 16). A membership references exactly one image or video with a foreign key, per-target uniqueness, and cascading cleanup. Deleting a collection never deletes media. Video identity is its video ID, not a sampled frame ID.
- API/UI boundary: A separate collection handler serves CRUD, membership lookup, and mixed saved-item pages; it uses existing NSFW SQL predicates before pagination and existing image/video hydration for response parity. Atelier can reuse its normal media cards and lightbox. See [collections](collections.md).
- Trade-offs: Pages are bounded to 200 items and hydrate targets individually to avoid duplicating complex media SQL; batch hydration can follow if profiling warrants it. Lists are instance-wide, not per-user or shareable. Newest-membership ordering avoids a separate manual-order system.

## ADR-010: Bounded Remote Annotation Admission With Draining
- Context: The remote HTTP client had a configurable concurrency cap, but the serial worker never filled more than one slot. Users expect the saved Settings value to control actual annotation overlap.
- Decision: Keep one dispatcher and reuse existing job execution/completion logic, admitting only remote image/video annotation jobs up to the saved limit independently of embedding batch size. Native/embedding/transcription work remains isolated. A resolver-side admission guard covers the interval between claiming and starting a call; changed backend configurations stop admission and drain active calls before replacing/closing resources.
- Safety: Active job IDs are excluded from this dispatcher's claims during explicit requeues. Existing owner-checked completion and lease renewal remain in use; shutdown cancels and joins jobs and renewers. Higher-priority work makes the dispatcher drain rather than continuously refill annotation slots.
- Trade-offs: Settings apply after the next dispatcher check (five-second polling throttle) and required drain; running embedding/transcription work can delay the check. Limits are per process; this is not a distributed limiter or a fairness redesign. Native serialization avoids duplicate GPU contexts/model loads. Concurrent videos that share an unannotated frame can still duplicate frame inference; that cache-coordination edge case is tracked separately.

## ADR-011: Opt-In Direct Strict Matching During Import
- Context: Users want unattended consolidation of resized/re-encoded copies, but the duplicate finder's transitive dHash groups are not proof of interchangeable content.
- Decision: `-dedup-strict-images` enables direct distance-2 matching in the upload transaction after its first write. Pick one largest existing standalone candidate, keep it on ties, or replace it with a larger incoming image. Preserve metadata provenance and collection ordering. Exact-content uploads, videos, and frame deduplication remain unchanged.
- Identity and safety: An upgrade uses the freshly inserted image ID, not in-place content mutation. This isolates stale worker and thumbnail results without changing their persistence interfaces. Remove old jobs/all-model embeddings atomically with reference transfer; preserve old disk files to avoid cross-process unlink races. No new migration or dependency.
- Trade-offs: Disabled by default; perceptual false positives and resolution-versus-quality differences remain possible. Old image-ID links are not redirected, superseded files consume disk, and other pre-existing duplicates are not swept. Candidate selection scans/sorts the library's standalone hashes per import; optimize only if measured ingestion cost warrants it. See [import deduplication](import-deduplication.md).

## ADR-012: Retire the Legacy Shell and Gate Atelier Delivery
- Context: The legacy escape route duplicated UI maintenance and could hide a missing Atelier build. CUDA builds depended on host dist and browser smoke previously skipped missing dist successfully.
- Decision: Supersede ADR-005's dual-shell rollout: serve only Atelier, return 404 for retired `/legacy` paths, and remove the legacy assets/navigation/proxy/smoke. Keep the source-only build-help fallback, without a legacy link. Root browser-cookie bootstrap and shared API/media/auth remain unchanged.
- Delivery: Reuse a dependency-free Node asset validator after Vite build, before release Go compilation (including skip-frontend-build mode), and at browser-smoke startup. Build Atelier inside a separate CUDA container stage; exclude host dist/dependencies. CI builds before Go tests. Validation checks index/app mount, emitted JS/CSS references and nonempty files; browser smoke checks runtime behavior.
- Product boundary: Retire legacy-only bulk-delete UI, tags explorer and live-connection panel while retaining backend APIs. Retry-failures UX follows existing UX-08, not a parity migration. See [frontend architecture](frontend.md).
- Trade-offs: Source-only `go build` still permits backend development without Node, but is not a release recipe. Packaging requires Node 24+ even when reusing an existing dist. Structural asset validation cannot prove all browser behavior; retain Playwright.
