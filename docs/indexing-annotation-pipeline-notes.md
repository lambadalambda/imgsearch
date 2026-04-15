# Indexing And Annotation Pipeline Notes

## Context

These notes capture the current bottlenecks and the likely direction for making indexing searchable sooner while keeping GPU utilization high.

Status update:
- The pipeline now uses separate `embed_image` and `annotate_image` jobs.
- Searchability is driven by the `embed_image` job reaching `done`.
- Annotation runs as follow-up work and can be processed later by a worker with an annotator available.

Relevant code today:
- `internal/worker/queue.go`: `embed_image` handles embedding/indexing and `annotate_image` handles descriptions/tags.
- `internal/db/index_jobs.go`: `embed_image` and `annotate_image` are now distinct queue kinds.
- `internal/upload/service.go`: uploads enqueue both job kinds when annotations are still missing.
- `internal/search/http.go`: text search needs the embedder at request time via `EmbedText`, so the embedder cannot disappear from the API process without affecting search.
- `internal/images/http.go`: gallery state is derived from `embed_image` job state and currently requeues the same job when annotations are missing.
- `cmd/imgsearch/main.go`: the app initializes both the embedder and the annotator at startup and keeps them resident for the process lifetime.

## Current Bottlenecks

- Searchability is delayed more than necessary.
- A single job only becomes `done` after both embedding and annotation complete.
- This means a slow annotator blocks embeddings from being written, indexed, and exposed to search.
- On a 24 GB GPU, Qwen 8B plus Gemma 26B are too heavy to keep fully resident together.
- Even when the smaller annotator fits, throughput is still lower than it should be because the pipeline is strictly one-image-at-a-time and mixes CPU preprocessing, GPU work, DB work, and JSON handling in one serial loop.

## Important Constraint

The user wants search to work immediately, even if descriptions and tags arrive later.

That implies:
- embedding is in the critical path for search,
- annotation is not in the critical path for search,
- the embedder must remain available for `EmbedText` in the serving process unless query embedding moves somewhere else.

This makes a pure in-process "unload the embedder, load the annotator, then switch back" design risky for the API process. It may be acceptable for a dedicated worker process, but not as the only runtime mode if text search must stay live.

## Recommended Direction

### 1. Split indexing into two job kinds

Use separate jobs:
- `embed_image`
- `annotate_image`

Desired behavior:
- upload enqueues `embed_image`,
- `embed_image` writes the embedding and updates the vector index,
- search starts working as soon as `embed_image` finishes,
- `annotate_image` runs later and fills `description` and `tags_json`.

Why this is the smallest correct change:
- it fixes the main UX problem immediately,
- it decouples annotation failure from embedding success,
- it lets us schedule annotation differently later without reworking the core indexing path again.

Note:
- `index_jobs.kind` already exists,
- the unique key is `(kind, image_id, model_id)`,
- so adding `annotate_image` should not require a schema migration if we reuse the existing job table shape.

### 2. Keep the embedder available for search

If text search must stay live, the embedder should remain loaded in the serving process.

This leads to two practical modes:
- with the smaller annotator: keep both models resident if VRAM and throughput are acceptable,
- with the 26B annotator: run annotation in a dedicated worker mode or dedicated process so the heavy annotator can load without taking the embedder away from live query embedding.

### 3. Treat model unloading as a worker concern, not an API concern

If the large annotator is required, the cleanest operational story is:
- API process keeps the embedder for query-time search,
- embedding worker prioritizes `embed_image`,
- annotation worker loads Gemma only when running `annotate_image`,
- process exit is the reliable way to fully release VRAM.

This can still be implemented incrementally:
- first split the job kinds,
- then decide whether annotation runs in the same binary with a worker mode flag or as a separate process.

## Throughput Ideas Worth Trying

These are ordered from likely-high-value and low-risk to more ambitious work.

### A. Separate embedding completion from annotation completion

This is the biggest win.

Even before any batching work, it:
- makes new images searchable earlier,
- prevents annotation latency from blocking the queue's most important output,
- reduces pressure to keep both models active at the same time.

### B. Add better timing and queue instrumentation

Measure at least:
- upload-to-searchable latency,
- embed duration,
- annotate duration,
- preprocess duration,
- DB write duration,
- queue depth by job kind and state,
- age of oldest pending job by kind.

Without this, it will be too easy to chase the wrong bottleneck.

### C. Batch claiming and reduce idle gaps

The current worker claims and processes one job at a time.

Likely useful next step:
- claim a small batch,
- keep the model hot,
- avoid per-image queue roundtrips between GPU calls.

This does not require changing search semantics and can be added after the job split.

### D. Overlap CPU preprocessing with GPU inference

Both embedding and annotation currently preprocess images before inference.

Likely improvements:
- preprocess the next image while the GPU is busy on the current one,
- avoid serial CPU -> GPU -> DB -> CPU gaps.

### E. Cache standardized preprocessed image artifacts

Potential follow-up:
- generate one reusable preprocessed image artifact for indexing and annotation,
- avoid repeating libvips resize and temp JPEG work across retries and across both phases.

This is useful if profiling shows preprocessing is a meaningful share of total latency.

## Implementation Notes To Remember

If we split job kinds, these places need attention:

- `internal/worker/queue.go`
  - stop doing annotation inline in `embed_image`,
  - add `annotate_image` processing,
  - decide when an `annotate_image` job is created.

- `internal/db/index_jobs.go`
  - add helpers for creating or requeueing `annotate_image` jobs,
  - stop reusing `embed_image` as the annotation backfill mechanism.

- `internal/images/http.go`
  - current gallery state is tied to `embed_image`,
  - current missing-annotation repair path requeues `embed_image`, which would become wrong after the split,
  - UI likely needs distinct notions of searchable vs annotated.

- `cmd/imgsearch/main.go`
  - avoid eager heavy annotator initialization when it is not needed yet,
  - consider dedicated worker mode flags later if the 26B path remains important.

- tests
  - queue tests should cover searchability after embedding but before annotation,
  - DB/job tests should cover idempotent `annotate_image` creation and requeue behavior.

## Open Product And Design Questions

- Is the smaller annotator good enough when the goal is continuous live indexing on a 24 GB card?
- Should annotation overwrite existing descriptions and tags, or only fill blanks?
- Do we want one overall "index state" in the UI, or separate searchable and annotated states?
- Is the 26B annotator meant for continuous background use, or for explicit batch backfill runs?

## Proposed Order Of Work

1. Add measurement and queue visibility.
2. Split `embed_image` and `annotate_image`.
3. Make search results available immediately after embedding commits.
4. Update UI and backfill logic so annotation no longer requeues embedding.
5. Use the shipped `-mode=all|api|worker` process split as the base for later dedicated embed-vs-annotate worker routing.
6. Revisit throughput with batching, preprocessing overlap, and cached preprocessed artifacts.
