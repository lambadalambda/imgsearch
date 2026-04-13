# Changelog

All notable changes to this project are tracked in this file.

## Unreleased
- perf: speed up sqlite-ai image preprocessing by using a persistent vips helper process created before model load, avoiding expensive per-image fork/exec overhead.
- test: add sqlite-ai helper and integration benchmarks covering preprocess and `EmbedImage` throughput.
- perf: isolate sqlite-ai embedding in a dedicated in-memory SQLite runtime so long `llm_embed_generate` calls do not monopolize the main app DB connection.

## 2026-04-13
- test: restore webp and avif upload fixtures (`7563fa7`)

## 2026-04-12
- fix: default sqlite-ai context pooling to last token (`65ff255`)
- test: add sqlite-ai fixture retrieval quality check (`466be11`)
- docs: clarify sqlite-ai preprocessing options (`55cb673`)
- feat: preprocess sqlite-ai images with vips (`897bad7`)
- test: refresh retrieval fixture image set (`df192b0`)
- chore: add qwen3 fixture evaluation task (`4a2b982`)
- test: add qwen3 fixture retrieval evaluator (`759903f`)
- fix: make qwen3 setup and serve work on mac (`8315fef`)
- feat: add sqlite-ai in-db embedding path and e2e coverage (`e416d6d`)

## 2026-04-11
- feat: add optional negative prompt for text search (`103d4a0`)
- feat: add qwen3-vl-embedding-8b remote sidecar support (`3cb23d1`)
- fix: improve vector search isolation and browsing UX (`aee32db`)
- feat: support remote torch sidecar embedding flow (`d973d2e`)
- fix: reconcile missing index jobs and correct progress metrics (`d06ed43`)
- feat: add indexing observability and failed-job retry controls (`dba1a0e`)
- feat: add bulk import task and webp/avif upload support (`72dfe51`)
- feat: serve built-in UI with sqlite-vector bootstrap (`8b358fb`)
- test: cover Jina sidecar API flow and tidy local artifacts (`2d27f8e`)
- chore: add jina setup/test tasks and stabilize sidecar (`0fac66c`)
- test: add opt-in sidecar similarity integration check (`228d976`)
- feat: add jina mlx embedder backend via local sidecar (`b1cb1ed`)
- feat: add paginated images listing endpoint (`fb143d2`)
- feat: start background worker loop and add search endpoints (`f6dc72c`)
- feat: implement lease-based worker queue processing (`5bc8553`)
- feat: add upload pipeline with queued indexing jobs (`7532d41`)
- test: add image fixtures for similarity search (`e636542`)
- chore: adopt mise workflow and tighten TDD rule (`2795b89`)
- feat: bootstrap Go app with migrations and vector index contracts (`9c6c1c8`)
- docs: adopt sqlite-vector behind a swappable search interface (`d468b75`)
- docs: initialize imgsearch foundation and MVP architecture (`53494ce`)
