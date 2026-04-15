# Changelog

All notable changes to this project are tracked in this file.

## Unreleased
- perf: reduce default embedder context size from 8192 to 512, freeing ~1 GiB of KV cache memory (1152 MiB → 72 MiB) with no quality regression or latency change on the 100-image benchmark.
- feat: add `-llama-native-flash-attn`, `-llama-native-cache-type-k`, `-llama-native-cache-type-v` CLI flags for llama.cpp attention and KV cache tuning. Same flags available for the annotator with `-llama-native-annotator-*` prefix. Default is auto (-1) for all three, preserving existing behavior.
- perf: lower the default embedding image cap from 512 to 384, cutting representative `~/old` embedding benchmark time by about 42% while keeping fixture retrieval quality green.
- test: allow native embedding benchmarks to run against a real image directory via `LLAMA_NATIVE_BENCH_IMAGE_DIR` and `LLAMA_NATIVE_BENCH_IMAGE_LIMIT`.
- fix: make local `mise run serve:8b` self-heal sqlite-vector platform mismatches and add a startup smoke task.
- feat: split indexing into `embed_image` and `annotate_image` jobs so images become searchable before annotation completes.
- feat: add `-mode=all|api|worker` so the HTTP server and background worker can run in separate processes.
- ci: run the Go test suite on GitHub Actions for pull requests and main/master pushes.
- refactor: drop legacy sidecar and HTTP embedder variants so imgsearch uses `llama-cpp-native` only, removing the old `-embedder` switch and sidecar/server flags.
- feat: switch UI status/gallery refresh from 5s client polling to `/api/live` WebSocket snapshots, with automatic polling fallback on disconnect/unsupported clients.
- fix: harden `/api/live` websocket handling with read limits, ping/pong timeouts, safer origin validation, reconnect backoff, and a lower 2s push cadence.
- ux: add a live connection badge, reduce unnecessary rerenders/status spam, and de-emphasize manual refresh controls while live updates are healthy.

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
