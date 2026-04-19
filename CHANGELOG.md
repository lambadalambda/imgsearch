# Changelog

All notable changes to this project are tracked in this file.

## Unreleased
- feat(deploy): add Ubuntu-based `Containerfile.cuda` for Podman GPU deployments (`--gpus=all`), plus container entrypoint defaults for sqlite-vector/model paths and a deployment guide (`docs/podman-cuda-ubuntu.md`).
- feat(annotations): upgrade image annotation prompts for richer retrieval detail (including explicit people-focused descriptors and translation of non-English visible text), allow longer descriptions when useful (up to ~500 words), and pass original filenames into annotation context only when they appear semantically meaningful.
- feat(video): add model-driven `annotate_video` jobs that synthesize video-level descriptions/tags from sampled frame annotations plus transcript context, persist them on `videos`, and surface them in `/api/videos` and search results.
- ux(search): include search debug details in success status text (duration, index backend/strategy, and quantization on/off) for text and similar-search responses.
- perf(search): switch sqlite-vector search to `vector_quantize_scan` using request-sized top-k (instead of corpus-sized scan windows), with automatic re-quantization/preload when embeddings change and full-scan fallback when quantization functions are unavailable.
- feat(ui): hide NSFW-tagged media by default with a workspace-level `Show NSFW` toggle, persisting preference and applying backend `include_nsfw` filtering across gallery/video/search/tag-cloud APIs so pagination totals stay correct.
- fix(ui): use touch-first card interactions on coarse-pointer devices by showing thumbnail actions without hover, suppressing hover-only detail overlays, and preventing tap collisions between action controls and media-open handlers.
- fix(ui): prevent slight mobile horizontal scrolling by tightening small-screen shell/popover widths, clamping tab button shrink behavior, and adding a horizontal overflow guard.
- feat: extend `scripts/import_images.sh` to accept 4chan thread URLs and import full-size thread pictures plus `.webm` from `i.4cdn.org` (skipping thumbnails and unsupported thread files), with browser-like 4chan media headers, HTTP 429 retry handling, and paced per-file 4chan download timing with jitter.
- feat: add tag discovery and search workflows with `/api/search/tag-cloud` and paginated `/api/search/tags` (including grouped video matches), a workspace `Tags` tab with clickable cloud chips, advanced-search tag restrictions with autocomplete + `all/any` matching, and clickable card tags that trigger tag search.
- fix: treat empty/no-audio video transcripts as successful transcribe jobs, storing an empty transcript and skipping transcript embeddings instead of retrying/failing the job.
- fix: exclude derived video-frame images from "annotation gaps" stats so the dashboard only reports missing annotations for standalone images.
- ux: shift to a search-first masthead (issue 022) by shrinking top chrome, moving indexing controls into a quiet ops bar/disclosure, and demoting infrequent controls so Search stays primary.
- ux: add layered hover/focus card detail overlays (issue 020) so clipped title/supporting content/tags expand above neighboring cards without changing grid row height.
- ux: move similar/delete controls into thumbnail action overlays (issue 021), keeping cards calmer at rest while preserving keyboard and delete-confirmation behavior.
- ux: tighten UI chrome/radius language (issue 018) by collapsing to shared panel/control radius tokens, reducing over-pill styling, and calming default tab/card framing.
- ux: standardize card resting density (issue 019) with fixed media/meta proportions, aggressive title/path/supporting-text clamping, and single-row tag overflow chips to keep image/video/result grids visually consistent.
- feat: add configuration-gated transcript-backed video search using a local Parakeet ONNX pipeline. When `-parakeet-onnx-bundle-dir` and `-parakeet-onnxruntime-lib` are set, videos are transcribed in the worker, transcript text is shown in the Videos tab, transcript embeddings are stored with the active Qwen model, and text search can match videos by transcript content. The integrated recognizer now reuses ONNX sessions across jobs, and CoreML stays off by default because the CPU path has been more stable and memory-efficient in practice.
- feat: add MVP video search by sampling a fixed number of representative frames per uploaded video, embedding those frames through the existing image pipeline, and grouping frame hits back into video search results with timestamps and preview frames.
- feat: extend the bulk import task so `mise run import-images <dir>` also imports supported videos (`.mp4`, `.mov`, `.webm`, `.mkv`) through the same upload pipeline, while skipping videos larger than 20 MB by default.
- feat: add `EmbedImages`, worker-side grouped image embedding plumbing, and native inspection/spike tests for issue 014. The current native implementation still uses the stable single-image bridge internally, the 100-image benchmark showed no throughput gain (`batch-2` ~= sequential, `batch-4` slower), and follow-up Metal spikes showed no easy win from tighter context sizing assumptions or dual-context parallelism.
- feat: add batch job claiming (`claimBatch`, `ProcessBatch`) and `-worker-batch-size` flag as worker infrastructure for issue 014 (batched inference). This reduces queue/DB overhead, but did not show a reliable standalone end-to-end speedup in the 100-image real-embedder benchmark.
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
