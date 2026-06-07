# Add real ingestion benchmark harness

## Summary

Add a reproducible benchmark harness that measures real `imgsearch` ingestion throughput instead of isolated model-server microbenchmarks. The benchmark must support quick smoke runs and a realistic load such as 50 images plus 5 videos, and it must run in Podman on `aiko` for both CPU-only and CUDA configurations.

## Requirements

- Run benchmark containers with Podman on `aiko` to keep tests reproducible and avoid polluting host accounts.
- Support a small smoke dataset for iteration and a real dataset target of 50 images plus 5 videos.
- Exercise the app ingestion path, including upload/store, embedding, annotation, video frame processing, and DB writes.
- Support CPU-only and CUDA runs with identical dataset manifests and comparable configuration.
- Capture total wall time, per-job timings, retries/failures, generated annotation counts, and model/runtime configuration.
- Keep output artifacts in a run directory so baseline and experiment runs can be compared.

## Acceptance Criteria

- A documented command can run a smoke benchmark in Podman on `aiko`.
- A documented command can run the full 50-image plus 5-video benchmark in Podman on `aiko`.
- Benchmark output includes total wall time, jobs completed by kind, and enough metadata to identify the model/runtime/container configuration.
- CPU and CUDA benchmark modes are both represented, even if one is initially marked experimental.

## Notes

- Quick exploration can use smaller sets, but optimization decisions should be based on the full realistic load.
- Final comparisons should report median results across repeated runs where feasible.
- 2026-06-06: CUDA image `imgsearch:cuda-bench-20260606T104932Z` built successfully on `aiko` from isolated workspace `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z`.
- 2026-06-06: one-image CUDA smoke passed with Qwen 8B embedder + Gemma E4B annotator in `12s`; summary at `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/smoke-cuda-1img-e4b-20260606T1139Z/summary.json`.
- 2026-06-06: `19 images + 1 video` CUDA benchmark passed in `62-63s`, with 10 sampled video frames stored as images; summary dirs `cuda-19img-1vid-384-20260606` and `cuda-19img-1vid-1024-20260606`.
- 2026-06-06: CPU mode is represented by a `5 images @384` run that passed in `168s`; matching CUDA run passed in `19s`.
- 2026-06-06: exact production model files downloaded to `aiko` under `/home/lain/imgsearch-ingestion-bench-work/prod-models`; production-model CUDA `19 images` runs passed at both `384` and `1024` annotator max side.
- 2026-06-06: production-model CUDA `19 images + 1 video @384` passed in `83s`, with 10 video frames embedded and one video annotation job completed.
- 2026-06-06: production-model CUDA default-5 run `19 images + 1 video @384` passed in `73s` with image `imgsearch:cuda-bench-default5-20260606T1315`; `embed_image` handled 24 images including 5 video frames, standalone `annotate_image` sum `48.535s`, and `annotate_video` sum `17.618s`.
- 2026-06-06: sourced additional benchmark media from `https://boards.4chan.org/wsg/thread/6147710` and `https://boards.4chan.org/s/thread/22404595`; expanded source `/home/lain/imgsearch-ingestion-bench-work/media-source-expanded-50img-5vid-20260606` now has `64` images and `9` videos available.
- 2026-06-06: full production-model CUDA default-5 run `50 images + 5 videos @384` passed in `215s` with image `imgsearch:cuda-bench-default5-20260606T1315`; `embed_image` handled 75 images including 25 video frames, standalone `annotate_image` sum `129.161s`, and `annotate_video` sum `76.439s`.
- 2026-06-06: matching full production-model CUDA control run with `IMGSEARCH_BENCH_VIDEO_FRAME_COUNT=10` passed in `283s`; `embed_image` handled 100 images including 50 video frames, standalone `annotate_image` sum `129.981s`, and `annotate_video` sum `142.202s`.
- Full-load default-5 vs 10-frame result: default-5 saved `68s` wall time; video annotation dropped by `65.763s`, while standalone image annotation was effectively unchanged.
