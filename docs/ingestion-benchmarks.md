# Ingestion Benchmarks

Use `scripts/benchmark_ingestion_container.sh` for end-to-end ingestion measurements. The harness runs an `imgsearch` container, stages a fixed media subset, uploads it through the HTTP API, waits for indexing jobs to drain, and writes run artifacts.

## CUDA on `aiko`

```sh
podman build -f Containerfile.cuda -t imgsearch:cuda .
scripts/benchmark_ingestion_container.sh /path/to/media ./bench-results/cuda-baseline
```

## CPU-Only Container Run

```sh
IMGSEARCH_BENCH_ACCELERATOR=cpu \
  scripts/benchmark_ingestion_container.sh /path/to/media ./bench-results/cpu-baseline
```

## Dataset Size

By default, the harness stages 50 images and 5 videos:

```sh
IMGSEARCH_BENCH_IMAGE_LIMIT=50 \
IMGSEARCH_BENCH_VIDEO_LIMIT=5 \
  scripts/benchmark_ingestion_container.sh /path/to/media ./bench-results/full
```

Use lower limits for smoke tests while iterating.

## Experiment Knobs

Annotator resolution sweep:

```sh
IMGSEARCH_BENCH_ANNOTATOR_IMAGE_MAX_SIDE=384 \
  scripts/benchmark_ingestion_container.sh /path/to/media ./bench-results/annotator-384
```

Video sampled-frame sweep:

```sh
IMGSEARCH_BENCH_VIDEO_FRAME_COUNT=5 \
  scripts/benchmark_ingestion_container.sh /path/to/media ./bench-results/video-frames-5
```

Arbitrary app flags can be passed through `IMGSEARCH_BENCH_APP_ARGS`.

Experimental annotation n-gram speculation:

```sh
IMGSEARCH_BENCH_APP_ARGS='-llama-native-annotation-ngram-speculation' \
  scripts/benchmark_ingestion_container.sh /path/to/media ./bench-results/ngram-spec
```

The harness starts the container on `0.0.0.0:8080`, so it always injects an explicit API key. Override the default with `IMGSEARCH_BENCH_API_KEY` when needed; the same key is used for the importer.

If the container exits before `/healthz` is ready, the harness fails immediately and prints the container logs instead of polling until the timeout.

## Outputs

Each run directory contains:

- `dataset-manifest.json`: staged files and source paths.
- `import.log`: upload/import output.
- `events.log`: benchmark progress events.
- `container.log`: container logs captured at shutdown.
- `summary.json`: elapsed time, media counts, and job states.

For real comparisons, run one warmup and at least three measured runs, then compare median `elapsed_seconds` from `summary.json`.

## Current Observations

On `aiko`, with the exact production Qwen 2B Q6 embedder and HauhauCS Gemma E4B Q4_K_P annotator, a single-video CUDA sweep at annotator max side `384` showed:

| Video frames | Elapsed | `annotate_video` | Quality note |
| --- | ---: | ---: | --- |
| 10 | `37s` | `31.342s` | Most complete multi-scene summary and tags. |
| 5 | `22s` | `16.249s` | Preserved broad Simpsons/cartoon/Homer theme, but missed the explicit SpongeBob tag on this sample. |
| 3 | `17s` | `11.525s` | Fastest; kept broad cartoon/SpongeBob/Simpsons content but lost more sequence detail. |

The runtime default is now `-video-frame-count=5` to cut video annotation work while preserving configurability. Use `10` frames when broader scene coverage matters more than throughput, and keep `3` frames as an aggressive speed mode until varied-video quality is checked.

The first production-like CUDA run using the default `5` frames, with the available `19 images + 1 video` source set, passed in `73s` using image `imgsearch:cuda-bench-default5-20260606T1315`. It embedded `24` images including `5` video frames, spent `48.535s` across `19` standalone image annotations, and spent `17.618s` on the single video annotation. The run directory is `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-default5-19img-1vid-20260606`.

After adding media from the requested 4chan threads, the expanded source directory `/home/lain/imgsearch-ingestion-bench-work/media-source-expanded-50img-5vid-20260606` contains `64` images and `9` videos. The first full default-5 CUDA run staged `50 images + 5 videos` and passed in `215s` with image `imgsearch:cuda-bench-default5-20260606T1315`. It embedded `75` images including `25` video frames, spent `129.161s` across `50` standalone image annotations, and spent `76.439s` across `5` video annotations. The run directory is `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-default5-50img-5vid-20260606`.

The matching full CUDA control run with `IMGSEARCH_BENCH_VIDEO_FRAME_COUNT=10` passed in `283s` on the same expanded source. It embedded `100` images including `50` video frames, spent `129.981s` across `50` standalone image annotations, and spent `142.202s` across `5` video annotations. Compared with the default `5`-frame run, the default saves `68s` wall time; nearly all of that comes from video annotation, which drops by `65.763s`. The run directory is `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-10frames-50img-5vid-20260606`.

Native annotation timing on a focused `5 images + 1 video` CUDA run showed that video annotation was still doing rich per-frame evidence generation. After adding the compact video-frame evidence path and exposing it through the model switchboard, the same focused run dropped from `37s` to `22s`; `annotate_video` job time dropped from `18.294s` to `6.190s`. The compact path logged `5` `annotate_video_frame` native calls averaging `0.707s` generation and `95.0` generated tokens, versus the old hidden rich frame calls averaging `2.592s` generation and `349.2` generated tokens.

The full CUDA run with compact video-frame evidence used image `imgsearch:cuda-bench-compact-video-frames-switchboard-20260606T1530` on the same expanded source and passed in `165s`. It embedded `75` images including `25` video frames, spent `128.264s` across `50` standalone image annotations, and spent `29.619s` across `5` video annotations. Compared with the earlier default-5 run, wall time improved by `50s` and video annotation work dropped by `46.820s`, while standalone image annotation remained effectively unchanged. The run directory is `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-compact-video-frames-full-50img-5vid-20260606`.

A rich image prompt-tightening experiment reduced standalone image annotation time but was rejected for quality. The tuned prompt image `imgsearch:cuda-bench-rich-prompt-260-20260606T2130` reduced the full `50 images + 5 videos` compact-frame run from `165s` to `145s` and standalone image annotation from `128.264s` to `105.334s`, but deterministic image-only comparison with `-llama-native-annotation-seed 42` showed NSFW tag coverage dropping from `10` images to `0` on the 50-image set. Stronger NSFW wording overcorrected on a 12-image smoke (`2` baseline NSFW tags became `6`), and balanced wording still mismatched (`4`). Keep the richer baseline prompt until a quality-gated approach preserves NSFW classification.

Speculative decode review found no safe production default yet. The native bridge already uses JSON-schema grammar for annotation output, and llama.cpp disables backend sampling when grammar is active. Draft-model speculation is blocked for the multimodal annotator path upstream. N-gram self-speculation is now wired as an opt-in benchmark experiment via `-llama-native-annotation-ngram-speculation`; benchmark logs include `speculative_drafted_tokens` and `speculative_accepted_tokens` so runs can show whether the drafts are useful. Expected benefit is uncertain for rich descriptions, and the flag should stay off unless measured on the target model/hardware. Prompt/KV prefix caching is lower priority on CUDA because standalone image annotations in the compact full run spent about `3.458s` in prefill versus `122.937s` in generation across `50` images.

N-gram speculation benchmark on `aiko` used image `imgsearch:cuda-bench-ngram-spec-20260607T040630Z` with `-llama-native-annotation-ngram-speculation`. The focused stochastic `5 images + 1 video` run passed in `27s`, versus a same-image no-flag control at `22s`; it drafted `197` tokens, accepted `42`, and increased total native generation time from `17.013s` to `17.870s`. A deterministic focused pair with `-llama-native-annotation-seed 42` passed at `27s` for both control and n-gram; n-gram drafted `195`, accepted `65`, and only changed total native generation time from `18.150s` to `17.914s`, with no wall-clock win. The full `50 images + 5 videos` n-gram run passed in `171s`, drafted `2331`, accepted `755` (`32.4%`), and spent `152.993s` in generation. The previous compact-frame full baseline passed in `165s` with `150.128s` generation. Recommendation: keep n-gram speculation disabled for production; current draft acceptance and loop overhead do not produce an ingestion speedup.
