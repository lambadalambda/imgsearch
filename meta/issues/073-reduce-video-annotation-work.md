# Reduce video annotation work without losing useful summaries

## Summary

Video annotation can run full image annotations for every sampled frame before generating the video-level summary. For a realistic benchmark with 5 videos, this can dominate total ingestion time. Reduce redundant frame work while preserving useful video search metadata.

## Requirements

- Benchmark current default video processing with the real ingestion harness.
- Compare frame count policies such as 3, 5, and 10 sampled frames.
- Explore scene/keyframe sampling or duplicate-frame filtering instead of uniform-only sampling.
- Evaluate whether every sampled frame needs a full Gemma annotation before the video summary.
- Preserve video search quality for varied videos, including text-heavy, multi-scene, and NSFW examples.

## Acceptance Criteria

- Full-load benchmark results show the impact of at least two video frame/work policies.
- Quality notes compare video summaries/tags for the tested policies.
- A recommended default or configurable policy is documented.

## Notes

- Embedding frames for video search and annotating frames for summaries can be treated as separate workloads.
- A useful policy may embed more frames than it annotates.
- 2026-06-06 CUDA sample with 19 images + 1 video, Qwen 8B embedder + Gemma E4B annotator:
  - `384`: elapsed `63s`; `embed_image` handled 29 images including 10 video frames, sum `2.090s`, avg `0.072s`; `annotate_image` for standalone images sum `34.370s`; `annotate_video` sum `20.407s`.
  - `1024`: elapsed `62s`; `embed_image` handled 29 images including 10 video frames, sum `2.013s`, avg `0.069s`; `annotate_image` for standalone images sum `36.498s`; `annotate_video` sum `22.247s`.
- Initial signal: video annotation adds roughly another 10-frame annotation workload plus summary generation; frame count/policy is likely a more useful optimization target than annotator max-side alone.
- 2026-06-06 exact production-model CUDA sample with 19 images + 1 video at annotator max side `384`, Qwen 2B Q6 embedder + HauhauCS Gemma E4B Q4_K_P annotator:
  - Elapsed `83s`; `embed_image` handled 29 images including 10 video frames, sum `1.468s`, avg `0.051s`; standalone `annotate_image` sum `47.448s`, avg `2.497s`; `annotate_video` sum `31.571s`.
- Production-model initial signal: the earlier 10-frame video annotation path was the largest single per-media cost observed; reducing annotated frame count or avoiding full per-frame rich annotations should be prioritized for video throughput.
- 2026-06-06 exact production-model CUDA single-video sweep at annotator max side `384`, Qwen 2B Q6 embedder + HauhauCS Gemma E4B Q4_K_P annotator, run from `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z`:
  - `10` sampled frames: elapsed `37s`; `embed_image` handled `10` video frames; `annotate_video` total `31.342s`; description length `1900` chars; tags included `the simpsons`, `spongebob squarepants`, `cartoon`, `animation`, `yellow characters`, `homer simpson`, `family show`, `tv show`, `mr job`, `underwater`.
  - `5` sampled frames: elapsed `22s`; `embed_image` handled `5` video frames; `annotate_video` total `16.249s`; description length `1422` chars; tags included `cartoon`, `simpsons`, `yellow characters`, `illustration`, `group portrait`, `cheerful`, `bright colors`, `homer simpson`, `animation`, `blue shirt`.
  - `3` sampled frames: elapsed `17s`; `embed_image` handled `3` video frames; `annotate_video` total `11.525s`; description length `1411` chars; tags included `cartoon`, `yellow character`, `david silverman`, `cheerful`, `animation`, `spongebob squarepants`, `simpsons`, `illustration`, `bright colors`, `teacher`.
- Quality notes for the single-video sweep: `10` frames produced the most complete multi-scene summary and tag set. `5` frames cut elapsed time by `15s` and annotation time by about half, while preserving the broad Simpsons/cartoon/Homer theme but missing the explicit SpongeBob tag on this sample. `3` frames was fastest and still captured broad cartoon/SpongeBob/Simpsons content, but the summary is more dependent on which frames are sampled and loses more sequence detail.
- Decision: make `5` sampled frames the configurable runtime default via `-video-frame-count`, based on the single-video CUDA sweep and the throughput/quality trade-off. Use `10` frames when broader scene coverage matters more than throughput; treat `3` frames as an aggressive speed mode until varied-video quality is checked.
- Default-5 confirmation run: image `imgsearch:cuda-bench-default5-20260606T1315` ingested the available `19 images + 1 video @384` production-model CUDA sample in `73s` from `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-default5-19img-1vid-20260606`. The upload stored `5` video frames, the worker logged `using 5 sampled frames`, and timings were `embed_image` 24 jobs sum `1.284s`, standalone `annotate_image` 19 jobs sum `48.535s`, and `annotate_video` 1 job sum `17.618s`.
- Full-load default-5 run: using expanded media source `/home/lain/imgsearch-ingestion-bench-work/media-source-expanded-50img-5vid-20260606`, image `imgsearch:cuda-bench-default5-20260606T1315` ingested `50 images + 5 videos @384` in `215s`. All 5 videos stored 5 sampled frames (`25` total video frames); timings were `embed_image` 75 jobs sum `3.288s`, standalone `annotate_image` 50 jobs sum `129.161s`, and `annotate_video` 5 jobs sum `76.439s`.
- Full-load 10-frame control: same source/image/models with `IMGSEARCH_BENCH_VIDEO_FRAME_COUNT=10` ingested `50 images + 5 videos @384` in `283s`. All 5 videos stored 10 sampled frames (`50` total video frames); timings were `embed_image` 100 jobs sum `4.878s`, standalone `annotate_image` 50 jobs sum `129.981s`, and `annotate_video` 5 jobs sum `142.202s`.
- Full-load comparison: `5` frames saved `68s` wall time versus `10` frames. The savings were concentrated in video annotation (`65.763s` less); extra video-frame embedding cost at `10` frames was only `1.590s`.
- Full-load quality spot-check: `10` frames improved detail for the multi-scene cartoon sample, including an extra character/source tag that `5` frames missed. The other four videos had broadly similar portrait/fashion-style tags and summaries between `5` and `10` frames, with wording variance but no major metadata gap observed. This supports `5` as the default and `10` as a higher-coverage override.
- Compact video-frame evidence path: added a `VideoFrameAnnotator` interface so sampled frames used as video-summary evidence can use a shorter prompt/output budget while standalone image annotations keep the rich prompt and `1024` token budget. The model switchboard now exposes this interface for the separate annotator model path; without that wrapper method, the first compact benchmark still fell back to rich `annotate_image` frame evidence.
- Focused compact-frame benchmark: image `imgsearch:cuda-bench-compact-video-frames-switchboard-20260606T1530` ingested the same `5 images + 1 video @384` production-model CUDA sample in `22s`, versus `37s` for rich frame evidence. Timings were `embed_image` 10 jobs sum `0.548s`, standalone `annotate_image` 5 jobs sum `11.732s`, and `annotate_video` 1 job sum `6.190s`. Native timing confirmed `annotate_video_frame` count `5`, average `0.707s` generation and `95.0` generated tokens per frame.
- Full-load compact-frame benchmark: using expanded media source `/home/lain/imgsearch-ingestion-bench-work/media-source-expanded-50img-5vid-20260606`, image `imgsearch:cuda-bench-compact-video-frames-switchboard-20260606T1530` ingested `50 images + 5 videos @384` in `165s`. All 5 videos stored 5 sampled frames (`25` total video frames); timings were `embed_image` 75 jobs sum `3.381s`, standalone `annotate_image` 50 jobs sum `128.264s`, and `annotate_video` 5 jobs sum `29.619s`.
- Compact-frame comparison: versus the default-5 rich-frame-evidence run, compact video-frame evidence saved `50s` wall time and reduced video annotation by `46.820s`; standalone image annotation stayed effectively unchanged (`129.161s` to `128.264s`). Average stored frame description length intentionally dropped from `1333.8` to `266.7` chars because video-frame descriptions are now evidence, not user-facing rich image records.
- Compact-frame quality spot-check: the five full-run video summaries remained useful and produced appropriate broad tags for the cartoon and portrait/fashion clips. They were shorter than rich-frame evidence summaries and showed wording/tag variance, but no major metadata gap was observed in this sample. Recommended current policy: keep `5` sampled frames as the default and use compact video-frame evidence for video summaries; keep `10` frames as a higher-coverage override for multi-scene videos. Scene/keyframe sampling and duplicate-frame filtering remain open follow-up work.
- Duplicate-frame evidence reuse: within one video annotation job, compact frame evidence is now cached by sampled frame `image_id`. This avoids duplicate generation and duplicate image annotation writes when static or looping videos sample identical frames while keeping every timeline frame entry in the video-summary input. The current `50 images + 5 videos` benchmark did not contain duplicate `(video_id, image_id)` frame references, so this is a targeted improvement for duplicate-heavy videos rather than a measured speedup on that dataset.
