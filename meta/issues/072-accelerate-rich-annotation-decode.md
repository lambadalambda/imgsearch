# Accelerate rich annotation decode without shrinking output

## Summary

Rich image descriptions are valuable, so speed work should not start by capping output length. Explore decode/runtime optimizations that preserve rich output, especially speculative decoding and reusable prompt/KV state.

## Requirements

- Benchmark current rich-output annotation generation to estimate how much time is spent in generated tokens versus image/prompt prefill.
- Evaluate speculative decoding approaches that are compatible with multimodal Gemma annotation, such as n-gram speculative decoding or a draft model if practical.
- Evaluate native prompt/KV prefix caching for the stable system/user prompt before the image marker.
- Preserve rich annotation prompts and output budgets during tests.
- Test CPU-only and CUDA paths where the feature applies.

## Acceptance Criteria

- Timing data separates image/prompt prefill from autoregressive generation for representative annotations.
- At least one speculative decode strategy is benchmarked or rejected with a concrete compatibility/performance reason.
- Prompt/KV prefix caching is prototyped or a concrete llama.cpp API/design blocker is documented.
- The issue concludes with a recommendation for which rich-output decode optimization, if any, should move into production.

## Notes

- The native bridge currently keeps the model loaded but clears llama memory before each image, so prompt caching is expected to save repeated prefix evaluation rather than model load time.
- Speculative decoding only helps if generation is a meaningful fraction of wall time.
- 2026-06-06 native timing instrumentation now logs `preprocess`, `native_decode`, `tokenize`, `prefill`, `generate`, `generated_tokens`, and `prompt_tokens` for image, video-frame, and video annotation calls.
- Focused production-model CUDA timing run, `5 images + 1 video @384`, Qwen 2B Q6 embedder + HauhauCS Gemma E4B Q4_K_P annotator, run dir `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-annotation-timing-5img-1vid-20260606`:
  - Elapsed `37s`; `embed_image` 10 jobs sum `0.576s`; standalone `annotate_image` 5 jobs sum `12.488s`; `annotate_video` 1 job sum `18.294s`.
  - Native timings showed `annotate_image` count `10` because the video summary path performed hidden rich image annotation for 5 sampled frames before the final video summary.
  - Rich image/frame annotation generation dominated: `generate` sum `25.916s`, average `2.592s`, average `349.2` generated tokens. Prompt/image work was small by comparison: prefill averaged `0.074s`, preprocessing averaged `0.010s`, tokenization averaged `0.002s`.
  - Final video summary generation was smaller: `generate=2.884s`, `prefill=0.215s`, `prompt_tokens=2380`, `generated_tokens=370`.
- Current decode recommendation: prioritize reducing unnecessary generation calls before prompt/KV caching. Prompt/prefill cost is measurable but much smaller than autoregressive generation in the current production-model CUDA path.
- Speculative/MTP status: native imgsearch generation does not currently expose llama.cpp speculative decoding, and the checked `ik_llama.cpp` multimodal MTP path rejected the use case with `speculative decode is not supported by multimodal`; keep speculative decoding as future work unless upstream multimodal support is available.
- Rejected rich prompt-tightening experiment: changing the image prompt from the existing up-to-500-word rich budget to a 90-180 word / 260-word complex-scene target and lowering primary/retry output budgets to `768`/`384` improved speed but regressed classification. On the full compact-frame benchmark, image `imgsearch:cuda-bench-rich-prompt-260-20260606T2130` reduced elapsed time `165s -> 145s` and standalone image annotation `128.264s -> 105.334s`. However, deterministic image-only runs with `-llama-native-annotation-seed 42` showed NSFW tag coverage dropping from `10` images to `0` on the 50-image set. Stronger NSFW prompt wording over-tagged the first 12-image smoke (`2` baseline NSFW tags became `6`), and balanced wording still mismatched (`4`). Decision: do not ship this prompt/token tightening; preserve the richer baseline prompt until a quality-gated alternative can maintain NSFW behavior.
- 2026-06-07 native bridge speculative-decode review:
  - The current bridge already converts annotation JSON schemas to a llama.cpp grammar before generation, so enabling constrained JSON is not a new optimization. llama.cpp disables backend sampling when a grammar is active, which limits sampler-side offload options for annotation.
  - Upstream speculative decoding is not a sampler-only switch. `common_speculative_*` changes the decode loop: draft tokens are evaluated in the target context, `common_sampler_sample_and_accept_n` verifies them, and rejected draft KV entries must be rolled back or removed.
  - Draft-model speculation remains blocked for the multimodal annotator path. The upstream server path reports speculative decoding as unsupported with multimodal requests, and imgsearch does not have a compatible vision draft model path.
  - N-gram self-speculation is now wired as an opt-in experiment behind `-llama-native-annotation-ngram-speculation`. The bridge preserves the existing first-token path, then speculates only after a generated token has already been decoded and current logits are available. It builds text-token history from `mtmd_input_chunks` using `mtmd_input_chunk_get_tokens_text`, initializes `COMMON_SPECULATIVE_TYPE_NGRAM_MOD` with `n_match=8`, `n_min=2`, and `n_max=8`, verifies drafts through target-model sampling, removes unaccepted draft KV with `llama_memory_seq_rm`, and decodes the final mismatch token to restore the normal one-token-ahead state.
  - The n-gram implementation disables itself unless the target context supports partial sequence removal. Without reliable KV cleanup, rejected drafts can corrupt later tokens.
  - The timing log now exposes `speculative_drafted_tokens` and `speculative_accepted_tokens`. Run first on the focused `5 images + 1 video` CUDA benchmark before a full `50 images + 5 videos` run. Expected benefit is uncertain because rich image descriptions may not repeat long token spans often; gains are more likely on repeated JSON structure or repetitive text.
  - Prompt/KV prefix caching remains lower priority on CUDA. In the compact full run, rich standalone image annotation spent about `3.458s` in prefill versus `122.937s` in generation across `50` images, so even perfect prefix reuse would be much smaller than decode-side gains. Prefix caching would also need careful mtmd position handling around the image marker.
- 2026-06-07 n-gram speculation benchmark results with image `imgsearch:cuda-bench-ngram-spec-20260607T040630Z`:
  - Focused stochastic no-flag control, `5 images + 1 video`, run dir `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-ngram-control-focused-5img-1vid-20260607`: elapsed `22s`, total native generation `17.013s`, generated tokens `2157`.
  - Focused stochastic n-gram run, run dir `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-ngram-spec-focused-5img-1vid-20260607`: elapsed `27s`, total native generation `17.870s`, generated tokens `2295`, drafted `197`, accepted `42`, acceptance `21.3%`.
  - Focused deterministic no-flag control with `-llama-native-annotation-seed 42`, run dir `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-ngram-control-seed42-focused-5img-1vid-20260607`: elapsed `27s`, total native generation `18.150s`, generated tokens `2325`.
  - Focused deterministic n-gram run with seed `42`, run dir `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-ngram-spec-seed42-focused-5img-1vid-20260607`: elapsed `27s`, total native generation `17.914s`, generated tokens `2337`, drafted `195`, accepted `65`, acceptance `33.3%`.
  - Full n-gram run, `50 images + 5 videos`, run dir `/home/lain/imgsearch-ingestion-bench-work/imgsearch-20260606T104932Z/bench-results/prod-cuda-ngram-spec-full-50img-5vid-20260607`: elapsed `171s`, total native generation `152.993s`, generated tokens `19970`, drafted `2331`, accepted `755`, acceptance `32.4%`. Breakdown: `annotate_image generate=125.559s`, `annotate_video_frame generate=16.958s`, `annotate_video generate=10.476s`.
  - Previous compact-frame full baseline run was `165s` elapsed with total generation `150.128s` (`annotate_image=122.937s`, `annotate_video_frame=17.023s`, `annotate_video=10.168s`). Recommendation: keep `-llama-native-annotation-ngram-speculation` disabled; current n-gram settings do not improve ingestion wall time or total generation time.
