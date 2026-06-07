# Benchmark and tune annotator image resolution

## Summary

The annotator currently defaults to a 1024 pixel maximum side while embedding defaults to 384 pixels. Since 384 worked well for image retrieval, benchmark whether lower annotation resolutions preserve useful rich descriptions while reducing ingestion time.

## Requirements

- Benchmark annotator max-side settings such as 384, 512, 768, and 1024.
- Measure both quick per-image annotation timing and full ingestion timing with the real benchmark harness.
- Compare annotation quality on diverse images, including text-heavy, detailed, and NSFW examples.
- Preserve rich descriptions; do not use output caps as the primary speed strategy.
- Decide whether the default annotator max side should change or whether a dynamic policy is needed.

## Acceptance Criteria

- Benchmark results show CPU and CUDA timing for at least three annotator resolution settings.
- A small quality review records whether lower resolution loses important text, object detail, or NSFW signals.
- The chosen default or policy is documented and covered by configuration/tests where applicable.

## Notes

- This may turn out to be a configuration bug rather than an advanced optimization.
- Reannotation can remain a higher-resolution path if quality review supports a fast default plus high-quality retry policy.
- 2026-06-06 CUDA image-only sample, Qwen 8B embedder + Gemma E4B annotator, 19 images:
  - `384`: elapsed `42s`; `annotate_image` sum `34.100s`, avg `1.795s`; `embed_image` sum `1.457s`, avg `0.077s`; average description length `896.8` chars.
  - `1024`: elapsed `42s`; `annotate_image` sum `36.323s`, avg `1.912s`; `embed_image` sum `1.480s`, avg `0.078s`; average description length `931.4` chars.
- 2026-06-06 CPU/CUDA spot comparison at `384`, 5 images:
  - CUDA: elapsed `19s`; annotation avg `1.894s`; embedding avg `0.102s`.
  - CPU: elapsed `168s`; annotation avg `26.311s`; embedding avg `6.752s`.
- Initial signal: `1024` only added about `6.5%` CUDA annotation time on the 19-image sample, so output decode/model work appears to dominate over image max-side preprocessing for this model/dataset. Quality review and at least one more resolution point are still needed before changing defaults.
- 2026-06-06 CUDA image-only sample with exact production models, Qwen 2B Q6 embedder + HauhauCS Gemma E4B Q4_K_P annotator, 19 images:
  - `384`: elapsed `53s`; `annotate_image` sum `47.898s`, avg `2.521s`; `embed_image` sum `1.058s`, avg `0.056s`; average description length `1354.8` chars.
  - `512`: elapsed `53s`; `annotate_image` sum `48.714s`, avg `2.564s`; `embed_image` sum `1.055s`, avg `0.056s`; average description length `1360.1` chars.
  - `1024`: elapsed `57s`; `annotate_image` sum `49.958s`, avg `2.629s`; `embed_image` sum `1.109s`, avg `0.058s`; average description length `1380.9` chars.
- Production-model initial signal: `1024` added about `4.3%` CUDA annotation time over `384` on the 19-image sample, while `512` was about `1.7%` slower. The higher default may not be the main throughput bottleneck on CUDA, but a quality comparison is still needed before deciding whether `384` or `512` is safe as the default.
