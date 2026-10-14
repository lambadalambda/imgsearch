# Investigate mtmd vision embedding gap against transformers

## Summary

During the WeMM-Embedding-2B evaluation, our GGUF text embeddings matched the transformers reference at cosine 0.9999, but image embeddings only matched at 0.964–0.979. The inputs were identical 384×256 PNGs with the same 96 image tokens. The gap does not change with Q8_0 versus f16 weights, with an f16 versus f32 projector, or with Metal versus CPU. So it is a systematic difference in the llama.cpp mtmd Qwen3-VL/Qwen3.5 vision path (likely position-embedding interpolation or resampling), not quantization.

## Requirements

- Compare vision-tower outputs layer by layer (patch embedding, position embedding, merger) between mtmd and transformers for one image.
- Check whether Qwen3-VL-Embedding-2B (current default) shows the same gap.
- If a fix exists upstream or locally, measure its effect with `mise run bench:retrieval`.

## Acceptance Criteria

- The cause is documented, and either fixed (image cosine ≥ 0.995) or recorded as an accepted upstream limitation.

## Notes

- Upstream llama.cpp `45451c3` narrows the gap (Q4_K_M images 0.985–0.988 vs 0.946–0.970) but lowers Flickr recall by 0.7–1.2 points. A closer match to transformers does not automatically mean better retrieval in our pipeline, so measure any fix with the benchmark.

- Reproduction (scripts were temporary and are deleted): crop fixtures to 384×256 PNG, embed them with the transformers model (`AutoModel.from_pretrained(..., trust_remote_code=True).embedding(...)`, fp32) and with the GGUF via `embedImagePathLocked` (skips our JPEG re-encode), then compare cosine per input.
- Parent: [Evaluate WeMM-Embedding-2B](evaluate-wemm-embedding-2b.md).
