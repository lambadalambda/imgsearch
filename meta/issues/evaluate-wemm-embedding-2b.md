# Evaluate WeMM-Embedding-2B as the native embedder

## Summary

Tencent's WeMM-Embedding-2B (Apache 2.0, released 2026-08-26) reports MMEB v2 Image 79.6 / Video 70.8 / Overall 77.9, against 75.0 / 61.9 / 73.2 for our default Qwen3-VL-Embedding-2B and 77.8 overall for Qwen3-VL-Embedding-8B. It is built on Qwen3.5-2B, outputs 2048 dimensions with Matryoshka truncation, and uses last-token pooling. Our vendored llama.cpp (2026-06-06) already converts `Qwen3_5ForConditionalGeneration` and maps its vision tower to the `QWEN3VL` projector, so it may run through the existing native path without a llama.cpp bump.

The scores are vendor-reported. Decide from our own measurements whether to adopt it as the default, offer it as an optional profile, or reject it.

Sub-issues:

- [Add text-to-image retrieval benchmark](add-text-to-image-retrieval-benchmark.md)

## Requirements

- Convert the official `tencent/WeMM-Embedding-2B` weights with the vendored llama.cpp converter (text GGUF + mmproj). Use community GGUFs only as a cross-check.
- Verify the GGUF against the reference PyTorch model (cosine similarity for text and image inputs).
- Find the reference prompt template and instructions, and check that `compose_prompt` in `bridge.cc` matches them, or make the template configurable.
- Compare WeMM-2B with Qwen3-VL-2B (and 8B where possible) on the retrieval benchmark, the existing fixture integration tests, and embedding throughput/memory.
- Check that multi-sequence batching gives the same vectors as sequential embedding (Qwen3.5 has recurrent Gated DeltaNet layers).

## Acceptance Criteria

- `docs/` records the results and a decision: adopt as default, optional `serve:*` profile, or reject.
- If adopted in any form, the existing native integration tests pass with the WeMM model.

## Notes

- Model card: https://huggingface.co/tencent/WeMM-Embedding-2B; report: https://arxiv.org/pdf/2608.24053.
- Switching models needs a full re-embed and touches [mixed-model search](050-sqlite-vector-quantized-mixed-models.md).
- Related candidates: Ovis-VL-Embedding-2B (weights not confirmed public), Ovis-Embedding-Omni-3B (audio; see [audio embedding watch](watch-ovis-omni-audio-embeddings.md)).
