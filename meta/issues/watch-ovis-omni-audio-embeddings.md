# Watch Ovis-Embedding-Omni-3B for audio embeddings

## Summary

Video search uses frame embeddings and Parakeet transcripts, so sounds without speech (music, animals, crowds) are not searchable. Ovis-Embedding-Omni-3B (Qwen2.5-Omni-3B base) reports MMEB-v3 Image 77.6, Video 65.0, Audio 50.1 and MAEB 57.3, which is better than EmbeddingGemma 2 on every modality. The paper says the weights "will be open-sourced"; they were not confirmed public on 2026-10-13.

## Requirements

- Check for released weights and llama.cpp support (`qwen2.5o` / `qwen2a` projectors exist in the vendored mtmd).
- If available, scope audio-track embeddings for videos as a separate issue.

## Acceptance Criteria

- Either a scoped follow-up issue exists, or this issue records why audio embeddings are not pursued.

## Notes

- Paper: https://arxiv.org/html/2609.25165v1.
- EmbeddingGemma 2 was rejected for image/video search (MMEB v2 Image 57.3) but remains a small audio fallback.
- jina-embeddings-v5-omni is CC-BY-NC and weak on video retrieval.
