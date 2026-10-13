# Deploy WeMM-Embedding-2B Q4_K_M to the home instance

## Summary

Switch CT107 production from Qwen3-VL-Embedding-8B Q4_K_M to WeMM-Embedding-2B Q4_K_M. On the 780M it is 2.5× faster per image and 4.7× faster per query, with better retrieval.

## Requirements

- Follow the Obsidian LXC deployment workflow (commit-specific release, rollback artifacts, SQLite snapshot).
- Install the model files under `/models/tencent/` and update the service flags.
- Expect a full re-embed (about 4,900 images and the sampled video frames, roughly 50 minutes); old embeddings are purged on start.

## Acceptance Criteria

- Service healthy with the new model version; re-embed completes with no new failed jobs.
- Search smoke checks return sensible results; rollback path recorded.

## Notes

- Eval artifacts on CT107: `/opt/imgsearch/eval/wemm-924c9bb` (model files with verified SHA-256).
- Production had 2 failed index jobs before this work (2026-10-13), not caused by it.
