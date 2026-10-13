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

## Outcome — 2026-10-13

Deployed `178fa7190d528d66176f7f4d37bbfba061105dc6` as `/opt/imgsearch/releases/178fa71-vulkan` at 18:36Z, replacing `a7fd679` and the Qwen3-VL-Embedding-8B Q4_K_M embedder.

- Gates: full local `mise run test:full`, `git diff --check`, CT107 full Go suite (32 packages), clean `ldd`, stamped `-version`, service-user sqlite-vector and model-read probes, model SHA-256 match. Llama.cpp pin and frontend dist unchanged (copied from `a7fd679-vulkan`). Independent review of the cutover script; all should-fix findings applied.
- Service flags: `-llama-native-model-path /models/tencent/WeMM-Embedding-2B-GGUF/WeMM-Embedding-2B-Q4_K_M.gguf -llama-native-mmproj-path /models/tencent/WeMM-Embedding-2B-GGUF/mmproj-WeMM-Embedding-2B-f16.gguf -llama-native-prompt-format wemm -llama-native-dimensions 2048`.
- Startup: new `model_id=54`; purged 5,158 embeddings and 7,179 old-model jobs; enqueued 5,158 embed jobs; recorded 1,541 image and 479 video annotations as done (no reannotation). One previously failed video annotation was retried and succeeded. Vulkan `KHR_coopmat`, 25/25 layers offloaded; remote annotation backend and disabled transcription unchanged.
- Re-embed finished at 19:26:39Z (about 49 minutes, ≈0.57 s per row in the app path). Final queue: 5,158 embed, 1,541 image and 480 video annotations done, 0 failed. Service active, 0 restarts, no journal errors; public `/healthz` 200.
- Search smoke checks (text: cat, dog, girl with headphones, chat screenshot, car on a street, food on a plate; similar-image for a cat) returned results whose existing annotations match the query.

### Rollback artifacts (stamp `20261013T183625Z`)

- Binary: `/opt/imgsearch/bin/imgsearch.backup-20261013T183625Z`
- Unit file: `/root/imgsearch.service.before-178fa71-vulkan-20261013T183625Z`
- Prior link: `/opt/imgsearch/releases/current-before-178fa71-vulkan-20261013T183625Z` → `a7fd679-vulkan`
- Integrity-checked SQLite snapshot (before the purge): `/var/lib/imgsearch/data/imgsearch.sqlite.before-178fa71-vulkan-20261013T183625Z`

A code-only rollback restarts the 8B embedder with no 8B vectors, which re-embeds for more than 2 hours. Restoring the snapshot avoids that but drops changes made since the switch; decide manually.
