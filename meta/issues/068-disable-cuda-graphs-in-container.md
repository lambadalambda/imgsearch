# 068: Disable CUDA Graphs in the CUDA Container by Default

## Priority

P1

## Status

Resolved (2026-06-06). The CUDA container entrypoint now disables llama.cpp CUDA graphs by default and documents `IMGSEARCH_CUDA_GRAPHS=1` as the opt-out.

## Summary

GitHub issue #3 reports CUDA out-of-memory crashes on a 24 GiB RTX 4090 during video annotation with `-llama-native-annotator-variant 26b`. The crash occurs after repeated frame annotations and when the video-level summary starts.

## Context

- llama.cpp CUDA builds support CUDA graphs when compiled with `GGML_CUDA_GRAPHS`.
- The bundled llama.cpp source checks `GGML_CUDA_DISABLE_GRAPHS`; if the variable is present, CUDA graph execution is disabled.
- Issue #3 points to CUDA graph capture/replay memory growth or fragmentation as the likely failure mode.
- The CUDA container runner is the main supported deployment shape for GPU users.

## Acceptance Criteria

- [x] Add regression coverage for the CUDA runner exporting `GGML_CUDA_DISABLE_GRAPHS=1` by default.
- [x] Allow explicit opt-out to re-enable CUDA graphs.
- [x] Document the CUDA graph default and opt-out.

## Related Files

- `scripts/run_imgsearch_cuda_container.sh`
- `scripts/run_imgsearch_cuda_container_test.sh`
- `docs/podman-cuda-ubuntu.md`
