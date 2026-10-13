# Bump llama.cpp for embedding speed

## Summary

The vendored llama.cpp is pinned at `5343f45` (2026-06-06); upstream has about 2,000 newer commits. Candidates relevant to us: Vulkan int8 coopmat1 matmul for AMD RDNA3 (NAS 780M), a tuned Gated DeltaNet kernel (Qwen3.5/WeMM), small-M Qwen matmul tuning, Metal SSM_CONV fusion, and native Qwen3.5 embedding conversion. Public headers (`llama.h`, `mtmd.h`, `common.h`, `chat.h`) changed substantially, so `bridge.cc` likely needs porting. Do this before switching the default embedder to WeMM.

## Requirements

- Move the submodule to a current upstream commit and port `bridge.cc` for embedding and Gemma annotation.
- Measure before/after with `mise run bench:retrieval` for WeMM Q4_K_M on the Mac (Metal) and CT107 (Vulkan), and check that quality does not drop.
- Check Gemma annotation still works (native annotation integration tests) and its speed does not regress.
- Check whether upstream conversion of WeMM works without the tokenizer workaround.

## Acceptance Criteria

- Full test suite and native embedding/annotation integration tests pass.
- Before/after numbers are recorded in `docs/embedding-model-evaluation.md`; the bump is kept only if it does not regress quality or speed.

## Notes

- Annotation uses the same libraries; a regression there blocks the bump.
- The NAS needs a Vulkan rebuild inside CT107 (`-DGGML_VULKAN=ON`, `-j2`).

## Outcome — 2026-10-13

Not adopted. Upstream `45451c3` lowered WeMM retrieval by 0.7–1.2 T2I@1 points and was at most 3% faster on the NAS, because the RDNA3 int8 matmul path is not used on the 780M. Gemma annotation showed no regression. The numbers and porting notes are in `docs/embedding-model-evaluation.md`; the port stays on branch `chore/llama-cpp-bump` (`ce4bc76`) for a later attempt.
