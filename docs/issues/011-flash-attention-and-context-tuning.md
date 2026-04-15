# 011: Enable Flash Attention and Tune llama.cpp Context Parameters

**Status: Done**

## Context

The llama.cpp bridge (`internal/embedder/llamacppnative/bridge.cc`) used default context parameters with no explicit Flash Attention control, no KV cache quantization, and a fixed `n_ctx=8192` regardless of actual token usage. The embedding model at 384px produces well under 1024 tokens per image, meaning the 8192-context KV cache wastes VRAM.

## Benchmark Results (M2 Max, Qwen3-VL-Embedding-8B Q4_K_M, 100 images from ~/old)

| Metric | Before (n_ctx=8192) | After (n_ctx=512) | Delta |
|--------|---------------------|-------------------|-------|
| Per image latency | 538ms | 530ms | ~1.5% (noise) |
| KV cache size | **1152 MiB** | **72 MiB** | **-1080 MiB** |
| CPU compute buffer | 56 MiB | 41 MiB | -15 MiB |
| Flash attention | auto → enabled | auto → enabled | unchanged |
| Image tokens/image | 49-120 | 49-120 | unchanged |
| Retrieval quality | 4/4 queries pass | 4/4 queries pass | unchanged |
| Semantic similarity | woman>dog verified | woman>dog verified | unchanged |

## What Was Done

### A. Flash Attention (already auto-enabled, now explicitly controllable)
Flash Attention was already being set to AUTO by llama.cpp's defaults, which auto-enables it. The change makes this controllable via `-llama-native-flash-attn` (-1=auto, 0=off, 1=on) and `-llama-native-annotator-flash-attn`.

### B. Reduced Embedder Context Size (default 8192 → 512)
Profiled actual token count: 49-120 tokens per image at 384px. Changed default `n_ctx` from 8192 to 512. This freed **~1080 MiB** of VRAM with no measurable latency impact or quality regression.

### C. KV Cache Quantization (exposed as CLI flags, not enabled by default)
Added `-llama-native-cache-type-k` and `-llama-native-cache-type-v` flags (accepts ggml_type values: -1=default, 0=f32, 1=f16, 8=q8_0). Not enabled by default because: (1) KV cache is already f16 by default on Metal, (2) llama.cpp source shows KV cache quantization has complex interactions with flash attention, and (3) with n_ctx=512 the KV cache is only 72 MiB so savings are minimal.

### D. CLI Flags Added
- `-llama-native-flash-attn` (default: -1/auto)
- `-llama-native-cache-type-k` (default: -1/auto)
- `-llama-native-cache-type-v` (default: -1/auto)
- `-llama-native-annotator-flash-attn` (default: -1/auto)
- `-llama-native-annotator-cache-type-k` (default: -1/auto)
- `-llama-native-annotator-cache-type-v` (default: -1/auto)

## Risks Encountered
- Go zero-value for int is 0 (LLAMA_FLASH_ATTN_TYPE_DISABLED), not -1 (AUTO). Fixed by mapping 0 → -1 at the Go level in both `New()` and `newGemmaNativeRuntime()`.
- KV cache quantization (type_k/type_v) has conflicting behavior in llama.cpp: requires flash attention, but also errors when flash attention is on and cache is quantized. Left as opt-in via flags for expert tuning on CUDA.

## Files Changed
- `internal/embedder/llamacppnative/bridge.h` — added 3 params to `imgsearch_llama_new`
- `internal/embedder/llamacppnative/bridge.cc` — apply flash_attn_type, type_k, type_v to context params
- `internal/embedder/llamacppnative/embedder_native.go` — Config struct + default mapping
- `internal/embedder/llamacppnative/gemma_runtime_native.go` — AnnotatorConfig struct + default mapping
- `cmd/imgsearch/main.go` — 6 new CLI flags + wiring
- `cmd/imgsearch/llama_cpp_native_embedder.go` — options struct + wiring
- `cmd/imgsearch/llamacpp_native_annotator.go` — options struct + wiring
- `cmd/imgsearch/default_model_assets.go` — new constants
- `internal/embedder/llamacppnative/benchmark_integration_test.go` — updated default
- `internal/embedder/llamacppnative/integration_test.go` — updated default
