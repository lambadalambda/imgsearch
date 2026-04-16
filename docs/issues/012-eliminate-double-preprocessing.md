# 012: Eliminate Double Image Preprocessing Round-Trip

**Status: Cancelled — benchmarked negative on M2 Max**

## Context

Currently every image goes through two separate processing stages that duplicate work:

1. **Go side** (`internal/embedder/llamacppnative/image_preprocess_vipsgen.go`): libvips thumbnail to target size (384px for embed, 1024px for annotate) + JPEG encode at quality 90 → writes temp file
2. **C++ side** (`bridge.cc`): `mtmd_helper_bitmap_init_from_file()` reads the temp file, decodes it via stb_image, then `maybe_resize_bitmap()` may resize again

For embedding at 384px, the Go side produces a 384px JPEG (~30KB), then C++ reads it, decodes it, and `maybe_resize_bitmap` sees it's already at target and skips. The JPEG decode is wasted work.

## What Was Tried

### Option A: Pass raw pixel data via CGo
Implemented `preprocessImageToPixels()` that uses vips `WriteToMemory()` to export raw RGB bytes, then `mtmd_bitmap_init()` on the C side to accept them directly — skipping the temp file write/read/decode cycle entirely.

New C bridge functions added:
- `imgsearch_llama_embed_image_pixels()` in bridge.h/bridge.cc
- `imgsearch_llama_generate_image_pixels()` in bridge.h/bridge.cc
- Refactored internal `generate_image()` to accept a pre-created `mtmd_bitmap*` instead of a file path

### Benchmark Results (M2 Max, 100 images, n_ctx=512)

| Metric | JPEG temp file (before) | Raw pixels (after) | Delta |
|--------|------------------------|-------------------|-------|
| Per image latency | ~530ms | **~720ms** | **+36%** |
| Allocations/op | 39 | 34 | -5 |
| Bytes/op | ~18KB | **~299KB** | **+16x** |

## Why It Was Reverted

The raw pixel approach is **significantly slower** because:
- 384px RGB = ~442KB vs JPEG = ~30KB — 15x more data across the Go→CGo boundary
- vips `WriteToMemory()` forces full image materialization and is slower than JPEG encode
- On Apple Silicon (unified memory), the large pixel buffer competes with the GPU for memory bandwidth
- The C++ side still calls `maybe_resize_bitmap()` which is a no-op when the image is already at target — but now it's working on a 442KB buffer instead of a decoded ~442KB JPEG (same actual work)

The bottleneck isn't the JPEG encode/decode — it's the GPU inference (~500ms). The preprocessing (including JPEG round-trip) is only ~30ms. Eliminating it saves at most a few milliseconds but the larger pixel buffer adds more overhead.

## Future Consideration

This may still be beneficial on discrete GPU systems (CUDA) where:
- System RAM → GPU transfers are the bottleneck (smaller JPEG might actually be better)
- The pixel buffer doesn't compete with GPU memory bandwidth
- But the temp file I/O on NVMe SSD is already very fast (~0.1ms)

The new C bridge functions (`_pixels` variants) were kept in bridge.h/bridge.cc as they may be useful for future work (e.g., batched inference where preprocessing happens once per batch).

## Proposed Changes

### Option A: Pass raw pixel data via CGo (recommended)
- Have libvips produce raw RGB pixels instead of JPEG
- Pass the pixel buffer + dimensions directly to a new C function
- Skip the temp file write/read/decode cycle entirely
- `mtmd_helper_bitmap_init_from_data()` or similar can accept raw pixels

### Option B: Let C++ handle everything from the original file
- Skip Go-side libvips preprocessing entirely
- Pass the original image path to the C++ bridge
- Let `mtmd_helper_bitmap_init_from_file()` + `maybe_resize_bitmap()` handle decoding and resizing
- Downside: loses libvips format support (WebP, AVIF) — stb_image only handles JPEG/PNG

Option A is preferred because it preserves libvips format support while eliminating the round-trip.

## Risks

- CGo memory management for passing pixel buffers requires careful handling (no GC, must free on the right side)
- Raw pixel data is larger than JPEG (384x384x3 = ~440KB vs ~30KB JPEG), but it's transient and avoids the encode+decode cost
- Must handle cleanup of the raw buffer on both happy and error paths

## Acceptance Criteria

- [ ] No temp file created during image preprocessing for the normal path
- [ ] Raw pixel data passed directly from Go/libvips to C/mtmd
- [ ] Per-image preprocessing time reduced (measure with existing timing instrumentation)
- [ ] All supported input formats still work (JPEG, PNG, WEBP, AVIF)
- [ ] Embedding quality unchanged
- [ ] Temp file cleanup still works for error paths

## Priority

High. Low-to-medium effort, removes unnecessary I/O and compute per image. Also simplifies the pipeline for issue 015 (pipelining).

## Benchmarking Protocol

### Baseline (before changes)
1. Use the same 100-image dataset from issue 011's benchmark.
2. Record per-job `embed` timing (which includes preprocessing) from worker logs.
3. Separately benchmark `preprocessImageForEmbeddingWithVipsgen` with 20 images to isolate preprocessing cost.

### After changes
1. Same dataset, same measurements.
2. Compare the `embed` stage timing (preprocessing + inference combined).
3. Verify no temp files are created in `data/tmp/` during normal processing.

### Success criteria
- Preprocessing time reduced (measured in isolation)
- Total `embed` stage time reduced
- No quality regression in embeddings or annotations

## Estimated Effort

Medium. Requires changes to the Go preprocessing function, a new C API in bridge.cc for accepting raw pixels, and CGo memory management. ~50-80 lines changed.
