# 012: Eliminate Double Image Preprocessing Round-Trip

## Context

Currently every image goes through two separate processing stages that duplicate work:

1. **Go side** (`internal/embedder/llamacppnative/image_preprocess_vipsgen.go`): libvips thumbnail to target size (384px for embed, 1024px for annotate) + JPEG encode at quality 90 → writes temp file
2. **C++ side** (`bridge.cc`): `mtmd_helper_bitmap_init_from_file()` reads the temp file, decodes it via stb_image, then `maybe_resize_bitmap()` may resize again

For embedding at 384px, the Go side produces a 384px JPEG, then C++ reads it, decodes it, and `maybe_resize_bitmap` sees it's already at target and skips. The JPEG decode is wasted work. For annotation at 1024px, same pattern.

The full per-image overhead: one libvips resize + one JPEG encode + one temp file write + one temp file read + one stb_image JPEG decode + one potential resize.

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
