# 042: Enforce Vector Dimensions and Share Cosine Logic

## Priority

P2

## Status

Completed.

## Summary

Production search paths should not silently compare vectors with different dimensions, and cosine similarity should live in one shared implementation.

## Context

- `internal/vectorindex/bruteforce/index.go:156` computes cosine over the shorter vector length.
- `internal/search/http.go:870` duplicates the same truncating cosine helper for transcript embeddings.
- `internal/search/http.go:754` reads transcript vectors without selecting or checking their stored dimension.
- Test packages also carry local cosine helpers.

## Risks

- Corrupt or stale vectors can rank incorrectly instead of being rejected or skipped.
- Different search surfaces can drift in how they handle zero vectors or mismatched lengths.
- Bugs in similarity math must be fixed in multiple places.

## Acceptance Criteria

- [x] Introduce a shared cosine helper that validates equal non-zero dimensions.
- [x] Use the shared helper in bruteforce image search and transcript search.
- [x] Select and verify transcript embedding dimensions before scoring.
- [x] Decide whether dimension mismatches should return errors or skip corrupt rows, and test that behavior.
- [x] Add contract coverage for vector index dimension behavior.

## Resolution

- Added `vectorindex.Cosine` as the single cosine implementation. It rejects empty and mismatched dimensions, while preserving the previous zero-norm behavior of returning similarity `0`.
- Bruteforce image search now returns an `ErrVectorDimensionMismatch`-wrapped error when stored vectors for a model do not match the query dimension, surfacing corrupt image embeddings loudly.
- Transcript search now selects `video_transcript_embeddings.dim`, skips rows where decoded vector length does not match stored `dim`, and skips rows whose dimension does not match the query vector.
- Added regression tests for shared cosine validation, vector-index contract dimension behavior, bruteforce mismatch errors, and transcript mismatch skipping.

## Related Files

- `internal/vectorindex/bruteforce/index.go`
- `internal/vectorindex/codec.go`
- `internal/vectorindex/contract_test.go`
- `internal/search/http.go`
- `internal/search/http_test.go`
